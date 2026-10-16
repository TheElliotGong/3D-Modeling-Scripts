"""
add_leaf_joints.py

Adds a small chain of child "leaf" joints to every terminal joint (a
joint with no joint children) found in the selected skeleton
hierarchy. Gives Blender's FBX importer a real head/tail pair to infer
bone length and direction from, instead of falling back to a default
that can come out wildly oversized. Can also remove everything it
previously added.

The new leaf joints:
    - Have their jointOrient and rotate zeroed, so their world
      orientation exactly matches their parent's (no re-aiming) --
      this holds true down the whole new chain, since each new joint
      is oriented relative to the previous one the same way.
    - Are offset along a chosen local axis by a customizable length,
      each one further down the chain than the last.
    - Let you choose how many extra joints to add per terminal (1 for
      a single end cap, more if you want a short chain).
    - Take a customizable display radius.
    - Are marked with a hidden custom attribute (not a name suffix),
      so the tool won't get confused by rigs (Mixamo and others) that
      already use "_end"-style naming for their real terminal joints,
      and so it knows exactly what it added and can remove it cleanly
      later.
    - Are NOT added to any skinCluster. They're purely structural;
      don't weight vertices to them.

Usage (Script Editor, Python tab):

    import add_leaf_joints
    add_leaf_joints.show_ui()

    # or, non-interactively:
    add_leaf_joints.add_leaf_joints(length=2.0, axis='y', radius=0.5, count=3)
    add_leaf_joints.remove_leaf_joints()
"""

import maya.cmds as cmds

MARKER_ATTR = 'addLeafJoint_generated'


def _is_generated_leaf(joint):
    """True if this joint was created by a previous run of this tool."""
    return cmds.attributeQuery(MARKER_ATTR, node=joint, exists=True)


def _collect_hierarchy_joints(root_joints):
    """Every joint under the given joints, roots included."""
    all_joints = set()
    for j in root_joints:
        if not cmds.objExists(j):
            continue
        if cmds.nodeType(j) == 'joint':
            all_joints.add(j)
        descendants = cmds.listRelatives(
            j, allDescendents=True, type='joint', fullPath=True
        ) or []
        all_joints.update(descendants)
    return all_joints


def _find_terminal_joints(root_joints):
    """Collect every joint with zero joint children under the given
    joints. Joints this tool created previously are skipped via their
    marker attribute -- not by name -- so it's safe to run more than
    once even on rigs whose native naming already uses an "_end"
    suffix."""

    terminals = []
    for j in _collect_hierarchy_joints(root_joints):
        if not cmds.objExists(j):
            continue
        if _is_generated_leaf(j):
            continue  # this joint IS a leaf we made -- don't chain off it
        children = cmds.listRelatives(j, children=True, type='joint', fullPath=True) or []
        if not children:
            terminals.append(j)

    return terminals


def _find_generated_leaves(root_joints):
    """Every joint under root_joints that this tool created previously."""
    return [
        j for j in _collect_hierarchy_joints(root_joints)
        if cmds.objExists(j) and _is_generated_leaf(j)
    ]


def _unique_name(base_name):
    """Return base_name, or base_name + an incrementing number if that
    name is already taken in the scene."""
    name = base_name
    i = 1
    while cmds.objExists(name):
        name = '%s%d' % (base_name, i)
        i += 1
    return name


def add_leaf_joints(joints=None, length=1.0, axis='x', radius=1.0, count=1):
    """
    Add a short chain of leaf joints to every terminal joint under the
    given selection.

    Args:
        joints (list[str] | None): Joints to search. Defaults to the
            current selection. The full hierarchy under each given
            joint is searched, not just the joints selected directly.
        length (float): Distance between each joint in the new chain,
            in current scene units.
        axis (str): Local axis to offset along -- 'x', 'y', 'z', or the
            negative variants '-x', '-y', '-z'.
        radius (float): Display radius for the new leaf joints. Purely
            visual (affects the joint's on-screen size, not skinning
            or export).
        count (int): How many extra joints to add per terminal. 1 adds
            a single end cap (the original behavior); higher values
            build a short straight chain, each one length units past
            the last, all holding the same orientation.

    Returns:
        list[str]: Full paths of newly created leaf joints (all of
        them, across every terminal joint processed).
    """
    axis_map = {
        'x': (length, 0.0, 0.0), '-x': (-length, 0.0, 0.0),
        'y': (0.0, length, 0.0), '-y': (0.0, -length, 0.0),
        'z': (0.0, 0.0, length), '-z': (0.0, 0.0, -length),
    }
    axis_key = axis.lower()
    if axis_key not in axis_map:
        cmds.warning('Invalid axis "%s" -- defaulting to x.' % axis)
        axis_key = 'x'

    count = int(count)
    if count < 1:
        cmds.warning('count must be at least 1 -- using 1.')
        count = 1

    if joints is None:
        joints = cmds.ls(selection=True) or []
    if not joints:
        cmds.warning('Select the root joint (or any joint(s)) of a skeleton first.')
        return []

    terminals = _find_terminal_joints(joints)
    if not terminals:
        cmds.warning(
            'No terminal joints found under the current selection. '
            'Check that you selected the actual joint hierarchy (not the mesh), '
            'and that it hasn\'t already been fully processed by this tool.'
        )
        return []

    new_joints = []
    for j in terminals:
        short_name = j.split('|')[-1]
        parent_joint = j

        for idx in range(count):
            # First new joint in the chain is named "<terminal>_end";
            # subsequent ones get "_end2", "_end3", etc.
            suffix = '_end' if idx == 0 else '_end%d' % (idx + 1)
            leaf_name = _unique_name(short_name + suffix)

            parent_pos = cmds.xform(parent_joint, query=True, worldSpace=True, translation=True)

            # Creating with a joint selected and an explicit world-space
            # position places the new joint coincident with the parent,
            # already parented under it.
            cmds.select(parent_joint, replace=True)
            new_joint = cmds.joint(position=parent_pos, name=leaf_name)

            # Maintain the parent's orientation exactly: zeroing jointOrient
            # and rotate means the leaf's local axes equal the parent's
            # world axes -- no re-aiming, no drift. Chaining this down each
            # new joint keeps the whole chain aligned with the original
            # terminal joint's orientation.
            cmds.setAttr(new_joint + '.jointOrient', 0, 0, 0, type='double3')
            cmds.setAttr(new_joint + '.rotate', 0, 0, 0, type='double3')

            # Translate is always measured in the parent's coordinate space,
            # so this correctly extends the bone along the parent's own
            # direction regardless of the leaf's now-zeroed orientation.
            cmds.setAttr(new_joint + '.translate', *axis_map[axis_key], type='double3')

            # Display radius only -- doesn't affect skinning, export, or
            # deformation, just how big the joint icon draws in the viewport.
            cmds.setAttr(new_joint + '.radius', radius)

            # Mark this joint as tool-generated so future runs recognize it
            # as a leaf (don't chain another leaf off it) without relying on
            # its name -- and so remove_leaf_joints() can find it later.
            cmds.addAttr(new_joint, longName=MARKER_ATTR, attributeType='bool', defaultValue=True)
            cmds.setAttr(new_joint + '.' + MARKER_ATTR, lock=True)

            new_joints.append(new_joint)
            parent_joint = new_joint  # next joint in the chain parents off this one

    cmds.select(new_joints, replace=True)
    print('Added %d leaf joint(s) across %d terminal(s):\n  %s' % (
        len(new_joints), len(terminals), '\n  '.join(new_joints)
    ))
    return new_joints


def remove_leaf_joints(joints=None):
    """
    Remove every leaf joint (or chain of leaf joints) previously added
    by add_leaf_joints() under the given selection. Only ever removes
    joints carrying this tool's marker attribute -- it will never
    touch a rig's real joints, even ones natively named "_end".

    Args:
        joints (list[str] | None): Joints to search. Defaults to the
            current selection. The full hierarchy under each given
            joint is searched, not just the joints selected directly.

    Returns:
        list[str]: Full paths of the chain-root joints that were
        deleted (deleting a chain root also removes any generated
        joints parented beneath it, so this list won't include those
        children even though they're gone too).
    """
    if joints is None:
        joints = cmds.ls(selection=True) or []
    if not joints:
        cmds.warning('Select the root joint (or any joint(s)) of a skeleton first.')
        return []

    generated = _find_generated_leaves(joints)
    if not generated:
        cmds.warning('No tool-generated leaf joints found under the current selection.')
        return []

    # Deleting a joint in Maya removes everything parented beneath it
    # too, so only delete the top of each generated chain -- trying to
    # delete already-generated children individually afterward would
    # just error out on missing nodes.
    generated_set = set(generated)
    chain_roots = []
    for j in generated:
        parent = cmds.listRelatives(j, parent=True, fullPath=True)
        parent = parent[0] if parent else None
        if parent not in generated_set:
            chain_roots.append(j)

    cmds.delete(chain_roots)
    print('Removed %d leaf joint chain(s) (%d joint(s) total).' % (
        len(chain_roots), len(generated)
    ))
    return chain_roots


def show_ui():
    """Small UI for adding or removing leaf joints."""
    window_name = 'addLeafJointsWin'
    if cmds.window(window_name, exists=True):
        cmds.deleteUI(window_name)

    cmds.window(window_name, title='Add Leaf Joints', sizeable=False)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=6, columnAttach=('both', 10))

    cmds.text(label=' ')
    cmds.text(
        label='Select the root joint (or any joint(s)) of a skeleton, then run.',
        align='left'
    )
    cmds.text(label=' ')

    length_field = cmds.floatSliderGrp(
        label='Leaf Length', field=True, minValue=0.01, maxValue=50.0,
        value=1.0, step=0.01, fieldMinValue=0.001, fieldMaxValue=1000.0
    )

    radius_field = cmds.floatSliderGrp(
        label='Leaf Radius', field=True, minValue=0.01, maxValue=20.0,
        value=1.0, step=0.01, fieldMinValue=0.001, fieldMaxValue=1000.0
    )

    count_field = cmds.intSliderGrp(
        label='Joints Per Leaf', field=True, minValue=1, maxValue=20,
        value=1, fieldMinValue=1, fieldMaxValue=100
    )

    axis_menu = cmds.optionMenu(label='Axis')
    for a in ('x', 'y', 'z', '-x', '-y', '-z'):
        cmds.menuItem(label=a)

    cmds.text(label=' ')

    def add_button_cmd(*_args):
        length_val = cmds.floatSliderGrp(length_field, query=True, value=True)
        radius_val = cmds.floatSliderGrp(radius_field, query=True, value=True)
        count_val = cmds.intSliderGrp(count_field, query=True, value=True)
        axis_val = cmds.optionMenu(axis_menu, query=True, value=True)
        add_leaf_joints(length=length_val, axis=axis_val, radius=radius_val, count=count_val)

    def remove_button_cmd(*_args):
        remove_leaf_joints()

    cmds.button(label='Add Leaf Joints', height=32, command=add_button_cmd)
    cmds.text(label=' ')
    cmds.button(label='Remove Leaf Joints', height=32, backgroundColor=(0.6, 0.3, 0.3), command=remove_button_cmd)
    cmds.text(label=' ')

    cmds.showWindow(window_name)


if __name__ == '__main__':
    show_ui()
