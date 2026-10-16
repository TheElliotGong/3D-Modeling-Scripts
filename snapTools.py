"""
snapTools.py  --  Blender-style 3D Cursor + Snap menu for Autodesk Maya 2026
================================================================================

Replicates Blender's Shift+S snap pie menu (Cursor to Selection, Selection to
Cursor, Snap to Grid, etc.) inside Maya, including a persistent 3D cursor.

INSTALL
-------
1. Drop this file in your Maya scripts folder:
       Windows : C:/Users/<you>/Documents/maya/2026/scripts
       macOS   : ~/Library/Preferences/Autodesk/maya/2026/scripts
       Linux   : ~/maya/2026/scripts

2. In the Script Editor (Python tab) run:
       import snapTools
       snapTools.install()

   install() creates a custom hotkey set (Maya's default set is locked),
   binds Shift+S, and drops a shelf button on the active shelf.

   To remove:  snapTools.uninstall()
   To just open the menu without a hotkey:  snapTools.show_menu()

NOTES
-----
* Works with objects, vertices, edges, faces and NURBS CVs.
* The 3D cursor is a locator called "snapCursor" placed in a reference
  display layer so it is visible but never selectable. Its position also
  lives in an optionVar, so it survives scene changes and Maya restarts.
* Every operation is wrapped in a single undo chunk.
"""

import math
import functools

import maya.cmds as cmds
import maya.mel as mel
import maya.api.OpenMaya as om
from maya import OpenMayaUI as omui

# --- Qt (Maya 2025+ ships PySide6, older ships PySide2) ----------------------
try:
    from PySide6 import QtWidgets, QtGui, QtCore
    from shiboken6 import wrapInstance
except ImportError:                                        # pragma: no cover
    from PySide2 import QtWidgets, QtGui, QtCore
    from shiboken2 import wrapInstance


# =============================================================================
# Constants / settings
# =============================================================================

CURSOR_NAME = "snapCursor"
CURSOR_LAYER = "snapCursor_layer"

OPTVAR_POS = "snapToolsCursorPos"
OPTVAR_ROT = "snapToolsCursorRot"
OPTVAR_MODE = "snapToolsPivotMode"

RTC_NAME = "snapToolsMenu"
NAME_COMMAND = "snapToolsMenuNameCommand"
HOTKEY_SET = "snapTools"

# How the "center" of a multi-component / multi-object selection is computed.
#   median  -> average of all points (Blender's default)
#   bbox    -> center of the bounding box
#   active  -> the last-selected element only
PIVOT_MODES = ("median", "bbox", "active")


def get_pivot_mode():
    if cmds.optionVar(exists=OPTVAR_MODE):
        mode = cmds.optionVar(q=OPTVAR_MODE)
        if mode in PIVOT_MODES:
            return mode
    return "median"


def set_pivot_mode(mode):
    if mode in PIVOT_MODES:
        cmds.optionVar(sv=(OPTVAR_MODE, mode))
        _msg("Snap pivot mode: <hl>%s</hl>" % mode)


# =============================================================================
# Small utilities
# =============================================================================

def undoable(func):
    """Collapse everything the function does into one undo step."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cmds.undoInfo(openChunk=True, chunkName=func.__name__)
        try:
            return func(*args, **kwargs)
        finally:
            cmds.undoInfo(closeChunk=True)
    return wrapper


def preserve_selection(func):
    """Restore the user's selection after the function runs."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sel = cmds.ls(sl=True, long=True) or []
        try:
            return func(*args, **kwargs)
        finally:
            if sel:
                cmds.select(sel, r=True, ne=True)
            else:
                cmds.select(clear=True)
    return wrapper


def _msg(text):
    """Non-blocking heads-up display message in the viewport."""
    try:
        cmds.inViewMessage(amg=text, pos="midCenterTop", fade=True, fadeStayTime=900)
    except Exception:
        pass


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


# =============================================================================
# Selection parsing
# =============================================================================

def _split_selection():
    """Return (transform long names, component names) from the selection."""
    sel = cmds.ls(sl=True, fl=True) or []
    comps = [s for s in sel if "." in s]
    nodes = [s for s in sel if "." not in s]

    xforms = []
    for n in nodes:
        if cmds.objectType(n, isAType="transform"):
            xforms.append(n)
        else:
            # a shape was selected directly -> use its transform
            parent = cmds.listRelatives(n, parent=True, fullPath=True)
            if parent:
                xforms.append(parent[0])

    xforms = cmds.ls(xforms, long=True) or []
    # de-duplicate, preserve order
    seen, unique = set(), []
    for x in xforms:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique, comps


def _filter_descendants(long_names):
    """Drop objects whose ancestor is also selected (avoids double moves)."""
    out = []
    for n in long_names:
        if not any(n != o and n.startswith(o + "|") for o in long_names):
            out.append(n)
    return out


def _is_poly_component(c):
    return any(tag in c for tag in (".vtx[", ".e[", ".f[", ".vtxFace["))


def _expand_components(comps):
    """Convert poly components to a flat vertex list; pass others through."""
    poly = [c for c in comps if _is_poly_component(c)]
    other = [c for c in comps if not _is_poly_component(c)]
    result = list(other)
    if poly:
        verts = cmds.polyListComponentConversion(poly, toVertex=True) or []
        result += cmds.ls(verts, fl=True) or []
    return result


def _points_from_components(comps):
    pts = []
    for c in _expand_components(comps):
        try:
            if ".vtx[" in c or ".cv[" in c:
                pts.append(list(cmds.pointPosition(c, w=True)))
            else:
                vals = cmds.xform(c, q=True, ws=True, t=True) or []
                for i in range(0, len(vals) - 2, 3):
                    pts.append(list(vals[i:i + 3]))
        except (RuntimeError, ValueError):
            continue
    return pts


def selection_points():
    """World-space points representing everything currently selected."""
    xforms, comps = _split_selection()
    pts = _points_from_components(comps)
    for x in xforms:
        pts.append(list(cmds.xform(x, q=True, ws=True, rp=True)))
    return pts


def _median(pts):
    n = float(len(pts))
    return [sum(p[i] for p in pts) / n for i in range(3)]


def _bbox_center(pts):
    return [(min(p[i] for p in pts) + max(p[i] for p in pts)) * 0.5 for i in range(3)]


def active_point():
    """Position of the last-selected element (Blender's 'active' element)."""
    sel = cmds.ls(sl=True) or []
    if not sel:
        return None
    last = cmds.ls(sel[-1], fl=True)[-1]
    if "." in last:
        pts = _points_from_components([last])
        return _median(pts) if pts else None
    node = last
    if not cmds.objectType(node, isAType="transform"):
        parent = cmds.listRelatives(node, parent=True, fullPath=True)
        node = parent[0] if parent else node
    return list(cmds.xform(node, q=True, ws=True, rp=True))


def selection_center(mode=None):
    """Center of the current selection using the active pivot mode."""
    mode = mode or get_pivot_mode()
    if mode == "active":
        return active_point()
    pts = selection_points()
    if not pts:
        return None
    return _bbox_center(pts) if mode == "bbox" else _median(pts)


# =============================================================================
# Orientation from the selection (face normal / edge direction)
# =============================================================================

def _euler_from_normal(normal):
    """Build XYZ euler degrees that aims +Y down the given world normal."""
    n = om.MVector(normal).normalize()
    ref = om.MVector(0.0, 0.0, 1.0) if abs(n * om.MVector(0.0, 1.0, 0.0)) > 0.999 \
        else om.MVector(0.0, 1.0, 0.0)
    x_axis = (ref ^ n).normalize()
    z_axis = (n ^ x_axis).normalize()
    m = om.MMatrix([x_axis.x, x_axis.y, x_axis.z, 0.0,
                    n.x, n.y, n.z, 0.0,
                    z_axis.x, z_axis.y, z_axis.z, 0.0,
                    0.0, 0.0, 0.0, 1.0])
    rot = om.MTransformationMatrix(m).rotation()
    return [math.degrees(v) for v in (rot.x, rot.y, rot.z)]


def selection_orientation():
    """Average orientation of selected faces (+Y = normal). None if N/A."""
    sel = cmds.ls(sl=True, fl=True) or []
    faces = [s for s in sel if ".f[" in s]
    if not faces:
        return None

    total = om.MVector(0.0, 0.0, 0.0)
    for face in faces:
        try:
            obj, idx = face.split(".f[")
            idx = int(idx.rstrip("]"))
            mlist = om.MSelectionList()
            mlist.add(obj)
            mesh = om.MFnMesh(mlist.getDagPath(0))
            total += mesh.getPolygonNormal(idx, om.MSpace.kWorld)
        except Exception:
            continue

    if total.length() < 1e-6:
        return None
    return _euler_from_normal(total)


# =============================================================================
# The 3D cursor
# =============================================================================

def cursor_position():
    if cmds.objExists(CURSOR_NAME):
        return list(cmds.xform(CURSOR_NAME, q=True, ws=True, t=True))
    if cmds.optionVar(exists=OPTVAR_POS):
        vals = cmds.optionVar(q=OPTVAR_POS)
        if vals and len(vals) == 3:
            return [float(v) for v in vals]
    return [0.0, 0.0, 0.0]


def cursor_rotation():
    if cmds.objExists(CURSOR_NAME):
        return list(cmds.xform(CURSOR_NAME, q=True, ws=True, ro=True))
    if cmds.optionVar(exists=OPTVAR_ROT):
        vals = cmds.optionVar(q=OPTVAR_ROT)
        if vals and len(vals) == 3:
            return [float(v) for v in vals]
    return [0.0, 0.0, 0.0]


def _store_optionvar_vector(name, vec):
    cmds.optionVar(clearArray=name)
    for v in vec:
        cmds.optionVar(floatValueAppend=(name, float(v)))


@preserve_selection
def create_cursor():
    """Create (or return) the cursor locator, unselectable and non-renderable."""
    if cmds.objExists(CURSOR_NAME):
        return CURSOR_NAME

    loc = cmds.spaceLocator(name=CURSOR_NAME)[0]
    shape = cmds.listRelatives(loc, shapes=True, fullPath=True)[0]

    for axis in "XYZ":
        cmds.setAttr("%s.localScale%s" % (shape, axis), 1.0)
    cmds.setAttr(shape + ".overrideEnabled", 1)
    cmds.setAttr(shape + ".overrideColor", 17)          # yellow
    try:
        cmds.setAttr(loc + ".hiddenInOutliner", True)
    except Exception:
        pass

    if not cmds.objExists(CURSOR_LAYER):
        cmds.createDisplayLayer(name=CURSOR_LAYER, empty=True)
        cmds.setAttr(CURSOR_LAYER + ".displayType", 2)   # reference = unselectable
    cmds.editDisplayLayerMembers(CURSOR_LAYER, loc, noRecurse=True)

    pos, rot = cursor_position(), cursor_rotation()
    cmds.xform(loc, ws=True, t=pos, ro=rot)
    return loc


def set_cursor(pos, rot=None):
    _store_optionvar_vector(OPTVAR_POS, pos)
    if rot is not None:
        _store_optionvar_vector(OPTVAR_ROT, rot)
    loc = create_cursor()
    cmds.xform(loc, ws=True, t=pos)
    if rot is not None:
        cmds.xform(loc, ws=True, ro=rot)


def toggle_cursor_visibility():
    if not cmds.objExists(CURSOR_NAME):
        create_cursor()
        _msg("3D cursor <hl>shown</hl>")
        return
    vis = cmds.getAttr(CURSOR_NAME + ".visibility")
    cmds.setAttr(CURSOR_NAME + ".visibility", not vis)
    _msg("3D cursor <hl>%s</hl>" % ("hidden" if vis else "shown"))


def delete_cursor():
    if cmds.objExists(CURSOR_NAME):
        cmds.delete(CURSOR_NAME)
    if cmds.objExists(CURSOR_LAYER):
        cmds.delete(CURSOR_LAYER)


# =============================================================================
# Grid helpers
# =============================================================================

def grid_step():
    try:
        spacing = float(cmds.grid(q=True, spacing=True))
        divisions = float(cmds.grid(q=True, divisions=True))
        step = spacing / divisions if divisions else spacing
    except Exception:
        step = 1.0
    return step if step > 1e-9 else 1.0


def _snapped(point, step):
    return [round(v / step) * step for v in point]


# =============================================================================
# Move helpers
# =============================================================================

def _move_objects_to(target, objs, orient=None):
    for o in _filter_descendants(objs):
        try:
            rp = cmds.xform(o, q=True, ws=True, rp=True)
            cmds.xform(o, ws=True, r=True, t=_sub(target, rp))
            if orient is not None:
                cmds.xform(o, ws=True, ro=orient)
        except RuntimeError:
            cmds.warning("Could not move '%s' (locked or connected channels)." % o)


def _move_relative(delta, objs, comps):
    for o in _filter_descendants(objs):
        try:
            cmds.xform(o, ws=True, r=True, t=delta)
        except RuntimeError:
            cmds.warning("Could not move '%s' (locked or connected channels)." % o)
    if comps:
        cmds.xform(comps, ws=True, r=True, t=delta)


def _collapse_components_to(target, comps):
    verts = _expand_components(comps)
    if verts:
        cmds.xform(verts, ws=True, a=True, t=target)


# =============================================================================
# Public snap operations
# =============================================================================

@undoable
def cursor_to_selection(align=False):
    center = selection_center()
    if center is None:
        _msg("Nothing selected")
        return
    orient = selection_orientation() if align else None
    set_cursor(center, orient)
    _msg("Cursor to selection")


@undoable
def cursor_to_active():
    pos = active_point()
    if pos is None:
        _msg("Nothing selected")
        return
    set_cursor(pos)
    _msg("Cursor to active element")


@undoable
def cursor_to_world_origin():
    set_cursor([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    _msg("Cursor to world origin")


@undoable
def cursor_to_grid():
    set_cursor(_snapped(cursor_position(), grid_step()))
    _msg("Cursor to grid")


@undoable
@preserve_selection
def selection_to_cursor(keep_offset=False, orient=False):
    objs, comps = _split_selection()
    if not objs and not comps:
        _msg("Nothing selected")
        return

    target = cursor_position()
    rot = cursor_rotation() if orient else None

    if keep_offset:
        center = selection_center()
        if center is None:
            return
        _move_relative(_sub(target, center), objs, comps)
        if rot is not None:
            for o in _filter_descendants(objs):
                cmds.xform(o, ws=True, ro=rot)
    else:
        _move_objects_to(target, objs, rot)
        _collapse_components_to(target, comps)

    _msg("Selection to cursor")


@undoable
@preserve_selection
def selection_to_active(keep_offset=True):
    """Snap everything else onto the last-selected element."""
    sel = cmds.ls(sl=True) or []
    if len(sel) < 2:
        _msg("Select at least two things")
        return

    target = active_point()
    if target is None:
        return

    last = cmds.ls(sel[-1], fl=True)[-1]
    rest = cmds.ls(sel[:-1], fl=True) or []
    cmds.select(rest, r=True)

    objs, comps = _split_selection()
    if keep_offset:
        center = selection_center()
        if center is not None:
            _move_relative(_sub(target, center), objs, comps)
    else:
        _move_objects_to(target, objs)
        _collapse_components_to(target, comps)

    cmds.select(rest + [last], r=True)
    _msg("Selection to active")


@undoable
@preserve_selection
def selection_to_grid():
    objs, comps = _split_selection()
    if not objs and not comps:
        _msg("Nothing selected")
        return

    step = grid_step()
    for o in _filter_descendants(objs):
        rp = cmds.xform(o, q=True, ws=True, rp=True)
        try:
            cmds.xform(o, ws=True, r=True, t=_sub(_snapped(rp, step), rp))
        except RuntimeError:
            cmds.warning("Could not move '%s' (locked or connected channels)." % o)

    for v in _expand_components(comps):
        try:
            p = cmds.pointPosition(v, w=True)
            cmds.xform(v, ws=True, a=True, t=_snapped(p, step))
        except (RuntimeError, ValueError):
            continue

    _msg("Selection to grid (step %.4g)" % step)


@undoable
@preserve_selection
def selection_to_world_origin(keep_offset=False):
    objs, comps = _split_selection()
    if not objs and not comps:
        _msg("Nothing selected")
        return
    if keep_offset:
        center = selection_center()
        if center is not None:
            _move_relative([-center[0], -center[1], -center[2]], objs, comps)
    else:
        _move_objects_to([0.0, 0.0, 0.0], objs)
        _collapse_components_to([0.0, 0.0, 0.0], comps)
    _msg("Selection to world origin")


@undoable
@preserve_selection
def pivot_to_cursor():
    """Maya extra: move the pivot of selected objects without moving them."""
    objs, _ = _split_selection()
    if not objs:
        _msg("Select one or more objects")
        return
    target = cursor_position()
    for o in objs:
        cmds.xform(o, ws=True, piv=target)
    _msg("Pivot to cursor")


@undoable
@preserve_selection
def pivot_to_selection():
    """Maya extra: move an object's pivot to the center of selected components."""
    center = selection_center()
    if center is None:
        _msg("Nothing selected")
        return
    objs, comps = _split_selection()
    targets = list(objs)
    for c in comps:
        node = c.split(".")[0]
        parent = cmds.ls(node, long=True)
        if parent and parent[0] not in targets:
            targets.append(parent[0])
    for o in targets:
        cmds.xform(o, ws=True, piv=center)
    _msg("Pivot to selection center")


# =============================================================================
# Menu UI
# =============================================================================

def _maya_window():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget) if ptr else None


def _add(menu, label, callback, enabled=True):
    action = menu.addAction(label)
    action.setEnabled(enabled)
    action.triggered.connect(callback)
    return action


def show_menu():
    """Pop the snap menu at the mouse cursor. Bound to Shift+S by install()."""
    parent = _maya_window()
    menu = QtWidgets.QMenu(parent)
    menu.setSeparatorsCollapsible(False)
    menu.setToolTipsVisible(True)

    has_sel = bool(cmds.ls(sl=True))
    multi = len(cmds.ls(sl=True) or []) > 1

    header = menu.addAction("Snap")
    header.setEnabled(False)
    menu.addSeparator()

    _add(menu, "&1  Selection to Cursor",
         lambda: selection_to_cursor(False), has_sel)
    _add(menu, "&2  Selection to Cursor (Keep Offset)",
         lambda: selection_to_cursor(True), has_sel)
    _add(menu, "&3  Selection to Cursor (Keep Offset + Orient)",
         lambda: selection_to_cursor(True, True), has_sel)
    _add(menu, "&4  Selection to Grid",
         selection_to_grid, has_sel)
    _add(menu, "&5  Selection to Active",
         lambda: selection_to_active(True), multi)
    _add(menu, "&6  Selection to World Origin",
         lambda: selection_to_world_origin(True), has_sel)

    menu.addSeparator()

    _add(menu, "&7  Cursor to Selection",
         lambda: cursor_to_selection(False), has_sel)
    _add(menu, "&8  Cursor to Selection (Align to Normal)",
         lambda: cursor_to_selection(True), has_sel)
    _add(menu, "&9  Cursor to Active",
         cursor_to_active, has_sel)
    _add(menu, "&0  Cursor to World Origin", cursor_to_world_origin)
    _add(menu, "Cursor to &Grid", cursor_to_grid)

    menu.addSeparator()

    _add(menu, "&Pivot to Cursor", pivot_to_cursor, has_sel)
    _add(menu, "Pivot to Selection Center", pivot_to_selection, has_sel)

    menu.addSeparator()

    options = menu.addMenu("Options")
    group = QtWidgets.QActionGroup(options)
    current = get_pivot_mode()
    labels = {"median": "Median Point", "bbox": "Bounding Box Center",
              "active": "Active Element"}
    for mode in PIVOT_MODES:
        act = options.addAction(labels[mode])
        act.setCheckable(True)
        act.setChecked(mode == current)
        act.triggered.connect(functools.partial(set_pivot_mode, mode))
        group.addAction(act)

    options.addSeparator()
    _add(options, "Toggle Cursor Visibility", toggle_cursor_visibility)
    _add(options, "Show Cursor", lambda: create_cursor())
    _add(options, "Delete Cursor", delete_cursor)
    _add(options, "Grid step: %.4g" % grid_step(), lambda: None, False)

    # Center the menu on the mouse, pie-menu style.
    menu.adjustSize()
    size = menu.sizeHint()
    pos = QtGui.QCursor.pos() - QtCore.QPoint(int(size.width() / 2),
                                              int(size.height() / 2))
    executor = getattr(menu, "exec", None) or getattr(menu, "exec_")
    executor(pos)


# =============================================================================
# Install / uninstall
# =============================================================================

def _ensure_runtime_command():
    if cmds.runTimeCommand(RTC_NAME, exists=True):
        cmds.runTimeCommand(RTC_NAME, e=True, delete=True)
    cmds.runTimeCommand(
        RTC_NAME,
        annotation="Blender-style snap menu (3D cursor)",
        category="Custom Scripts",
        command="import snapTools\nsnapTools.show_menu()",
        commandLanguage="python",
    )
    cmds.nameCommand(
        NAME_COMMAND,
        annotation="Open the snapTools snap menu",
        command=RTC_NAME,
        sourceType="mel",
    )


def _ensure_hotkey_set():
    current = cmds.hotkeySet(q=True, current=True)
    if current != "Maya_Default":
        return current
    if not cmds.hotkeySet(HOTKEY_SET, exists=True):
        cmds.hotkeySet(HOTKEY_SET, source="Maya_Default", current=True)
    else:
        cmds.hotkeySet(HOTKEY_SET, e=True, current=True)
    return HOTKEY_SET


def add_shelf_button():
    try:
        shelf_top = mel.eval("$tmpVar = $gShelfTopLevel")
        current_shelf = cmds.tabLayout(shelf_top, q=True, selectTab=True)
        cmds.shelfButton(
            parent=current_shelf,
            label="Snap Menu",
            annotation="Blender-style snap menu (Shift+S)",
            image="pythonFamily.png",
            imageOverlayLabel="SNAP",
            command="import snapTools\nsnapTools.show_menu()",
            sourceType="python",
        )
        return True
    except Exception as exc:
        cmds.warning("Could not create shelf button: %s" % exc)
        return False


def install(key="s", shift=True, ctrl=False, alt=False, shelf=True):
    """Bind the menu to a hotkey (default Shift+S) and add a shelf button."""
    _ensure_runtime_command()
    hotkey_set = _ensure_hotkey_set()

    existing = cmds.hotkey(key, query=True, shiftModifier=shift,
                           ctrlModifier=ctrl, altModifier=alt, name=True)
    cmds.hotkey(k=key, sht=shift, ctl=ctrl, alt=alt, name=NAME_COMMAND)
    cmds.savePrefs(hotkeys=True)
    create_cursor()

    combo = "%s%s%s%s" % ("Ctrl+" if ctrl else "", "Alt+" if alt else "",
                          "Shift+" if shift else "", key.upper())
    print("snapTools installed on %s (hotkey set: %s)" % (combo, hotkey_set))
    if existing:
        print("  ...replaced previous command: %s" % existing)
    if shelf:
        add_shelf_button()


def uninstall(key="s", shift=True, ctrl=False, alt=False):
    """Clear the hotkey and remove the cursor. Leaves the hotkey set intact."""
    try:
        cmds.hotkey(k=key, sht=shift, ctl=ctrl, alt=alt, name="")
        cmds.savePrefs(hotkeys=True)
    except Exception as exc:
        cmds.warning("Could not clear hotkey: %s" % exc)
    delete_cursor()
    print("snapTools uninstalled.")
