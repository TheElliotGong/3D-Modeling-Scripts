"""
select_faces_by_material.py

Select all faces on a SINGLE mesh that are assigned to a specific material.

Why this is not a one-liner:
    Maya stores assignments on shadingEngine (shading group) sets. A mesh can be
    a member either as a whole object ("pCube1") or as face components
    ("pCubeShape1.f[0:5]"). This handles both, plus instances and namespaces,
    and ignores every other mesh in the scene.

Usage (Script Editor -> Python tab):

    import sys
    sys.path.append(r"C:/path/to/folder")   # folder containing this file
    import select_faces_by_material as sfbm

    sfbm.show_ui()                                  # small window
    sfbm.list_materials("pCube1")                   # what's on this mesh?
    sfbm.select_faces_by_material("pCube1", "lambert2")
    faces = sfbm.get_faces_by_material("pCube1", "lambert2")   # no selection change

If you edit the file, reload it:
    try: reload(sfbm)
    except NameError: from importlib import reload; reload(sfbm)
"""

import maya.cmds as cmds

WIN = "sfbmWindow"
MENU = "sfbmMaterialMenu"
FIELD = "sfbmMeshField"


# ---------------------------------------------------------------- helpers

def _get_shape(obj):
    """Return the long name of the mesh shape for a transform, shape, or component."""
    node = obj.split(".")[0]
    if not cmds.objExists(node):
        raise ValueError("Object does not exist: {0}".format(node))

    if cmds.nodeType(node) == "mesh":
        return cmds.ls(node, long=True)[0]

    shapes = cmds.listRelatives(node, shapes=True, noIntermediate=True,
                               fullPath=True) or []
    shapes = [s for s in shapes if cmds.nodeType(s) == "mesh"]
    if not shapes:
        raise ValueError("No mesh shape found under: {0}".format(node))
    return shapes[0]


def _uuid(node):
    """Node UUID. Instances share a shape node, so this compares identity,
    not DAG path — safer than string matching full paths."""
    ids = cmds.ls(node, uuid=True)
    return ids[0] if ids else None


def _shading_engines(material):
    """Shading groups driven by this material (or the SG itself, if given one)."""
    if not cmds.objExists(material):
        raise ValueError("Material does not exist: {0}".format(material))
    if cmds.nodeType(material) == "shadingEngine":
        return [material]

    sgs = cmds.listConnections(material, type="shadingEngine",
                               source=False, destination=True) or []
    if not sgs:
        raise ValueError(
            "'{0}' is not connected to any shading group. Is it a material?".format(material))
    return list(set(sgs))


# ---------------------------------------------------------------- core

def get_faces_by_material(mesh, material):
    """Return a flat list of face components on `mesh` assigned to `material`.

    Does not modify the selection. Returns [] if nothing matches.
    """
    shape = _get_shape(mesh)
    shape_id = _uuid(shape)
    face_count = cmds.polyEvaluate(shape, face=True)
    if not isinstance(face_count, int):
        raise ValueError("Could not evaluate faces on: {0}".format(shape))

    faces = []
    for sg in _shading_engines(material):
        for member in cmds.sets(sg, query=True) or []:
            node = member.split(".")[0]
            if not cmds.objExists(node):
                continue
            try:
                member_shape = _get_shape(node)
            except ValueError:
                continue
            if _uuid(member_shape) != shape_id:
                continue   # different mesh, skip

            if "." in member:
                comp = member.split(".", 1)[1]
                if comp.startswith("f["):
                    faces.append("{0}.{1}".format(shape, comp))
                else:
                    # vtx/edge/map assignment (rare) -> convert to faces
                    converted = cmds.polyListComponentConversion(
                        "{0}.{1}".format(shape, comp), toFace=True) or []
                    faces.extend(converted)
            else:
                # whole-object assignment: every face belongs to this material
                faces.append("{0}.f[0:{1}]".format(shape, face_count - 1))

    return cmds.ls(faces, flatten=True) if faces else []


def select_faces_by_material(mesh, material, add=False):
    """Select the matching faces. Returns the list that was selected."""
    faces = get_faces_by_material(mesh, material)
    if not faces:
        cmds.warning("No faces on '{0}' are assigned to '{1}'.".format(mesh, material))
        return []

    if add:
        cmds.select(faces, add=True)
    else:
        cmds.select(faces, replace=True)
    cmds.selectMode(component=True)
    cmds.selectType(facet=True)
    print("Selected {0} face(s) using '{1}'.".format(len(faces), material))
    return faces


def list_materials(mesh, verbose=True):
    """Return the materials assigned to `mesh`, with face counts."""
    shape = _get_shape(mesh)
    shape_id = _uuid(shape)

    materials = []
    for sg in cmds.ls(type="shadingEngine") or []:
        hit = False
        for member in cmds.sets(sg, query=True) or []:
            node = member.split(".")[0]
            if not cmds.objExists(node):
                continue
            try:
                if _uuid(_get_shape(node)) == shape_id:
                    hit = True
                    break
            except ValueError:
                continue
        if not hit:
            continue

        mats = cmds.ls(cmds.listConnections("{0}.surfaceShader".format(sg)) or [],
                       materials=True) or []
        for mat in mats:
            if mat not in materials:
                materials.append(mat)

    if verbose:
        print("Materials on {0}:".format(shape))
        for mat in materials:
            print("   {0}  ({1} faces)".format(mat, len(get_faces_by_material(shape, mat))))
    return materials


# ---------------------------------------------------------------- UI

def _selected_mesh():
    sel = cmds.ls(selection=True, objectsOnly=True, long=True) or []
    for node in sel:
        try:
            return _get_shape(node)
        except ValueError:
            continue
    return None


def _refresh(*_):
    mesh = _selected_mesh()
    if not mesh:
        cmds.warning("Select a mesh in the viewport, then hit Refresh.")
        return

    cmds.textField(FIELD, edit=True, text=mesh.split("|")[-1])
    for item in cmds.optionMenu(MENU, query=True, itemListLong=True) or []:
        cmds.deleteUI(item)

    materials = list_materials(mesh, verbose=False)
    if not materials:
        cmds.menuItem(label="<no materials found>", parent=MENU)
        return
    for mat in materials:
        cmds.menuItem(label=mat, parent=MENU)


def _do_select(*_):
    mesh = _selected_mesh() or cmds.textField(FIELD, query=True, text=True)
    if not mesh:
        cmds.warning("No mesh. Select one and hit Refresh.")
        return

    material = cmds.optionMenu(MENU, query=True, value=True)
    if not material or material.startswith("<"):
        cmds.warning("Pick a material from the list.")
        return

    select_faces_by_material(mesh, material)


def show_ui():
    if cmds.window(WIN, exists=True):
        cmds.deleteUI(WIN)

    cmds.window(WIN, title="Select Faces By Material", widthHeight=(320, 150))
    cmds.columnLayout(adjustableColumn=True, rowSpacing=6, columnOffset=("both", 10))
    cmds.text(label="Select a mesh, Refresh, pick a material, Select Faces.",
              align="left", height=24)
    cmds.textField(FIELD, editable=False, placeholderText="(no mesh)")
    cmds.optionMenu(MENU, label="Material")
    cmds.button(label="Refresh From Selection", height=28, command=_refresh)
    cmds.button(label="Select Faces", height=32, command=_do_select)
    cmds.showWindow(WIN)
    _refresh()
