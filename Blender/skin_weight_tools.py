"""
Skin Weight Tools - Maya-style skin weight editing for Blender.

Brings the parts of Maya's Paint Skin Weights / Component Editor workflow that
Blender doesn't ship with: a lockable influence list, a numeric weight table,
Hammer Weights, Prune, spatial Mirror with name flipping, and closest-point
weight transfer between meshes.
"""

bl_info = {
    "name": "Skin Weight Tools",
    "author": "Elliot",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar (N) > Skin Weights",
    "description": "Maya-style skin weight editing: influence locks, component editor, hammer, prune, mirror, transfer",
    "category": "Rigging",
}

import bpy
import bmesh
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, IntProperty,
    StringProperty, PointerProperty, CollectionProperty,
)
from bpy.types import Operator, Panel, PropertyGroup, UIList
from mathutils.kdtree import KDTree
from mathutils.interpolate import poly_3d_calc


EPS = 1e-6
_SUSPEND_SYNC = False


# ---------------------------------------------------------------------------
# Core data access
# ---------------------------------------------------------------------------

class WMap:
    """Unified vertex-weight access across Object, Weight Paint and Edit mode.

    Edit mode goes through BMesh deform layers; the other modes read and write
    the Mesh datablock directly. Nothing here toggles modes, which keeps the
    undo stack and the user's selection intact.
    """

    def __init__(self, obj):
        self.obj = obj
        self.mesh = obj.data
        self.mode = obj.mode
        self.is_edit = (self.mode == 'EDIT')
        self.bm = None
        self.dl = None
        if self.is_edit:
            self.bm = bmesh.from_edit_mesh(self.mesh)
            self.dl = self.bm.verts.layers.deform.verify()
            self.bm.verts.ensure_lookup_table()

    # -- topology / selection ------------------------------------------------

    def vert_count(self):
        return len(self.bm.verts) if self.is_edit else len(self.mesh.vertices)

    def all_indices(self):
        return range(self.vert_count())

    def co(self, vi):
        return self.bm.verts[vi].co if self.is_edit else self.mesh.vertices[vi].co

    def target_indices(self, require_selection=False):
        """Vertices an operation should affect, following Maya's 'selection or
        everything' convention."""
        if self.is_edit:
            sel = [v.index for v in self.bm.verts if v.select]
            if sel or require_selection:
                return sel
            return list(self.all_indices())

        masked = self.mesh.use_paint_mask_vertex or self.mesh.use_paint_mask
        if self.mode == 'WEIGHT_PAINT' and not masked:
            return list(self.all_indices())

        sel = [v.index for v in self.mesh.vertices if v.select]
        if sel or require_selection:
            return sel
        return list(self.all_indices())

    def active_index(self):
        """Index of the active vertex, or the first selected one."""
        if self.is_edit:
            for elem in reversed(self.bm.select_history):
                if isinstance(elem, bmesh.types.BMVert):
                    return elem.index
            for v in self.bm.verts:
                if v.select:
                    return v.index
            return -1
        for v in self.mesh.vertices:
            if v.select:
                return v.index
        return -1

    def adjacency(self):
        adj = [set() for _ in range(self.vert_count())]
        if self.is_edit:
            for e in self.bm.edges:
                a, b = e.verts[0].index, e.verts[1].index
                adj[a].add(b)
                adj[b].add(a)
        else:
            for e in self.mesh.edges:
                a, b = e.vertices
                adj[a].add(b)
                adj[b].add(a)
        return adj

    # -- weights -------------------------------------------------------------

    def get(self, vi):
        if self.is_edit:
            return {g: w for g, w in self.bm.verts[vi][self.dl].items()}
        return {g.group: g.weight for g in self.mesh.vertices[vi].groups}

    def set(self, vi, wmap):
        """Replace the vertex's weights with wmap. Entries at or below EPS are
        removed so we don't leave zero-weight influences behind."""
        if self.is_edit:
            dv = self.bm.verts[vi][self.dl]
            dv.clear()
            for g, w in wmap.items():
                if w > EPS:
                    dv[g] = min(w, 1.0)
            return

        existing = {g.group for g in self.mesh.vertices[vi].groups}
        groups = self.obj.vertex_groups
        for g in existing:
            if wmap.get(g, 0.0) <= EPS and g < len(groups):
                groups[g].remove([vi])
        for g, w in wmap.items():
            if w > EPS and g < len(groups):
                groups[g].add([vi], min(w, 1.0), 'REPLACE')

    def finish(self):
        if self.is_edit:
            bmesh.update_edit_mesh(self.mesh, loop_triangles=False, destructive=False)
        else:
            self.mesh.update()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_armature(obj):
    for m in obj.modifiers:
        if m.type == 'ARMATURE' and m.object:
            return m.object
    return None


def deform_indices(obj):
    """Vertex group indices that correspond to deforming bones. Falls back to
    every group when there is no armature."""
    arm = get_armature(obj)
    if not arm or arm.type != 'ARMATURE':
        return {vg.index for vg in obj.vertex_groups}
    names = {b.name for b in arm.data.bones if b.use_deform}
    return {vg.index for vg in obj.vertex_groups if vg.name in names}


def locked_indices(obj):
    return {vg.index for vg in obj.vertex_groups if vg.lock_weight}


_DEFORM_CACHE = {}


def deform_indices_cached(obj):
    """draw_item runs once per visible row, so recomputing the deform set there
    gets expensive on dense rigs. Cache it against a cheap signature."""
    arm = get_armature(obj)
    key = obj.name
    sig = (len(obj.vertex_groups), arm.name if arm else "",
           len(arm.data.bones) if arm else 0)
    cached = _DEFORM_CACHE.get(key)
    if cached and cached[0] == sig:
        return cached[1]
    result = deform_indices(obj)
    _DEFORM_CACHE[key] = (sig, result)
    return result


def normalize_map(wmap, locked, deform):
    """Scale unlocked deform weights so the total reaches 1.0, leaving locked
    influences untouched. This is Maya's 'interactive' normalization."""
    locked_sum = sum(w for g, w in wmap.items() if g in locked)
    unlocked = {g: w for g, w in wmap.items() if g not in locked and g in deform}
    remain = 1.0 - locked_sum

    if remain <= EPS:
        for g in unlocked:
            wmap[g] = 0.0
        return wmap

    total = sum(unlocked.values())
    if total > EPS:
        f = remain / total
        for g in unlocked:
            wmap[g] *= f
    elif unlocked:
        share = remain / len(unlocked)
        for g in unlocked:
            wmap[g] = share
    return wmap


def redistribute_map(wmap, held, locked, deform):
    """Set one influence and absorb the difference across the others, which is
    what Maya does when you type a number into the Component Editor."""
    fixed = wmap.get(held, 0.0) + sum(
        w for g, w in wmap.items() if g in locked and g != held
    )
    others = {g: w for g, w in wmap.items()
              if g != held and g not in locked and g in deform}
    remain = max(0.0, 1.0 - fixed)

    total = sum(others.values())
    if total > EPS:
        f = remain / total
        for g in others:
            wmap[g] *= f
    elif others and remain > EPS:
        share = remain / len(others)
        for g in others:
            wmap[g] = share
    return wmap


def flip_group_name(name):
    """Turn Arm_L into Arm_R and so on."""
    flipper = getattr(bpy.utils, "flip_name", None)
    if flipper:
        flipped = flipper(name)
        if flipped != name:
            return flipped

    pairs = (("_L", "_R"), (".L", ".R"), ("_l", "_r"), (".l", ".r"),
             ("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT"))
    for a, b in pairs:
        if name.endswith(a):
            return name[:-len(a)] + b
        if name.endswith(b):
            return name[:-len(b)] + a
        if name.startswith(a):
            return b + name[len(a):]
        if name.startswith(b):
            return a + name[len(b):]
    return name


def ensure_group(obj, name):
    vg = obj.vertex_groups.get(name)
    if vg is None:
        vg = obj.vertex_groups.new(name=name)
    return vg.index


def settings(context):
    return context.scene.skinw


def mesh_poll(context):
    obj = context.object
    return (obj is not None and obj.type == 'MESH'
            and obj.mode in {'WEIGHT_PAINT', 'EDIT', 'OBJECT'})


# ---------------------------------------------------------------------------
# Component editor rows
# ---------------------------------------------------------------------------

def _row_weight_update(self, context):
    global _SUSPEND_SYNC
    if _SUSPEND_SYNC:
        return

    obj = context.object
    if not obj or obj.type != 'MESH':
        return

    gi = obj.vertex_groups.find(self.name)
    if gi < 0:
        return

    locked = locked_indices(obj)
    if gi in locked:
        return

    st = settings(context)
    deform = deform_indices(obj)

    wm = WMap(obj)
    for vi in wm.target_indices(require_selection=True):
        m = wm.get(vi)
        m[gi] = self.weight
        if st.use_normalize:
            redistribute_map(m, gi, locked, deform)
        wm.set(vi, m)
    wm.finish()

    refresh_rows(obj, context)


class SKINW_ComponentRow(PropertyGroup):
    name: StringProperty(name="Influence")
    weight: FloatProperty(
        name="Weight", min=0.0, max=1.0, precision=4, step=1,
        update=_row_weight_update,
    )


def refresh_rows(obj, context):
    """Rebuild the numeric weight table from the current selection. Multiple
    vertices show the average, as in Maya."""
    global _SUSPEND_SYNC
    _SUSPEND_SYNC = True
    try:
        rows = obj.skinw_rows
        rows.clear()

        wm = WMap(obj)
        targets = wm.target_indices(require_selection=True)
        if not targets:
            return

        st = settings(context)
        deform = deform_indices(obj)

        totals = {}
        for vi in targets:
            for g, w in wm.get(vi).items():
                if g in deform:
                    totals[g] = totals.get(g, 0.0) + w

        count = float(len(targets))
        groups = obj.vertex_groups

        for g, total in sorted(totals.items(), key=lambda kv: -kv[1]):
            avg = total / count
            if avg <= EPS and not st.show_zero_influences:
                continue
            if g >= len(groups):
                continue
            row = rows.add()
            row.name = groups[g].name
            row.weight = avg

        if st.show_zero_influences:
            for g in sorted(deform):
                if g in totals or g >= len(groups):
                    continue
                row = rows.add()
                row.name = groups[g].name
                row.weight = 0.0
    finally:
        _SUSPEND_SYNC = False


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SKINW_Settings(PropertyGroup):
    use_normalize: BoolProperty(
        name="Normalize",
        description="Keep each vertex's deform weights summing to 1.0, "
                    "respecting locked influences",
        default=True,
    )
    deform_only: BoolProperty(
        name="Deform Bones Only",
        description="Hide vertex groups that don't map to a deforming bone",
        default=True,
    )
    show_zero_influences: BoolProperty(
        name="Show Zero Weights",
        description="Include influences with no weight in the component editor",
        default=False,
    )
    influence_filter: StringProperty(
        name="Filter", description="Filter the influence list by name", default="",
    )

    prune_threshold: FloatProperty(
        name="Prune Below", default=0.005, min=0.0, max=1.0, precision=4,
        description="Weights below this value are removed",
    )
    max_influences: IntProperty(
        name="Max Influences", default=4, min=1, max=32,
        description="Keep only this many highest weights per vertex",
    )

    hammer_falloff: EnumProperty(
        name="Falloff",
        items=(
            ('UNIFORM', "Uniform", "Average neighbours equally"),
            ('DISTANCE', "Distance", "Weight neighbours by inverse distance"),
        ),
        default='DISTANCE',
    )
    hammer_iterations: IntProperty(
        name="Iterations", default=1, min=1, max=20,
        description="Repeat the hammer pass this many times",
    )

    mirror_axis: EnumProperty(
        name="Axis",
        items=(('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")),
        default='X',
    )
    mirror_direction: EnumProperty(
        name="Direction",
        items=(
            ('POS_TO_NEG', "+ to -", "Copy from the positive side to the negative"),
            ('NEG_TO_POS', "- to +", "Copy from the negative side to the positive"),
        ),
        default='POS_TO_NEG',
    )
    mirror_tolerance: FloatProperty(
        name="Tolerance", default=0.001, min=0.0, max=1.0, precision=5,
        description="Maximum distance when matching mirrored vertices",
    )
    mirror_selected_only: BoolProperty(
        name="Selected Only", default=False,
        description="Only write to selected vertices",
    )

    source_object: PointerProperty(
        name="Source", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
        description="Mesh to copy skin weights from",
    )
    transfer_method: EnumProperty(
        name="Method",
        items=(
            ('SURFACE', "Closest Point on Surface",
             "Interpolate weights across the nearest source face"),
            ('VERTEX', "Closest Vertex",
             "Copy weights from the nearest source vertex"),
        ),
        default='SURFACE',
    )

    move_source: StringProperty(
        name="From", description="Influence to move weight out of", default="",
    )


# ---------------------------------------------------------------------------
# Influence list
# ---------------------------------------------------------------------------

class SKINW_UL_influences(UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        obj = context.object
        row = layout.row(align=True)

        icon_id = 'BONE_DATA' if item.index in deform_indices_cached(obj) else 'GROUP_VERTEX'
        row.prop(item, "name", text="", emboss=False, icon=icon_id)

        for r in obj.skinw_rows:
            if r.name == item.name:
                row.label(text=f"{r.weight:.3f}")
                break

        row.prop(item, "lock_weight", text="",
                 icon='LOCKED' if item.lock_weight else 'UNLOCKED', emboss=False)

    def filter_items(self, context, data, propname):
        groups = getattr(data, propname)
        st = settings(context)
        helper = bpy.types.UI_UL_list

        flags = [self.bitflag_filter_item] * len(groups)

        if st.influence_filter:
            flags = helper.filter_items_by_name(
                st.influence_filter, self.bitflag_filter_item, groups, "name",
            )

        if st.deform_only:
            deform = deform_indices_cached(context.object)
            flags = [
                (f if i in deform else f & ~self.bitflag_filter_item)
                for i, f in enumerate(flags)
            ]

        order = helper.sort_items_by_name(groups, "name")
        return flags, order


# ---------------------------------------------------------------------------
# Operators - normalization and cleanup
# ---------------------------------------------------------------------------

class SKINW_OT_normalize(Operator):
    bl_idname = "skinw.normalize"
    bl_label = "Normalize Weights"
    bl_description = "Scale unlocked deform weights so each vertex totals 1.0"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        obj = context.object
        locked = locked_indices(obj)
        deform = deform_indices(obj)

        wm = WMap(obj)
        targets = wm.target_indices()
        for vi in targets:
            m = wm.get(vi)
            wm.set(vi, normalize_map(m, locked, deform))
        wm.finish()

        self.report({'INFO'}, f"Normalized {len(targets)} vertices")
        return {'FINISHED'}


class SKINW_OT_prune(Operator):
    bl_idname = "skinw.prune"
    bl_label = "Prune Small Weights"
    bl_description = "Remove weights below the threshold, then renormalize"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        obj = context.object
        st = settings(context)
        locked = locked_indices(obj)
        deform = deform_indices(obj)

        wm = WMap(obj)
        targets = wm.target_indices()
        removed = 0

        for vi in targets:
            m = wm.get(vi)
            drop = [g for g, w in m.items()
                    if g in deform and g not in locked and w < st.prune_threshold]
            if not drop:
                continue
            for g in drop:
                del m[g]
            removed += len(drop)
            if st.use_normalize:
                normalize_map(m, locked, deform)
            wm.set(vi, m)
        wm.finish()

        self.report({'INFO'}, f"Pruned {removed} weights")
        return {'FINISHED'}


class SKINW_OT_limit_influences(Operator):
    bl_idname = "skinw.limit_influences"
    bl_label = "Limit Influences"
    bl_description = "Keep only the N largest weights per vertex and renormalize"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        obj = context.object
        st = settings(context)
        locked = locked_indices(obj)
        deform = deform_indices(obj)

        wm = WMap(obj)
        targets = wm.target_indices()
        affected = 0

        for vi in targets:
            m = wm.get(vi)
            candidates = {g: w for g, w in m.items() if g in deform}
            if len(candidates) <= st.max_influences:
                continue

            # Locked influences are never dropped.
            keep = {g for g in candidates if g in locked}
            slots = max(0, st.max_influences - len(keep))
            ranked = sorted(
                (g for g in candidates if g not in keep),
                key=lambda g: -candidates[g],
            )
            keep.update(ranked[:slots])

            for g in list(m):
                if g in deform and g not in keep:
                    del m[g]

            if st.use_normalize:
                normalize_map(m, locked, deform)
            wm.set(vi, m)
            affected += 1
        wm.finish()

        self.report({'INFO'}, f"Limited {affected} vertices to {st.max_influences}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators - painting helpers
# ---------------------------------------------------------------------------

class SKINW_OT_flood(Operator):
    bl_idname = "skinw.flood"
    bl_label = "Flood"
    bl_description = ("Apply the current paint weight to every affected vertex "
                      "on the active influence")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return mesh_poll(context) and obj.vertex_groups.active is not None

    def execute(self, context):
        obj = context.object
        st = settings(context)
        active = obj.vertex_groups.active

        if active.lock_weight:
            self.report({'WARNING'}, f"{active.name} is locked")
            return {'CANCELLED'}

        value = context.tool_settings.vertex_group_weight
        gi = active.index
        locked = locked_indices(obj)
        deform = deform_indices(obj)

        wm = WMap(obj)
        targets = wm.target_indices()
        for vi in targets:
            m = wm.get(vi)
            m[gi] = value
            if st.use_normalize:
                redistribute_map(m, gi, locked, deform)
            wm.set(vi, m)
        wm.finish()

        self.report({'INFO'}, f"Flooded {len(targets)} vertices to {value:.3f}")
        return {'FINISHED'}


class SKINW_OT_hammer(Operator):
    bl_idname = "skinw.hammer"
    bl_label = "Hammer Weights"
    bl_description = ("Discard the selected vertices' weights and rebuild them "
                      "from their unselected neighbours")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        obj = context.object
        st = settings(context)
        locked = locked_indices(obj)
        deform = deform_indices(obj)

        wm = WMap(obj)
        targets = wm.target_indices(require_selection=True)
        if not targets:
            self.report({'WARNING'}, "Select the vertices you want to hammer")
            return {'CANCELLED'}

        adj = wm.adjacency()
        target_set = set(targets)

        for _ in range(st.hammer_iterations):
            # Read every source map up front so one vertex's new weights don't
            # bleed into the next vertex within the same pass.
            snapshot = {}
            for vi in targets:
                for n in adj[vi]:
                    if n not in snapshot:
                        snapshot[n] = wm.get(n)

            updates = {}
            for vi in targets:
                donors = [n for n in adj[vi] if n not in target_set]
                if not donors:
                    donors = list(adj[vi])
                if not donors:
                    continue

                accum = {}
                total_influence = 0.0
                co = wm.co(vi)

                for n in donors:
                    if st.hammer_falloff == 'DISTANCE':
                        d = (wm.co(n) - co).length
                        factor = 1.0 / max(d, 1e-5)
                    else:
                        factor = 1.0
                    total_influence += factor
                    for g, w in snapshot.get(n, {}).items():
                        if g in deform and g not in locked:
                            accum[g] = accum.get(g, 0.0) + w * factor

                if total_influence <= EPS or not accum:
                    continue
                for g in accum:
                    accum[g] /= total_influence

                original = wm.get(vi)
                merged = {g: w for g, w in original.items() if g in locked}
                merged.update(accum)
                updates[vi] = normalize_map(merged, locked, deform)

            for vi, m in updates.items():
                wm.set(vi, m)

        wm.finish()
        self.report({'INFO'}, f"Hammered {len(targets)} vertices")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators - copy / paste / move
# ---------------------------------------------------------------------------

_CLIPBOARD = {}


class SKINW_OT_copy_vertex_weights(Operator):
    bl_idname = "skinw.copy_vertex_weights"
    bl_label = "Copy Vertex Weights"
    bl_description = "Copy the active vertex's full influence map"

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        global _CLIPBOARD
        obj = context.object
        wm = WMap(obj)
        vi = wm.active_index()
        if vi < 0:
            self.report({'WARNING'}, "No active vertex")
            return {'CANCELLED'}

        groups = obj.vertex_groups
        _CLIPBOARD = {
            groups[g].name: w
            for g, w in wm.get(vi).items() if g < len(groups)
        }
        self.report({'INFO'}, f"Copied {len(_CLIPBOARD)} influences")
        return {'FINISHED'}


class SKINW_OT_paste_vertex_weights(Operator):
    bl_idname = "skinw.paste_vertex_weights"
    bl_label = "Paste Vertex Weights"
    bl_description = "Apply the copied influence map to every selected vertex"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return mesh_poll(context) and bool(_CLIPBOARD)

    def execute(self, context):
        obj = context.object
        st = settings(context)

        # Groups are created before BMesh is grabbed so indices stay valid.
        name_to_index = {}
        for name in _CLIPBOARD:
            name_to_index[name] = ensure_group(obj, name)

        locked = locked_indices(obj)
        deform = deform_indices(obj)

        wm = WMap(obj)
        targets = wm.target_indices(require_selection=True)
        if not targets:
            self.report({'WARNING'}, "Nothing selected")
            return {'CANCELLED'}

        for vi in targets:
            original = wm.get(vi)
            m = {g: w for g, w in original.items() if g in locked}
            for name, w in _CLIPBOARD.items():
                gi = name_to_index[name]
                if gi not in locked:
                    m[gi] = w
            if st.use_normalize:
                normalize_map(m, locked, deform)
            wm.set(vi, m)
        wm.finish()

        self.report({'INFO'}, f"Pasted onto {len(targets)} vertices")
        return {'FINISHED'}


class SKINW_OT_move_weights(Operator):
    bl_idname = "skinw.move_weights"
    bl_label = "Move Weights to Active"
    bl_description = ("Transfer all weight from the chosen influence into the "
                      "active influence")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (mesh_poll(context) and obj.vertex_groups.active is not None
                and settings(context).move_source)

    def execute(self, context):
        obj = context.object
        st = settings(context)

        src = obj.vertex_groups.get(st.move_source)
        dst = obj.vertex_groups.active

        if src is None:
            self.report({'WARNING'}, f"No influence named '{st.move_source}'")
            return {'CANCELLED'}
        if src.index == dst.index:
            self.report({'WARNING'}, "Source and target are the same influence")
            return {'CANCELLED'}

        si, di = src.index, dst.index
        wm = WMap(obj)
        targets = wm.target_indices()
        moved = 0

        for vi in targets:
            m = wm.get(vi)
            w = m.pop(si, 0.0)
            if w <= EPS:
                continue
            m[di] = min(1.0, m.get(di, 0.0) + w)
            wm.set(vi, m)
            moved += 1
        wm.finish()

        self.report({'INFO'}, f"Moved weight on {moved} vertices to {dst.name}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators - mirror
# ---------------------------------------------------------------------------

class SKINW_OT_mirror(Operator):
    bl_idname = "skinw.mirror"
    bl_label = "Mirror Weights"
    bl_description = ("Mirror weights across an axis using position matching and "
                      "L/R name flipping. Works on asymmetric topology")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        obj = context.object
        st = settings(context)
        axis = {'X': 0, 'Y': 1, 'Z': 2}[st.mirror_axis]
        tol = st.mirror_tolerance

        # Build the flipped-name mapping first so no groups are created while a
        # BMesh is live.
        flip_index = {}
        for vg in list(obj.vertex_groups):
            flipped = flip_group_name(vg.name)
            flip_index[vg.index] = ensure_group(obj, flipped)

        locked = locked_indices(obj)
        deform = deform_indices(obj)

        wm = WMap(obj)
        count = wm.vert_count()

        kd = KDTree(count)
        for vi in range(count):
            kd.insert(wm.co(vi), vi)
        kd.balance()

        if st.mirror_direction == 'POS_TO_NEG':
            sign = 1.0
        else:
            sign = -1.0

        write_set = None
        if st.mirror_selected_only:
            write_set = set(wm.target_indices(require_selection=True))
            if not write_set:
                self.report({'WARNING'}, "Nothing selected")
                return {'CANCELLED'}

        matched = 0
        unmatched = 0
        updates = {}

        for vi in range(count):
            co = wm.co(vi)
            side = co[axis] * sign

            if side > tol:
                continue  # this is a source vertex

            if write_set is not None and vi not in write_set:
                continue

            mirrored = co.copy()
            mirrored[axis] = -co[axis]

            _, src_i, dist = kd.find(mirrored)
            if src_i is None or dist > max(tol, 1e-4) * 10.0:
                unmatched += 1
                continue

            src_map = wm.get(src_i)
            flipped = {}
            for g, w in src_map.items():
                tgt = flip_index.get(g, g)
                flipped[tgt] = flipped.get(tgt, 0.0) + w

            if abs(side) <= tol:
                # Centre vertex: blend its own weights with the flipped set so
                # the seam stays symmetrical.
                own = wm.get(vi)
                blended = dict(flipped)
                for g, w in own.items():
                    blended[g] = (blended.get(g, 0.0) + w) * 0.5
                flipped = blended

            # Locked influences on the destination keep their existing values.
            original = wm.get(vi)
            result = {g: w for g, w in flipped.items() if g not in locked}
            for g in locked:
                if g in original:
                    result[g] = original[g]

            if st.use_normalize:
                normalize_map(result, locked, deform)
            updates[vi] = result
            matched += 1

        for vi, m in updates.items():
            wm.set(vi, m)
        wm.finish()

        msg = f"Mirrored {matched} vertices"
        if unmatched:
            msg += f", {unmatched} had no match (raise tolerance)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators - transfer between meshes
# ---------------------------------------------------------------------------

class SKINW_OT_transfer(Operator):
    bl_idname = "skinw.transfer"
    bl_label = "Copy Skin Weights"
    bl_description = ("Copy weights from another mesh using closest point on "
                      "surface, matching influences by name")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        st = settings(context)
        return (mesh_poll(context) and st.source_object is not None
                and st.source_object != context.object)

    def execute(self, context):
        obj = context.object
        st = settings(context)
        src = st.source_object

        if src.type != 'MESH' or not src.data.vertices:
            self.report({'ERROR'}, "Source has no mesh data")
            return {'CANCELLED'}

        src_mesh = src.data
        src_names = {vg.index: vg.name for vg in src.vertex_groups}

        # Pre-create every destination group.
        name_to_index = {}
        for name in src_names.values():
            name_to_index[name] = ensure_group(obj, name)

        src_weights = [
            {g.group: g.weight for g in v.groups} for v in src_mesh.vertices
        ]

        kd = None
        if st.transfer_method == 'VERTEX':
            kd = KDTree(len(src_mesh.vertices))
            for v in src_mesh.vertices:
                kd.insert(v.co, v.index)
            kd.balance()

        to_src = src.matrix_world.inverted() @ obj.matrix_world

        locked = locked_indices(obj)
        deform = deform_indices(obj)

        wm = WMap(obj)
        targets = wm.target_indices()
        transferred = 0

        for vi in targets:
            co = to_src @ wm.co(vi)
            sampled = {}

            if st.transfer_method == 'VERTEX':
                _, si, _ = kd.find(co)
                if si is None:
                    continue
                sampled = dict(src_weights[si])
            else:
                hit, loc, _, face_i = src.closest_point_on_mesh(co)
                if not hit or face_i < 0:
                    continue
                poly = src_mesh.polygons[face_i]
                verts = list(poly.vertices)
                coords = [src_mesh.vertices[v].co for v in verts]
                bary = poly_3d_calc(coords, loc)
                for v, b in zip(verts, bary):
                    if b <= EPS:
                        continue
                    for g, w in src_weights[v].items():
                        sampled[g] = sampled.get(g, 0.0) + w * b

            remapped = {}
            for g, w in sampled.items():
                name = src_names.get(g)
                if name is None:
                    continue
                gi = name_to_index[name]
                if gi in locked:
                    continue
                remapped[gi] = remapped.get(gi, 0.0) + w

            original = wm.get(vi)
            for g in locked:
                if g in original:
                    remapped[g] = original[g]

            if st.use_normalize:
                normalize_map(remapped, locked, deform)
            wm.set(vi, remapped)
            transferred += 1

        wm.finish()
        self.report({'INFO'}, f"Transferred weights to {transferred} vertices")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators - influence list utilities
# ---------------------------------------------------------------------------

class SKINW_OT_lock_influences(Operator):
    bl_idname = "skinw.lock_influences"
    bl_label = "Lock Influences"
    bl_description = "Change lock state across the influence list"
    bl_options = {'REGISTER', 'UNDO'}

    action: EnumProperty(
        items=(
            ('LOCK', "Lock All", ""),
            ('UNLOCK', "Unlock All", ""),
            ('INVERT', "Invert", ""),
            ('LOCK_OTHERS', "Lock All But Active", ""),
        ),
        default='LOCK',
    )

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        obj = context.object
        active = obj.vertex_groups.active
        active_index = active.index if active else -1

        for vg in obj.vertex_groups:
            if self.action == 'LOCK':
                vg.lock_weight = True
            elif self.action == 'UNLOCK':
                vg.lock_weight = False
            elif self.action == 'INVERT':
                vg.lock_weight = not vg.lock_weight
            else:
                vg.lock_weight = (vg.index != active_index)
        return {'FINISHED'}


class SKINW_OT_sync_influences(Operator):
    bl_idname = "skinw.sync_influences"
    bl_label = "Sync Influences from Armature"
    bl_description = "Create a vertex group for every deforming bone that lacks one"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return mesh_poll(context) and get_armature(obj) is not None

    def execute(self, context):
        obj = context.object
        arm = get_armature(obj)
        created = 0
        for bone in arm.data.bones:
            if bone.use_deform and obj.vertex_groups.get(bone.name) is None:
                obj.vertex_groups.new(name=bone.name)
                created += 1
        self.report({'INFO'}, f"Added {created} influences")
        return {'FINISHED'}


class SKINW_OT_set_active_to_heaviest(Operator):
    bl_idname = "skinw.set_active_to_heaviest"
    bl_label = "Pick Influence Under Selection"
    bl_description = ("Make the strongest influence on the active vertex the "
                      "active influence")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        obj = context.object
        wm = WMap(obj)
        vi = wm.active_index()
        if vi < 0:
            self.report({'WARNING'}, "No active vertex")
            return {'CANCELLED'}

        deform = deform_indices(obj)
        m = {g: w for g, w in wm.get(vi).items() if g in deform}
        if not m:
            self.report({'WARNING'}, "Vertex has no deform weights")
            return {'CANCELLED'}

        best = max(m, key=m.get)
        obj.vertex_groups.active_index = best
        self.report({'INFO'}, f"{obj.vertex_groups[best].name}: {m[best]:.3f}")
        return {'FINISHED'}


class SKINW_OT_select_by_influence(Operator):
    bl_idname = "skinw.select_by_influence"
    bl_label = "Select Affected Vertices"
    bl_description = "Select every vertex with weight on the active influence"
    bl_options = {'REGISTER', 'UNDO'}

    threshold: FloatProperty(name="Above", default=0.0, min=0.0, max=1.0)
    extend: BoolProperty(name="Extend", default=False)

    @classmethod
    def poll(cls, context):
        obj = context.object
        return mesh_poll(context) and obj.vertex_groups.active is not None

    def execute(self, context):
        obj = context.object
        gi = obj.vertex_groups.active.index

        wm = WMap(obj)
        count = 0

        if wm.is_edit:
            for v in wm.bm.verts:
                w = v[wm.dl].get(gi, 0.0)
                hit = w > self.threshold
                if hit:
                    v.select = True
                    count += 1
                elif not self.extend:
                    v.select = False
            wm.bm.select_flush(True)
        else:
            for v in wm.mesh.vertices:
                w = 0.0
                for g in v.groups:
                    if g.group == gi:
                        w = g.weight
                        break
                hit = w > self.threshold
                if hit:
                    v.select = True
                    count += 1
                elif not self.extend:
                    v.select = False

        wm.finish()
        self.report({'INFO'}, f"Selected {count} vertices")
        return {'FINISHED'}


class SKINW_OT_refresh_rows(Operator):
    bl_idname = "skinw.refresh_rows"
    bl_label = "Load Selection"
    bl_description = "Read the selected vertices into the component editor"

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)

    def execute(self, context):
        refresh_rows(context.object, context)
        return {'FINISHED'}


class SKINW_OT_set_paint_mode(Operator):
    bl_idname = "skinw.set_paint_mode"
    bl_label = "Set Paint Mode"
    bl_description = "Switch the weight brush to a Maya-equivalent blend mode"

    mode: EnumProperty(
        items=(
            ('REPLACE', "Replace", ""),
            ('ADD', "Add", ""),
            ('SCALE', "Scale", ""),
            ('SMOOTH', "Smooth", ""),
        ),
        default='REPLACE',
    )

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.mode == 'WEIGHT_PAINT'

    def execute(self, context):
        brush = context.tool_settings.weight_paint.brush
        if brush is None:
            self.report({'WARNING'}, "No active weight paint brush")
            return {'CANCELLED'}

        mapping = {
            'REPLACE': 'MIX',
            'ADD': 'ADD',
            'SCALE': 'MUL',
            'SMOOTH': 'BLUR',
        }
        brush.blend = mapping[self.mode]
        if self.mode == 'REPLACE':
            brush.strength = 1.0
        return {'FINISHED'}


class SKINW_OT_toggle_grayscale(Operator):
    bl_idname = "skinw.toggle_grayscale"
    bl_label = "Toggle Grayscale Weights"
    bl_description = ("Switch between Blender's colour ramp and Maya's grayscale "
                      "weight display")

    def execute(self, context):
        prefs = context.preferences
        holders = [prefs.view]
        if prefs.themes:
            holders.append(prefs.themes[0].view_3d)
            holders.append(prefs.themes[0])

        for holder in holders:
            if hasattr(holder, "use_weight_color_range"):
                holder.use_weight_color_range = not holder.use_weight_color_range
                state = "colour ramp" if holder.use_weight_color_range else "grayscale"
                self.report({'INFO'}, f"Weight display: {state}")
                return {'FINISHED'}

        self.report(
            {'INFO'},
            "Not available in this build - see Preferences > Themes > 3D Viewport",
        )
        return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class SKINW_PT_base:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Skin Weights"

    @classmethod
    def poll(cls, context):
        return mesh_poll(context)


class SKINW_PT_influences(SKINW_PT_base, Panel):
    bl_idname = "SKINW_PT_influences"
    bl_label = "Influences"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        st = settings(context)

        if get_armature(obj) is None:
            layout.label(text="No armature modifier", icon='INFO')

        row = layout.row(align=True)
        row.prop(st, "influence_filter", text="", icon='VIEWZOOM')
        row.prop(st, "deform_only", text="", icon='BONE_DATA')

        layout.template_list(
            "SKINW_UL_influences", "", obj, "vertex_groups",
            obj.vertex_groups, "active_index", rows=8,
        )

        row = layout.row(align=True)
        row.operator("skinw.lock_influences", text="Lock All").action = 'LOCK'
        row.operator("skinw.lock_influences", text="Unlock All").action = 'UNLOCK'

        row = layout.row(align=True)
        row.operator("skinw.lock_influences", text="Invert").action = 'INVERT'
        row.operator("skinw.lock_influences", text="Hold Others").action = 'LOCK_OTHERS'

        col = layout.column(align=True)
        col.operator("skinw.set_active_to_heaviest", icon='EYEDROPPER')
        col.operator("skinw.select_by_influence", icon='RESTRICT_SELECT_OFF')
        col.operator("skinw.sync_influences", icon='FILE_REFRESH')


class SKINW_PT_component(SKINW_PT_base, Panel):
    bl_idname = "SKINW_PT_component"
    bl_label = "Component Editor"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        st = settings(context)

        row = layout.row(align=True)
        row.operator("skinw.refresh_rows", icon='FILE_REFRESH')
        row.prop(st, "show_zero_influences", text="", icon='GHOST_ENABLED')

        rows = obj.skinw_rows
        if not rows:
            layout.label(text="Select vertices, then Load Selection", icon='INFO')
            return

        total = 0.0
        col = layout.column(align=True)
        for r in rows:
            vg = obj.vertex_groups.get(r.name)
            sub = col.row(align=True)
            sub.enabled = not (vg and vg.lock_weight)
            sub.prop(r, "weight", text=r.name, slider=True)
            total += r.weight

        info = layout.row()
        info.label(text=f"Total: {total:.4f}")
        if abs(total - 1.0) > 0.001:
            info.label(text="", icon='ERROR')

        row = layout.row(align=True)
        row.operator("skinw.copy_vertex_weights", icon='COPYDOWN')
        row.operator("skinw.paste_vertex_weights", icon='PASTEDOWN')


class SKINW_PT_paint(SKINW_PT_base, Panel):
    bl_idname = "SKINW_PT_paint"
    bl_label = "Paint"

    @classmethod
    def poll(cls, context):
        return mesh_poll(context) and context.object.mode == 'WEIGHT_PAINT'

    def draw(self, context):
        layout = self.layout
        ts = context.tool_settings

        row = layout.row(align=True)
        brush = ts.weight_paint.brush
        current = brush.blend if brush else None
        for label, mode, blend in (
            ("Replace", 'REPLACE', 'MIX'),
            ("Add", 'ADD', 'ADD'),
            ("Scale", 'SCALE', 'MUL'),
            ("Smooth", 'SMOOTH', 'BLUR'),
        ):
            sub = row.row(align=True)
            sub.depress = (current == blend)
            sub.operator("skinw.set_paint_mode", text=label).mode = mode

        col = layout.column(align=True)
        col.prop(ts, "vertex_group_weight", text="Value", slider=True)
        if brush:
            col.prop(brush, "strength", slider=True)

        layout.operator("skinw.flood", icon='BRUSH_DATA')

        box = layout.box()
        box.label(text="Normalize")
        box.prop(ts, "use_auto_normalize", text="Auto Normalize")
        box.prop(ts, "use_lock_relative", text="Lock Relative")
        box.prop(ts, "use_multipaint", text="Multi-Paint")


class SKINW_PT_cleanup(SKINW_PT_base, Panel):
    bl_idname = "SKINW_PT_cleanup"
    bl_label = "Weight Ops"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        st = settings(context)

        layout.prop(st, "use_normalize", icon='NORMALIZE_FCURVES')

        box = layout.box()
        box.label(text="Hammer", icon='MOD_SMOOTH')
        box.prop(st, "hammer_falloff", text="")
        box.prop(st, "hammer_iterations")
        box.operator("skinw.hammer")

        box = layout.box()
        box.label(text="Cleanup", icon='BRUSH_DATA')
        row = box.row(align=True)
        row.prop(st, "prune_threshold", text="")
        row.operator("skinw.prune", text="Prune")
        row = box.row(align=True)
        row.prop(st, "max_influences", text="")
        row.operator("skinw.limit_influences", text="Limit")
        box.operator("skinw.normalize")

        box = layout.box()
        box.label(text="Move Weights", icon='FORWARD')
        box.prop_search(st, "move_source", obj, "vertex_groups", text="From")
        active = obj.vertex_groups.active
        box.label(text=f"To: {active.name if active else '-'}")
        box.operator("skinw.move_weights")


class SKINW_PT_mirror(SKINW_PT_base, Panel):
    bl_idname = "SKINW_PT_mirror"
    bl_label = "Mirror & Transfer"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        st = settings(context)

        box = layout.box()
        box.label(text="Mirror", icon='MOD_MIRROR')
        row = box.row(align=True)
        row.prop(st, "mirror_axis", expand=True)
        box.prop(st, "mirror_direction", text="")
        box.prop(st, "mirror_tolerance")
        box.prop(st, "mirror_selected_only")
        box.operator("skinw.mirror")

        box = layout.box()
        box.label(text="Copy From Mesh", icon='MOD_DATA_TRANSFER')
        box.prop(st, "source_object", text="")
        box.prop(st, "transfer_method", text="")
        box.operator("skinw.transfer")


class SKINW_PT_display(SKINW_PT_base, Panel):
    bl_idname = "SKINW_PT_display"
    bl_label = "Display"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = context.object

        col = layout.column(align=True)
        col.prop(obj.data, "use_paint_mask_vertex", text="Vertex Selection Mask")
        col.prop(obj.data, "use_paint_mask", text="Face Selection Mask")

        space = context.space_data
        if space and hasattr(space, "overlay"):
            overlay = space.overlay
            col = layout.column(align=True)
            col.prop(overlay, "weight_paint_mode_opacity", text="Opacity")
            col.prop(overlay, "show_paint_wire", text="Wireframe")
            col.prop(overlay, "show_wpaint_contours", text="Contours")

        layout.operator("skinw.toggle_grayscale", icon='COLOR')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    SKINW_ComponentRow,
    SKINW_Settings,
    SKINW_UL_influences,
    SKINW_OT_normalize,
    SKINW_OT_prune,
    SKINW_OT_limit_influences,
    SKINW_OT_flood,
    SKINW_OT_hammer,
    SKINW_OT_copy_vertex_weights,
    SKINW_OT_paste_vertex_weights,
    SKINW_OT_move_weights,
    SKINW_OT_mirror,
    SKINW_OT_transfer,
    SKINW_OT_lock_influences,
    SKINW_OT_sync_influences,
    SKINW_OT_set_active_to_heaviest,
    SKINW_OT_select_by_influence,
    SKINW_OT_refresh_rows,
    SKINW_OT_set_paint_mode,
    SKINW_OT_toggle_grayscale,
    SKINW_PT_influences,
    SKINW_PT_component,
    SKINW_PT_paint,
    SKINW_PT_cleanup,
    SKINW_PT_mirror,
    SKINW_PT_display,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.skinw = PointerProperty(type=SKINW_Settings)
    bpy.types.Object.skinw_rows = CollectionProperty(type=SKINW_ComponentRow)


def unregister():
    _DEFORM_CACHE.clear()
    del bpy.types.Object.skinw_rows
    del bpy.types.Scene.skinw
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
