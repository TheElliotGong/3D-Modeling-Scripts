"""
find_faces_in_udim_tile.py

Recovered from your Maya 2026 Script Editor Python tab (unsaved tab, last touched 2026-08-26).
Selects faces on the selected mesh whose UV centroid lands in TARGET_TILE, and reports
faces in invalid (non-UDIM) tiles and faces straddling tile borders.
Set TARGET_TILE = None to catch all invalid tiles.
"""

import math
import maya.api.OpenMaya as om
import maya.cmds as cmds

TARGET_TILE = (3, -2)   # the tile Painter reported; set to None to catch all invalid tiles

sel = om.MGlobal.getActiveSelectionList()
dag = sel.getDagPath(0)
dag.extendToShape()
fn = om.MFnMesh(dag)
name = dag.partialPathName()

us, vs = fn.getUVs()
counts, ids = fn.getAssignedUVs()

matches, straddling, tally = [], [], {}
i = 0
for face, c in enumerate(counts):
    if c == 0:
        continue
    fu = [us[ids[i + k]] for k in range(c)]
    fv = [vs[ids[i + k]] for k in range(c)]
    i += c

    corners = {(int(math.floor(u)), int(math.floor(v))) for u, v in zip(fu, fv)}
    tile = (int(math.floor(sum(fu) / c)), int(math.floor(sum(fv) / c)))
    valid = 0 <= tile[0] < 10 and tile[1] >= 0

    if len(corners) > 1:
        straddling.append(face)
    if TARGET_TILE is not None:
        if tile == tuple(TARGET_TILE):
            matches.append(face)
    elif not valid:
        matches.append(face)
    if not valid:
        tally[tile] = tally.get(tile, 0) + 1

for tile, n in sorted(tally.items()):
    udim = 1001 + tile[0] + tile[1] * 10
    print("Invalid tile {}  ->  would be UDIM {}   ({} faces)".format(tile, udim, n))
print("Faces straddling a tile border: {}".format(len(straddling)))

if matches:
    cmds.select(["{}.f[{}]".format(name, f) for f in matches], r=True)
    print("Selected {} faces in tile {}".format(len(matches), TARGET_TILE))
else:
    print("No faces found in that tile")