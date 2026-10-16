# Maya & Blender Python Scripts — 

## Maya
| File | What it does  
|---|---|---|
| `Maya/select_faces_by_material.py` | Selects faces on one mesh by assigned material (handles object/face-level assignment, instances, namespaces); includes a small UI 
| `Maya/snapTools.py` | Blender-style Shift+S snap menu + persistent 3D cursor for Maya 2026 (`import snapTools; snapTools.install()`) 
| `Maya/add_leaf_joints.py` | Adds/removes leaf joint chains on terminal joints for clean Blender FBX import; length, axis, radius, count; marker-attribute idempotency (`add_leaf_joints.show_ui()`) 
| `Maya/find_faces_in_udim_tile.py` | Finds/selects faces in a given UV tile, reports invalid UDIM tiles and border-straddling faces (OpenMaya 2.0) 

## Blender
| File | What it does
|---|---|---|
| `Blender/skin_weight_tools.py` | "Skin Weight Tools" add-on v1.0.0 — Maya-style influence locks, component editor, hammer, prune, mirror, transfer (N-panel > Skin Weights) 
| `Blender/skin_weight_tools_addon.zip` | Same add-on, installable zip (Edit > Preferences > Add-ons > Install from Disk)
