"""
Keyboard Plate Generator — Blender add-on entry point.

Thin Blender layer over the pure-Python geometry core in core.py. The core
returns (vertices, faces); this module pushes them into a bmesh and creates
a mesh object in the scene. Keeping geometry logic in core.py means it stays
testable and headless-runnable (see build_headless()).
"""

# Support Blender's "Reload Scripts": if bpy is already loaded, reload the
# core submodule rather than re-importing from scratch.
_needs_reload = "bpy" in locals()

import bpy
import bmesh
import json
import os

from bpy.props import (StringProperty, EnumProperty, FloatProperty, IntProperty,
                       BoolProperty, PointerProperty)
from bpy.types import Operator, Panel, PropertyGroup

from . import core
from . import keylist_gen

if _needs_reload:
    import importlib
    keylist_gen = importlib.reload(keylist_gen)
    core = importlib.reload(core)


# ---------------------------------------------------------------------------
# Settings (scene properties) + override helper
# ---------------------------------------------------------------------------

# Simple JSON fields exposed as editable Blender properties. Each entry is
# (json_key, blender_prop_name, kind) where kind is 'str', 'int' or 'float'.
# Only fields whose JSON value is a plain string or number appear here; the
# complex array/object fields (keylist, linked_keys, u_diff, ignored_keys,
# legends, switches, inserts) are edited as raw JSON in the Text Editor.
#
# For 'str' fields, an empty value means "leave the JSON's own value alone"
# (so an untouched algo field won't clobber the JSON default). Numeric fields
# always carry a value and are applied as-is when overrides are enabled.
_FIELD_SPECS = (
    ("name", "kb_name", "str"),
    ("width", "kb_width", "int"),
    ("height", "kb_height", "int"),
    ("x_algo", "x_algo", "str"),
    ("y_algo", "y_algo", "str"),
    ("z_algo", "z_algo", "str"),
    ("x_rot_algo", "x_rot_algo", "str"),
    ("y_rot_algo", "y_rot_algo", "str"),
    ("z_rot_algo", "z_rot_algo", "str"),
    ("hole_size", "hole_size", "float"),
    ("switch_border", "switch_border", "float"),
    ("tent_angle", "tent_angle", "float"),
    ("pitch_angle", "pitch_angle", "float"),
    ("key_1u", "key_1u", "float"),
    ("thickness", "thickness", "float"),
    ("flange_offset", "flange_offset", "float"),
    ("flange_z", "flange_z", "float"),
    ("plate_lip", "plate_lip", "float"),
    ("plate_gap", "plate_gap", "float"),
    ("wall_base_z", "wall_base_z", "float"),
    # Fused-skirt wall method.
    ("wall_thickness", "wall_thickness", "float"),
    ("skirt_flange", "skirt_flange", "float"),
    ("skirt_mode", "skirt_mode", "enum"),
    ("skirt_angle", "skirt_angle", "float"),
    ("skirt_flare", "skirt_flare", "float"),
    ("constant_thickness_walls", "constant_thickness_walls", "bool"),
    # Step-generator settings. core ignores these — they drive the generated
    # `skirt_profile` — but they are written out and read back so the GUI state
    # survives an export/load round trip.
    ("skirt_steps", "skirt_steps", "int"),
    ("skirt_angle_end", "skirt_angle_end", "float"),
    ("skirt_step_out", "skirt_step_out", "float"),
    # Baseplate (skirt only).
    ("baseplate_thickness", "baseplate_thickness", "float"),
    ("insert_clearance_d", "insert_clearance_d", "float"),
)

# Grouping for a tidy panel layout.
_ALGO_FIELDS = ("x_algo", "y_algo", "z_algo", "x_rot_algo", "y_rot_algo", "z_rot_algo")
_NUM_FIELDS = ("hole_size", "switch_border", "key_1u", "thickness")

# Wall-method controls. These are BUILD settings, not JSON overrides: when a
# wall method is explicitly chosen (not 'From JSON'), they are always applied,
# and they stay editable in the panel regardless of the override toggle.
_RECESS_FIELDS = ("flange_offset", "flange_z", "plate_lip", "plate_gap", "wall_base_z")
_SKIRT_FIELDS = ("wall_thickness", "skirt_flange", "skirt_mode",
                 "skirt_angle", "skirt_flare", "wall_base_z",
                 "constant_thickness_walls")
# Written alongside _SKIRT_FIELDS so an exported file restores the GUI state.
_SKIRT_GEN_FIELDS = ("skirt_steps", "skirt_angle_end", "skirt_step_out",
                     "baseplate_thickness", "insert_clearance_d")

# prop name -> (json_key, kind), for applying subsets of the specs.
_PROP_TO_SPEC = {prop: (jk, kind) for jk, prop, kind in _FIELD_SPECS}


def _set_field(entry, settings, prop):
    """Write one settings property into `entry` using its declared kind."""
    json_key, kind = _PROP_TO_SPEC[prop]
    val = getattr(settings, prop)
    if kind == 'str':
        if str(val).strip() == "":
            return
        entry[json_key] = val
    elif kind == 'enum':
        entry[json_key] = str(val)
    elif kind == 'int':
        entry[json_key] = int(val)
    elif kind == 'bool':
        entry[json_key] = bool(val)
    else:
        entry[json_key] = float(val)


class KEYBOARD_settings(PropertyGroup):
    """Add-on settings stored on the Scene."""

    json_text: PointerProperty(
        type=bpy.types.Text,
        name="JSON",
        description="Text block holding the keyboard/keylist JSON to build from. "
                    "Edit it in a Text Editor area.",
    )

    parts: EnumProperty(
        name="Generate",
        items=[
            ('PLATE', "Plate only", "The key plate shell"),
            ('WALLS', "Walls only", "The perimeter walls the plate seats into"),
            ('BOTH', "Plate + Walls", "Both, as two separate objects"),
        ],
        default='BOTH',
    )

    edges: EnumProperty(
        name="Plate edges",
        items=[
            ('AUTO', "Auto", "Vertical when building walls, perpendicular for plate-only"),
            ('VERTICAL', "Vertical", "Outer edge vertical so the plate drops into the walls"),
            ('PERPENDICULAR', "Perpendicular", "Constant-thickness edge (for printing the plate alone)"),
        ],
        default='AUTO',
    )

    use_overrides: BoolProperty(
        name="Edit fields (override JSON)",
        description="When on, the fields below override the JSON at build time. "
                    "Use 'Pull from JSON' to load current values first",
        default=False,
    )

    # Identity / grid dimensions.
    kb_name: StringProperty(name="Name", default="")
    kb_width: IntProperty(name="Width (cols)", default=6, min=1, soft_max=30)
    kb_height: IntProperty(name="Height (rows)", default=4, min=1, soft_max=30)

    # Position / rotation algorithm expressions (blank = use JSON's own value).
    x_algo: StringProperty(name="x_algo", default="")
    y_algo: StringProperty(name="y_algo", default="")
    z_algo: StringProperty(name="z_algo", default="")
    x_rot_algo: StringProperty(name="x_rot_algo", default="")
    y_rot_algo: StringProperty(name="y_rot_algo", default="")
    z_rot_algo: StringProperty(name="z_rot_algo", default="")

    hole_size: FloatProperty(name="Hole size", default=14.5, min=0.0, soft_max=20.0)
    switch_border: FloatProperty(
        name="Switch border", default=1.5, min=0.0, soft_max=10.0,
        description="Minimum flat plate around each switch cutout, mm. The cell "
                    "grows so there's always this much flat rim; wide/tall keys "
                    "keep their larger footprint")
    key_1u: FloatProperty(name="Key 1u", default=19.05, min=1.0, soft_max=30.0)
    thickness: FloatProperty(name="Thickness", default=5.0, min=0.1, soft_max=15.0)
    flange_offset: FloatProperty(name="Flange offset", default=3.0, min=0.0, soft_max=15.0)
    flange_z: FloatProperty(name="Flange Z (recess)", default=2.0, min=0.0, soft_max=15.0)
    plate_lip: FloatProperty(name="Plate lip", default=1.5, min=0.0, soft_max=5.0)
    plate_gap: FloatProperty(name="Plate gap", default=0.25, min=0.0, soft_max=2.0)
    wall_base_z: FloatProperty(name="Wall base Z", default=0.0, soft_min=-10.0, soft_max=10.0)
    tent_angle: FloatProperty(
        name="Tent angle", default=0.0, soft_min=-45.0, soft_max=45.0,
        description="Tilt the key plate side-to-side (about Y), degrees. Applied "
                    "before the walls, which still drop to the flat base")
    pitch_angle: FloatProperty(
        name="Pitch angle", default=0.0, soft_min=-45.0, soft_max=45.0,
        description="Tilt the key plate front-to-back (about X), degrees. Applied "
                    "before the walls, which still drop to the flat base")

    # --- Fused skirt walls ---
    wall_mode: EnumProperty(
        name="Wall method",
        description="How the walls are made",
        items=[
            ('AUTO', "From JSON", "Use the JSON's own 'skirt' setting"),
            ('RECESS', "Recess frame", "Separate wall object the plate drops into"),
            ('SKIRT', "Fused skirt", "Walls swept down from the plate edge, one object"),
        ],
        default='AUTO',
    )
    wall_thickness: FloatProperty(name="Wall thickness", default=2.0, min=0.1, soft_max=10.0)
    skirt_flange: FloatProperty(name="Skirt flange", default=0.0, min=0.0, soft_max=10.0,
                                description="Small outward step at the top of the skirt")
    skirt_mode: EnumProperty(
        name="Flare mode",
        description="How the skirt flares outward as it drops",
        items=[
            ('angle', "Constant angle", "Fixed draft angle; outward run varies with height"),
            ('flare', "Constant flare", "Fixed outward run; draft angle varies with height"),
        ],
        default='angle',
    )
    skirt_angle: FloatProperty(name="Skirt angle (deg)", default=0.0, soft_min=-30.0, soft_max=30.0,
                               description="Outward draft angle from vertical (constant-angle mode)")
    skirt_flare: FloatProperty(name="Skirt flare (mm)", default=0.0, soft_min=-10.0, soft_max=10.0,
                               description="Outward run at the base (constant-flare mode)")
    constant_thickness_walls: BoolProperty(
        name="Constant-thickness walls",
        description="Inner wall face parallels the outer skirt profile at a "
                    "fixed wall_thickness offset, blending to the plate edge at "
                    "the top. Off = a single sloped inner face, so the wall "
                    "thickens as the outer face flares away",
        default=False,
    )

    # Step generator: builds a multi-segment skirt_profile without hand-editing
    # JSON. With steps > 1 the draft angle is interpolated from `skirt_angle`
    # (top) to `skirt_angle_end` (bottom) across evenly-sized segments.
    skirt_steps: IntProperty(
        name="Steps", default=1, min=1, max=12,
        description="Number of rotation steps between the plate and the base. "
                    "1 = a single straight run",
    )
    skirt_angle_end: FloatProperty(
        name="End angle (deg)", default=0.0, soft_min=-30.0, soft_max=30.0,
        description="Draft angle of the LAST step (the first uses 'Skirt angle')",
    )
    skirt_step_out: FloatProperty(
        name="Step ledge (mm)", default=0.0, min=0.0, soft_max=5.0,
        description="Horizontal ledge inserted between steps (0 = smooth)",
    )

    # --- Baseplate (fused-skirt only) ---
    make_baseplate: BoolProperty(
        name="Also build baseplate",
        description="Build a flat bottom cover matching the skirt's outer "
                    "footprint, as a separate object",
        default=False,
    )
    build_assembly: BoolProperty(
        name="Build assembly if present",
        description="When the JSON contains a combined entry (one with an "
                    "'items' array), build that assembly — placing each "
                    "referenced board by its pos/rot/mirror. Turn off to build "
                    "just the single selected board instead",
        default=True,
    )
    baseplate_thickness: FloatProperty(
        name="Baseplate thickness", default=2.0, min=0.1, soft_max=10.0,
        description="Thickness of the bottom cover; the case rests on its top face",
    )
    insert_clearance_d: FloatProperty(
        name="Screw clearance", default=3.0, min=0.0, soft_max=10.0,
        description="Diameter of the baseplate screw hole under each threaded "
                    "insert (0 = no hole). Per-insert 'clearance_d' overrides it",
    )

    grid_interval: EnumProperty(
        name="Grid interval",
        description="Spacing of the viewport grid lines when setting up mm units",
        items=[
            ('1', "1 mm", "Grid lines every 1 mm"),
            ('10', "10 mm", "Grid lines every 10 mm"),
            ('100', "100 mm", "Grid lines every 100 mm"),
        ],
        default='10',
    )


def _apply_wall_settings(entry, settings):
    """
    Write the edge style and the chosen wall method (with its parameters, and
    the generated `skirt_profile`) into `entry`, in place.

    Shared by the build path and the JSON-export path so an exported file
    always reproduces exactly the geometry that was built.
    """
    # Edge style.
    if settings.edges == 'VERTICAL':
        entry['vertical_edges'] = True
    elif settings.edges == 'PERPENDICULAR':
        entry['vertical_edges'] = False
    elif 'vertical_edges' not in entry:
        entry['vertical_edges'] = settings.parts in ('WALLS', 'BOTH')

    # Tent/pitch tilt the finished plate before the walls; they are build
    # controls applied from the panel. A non-zero panel value takes effect; at
    # zero we leave any JSON value untouched so a tilted board loaded from JSON
    # still builds tilted until you dial the panel.
    if settings.tent_angle:
        entry['tent_angle'] = settings.tent_angle
    if settings.pitch_angle:
        entry['pitch_angle'] = settings.pitch_angle

    # Wall method. Choosing a method explicitly also applies that method's
    # parameters — they are build controls, not "overrides". 'From JSON'
    # (AUTO) defers entirely to whatever the JSON says.
    if settings.wall_mode == 'SKIRT':
        entry['skirt'] = True
        for prop in _SKIRT_FIELDS + _SKIRT_GEN_FIELDS:
            _set_field(entry, settings, prop)
        # >1 step builds a multi-segment profile, interpolating the draft angle
        # from skirt_angle (top) to skirt_angle_end (bottom), optionally with a
        # horizontal ledge between steps. With 1 step we leave any JSON
        # `skirt_profile` untouched.
        if settings.skirt_steps > 1:
            steps = settings.skirt_steps
            a0, a1 = settings.skirt_angle, settings.skirt_angle_end
            frac = 1.0 / steps
            prof = []
            for i in range(steps):
                ang = a0 + (a1 - a0) * (i / (steps - 1))
                prof.append({"fraction": frac, "angle": ang})
                if settings.skirt_step_out > 0.0 and i < steps - 1:
                    prof.append({"fraction": 0.0, "out": settings.skirt_step_out})
            entry['skirt_profile'] = prof
    elif settings.wall_mode == 'RECESS':
        entry['skirt'] = False
        for prop in _RECESS_FIELDS:
            _set_field(entry, settings, prop)
    return entry


def _apply_settings(entry, settings):
    """
    Return a copy of `entry` with the scene settings applied: the wall method
    and edge style always, and the simple JSON fields when overrides are on.
    """
    entry = dict(entry)
    _apply_wall_settings(entry, settings)

    # Simple-field overrides (name/size/algos/dimensions).
    if settings.use_overrides:
        for json_key, prop, kind in _FIELD_SPECS:
            _set_field(entry, settings, prop)

    return entry


def _load_entry_into_settings(entry, settings):
    """Populate the editable fields from a loaded JSON entry (best-effort).
    Fields absent from the JSON are left as-is (string fields cleared so they
    won't override)."""
    for json_key, prop, kind in _FIELD_SPECS:
        if json_key in entry:
            try:
                if kind in ('str', 'enum'):
                    setattr(settings, prop, str(entry[json_key]))
                elif kind == 'int':
                    setattr(settings, prop, int(entry[json_key]))
                elif kind == 'bool':
                    setattr(settings, prop, bool(entry[json_key]))
                else:
                    setattr(settings, prop, float(entry[json_key]))
            except (TypeError, ValueError):
                pass
        elif kind == 'str':
            # Not present -> clear so it doesn't override on build.
            setattr(settings, prop, "")


def _write_settings_into_data(data, settings, include_fields=True):
    """
    Write the settings back into the buildable entry of `data` (a parsed JSON
    structure — dict or list of entries), returning the modified data.

    Mirrors `_apply_settings` exactly, so the exported JSON reproduces the
    geometry that Generate would produce:
      * wall method + edge style + generated `skirt_profile` are always written
      * the simple JSON fields (name/size/algos/dimensions) are written only
        when the override toggle is on, so an untouched field never clobbers
        the file's own value.
    Blank string fields are skipped so we don't inject empty values.
    """
    entry = select_buildable_entry(data)
    if entry is None:
        return data
    if include_fields and settings.use_overrides:
        for json_key, prop, kind in _FIELD_SPECS:
            _set_field(entry, settings, prop)
    _apply_wall_settings(entry, settings)
    return data


# ---------------------------------------------------------------------------
# Mesh construction (Blender side)
# ---------------------------------------------------------------------------

def select_buildable_entry(data):
    """
    Given parsed JSON that may be:
      * a single keylist dict,
      * a single keyboard-definition dict (with x_algo/... position algos), or
      * a list mixing keyboard definitions and combined/assembly entries,
    return the first entry that can be built directly (a keylist or a keyboard
    definition), skipping combined/assembly entries (those with 'items', which
    reference other keyboards and need the multi-file pipeline).

    Returns None if nothing buildable is found.
    """
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if 'items' in entry:
            continue  # combined/assembly entry — skip
        # Either an explicit keylist or a keyboard definition is buildable.
        if 'keylist' in entry or core.keylist_gen.is_keyboard_def(entry):
            return entry
    return None


def _object_from_verts_faces(name, verts, faces):
    """Create a Blender mesh object from explicit verts/faces."""
    mesh = bpy.data.meshes.new(name + "_mesh")
    bm = bmesh.new()
    bm_verts = [bm.verts.new(v) for v in verts]
    bm.verts.ensure_lookup_table()
    dropped = 0
    for face in faces:
        try:
            bm.faces.new([bm_verts[i] for i in face])
        except ValueError:
            # A face on these exact verts already exists. This should not happen
            # for our disjoint shell/insert index ranges; if it ever does, count
            # it so the (silent) loss is at least visible during debugging.
            dropped += 1
    bm.normal_update()
    if dropped:
        print(f"[keyboard] warning: {dropped} duplicate face(s) skipped in {name}")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _weld_object(obj, dist=1e-4):
    """Merge coincident vertices in-place (remove doubles)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bm.to_mesh(obj.data)
    bm.free()


def build_plate_object(data, name="Keyboard"):
    """
    Build the key-plate shell object from keyboard/keylist data.

    The shell is one closed watertight manifold. Threaded-insert holders are
    built as separate overlapping bodies and then merged into the shell with a
    Boolean UNION so the result is a SINGLE solid — otherwise the inserts are
    loose islands that some exporters (notably File > Export > STL) can drop,
    even though they render in the viewport. After the union we weld coincident
    verts so the mesh is clean for slicing.
    """
    verts, faces = core.build_shell_from_any(data)
    plate = _object_from_verts_faces(name + "_plate", verts, faces)

    ivs, ifs = core.build_inserts_from_any(data)
    if not ivs:
        return plate

    inserts = _object_from_verts_faces(name + "_inserts_tmp", ivs, ifs)

    # Boolean-union the inserts into the plate (single evaluated solid).
    mod = plate.modifiers.new(name="insert_union", type='BOOLEAN')
    mod.operation = 'UNION'
    mod.solver = 'EXACT'
    mod.object = inserts

    # Apply the modifier. Make the plate the sole selected+active object so the
    # operator's context is unambiguous even when called repeatedly in a loop
    # (e.g. building an assembly of several boards).
    prev_active = bpy.context.view_layer.objects.active
    for o in bpy.context.selected_objects:
        o.select_set(False)
    plate.select_set(True)
    bpy.context.view_layer.objects.active = plate
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except RuntimeError:
        # If the boolean fails for any reason, fall back to leaving the inserts
        # as a joined (but un-unioned) mesh so nothing is lost.
        plate.modifiers.remove(mod)
        _join_objects(plate, [inserts])
        bpy.context.view_layer.objects.active = prev_active
        _weld_object(plate)
        return plate

    # Remove the now-consumed temporary insert object.
    bpy.data.objects.remove(inserts, do_unlink=True)
    bpy.context.view_layer.objects.active = prev_active
    _weld_object(plate)
    return plate


def _join_objects(target, others):
    """Join `others` into `target` (fallback when boolean is unavailable)."""
    for o in bpy.context.selected_objects:
        o.select_set(False)
    target.select_set(True)
    for o in others:
        o.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.join()


def _place_object(obj, pos, rot, mirror):
    """
    Apply an assembly item's transform to a freshly built object, in the order
    mirror -> rotate -> translate, and bake it into the mesh.

        mirror : (x, y, z) axis flags; a set flag scales that axis by -1.
        rot    : Euler angles in degrees, OpenSCAD order (Z, then Y, then X).
        pos    : translation in mm.

    A reflection (odd number of -1 scales) inverts face winding, so we reverse
    every face afterwards to keep the mesh solid with outward normals.

    This works directly on the mesh data (mesh.transform + bmesh) rather than
    through bpy.ops, so it doesn't depend on the current selection / active
    object / mode — those context dependencies were silently dropping mirrored
    boards.
    """
    import math
    import bmesh
    from mathutils import Matrix

    sx = -1.0 if mirror[0] else 1.0
    sy = -1.0 if mirror[1] else 1.0
    sz = -1.0 if mirror[2] else 1.0

    scale_m = Matrix.Diagonal((sx, sy, sz, 1.0))
    # OpenSCAD rotate([rx,ry,rz]) applies Z, then Y, then X.
    rot_m = (Matrix.Rotation(math.radians(rot[0]), 4, 'X')
             @ Matrix.Rotation(math.radians(rot[1]), 4, 'Y')
             @ Matrix.Rotation(math.radians(rot[2]), 4, 'Z'))
    trans_m = Matrix.Translation((pos[0], pos[1], pos[2]))

    # Bake mirror -> rotate -> translate straight into the mesh coordinates.
    mesh = obj.data
    mesh.transform(trans_m @ rot_m @ scale_m)

    if (sx * sy * sz) < 0.0:
        # Reflection inverted winding — reverse every face so normals point out.
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()

    mesh.update()
    return obj


def build_assembly_objects(catalog, entry, name=None, prepare_board=None):
    """
    Build a combined/assembly entry: one SEPARATE object per item, each a full
    plate (shell + inserts) of the board the item references by name, placed by
    the item's pos / rot / mirror (mirror -> rotate -> translate).

    `catalog` is the full parsed JSON (list or single entry) so item names can
    be resolved; `entry` is the assembly entry itself. `prepare_board`, if
    given, is called as prepare_board(board_dict) and must return the board
    dict to actually build — used to apply the GUI's wall/skirt settings to
    each referenced board so an assembly matches what a single build produces.

    Returns the list of created objects. Items whose board name can't be
    resolved are skipped (the caller may warn).
    """
    base = name or entry.get("name", "Assembly")
    created = []
    for idx, item in enumerate(entry.get("items", [])):
        board_name = item.get("name")
        board = core.find_named_entry(board_name, catalog)
        if board is None:
            continue
        if prepare_board is not None:
            board = prepare_board(board)
        obj_name = f"{base}_{board_name or 'item'}_{idx}"
        obj = build_plate_object(board, name=obj_name)
        _place_object(obj,
                      item.get("pos", (0.0, 0.0, 0.0)),
                      item.get("rot", (0.0, 0.0, 0.0)),
                      item.get("mirror", (0, 0, 0)))
        created.append(obj)
    return created


def build_walls_object(data, name="Keyboard"):
    """
    Build the perimeter walls object (a separate frame the plate seats into),
    honouring flange_offset / flange_z / plate_lip. Watertight manifold.
    """
    verts, faces = core.build_walls_from_any(data)
    return _object_from_verts_faces(name + "_walls", verts, faces)


def build_baseplate_object(data, name="Keyboard"):
    """
    Build the baseplate: a flat bottom cover matching the fused skirt's outer
    footprint, extruded down from wall_base_z by baseplate_thickness. The case
    rests flush on its top face. Requires the skirt wall method.
    """
    verts, faces = core.build_baseplate_from_any(data)
    return _object_from_verts_faces(name + "_baseplate", verts, faces)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class KEYBOARD_OT_load_json(Operator):
    """Load a JSON file into a Text block for editing in Blender"""
    bl_idname = "keyboard.load_json"
    bl_label = "Load JSON into editor"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "Please choose a valid JSON file")
            return {'CANCELLED'}
        try:
            with open(self.filepath) as f:
                raw = f.read()
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            self.report({'ERROR'}, f"Could not read JSON: {e}")
            return {'CANCELLED'}

        # Create/replace a text datablock named after the file.
        base = os.path.basename(self.filepath)
        text = bpy.data.texts.get(base)
        if text is None:
            text = bpy.data.texts.new(base)
        text.clear()
        text.write(raw)

        settings = context.scene.keyboard_settings
        settings.json_text = text

        # Best-effort: populate override fields from the buildable entry.
        entry = select_buildable_entry(data)
        if entry is not None:
            _load_entry_into_settings(entry, settings)

        self.report({'INFO'}, f"Loaded '{base}' into the Text editor")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


_TEMPLATE_JSON = """\
{
  "name": "MyKeyboard",
  "width": 6,
  "height": 4,
  "hole_size": 14.5,
  "key_1u": 19.05,
  "thickness": 5,
  "flange_offset": 3,
  "flange_z": 2,
  "plate_lip": 1.5,
  "plate_gap": 0.25,
  "vertical_edges": true,
  "x_algo": "x*key_1u",
  "y_algo": "-y*key_1u",
  "z_algo": "10",
  "x_rot_algo": "0",
  "y_rot_algo": "0",
  "z_rot_algo": "0",
  "ignored_keys": [],
  "u_diff": [],
  "linked_keys": []
}
"""


class KEYBOARD_OT_new_template(Operator):
    """Create a new JSON template Text block to edit"""
    bl_idname = "keyboard.new_template"
    bl_label = "New template"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        text = bpy.data.texts.new("keyboard.json")
        text.write(_TEMPLATE_JSON)
        context.scene.keyboard_settings.json_text = text
        self.report({'INFO'}, "Created template 'keyboard.json' — edit it in a Text editor")
        return {'FINISHED'}


class KEYBOARD_OT_edit_json(Operator):
    """Open the JSON text block in a Text Editor so you can edit it"""
    bl_idname = "keyboard.edit_json"
    bl_label = "Edit JSON"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.keyboard_settings
        text = settings.json_text
        if text is None:
            self.report({'ERROR'}, "No JSON text block. Load a file or make a template first.")
            return {'CANCELLED'}

        # 1) If a Text Editor area already exists anywhere, point it at our text.
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'TEXT_EDITOR':
                    area.spaces.active.text = text
                    return {'FINISHED'}

        # 2) Otherwise, split the largest area in the current screen and turn
        #    the new half into a Text Editor showing our text.
        screen = context.screen
        area = max(screen.areas, key=lambda a: a.width * a.height)
        before = set(screen.areas)
        try:
            with context.temp_override(window=context.window, screen=screen, area=area):
                bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
        except Exception:
            # Fallback for older API without temp_override.
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)

        # Identify the area that was just created (robust across versions).
        new_areas = [a for a in screen.areas if a not in before]
        new_area = new_areas[0] if new_areas else screen.areas[-1]
        new_area.type = 'TEXT_EDITOR'
        st = new_area.spaces.active
        st.text = text
        # Handy editor options for JSON.
        try:
            st.show_line_numbers = True
            st.show_syntax_highlight = True
            st.show_word_wrap = True
        except Exception:
            pass
        return {'FINISHED'}


class KEYBOARD_OT_pull_fields(Operator):
    """Populate the editable fields from the current JSON text block"""
    bl_idname = "keyboard.pull_fields"
    bl_label = "Pull from JSON"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.keyboard_settings
        text = settings.json_text
        if text is None:
            self.report({'ERROR'}, "No JSON text block set")
            return {'CANCELLED'}
        try:
            data = json.loads(text.as_string())
        except json.JSONDecodeError as e:
            self.report({'ERROR'}, f"JSON error: {e}")
            return {'CANCELLED'}
        entry = select_buildable_entry(data)
        if entry is None:
            self.report({'ERROR'}, "No buildable keyboard/keylist found in JSON")
            return {'CANCELLED'}
        _load_entry_into_settings(entry, settings)
        self.report({'INFO'}, "Fields populated from JSON")
        return {'FINISHED'}


class KEYBOARD_OT_push_fields(Operator):
    """Write the editable fields back into the JSON text block"""
    bl_idname = "keyboard.push_fields"
    bl_label = "Push to JSON"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.keyboard_settings
        text = settings.json_text
        if text is None:
            self.report({'ERROR'}, "No JSON text block set")
            return {'CANCELLED'}
        try:
            data = json.loads(text.as_string())
        except json.JSONDecodeError as e:
            self.report({'ERROR'}, f"JSON error: {e}")
            return {'CANCELLED'}
        data = _write_settings_into_data(data, settings)
        text.clear()
        text.write(json.dumps(data, indent=2))
        self.report({'INFO'}, "Fields written into the JSON text block")
        return {'FINISHED'}


class KEYBOARD_OT_export_json(Operator):
    """Save the current JSON (with field edits applied) to a file"""
    bl_idname = "keyboard.export_json"
    bl_label = "Export JSON"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    apply_fields: BoolProperty(
        name="Apply field edits",
        description="Write the editable fields into the exported JSON",
        default=True,
    )

    def execute(self, context):
        settings = context.scene.keyboard_settings
        text = settings.json_text
        if text is None:
            self.report({'ERROR'}, "No JSON text block set")
            return {'CANCELLED'}
        try:
            data = json.loads(text.as_string())
        except json.JSONDecodeError as e:
            self.report({'ERROR'}, f"JSON error: {e}")
            return {'CANCELLED'}

        # Always record the wall method / edge style / skirt_profile so the
        # exported file rebuilds the same geometry. The simple fields are only
        # baked in when the user asked for it (and has overrides enabled).
        data = _write_settings_into_data(data, settings,
                                         include_fields=self.apply_fields)

        path = self.filepath
        if not path:
            self.report({'ERROR'}, "No file path given")
            return {'CANCELLED'}
        if not path.lower().endswith('.json'):
            path += '.json'
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            self.report({'ERROR'}, f"Could not write file: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported JSON to {os.path.basename(path)}")
        return {'FINISHED'}

    def invoke(self, context, event):
        settings = context.scene.keyboard_settings
        if settings.json_text is not None and not self.filepath:
            # Suggest a filename from the text block name.
            self.filepath = settings.json_text.name
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class KEYBOARD_OT_setup_units(Operator):
    """Set the scene units to millimetres for 3D printing"""
    bl_idname = "keyboard.setup_units"
    bl_label = "Set up mm units"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # The geometry is authored in millimetres (key_1u = 19.05 mm, etc.).
        # Blender's default scene treats 1 unit = 1 metre, so without this the
        # viewport shows the plate as ~140 m. Setting METRIC + scale_length
        # 0.001 makes 1 Blender unit display as 1 mm. STL export writes the raw
        # unit numbers (we keep apply_scene_unit off in the exporter), so a
        # vertex at 19.05 exports as 19.05 mm — correct for slicers.
        us = context.scene.unit_settings
        us.system = 'METRIC'
        us.scale_length = 0.001
        try:
            us.length_unit = 'MILLIMETERS'
        except Exception:
            pass

        # A clip range that comfortably covers a keyboard at mm scale, plus a
        # viewport grid whose major lines fall on a sensible mm interval. With
        # scale_length = 0.001 (1 BU = 1 mm), the metric grid multiplies
        # overlay.grid_scale by the unit scale, so grid_scale = interval_mm *
        # 0.001 puts major grid lines every `interval_mm` millimetres.
        interval_mm = float(context.scene.keyboard_settings.grid_interval)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.clip_start = 0.1
                        space.clip_end = 10000.0
                        try:
                            space.overlay.grid_scale = interval_mm * us.scale_length
                            # 10 minor subdivisions between major lines.
                            space.overlay.grid_subdivisions = 10
                        except Exception:
                            pass

        self.report({'INFO'},
                    "Scene units set to millimetres (grid every %g mm)" % interval_mm)
        return {'FINISHED'}


def _printing_status(scene):
    """Return (ok, list of (message, icon)) describing 3D-print unit readiness."""
    us = scene.unit_settings
    msgs = []
    ok = True
    if us.system != 'METRIC':
        ok = False
        msgs.append(("Unit system is not Metric", 'ERROR'))
    if abs(us.scale_length - 0.001) > 1e-9:
        ok = False
        msgs.append(("Unit scale is %g (want 0.001 for mm)" % us.scale_length, 'ERROR'))
    length_unit = getattr(us, 'length_unit', 'MILLIMETERS')
    if length_unit not in ('MILLIMETERS', 'ADAPTIVE'):
        msgs.append(("Length unit is %s" % length_unit, 'INFO'))
    if ok:
        msgs.append(("Scene is set for mm 3D printing", 'CHECKMARK'))
    return ok, msgs


class KEYBOARD_OT_generate(Operator):
    """Generate the keyboard plate and/or walls from the JSON text block"""
    bl_idname = "keyboard.generate_plate"
    bl_label = "Generate Keyboard Plate"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.keyboard_settings
        text = settings.json_text
        if text is None:
            self.report({'ERROR'}, "No JSON text block set. Load a file or make a template.")
            return {'CANCELLED'}

        try:
            data = json.loads(text.as_string())
        except json.JSONDecodeError as e:
            self.report({'ERROR'}, f"JSON error: {e}")
            return {'CANCELLED'}

        # If the file contains a combined/assembly entry (one with 'items'),
        # that's the thing to build — it references the raw boards as
        # ingredients and places them. Look for one anywhere in the file, not
        # just first, since the referenced boards usually come before it.
        assembly = None
        if settings.build_assembly:
            for e in (data if isinstance(data, list) else [data]):
                if core.is_assembly(e):
                    assembly = e
                    break

        if assembly is not None:
            asm_name = assembly.get("name", text.name.rsplit('.', 1)[0])

            # Apply the GUI's wall/skirt/edge settings to each referenced board
            # (but NOT the name/dimension overrides, which describe a single
            # authored board, not assembly members).
            def _prepare(board):
                b = dict(board)
                _apply_wall_settings(b, settings)
                return b

            try:
                created = build_assembly_objects(
                    data, assembly, name=asm_name, prepare_board=_prepare)
                # A baseplate per board when the skirt is fused and requested.
                if settings.make_baseplate:
                    extra = []
                    for idx, item in enumerate(assembly.get("items", [])):
                        board = core.find_named_entry(item.get("name"), data)
                        if board is None:
                            continue
                        b = _prepare(board)
                        if not b.get('skirt', False):
                            continue
                        bp = build_baseplate_object(
                            b, name=f"{asm_name}_{item.get('name')}_{idx}")
                        _place_object(bp,
                                      item.get("pos", (0.0, 0.0, 0.0)),
                                      item.get("rot", (0.0, 0.0, 0.0)),
                                      item.get("mirror", (0, 0, 0)))
                        extra.append(bp)
                    created.extend(extra)
            except Exception as e:
                self.report({'ERROR'}, f"Assembly build failed: {e}")
                return {'CANCELLED'}
            if not created:
                self.report({'ERROR'},
                            "Assembly has no buildable items — check that each "
                            "item's 'name' matches a board in this file")
                return {'CANCELLED'}
            for o in context.selected_objects:
                o.select_set(False)
            for o in created:
                o.select_set(True)
            context.view_layer.objects.active = created[-1]
            total_v = sum(len(o.data.vertices) for o in created)
            total_f = sum(len(o.data.polygons) for o in created)
            self.report({'INFO'},
                        f"Generated assembly '{asm_name}': {len(created)} "
                        f"object(s), {total_v} verts, {total_f} faces")
            return {'FINISHED'}

        entry = select_buildable_entry(data)
        if entry is None:
            self.report({'ERROR'},
                        "No buildable keyboard/keylist or assembly found in JSON")
            return {'CANCELLED'}

        name = entry.get("name", text.name.rsplit('.', 1)[0])
        entry = _apply_settings(entry, settings)

        # With a fused skirt the walls are part of the plate, so a separate
        # walls object is neither built nor meaningful.
        fused = bool(entry.get('skirt', False))

        created = []
        try:
            if fused:
                created.append(build_plate_object(entry, name=name))
                if settings.make_baseplate:
                    created.append(build_baseplate_object(entry, name=name))
            else:
                if settings.parts in ('PLATE', 'BOTH'):
                    created.append(build_plate_object(entry, name=name))
                if settings.parts in ('WALLS', 'BOTH'):
                    created.append(build_walls_object(entry, name=name))
                elif settings.wall_mode == 'RECESS':
                    self.report({'WARNING'},
                                "Wall method is 'Recess frame' but Parts is "
                                "'Plate only' — no walls were built")
                if settings.make_baseplate:
                    self.report({'WARNING'},
                                "Baseplate needs the 'Fused skirt' wall method "
                                "— none was built")
        except Exception as e:
            self.report({'ERROR'}, f"Build failed: {e}")
            return {'CANCELLED'}

        for o in context.selected_objects:
            o.select_set(False)
        for o in created:
            o.select_set(True)
        if created:
            context.view_layer.objects.active = created[-1]

        total_v = sum(len(o.data.vertices) for o in created)
        total_f = sum(len(o.data.polygons) for o in created)
        self.report({'INFO'}, f"Generated '{name}': {len(created)} object(s), "
                              f"{total_v} verts, {total_f} faces")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class KEYBOARD_PT_panel(Panel):
    bl_label = "Keyboard Plate"
    bl_idname = "KEYBOARD_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Keyboard"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.keyboard_settings

        # --- 3D-printing units check ---
        box = layout.box()
        box.label(text="3D printing:", icon='EXPORT')
        ok, msgs = _printing_status(context.scene)
        for msg, icon in msgs:
            box.label(text=msg, icon=icon)
        box.prop(settings, "grid_interval", text="Grid")
        if not ok:
            box.operator(KEYBOARD_OT_setup_units.bl_idname,
                         text="Set up mm units", icon='MODIFIER')
        else:
            box.operator(KEYBOARD_OT_setup_units.bl_idname,
                         text="Re-apply mm units", icon='FILE_REFRESH')

        # --- JSON source ---
        box = layout.box()
        box.label(text="JSON source:", icon='TEXT')
        row = box.row(align=True)
        row.operator(KEYBOARD_OT_load_json.bl_idname, text="Load file", icon='FILEBROWSER')
        row.operator(KEYBOARD_OT_new_template.bl_idname, text="Template", icon='FILE_NEW')
        box.prop(settings, "json_text", text="")
        row = box.row(align=True)
        row.enabled = settings.json_text is not None
        row.operator(KEYBOARD_OT_edit_json.bl_idname, text="Edit JSON", icon='GREASEPENCIL')
        row.operator(KEYBOARD_OT_export_json.bl_idname, text="Export", icon='EXPORT')

        # --- Editable fields (override JSON) ---
        box = layout.box()
        row = box.row(align=True)
        row.prop(settings, "use_overrides")
        row.operator(KEYBOARD_OT_pull_fields.bl_idname, text="", icon='IMPORT')
        row.operator(KEYBOARD_OT_push_fields.bl_idname, text="", icon='EXPORT')

        col = box.column()
        col.enabled = settings.use_overrides

        col.label(text="Identity / size:")
        sub = col.column(align=True)
        sub.prop(settings, "kb_name")
        r = sub.row(align=True)
        r.prop(settings, "kb_width")
        r.prop(settings, "kb_height")

        col.label(text="Algorithms (blank = keep JSON):")
        sub = col.column(align=True)
        for prop in _ALGO_FIELDS:
            sub.prop(settings, prop)

        col.label(text="Dimensions (mm):")
        sub = col.column(align=True)
        for prop in _NUM_FIELDS:
            sub.prop(settings, prop)

        # --- Build options (wall fields always editable) ---
        box = layout.box()
        box.label(text="Build:", icon='MESH_GRID')
        tilt = box.row(align=True)
        tilt.prop(settings, "tent_angle")
        tilt.prop(settings, "pitch_angle")
        box.prop(settings, "wall_mode", text="")

        if settings.wall_mode == 'SKIRT':
            box.label(text="Fused skirt (one object):")
            sub = box.column(align=True)
            for prop in _SKIRT_FIELDS:
                if prop == 'skirt_angle' and settings.skirt_mode != 'angle':
                    continue
                if prop == 'skirt_flare' and settings.skirt_mode != 'flare':
                    continue
                sub.prop(settings, prop)
            step = box.column(align=True)
            step.prop(settings, "skirt_steps")
            if settings.skirt_steps > 1:
                step.prop(settings, "skirt_angle_end")
                step.prop(settings, "skirt_step_out")
                step.label(text="Overrides JSON skirt_profile", icon='INFO')
            else:
                step.label(text="JSON 'skirt_profile' respected", icon='INFO')
            bp = box.column(align=True)
            bp.prop(settings, "make_baseplate")
            row = bp.row()
            row.enabled = settings.make_baseplate
            row.prop(settings, "baseplate_thickness")
            row = bp.row()
            row.enabled = settings.make_baseplate
            row.prop(settings, "insert_clearance_d")
        elif settings.wall_mode == 'RECESS':
            box.label(text="Recess walls:")
            sub = box.column(align=True)
            for prop in _RECESS_FIELDS:
                sub.prop(settings, prop)
        else:
            box.label(text="Wall settings taken from JSON", icon='INFO')

        row = box.row()
        row.enabled = settings.wall_mode != 'SKIRT'
        row.prop(settings, "parts", text="")
        box.prop(settings, "edges", text="")
        box.prop(settings, "build_assembly")
        box.operator(KEYBOARD_OT_generate.bl_idname, text="Generate", icon='PLAY')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    KEYBOARD_settings,
    KEYBOARD_OT_setup_units,
    KEYBOARD_OT_load_json,
    KEYBOARD_OT_new_template,
    KEYBOARD_OT_edit_json,
    KEYBOARD_OT_pull_fields,
    KEYBOARD_OT_push_fields,
    KEYBOARD_OT_export_json,
    KEYBOARD_OT_generate,
    KEYBOARD_PT_panel,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.keyboard_settings = PointerProperty(type=KEYBOARD_settings)


def unregister():
    del bpy.types.Scene.keyboard_settings
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


# ---------------------------------------------------------------------------
# Headless entry point (usable via: blender --background --python ... )
# ---------------------------------------------------------------------------

def build_headless(json_path, parts='BOTH', export_stl_plate=None,
                   export_stl_walls=None, export_stl_baseplate=None):
    """
    Build a keyboard headlessly and optionally export STL(s). Accepts a
    keyboard-definition JSON or a keylist JSON (picks the buildable entry if
    the file is a list).

    parts: 'PLATE', 'WALLS', or 'BOTH'.
    export_stl_plate / export_stl_walls: optional output paths.

    Example:
        blender --background --python-expr \
          "import keyboard_mesh_addon as k; \
           k.build_headless('/p/x.json','BOTH','/tmp/plate.stl','/tmp/walls.stl')"
    """
    with open(json_path) as f:
        data = json.load(f)

    entry = select_buildable_entry(data)
    if entry is None:
        raise ValueError("No buildable keyboard/keylist found in JSON")

    name = entry.get("name", "Keyboard")

    plate_obj = walls_obj = None
    if parts in ('PLATE', 'BOTH'):
        plate_obj = build_plate_object(entry, name=name)
    if parts in ('WALLS', 'BOTH'):
        walls_obj = build_walls_object(entry, name=name)
    base_obj = None
    if entry.get('skirt', False) and export_stl_baseplate:
        base_obj = build_baseplate_object(entry, name=name)

    def _export(obj, path):
        if not (obj and path):
            return
        for o in bpy.context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # global_scale=1.0 and apply_scene_unit=False => write the raw unit
        # numbers (which are millimetres). This keeps the STL correct for
        # slicers whether or not the scene uses the mm unit scale (0.001);
        # without apply_scene_unit=False, a mm-configured scene would shrink
        # the export 1000x.
        try:
            bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True,
                                  global_scale=1.0, apply_scene_unit=False)
        except TypeError:
            # Older exporter signature without these kwargs.
            bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)

    _export(plate_obj, export_stl_plate)
    _export(walls_obj, export_stl_walls)
    _export(base_obj, export_stl_baseplate)
    return plate_obj, walls_obj, base_obj


if __name__ == "__main__":
    register()
