"""
Self-contained keylist generation.

Ports the matrix-generation logic from the standalone keylist.py so the
Blender add-on can consume a *keyboard* JSON (with x_algo/y_algo/... position
expressions) directly, without needing to run keylist.py first. Given a
keyboard-definition dict it returns a keylist dict of the same shape that
build_shell() already consumes:

    {
        "name": ..., "hole_size": ..., "key_1u": ..., "thickness": ...,
        "flange_offset": ..., "flange_z": ...,
        "keylist": [ {col,row,pos,rotation,u_width,...}, ... ]
    }

parse_algo() is a sandboxed arithmetic evaluator (only whitelisted names and
operators), identical in behaviour to keylist.py's, so untrusted expressions
in a keyboard JSON can't execute arbitrary code.
"""

import ast
import math


# Default plate settings, mirroring defaults.py from the standalone pipeline.
_DEFAULTS = {
    "hole_size": 14.5,
    "key_1u": 19.05,
    "thickness": 5,
}


def parse_algo(expr, x, y, z, w, h, key_1u=19.05):
    """
    Safely evaluate a simple arithmetic expression using the variables
    x, y, z, width, height, key_1u and a whitelist of functions
    (abs, min, max, floor, ceil, round). Anything outside that whitelist —
    attribute access, calls to other names, non-numeric literals — is
    rejected before evaluation, so keyboard JSONs can't run arbitrary code.
    """
    allowed_funcs = {'abs': abs, 'min': min, 'max': max,
                     'floor': math.floor, 'ceil': math.ceil, 'round': round}
    names = {'x': x, 'y': y, 'z': z, 'width': w, 'height': h, 'key_1u': key_1u}

    tree = ast.parse(expr, mode='eval')

    def _check(node):
        if isinstance(node, ast.Expression):
            return _check(node.body)
        if isinstance(node, ast.IfExp):
            _check(node.test); _check(node.body); _check(node.orelse)
            return
        if isinstance(node, ast.BinOp):
            _check(node.left); _check(node.right)
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                        ast.FloorDiv, ast.Mod, ast.Pow)):
                raise ValueError("Disallowed operator")
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                raise ValueError("Disallowed unary operator")
            return _check(node.operand)
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                raise ValueError("Disallowed boolean operator")
            for v in node.values:
                _check(v)
            return
        if isinstance(node, ast.Compare):
            _check(node.left)
            for comp in node.comparators:
                _check(comp)
            for op in node.ops:
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                                       ast.Gt, ast.GtE)):
                    raise ValueError("Disallowed comparison operator")
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                raise ValueError("Disallowed function")
            for a in node.args:
                _check(a)
            return
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError("Disallowed name")
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants allowed")
            return
        if isinstance(node, ast.Num):  # older AST node
            return
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    _check(tree)
    return eval(compile(tree, "<expr>", "eval"),
                {"__builtins__": {}}, {**allowed_funcs, **names})


def is_keyboard_def(data):
    """
    True if `data` looks like a keyboard definition (has position algos /
    width/height) rather than an already-generated keylist (has a 'keylist'
    array). Used to decide whether generation is needed.
    """
    if isinstance(data, dict):
        if 'keylist' in data:
            return False
        if 'items' in data:
            return False   # combined/assembly entry — handled separately
        return any(k in data for k in ('x_algo', 'y_algo', 'width', 'height'))
    return False


def generate_keylist(data, settings=None):
    """
    Turn a keyboard-definition dict into a keylist dict that build_shell()
    can consume. Faithful port of keylist.py's gen_matrix() (minus the file
    I/O and the OpenSCAD-only fields), including ignored_keys, u_diff,
    linked_keys, legends and switches handling.
    """
    if settings is None:
        settings = _DEFAULTS

    name = data.get('name', "default")
    ignored_keys = {tuple(k) for k in data.get('ignored_keys', [])}
    hole_size = data.get('hole_size', settings.get('hole_size', 14.5))
    key_1u = data.get('key_1u', settings.get('key_1u', 19.05))
    thickness = data.get('thickness', settings.get('thickness', 5))
    flange_offset = data.get('flange_offset', 0)
    flange_z = data.get('flange_z', 0)

    width = data.get('width', 6)
    height = data.get('height', 4)
    x_algo = data.get('x_algo', f"x*{key_1u}")
    y_algo = data.get('y_algo', f"-y*{key_1u}")
    z_algo = data.get('z_algo', "10")
    x_rot_algo = data.get('x_rot_algo', "0")
    y_rot_algo = data.get('y_rot_algo', "0")
    z_rot_algo = data.get('z_rot_algo', "0")


    def pa(expr, x, y):
        return parse_algo(expr, x, y, 0, width, height, key_1u)

    keys = [
        {
            "u_width": 1,
            "u_height": 1,
            "col": x,
            "row": y,
            "linked_keys": {},
            "pos": {
                "x": pa(x_algo, x, y),
                "y": pa(y_algo, x, y),
                "z": pa(z_algo, x, y),
            },
            "rotation": {
                "x": pa(x_rot_algo, x, y),
                "y": pa(y_rot_algo, x, y),
                "z": pa(z_rot_algo, x, y),
            },
            "insert": {},
            "legend": "",
            "switch_profile": "asa",
            "switch_rotation": 0,
        }
        for x in range(width) for y in range(height)
    ]

    keylist = [k for k in keys if (k['col'], k['row']) not in ignored_keys]

    # u_diff: wider/taller keys, repositioned so the widened cell stays centred
    for u in data.get('u_diff', []):
        u_keys = {tuple(key) for key in u.get('keys', [])}
        for k in keylist:
            if (k['col'], k['row']) in u_keys:
                k['u_width'] = u.get('u_width', 1)
                if k['u_width'] != 1:
                    if k['u_width'] > 0:
                        k['pos']['x'] = pa(x_algo, k['col'] + (k['u_width'] - 1) / 2, k['row'])
                    else:
                        k['u_width'] = abs(k['u_width'])
                        k['pos']['x'] = pa(x_algo, k['col'] - (k['u_width'] - 1) / 2, k['row'])
                k['u_height'] = u.get('u_height', 1)
                if k['u_height'] != 1:
                    k['pos']['y'] = pa(y_algo, k['col'], k['row'] + (k['u_height'] - 1) / 2)

    # linked_keys: explicit joins between (possibly non-grid) keys.
    # A side value is [col, row] or [col, row, "corner"], where the corner
    # names a corner of the KEY THAT SIDE REFERS TO (the source of that side's
    # half of the link). We split each side value into its (col,row) cell and
    # optional corner, then record, on each participating key, an outgoing link
    # toward its partner — carrying THAT key's own corner if one was given.
    def _split(val):
        if val is None:
            return None, None
        if len(val) >= 3:
            return (int(val[0]), int(val[1])), str(val[2])
        return (int(val[0]), int(val[1])), None

    for group in data.get('linked_keys', {}):
        l_cell, l_corner = _split(group.get('l'))
        r_cell, r_corner = _split(group.get('r'))
        t_cell, t_corner = _split(group.get('t'))
        b_cell, b_corner = _split(group.get('b'))

        for k in keylist:
            pos = (k['col'], k['row'])
            # The 'l' key links rightward to the 'r' key (using l's corner);
            # the 'r' key links leftward to the 'l' key (using r's corner).
            if pos == l_cell and r_cell is not None:
                k['linked_keys']['r'] = ([r_cell[0], r_cell[1], l_corner]
                                         if l_corner else [r_cell[0], r_cell[1]])
            elif pos == r_cell and l_cell is not None:
                k['linked_keys']['l'] = ([l_cell[0], l_cell[1], r_corner]
                                         if r_corner else [l_cell[0], l_cell[1]])
            elif pos == t_cell and b_cell is not None:
                k['linked_keys']['b'] = ([b_cell[0], b_cell[1], t_corner]
                                         if t_corner else [b_cell[0], b_cell[1]])
            elif pos == b_cell and t_cell is not None:
                k['linked_keys']['t'] = ([t_cell[0], t_cell[1], b_corner]
                                         if b_corner else [t_cell[0], t_cell[1]])

    for inserts in data.get('inserts', []):
        for k in keylist:
            if k['col'] == inserts.get('col') and k['row'] == inserts.get('row'):
                # Copy every insert field through
                for field in ('x', 'y', 'rot',
                              'id', 'od', 'depth', 'height', 'clearance_d',
                              'hole_x', 'hole_y', 'leg_0', 'leg_1', 'leg_2'):
                    if field in inserts:
                        k['insert'][field] = inserts.get(field)

    for legends in data.get('legends', []):
        for k in keylist:
            if k['col'] == legends.get('col') and k['row'] == legends.get('row'):
                k['legend'] = legends.get('legend')

    for switches in data.get('switches', []):
        for k in keylist:
            if k['col'] == switches.get('col') and k['row'] == switches.get('row'):
                k['switch_profile'] = switches.get('profile', 'asa')
                k['switch_rotation'] = switches.get('rot', 0)

    out = {
        "name": name,
        "hole_size": hole_size,
        "key_1u": key_1u,
        "thickness": thickness,
        "flange_offset": flange_offset,
        "flange_z": flange_z,
        "keylist": keylist,
    }
    # Carry through optional build parameters that the mesh/wall builders read,
    # so they survive keyboard-definition -> keylist expansion rather than
    # being silently dropped.
    for opt in ('plate_lip', 'plate_gap', 'wall_base_z', 'vertical_edges',
                'skirt', 'wall_thickness', 'skirt_flange', 'skirt_mode',
                'skirt_angle', 'skirt_flare', 'skirt_profile',
                'skirt_steps', 'skirt_angle_end', 'skirt_step_out',
                'constant_thickness_walls',
                'switch_border',
                'tent_angle', 'pitch_angle', 'plate_min_wall',
                'baseplate_thickness', 'insert_clearance_d', 'insert_hole_segments'):
        if opt in data:
            out[opt] = data[opt]
    return out
