"""
Pure-Python geometry core for keyboard plate generation.

No Blender dependency — this module takes a keylist (the same dict structure
produced by keylist.py / stored in *_keylist.json) and returns an explicit
mesh as (vertices, faces):

    vertices : list[tuple[float, float, float]]
    faces    : list[tuple[int, ...]]   # indices into vertices, CCW when
                                       # viewed from outside the solid

Keeping this Blender-free means it can be unit-tested and run headless, and
the Blender add-on layer only has to push these verts/faces into a bmesh.

Milestone 1 scope: each key is a sealed box — a rectangular top face at the
key's cell boundary (in the key's own rotated/translated frame) with four
vertical side walls dropping straight down to z=0 and a bottom face at z=0.
No inter-key bridging, no flange, no texture yet. The goal is a provably
watertight (manifold) mesh that matches the layout.
"""

import math

try:
    # When imported as part of the Blender add-on package.
    from . import keylist_gen
except ImportError:
    # When run headless / standalone from this directory.
    import keylist_gen


def _rot_xyz(p, rx_deg, ry_deg, rz_deg):
    """
    Rotate point p=(x,y,z) by the given Euler angles (degrees), matching
    OpenSCAD's rotate([rx,ry,rz]) convention: Z first, then Y, then X.
    """
    x, y, z = p
    rz = math.radians(rz_deg)
    ry = math.radians(ry_deg)
    rx = math.radians(rx_deg)

    # Z
    x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
    # Y
    x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
    # X
    y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
    return (x, y, z)


def _place(p_local, key):
    """
    Transform a point from a key's local frame into world space:
    rotate by the key's rotation, then translate by the key's position.
    Mirrors OpenSCAD's translate(pos){ rotate(rot){ ... } }.
    """
    r = key['rotation']
    px, py, pz = _rot_xyz(p_local, r['x'], r['y'], r['z'])
    pos = key['pos']
    return (px + pos['x'], py + pos['y'], pz + pos['z'])


def _norm(a):
    m = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5
    if m < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / m, a[1] / m, a[2] / m)


def _face_normal(pts):
    """
    Newell's method: robust polygon normal (not unit-length; magnitude is
    2x the polygon area, which conveniently area-weights when accumulated).
    """
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return (nx, ny, nz)


class TopSurface:
    """
    Accumulates the keyboard's TOP surface as one connected mesh with a shared
    vertex pool: top vertices from different pieces (key cells, bridges, corner
    patches) that land on the same world point are merged into a single vertex.

    As each top face is added, its (area-weighted) normal is accumulated onto
    every vertex it touches. After all faces are in, each vertex has an averaged
    outward normal, which is used to offset the bottom surface a uniform
    `thickness` PERPENDICULAR to the top — so wherever two top pieces share an
    edge, their bottoms share it too (no troughs), giving a true constant-
    thickness shell.

    Switch-cutout rings are added the same way but flagged so their offset
    still follows the shared normal, keeping the cutout walls perpendicular to
    the tilted top rather than dropped straight down in world Z.
    """

    def __init__(self, weld_tol=1e-4):
        self._inv = 1.0 / weld_tol
        self._remap = {}          # quantized xyz -> vertex index
        self.points = []          # world-space top points
        self.normals = []         # accumulated (unnormalized) normals
        self.faces = []           # top faces as vertex-index tuples

    def _key(self, p):
        return (round(p[0] * self._inv), round(p[1] * self._inv), round(p[2] * self._inv))

    def vert(self, p):
        k = self._key(p)
        idx = self._remap.get(k)
        if idx is None:
            idx = len(self.points)
            self._remap[k] = idx
            self.points.append(p)
            self.normals.append([0.0, 0.0, 0.0])
        return idx

    def face(self, world_pts):
        """Add a top face given its world points (CCW from above)."""
        idxs = [self.vert(p) for p in world_pts]
        # Skip degenerate faces (repeated welded verts).
        if len(set(idxs)) < 3:
            return None
        nrm = _face_normal(world_pts)
        for i in idxs:
            self.normals[i][0] += nrm[0]
            self.normals[i][1] += nrm[1]
            self.normals[i][2] += nrm[2]
        self.faces.append(tuple(idxs))
        return tuple(idxs)

    def unit_normals(self):
        """Return per-vertex averaged unit normals (falling back to +Z)."""
        out = []
        for n in self.normals:
            u = _norm(tuple(n))
            if u == (0.0, 0.0, 0.0):
                u = (0.0, 0.0, 1.0)
            out.append(u)
        return out


def _cell_half_extents(key, key_1u, hole_size=14.5, switch_border=1.5):
    """
    Half-width and half-height of a key's cell footprint in its local frame.

    The cell is the flat top around the switch cutout and the surface bridges
    attach to. Historically it was simply (u * key_1u - 3) / 2, which for a 1u
    key leaves only (key_1u - 3 - hole_size)/2 ~= 0.8 mm of flat plate around a
    14.5 mm cutout — too little to read as a flat rim, especially between close
    thumb keys. We now guarantee at least `switch_border` mm of flat plate on
    every side of the cutout, while still letting wide/tall keys (u_width /
    u_height > 1) keep their larger footprint.
    """
    u = key.get('u_width', 1)
    h = key.get('u_height', 1)
    hw = max((u * key_1u - 3) / 2.0, hole_size / 2.0 + switch_border)
    hh = max((h * key_1u - 3) / 2.0, hole_size / 2.0 + switch_border)
    return hw, hh


def key_cell_corners(key, key_1u, thickness=5.0, hole_size=14.5,
                     switch_border=1.5):
    """
    Return the four top corners of a key's cell in the key's LOCAL frame,
    at local z=0 (the top plane of the switch plate). Order is CCW when
    viewed from above (+z looking down): tl, bl, br, tr.

    u_width / u_height scale the cell so wide keys (e.g. 1.25u) get a
    correspondingly wider footprint; `switch_border` guarantees a minimum flat
    rim around the switch cutout (see _cell_half_extents).
    """
    hw, hh = _cell_half_extents(key, key_1u, hole_size, switch_border)
    # CCW from top view: top-left, bottom-left, bottom-right, top-right
    return [[
        (-hw,  hh, 0.0),
        (-hw, -hh, 0.0),
        ( hw, -hh, 0.0),
        ( hw,  hh, 0.0),
    ],[
        (-hw,  hh, -thickness),
        (-hw, -hh, -thickness),
        ( hw, -hh, -thickness),
        ( hw,  hh, -thickness),
    ]]

def key_hole_corners(key, key_1u, hole_size=14.5, thickness=5.0):
    """
    Return the four corners of a key's switch cutout in the key's LOCAL frame,
    at local z=0. Order is CCW when viewed from above (+z looking down):
    tl, bl, br, tr. The cutout is a square of side `hole_size` (mm), read from
    the keyboard/keylist data rather than hardcoded.
    """
    u = hole_size / 2.0
    # CCW from top view: top-left, bottom-left, bottom-right, top-right
    return [[
        (-u,  u, 0.0),
        (-u, -u, 0.0),
        ( u, -u, 0.0),
        ( u,  u, 0.0),
    ],[
        (-u,  u, -thickness),
        (-u, -u, -thickness),
        ( u, -u, -thickness),
        ( u,  u, -thickness),
    ]]


def key_edges_world(key, key_1u, thickness=5.0, hole_size=14.5,
                    switch_border=1.5):
    """
    Return the key's four top corners in WORLD space, keyed by name.
    Corner names use the cell's local orientation:
        tl (-x,+y), bl (-x,-y), br (+x,-y), tr (+x,+y)
    """
    tl, bl, br, tr = [_place(p, key) for p in
                      key_cell_corners(key, key_1u, thickness, hole_size,
                                       switch_border)[0]]
    btl, bbl, bbr, btr = [_place(p, key) for p in
                          key_cell_corners(key, key_1u, thickness, hole_size,
                                           switch_border)[1]]
    return [{'tl': tl, 'bl': bl, 'br': br, 'tr': tr}, {'tl': btl, 'bl': bbl, 'br': bbr, 'tr': btr}]


def _linked_targets(key):
    """
    Return this key's explicit links as {side: ((col,row), corner_or_None)}.

    A link value may be [col, row] (use the neighbour's full facing edge) or
    [col, row, "br"] where the third element names ONE corner of THIS key to
    attach the link to (tl/tr/br/bl). When a corner is named, the link bridge
    uses just that corner of this key instead of its full facing edge — which
    lets a key fan out to a cardinal neighbour on part of its edge and to an
    explicit link on another part, without the two bridges fighting over the
    whole edge.
    """
    out = {}
    lk = key.get('linked_keys', {}) or {}
    for side in ('l', 'r', 't', 'b'):
        v = lk.get(side)
        if v is None:
            continue
        if len(v) >= 3:
            out[side] = ((int(v[0]), int(v[1])), str(v[2]))
        else:
            out[side] = ((int(v[0]), int(v[1])), None)
    return out


def _explicit_link_pairs(keys, full_edge_only=False):
    """
    Collect explicit links as canonical unordered pairs of (col,row) cells.

    If full_edge_only is True, only links that use the FULL facing edge (no
    named corner on either endpoint) are returned. Single-corner links are
    excluded, because those don't span the junction and a corner patch is
    still wanted there to close the notch.
    """
    # Gather, per unordered pair, whether either endpoint named a corner.
    pair_has_corner = {}
    for k in keys:
        a = (k['col'], k['row'])
        for _side, (b, corner) in _linked_targets(k).items():
            pr = frozenset((a, b))
            pair_has_corner[pr] = pair_has_corner.get(pr, False) or (corner is not None)

    if not full_edge_only:
        return set(pair_has_corner.keys())
    return {pr for pr, has_c in pair_has_corner.items() if not has_c}


def find_neighbours(key, keys_by_cr):
    """
    Determine this key's neighbours on each side (l/r/t/b).

    Grid adjacency: same row & col±1 (l/r); same col & row±1 (t/b).
    Explicit links from the key's 'linked_keys' add EXTRA connections (e.g. a
    split thumb cluster joined diagonally). A key keeps its cardinal grid
    neighbours AND its explicit links — both are bridged. Because an explicit
    link may name a single corner to attach to (see _linked_targets), a link
    and a cardinal bridge can share that corner vertex without conflict.

    Returns a dict {side: [neighbour_key, ...]} — a list per side, since a side
    may have both a grid neighbour and one or more explicit links.
    """
    col, row = key['col'], key['row']
    neighs = {}

    grid = {
        'l': (col - 1, row),
        'r': (col + 1, row),
        't': (col, row - 1),
        'b': (col, row + 1),
    }
    my_links = _linked_targets(key)
    _opp = {'l': 'r', 'r': 'l', 't': 'b', 'b': 't'}

    for side, cr in grid.items():
        if cr not in keys_by_cr:
            continue
        # A FULL-EDGE link on this side (no named corner) claims the whole
        # edge, so the cardinal grid neighbour there yields to avoid two
        # bridges overlapping. A CORNER link only takes one corner, so the
        # cardinal neighbour is kept (they share just that corner vertex).
        if side in my_links and my_links[side][1] is None:
            continue
        # Reciprocal: if the grid neighbour full-edge-links back along the
        # shared edge, yield too.
        neigh = keys_by_cr[cr]
        neigh_links = _linked_targets(neigh)
        opp = _opp[side]
        if opp in neigh_links and neigh_links[opp][1] is None:
            continue
        neighs.setdefault(side, []).append(neigh)

    # Explicit links add to (don't replace) the neighbours on their side.
    for side, (cr, _corner) in my_links.items():
        if cr in keys_by_cr:
            neighs.setdefault(side, []).append(keys_by_cr[cr])

    return neighs


# Which pair of this key's corners face a neighbour on each side, and which
# pair of the neighbour's corners face back. Ordered so the two edges run in
# a consistent direction for building a non-self-intersecting bridge quad.
_FACING = {
    #        this-key edge        neighbour edge
    'r': (('tr', 'br'),          ('tl', 'bl')),
    'l': (('bl', 'tl'),          ('br', 'tr')),
    'b': (('br', 'bl'),          ('tr', 'tl')),
    't': (('tl', 'tr'),          ('bl', 'br')),
}


def _degenerate(p, q, tol=1e-6):
    """True if two points are effectively coincident."""
    return (abs(p[0]-q[0]) < tol and abs(p[1]-q[1]) < tol and abs(p[2]-q[2]) < tol)


def _polygon_scale(top):
    """
    Estimate the XY footprint scale of an N-sided (3 or 4 point) top
    polygon, used to detect degenerate corner patches.
    """
    n = len(top)
    xy = [(p[0], p[1]) for p in top]
    area2 = 0.0
    for i in range(n):
        x0, y0 = xy[i]
        x1, y1 = xy[(i + 1) % n]
        area2 += x0 * y1 - x1 * y0
    area = abs(area2) * 0.5
    avg_h = sum(p[2] for p in top) / n
    return area * abs(avg_h)


def diagonal_corner_patches(keys_by_cr, key_1u, link_pairs=None,
                            hole_size=14.5, switch_border=1.5):
    """
    Find the small gaps left at every grid corner where up to four key cells
    meet, and return a top-polygon (world points) to seal each one.

    Since each key's own footprint is inset from its cell (key_cell_corners
    shrinks the outer wall so switches don't touch), the edge-to-edge bridges
    built in build_shell() close the gap along each shared side, but the very
    centre of every 2x2 block of grid positions — where up to four keys'
    corners nearly meet — is left as a small hole. This is the mesh
    equivalent of the old OpenSCAD pipeline's intersections().

    For a block at grid origin (c, r), the four potential corners are:
        A = (c,   r)    -> contributes its 'br' corner (points into the block)
        B = (c+1, r)    -> contributes its 'bl' corner
        C = (c,   r+1)  -> contributes its 'tr' corner
        D = (c+1, r+1)  -> contributes its 'tl' corner
    Whichever of A/B/C/D exist are connected in cyclic order (A->B->D->C->A)
    to form a patch — a quad when all four keys are present, a triangle when
    exactly three are (one corner of the block is empty/ignored). Blocks
    where fewer than three of the four keys exist are skipped: two adjacent
    keys are already joined by an ordinary edge bridge, and two purely
    diagonal keys (no shared cardinal neighbour) are left alone since that
    gap may be intentional (e.g. a deliberately separated key group).

    If `link_pairs` is given (a set of frozenset({a_cr,b_cr})), any block whose
    diagonal (A-D or B-C) is an explicit link is SKIPPED here — a full edge-to-
    edge link bridge fills that junction instead, and a corner patch would
    overlap it.
    """
    if not keys_by_cr:
        return []

    link_pairs = link_pairs or set()
    cols = [c for c, r in keys_by_cr]
    rows = [r for c, r in keys_by_cr]
    patches = []

    order = ['A', 'C', 'D', 'B']
    corner_name = {'A': 'br', 'B': 'bl', 'D': 'tl', 'C': 'tr'}

    for c in range(min(cols), max(cols) + 1):
        for r in range(min(rows), max(rows) + 1):
            block = {
                'A': (c, r),
                'B': (c + 1, r),
                'C': (c, r + 1),
                'D': (c + 1, r + 1),
            }
            # Yield to an explicit link on either diagonal of this block.
            if (frozenset((block['A'], block['D'])) in link_pairs or
                    frozenset((block['B'], block['C'])) in link_pairs):
                continue

            present = {name: keys_by_cr[cr] for name, cr in block.items() if cr in keys_by_cr}
            if len(present) < 3:
                continue

            pts = []
            for name in order:
                if name in present:
                    key = present[name]
                    corners = key_edges_world(key, key_1u, 5.0, hole_size,
                                              switch_border)[0]
                    pts.append(corners[corner_name[name]])

            if len(pts) >= 3:
                patches.append(pts)

    return patches


def resolve_keylist(data):
    """
    Accept either an already-generated keylist dict (has a 'keylist' array)
    or a keyboard-definition dict (has x_algo/y_algo/width/height), and return
    a keylist dict ready for build_shell(). Keyboard definitions are expanded
    via keylist_gen.generate_keylist(); keylists are returned unchanged.

    Combined/assembly entries (those with an 'items' array) are not handled
    here — they reference other keyboards and need the multi-file pipeline.
    """
    if keylist_gen.is_keyboard_def(data):
        return keylist_gen.generate_keylist(data)
    return data


def is_assembly(data):
    """True if `data` is a combined/assembly entry (has an 'items' array)."""
    return isinstance(data, dict) and 'items' in data


def find_named_entry(name, catalog):
    """
    Resolve a board referenced by name from `catalog` (the full parsed JSON:
    a list of entries, or a single entry). Returns the matching buildable
    entry (a keyboard definition or keylist), or None. Assembly entries are
    skipped so an assembly can't reference another assembly by name.
    """
    entries = catalog if isinstance(catalog, list) else [catalog]
    for entry in entries:
        if not isinstance(entry, dict) or is_assembly(entry):
            continue
        if entry.get('name') == name:
            return entry
    return None


def transform_mesh(vertices, faces, pos=(0.0, 0.0, 0.0),
                   rot=(0.0, 0.0, 0.0), mirror=(0, 0, 0)):
    """
    Place a mesh in an assembly: apply mirror -> rotate -> translate to every
    vertex, in that order.

        mirror : per-axis flags (x, y, z); a non-zero flag reflects that axis
                 about 0 (so mirror=(1,0,0) flips X — a reflection across the
                 YZ plane). An odd number of flips reverses handedness, so we
                 also reverse face winding to keep normals pointing outward.
        rot    : Euler angles in degrees, OpenSCAD order (Z, then Y, then X).
        pos    : translation in mm, applied last.

    Returns (new_vertices, new_faces). Face index values are unchanged (only
    their order within each face may flip); vertices are new tuples.
    """
    sx = -1.0 if mirror[0] else 1.0
    sy = -1.0 if mirror[1] else 1.0
    sz = -1.0 if mirror[2] else 1.0
    reflected = (sx * sy * sz) < 0.0   # odd number of axis flips

    out_v = []
    for (x, y, z) in vertices:
        p = _rot_xyz((x * sx, y * sy, z * sz), rot[0], rot[1], rot[2])
        out_v.append((p[0] + pos[0], p[1] + pos[1], p[2] + pos[2]))

    if reflected:
        out_f = [tuple(reversed(f)) for f in faces]
    else:
        out_f = [tuple(f) for f in faces]
    return out_v, out_f


def build_shell_from_any(data):
    """
    Produce the constant-thickness perpendicular SHELL of the plate
    (build_shell), accepting either a keyboard JSON or a keylist JSON. Returns
    (vertices, faces) for a single closed manifold — no boolean union needed
    downstream. This is the plate entry point the Blender add-on uses.
    """
    return build_shell(resolve_keylist(data))


def _build_top_surface(keylist_data):
    """
    Build the connected TOP surface of the plate (key annuli + bridges +
    corner/diagonal patches) into a TopSurface, returning:
        (top, hole_vert_ids)
    where hole_vert_ids is the set of welded vertex indices that belong to
    switch-cutout rims (so callers can distinguish the outer perimeter from
    the switch holes).

    Shared by build_shell (which offsets a bottom) and build_walls (which
    sweeps a wall down from the outer perimeter), so the plate and walls are
    guaranteed to reference the same perimeter geometry and mate exactly.
    """
    keylist_data = resolve_keylist(keylist_data)
    key_1u = keylist_data.get('key_1u', 19.05)
    hole_size = keylist_data.get('hole_size', 14.5)
    switch_border = keylist_data.get('switch_border', 1.5)
    keys = keylist_data.get('keylist', [])
    keys_by_cr = {(k['col'], k['row']): k for k in keys}

    top = TopSurface()
    hole_vert_ids = set()
    # Per welded vertex: the offset normal to use for the constant-thickness
    # bottom. Cell-corner (outer ring) vertices belong to exactly one key (the
    # cells are inset, so no two keys share a corner), and their bottom should
    # drop along THAT KEY'S plane normal — not the face-average, which gets
    # dragged sideways by bridge/patch faces fanning off the vertex and would
    # bow the underside wall outward. Hole-ring and any other vertices fall
    # back to the averaged normal.
    offset_normal = {}

    def _key_plane_normal(key):
        o = _place((0.0, 0.0, 0.0), key)
        z = _place((0.0, 0.0, 1.0), key)
        return _norm((z[0] - o[0], z[1] - o[1], z[2] - o[2]))

    def cell_top_world(key):
        return [_place(p, key) for p in
                _cell_ring(key, key_1u, hole_size, switch_border)]

    def hole_top_world(key):
        return [_place(p, key) for p in _hole_ring(key, key_1u, hole_size)]

    # --- 1) Key tops: annulus (outer cell ring minus inner hole ring) ---
    for key in keys:
        outer = cell_top_world(key)
        inner = hole_top_world(key)
        for i in range(4):
            j = (i + 1) % 4
            top.face([outer[i], outer[j], inner[j], inner[i]])
        kn = _key_plane_normal(key)
        for p in outer:
            offset_normal[top.vert(p)] = kn      # cell corner -> key normal
        for p in inner:
            hole_vert_ids.add(top.vert(p))

    # --- 2) Bridge tops between adjacent keys ---
    # Two kinds of connection, both bridged here:
    #   * cardinal grid neighbours -> full facing-edge bridge (tr,br <-> tl,bl)
    #   * explicit links           -> full facing-edge bridge, OR, if the link
    #                                 names a corner of THIS key, a bridge from
    #                                 just that one corner to the neighbour's
    #                                 full facing edge (a triangle). This lets a
    #                                 key fan out to a cardinal neighbour on part
    #                                 of its edge and a link on another corner.
    #
    # A key keeps BOTH its cardinal bridges and its link bridges. Where a link
    # names a corner, that link bridge shares only a corner vertex with the
    # cardinal bridge (manifold-safe), not a whole edge.
    link_pairs = _explicit_link_pairs(keys)

    # Collect relationships, de-duplicated per unordered key pair. When the same
    # pair is both a grid neighbour and an explicit link, the explicit link
    # (which may carry a corner) takes the slot.
    rels = {}   # frozenset(pair) -> (this_key, neigh_key, side, corner)
    for key in keys:
        this_cr = (key['col'], key['row'])
        my_links = _linked_targets(key)
        for side, neighbours in find_neighbours(key, keys_by_cr).items():
            for neigh in neighbours:
                neigh_cr = (neigh['col'], neigh['row'])
                pair = frozenset((this_cr, neigh_cr))
                # Determine if THIS relationship (key -> neigh on `side`) is an
                # explicit link, and grab its corner if so.
                corner = None
                is_link = False
                if side in my_links and my_links[side][0] == neigh_cr:
                    is_link = True
                    corner = my_links[side][1]
                # Prefer a link record (carries corner) over a plain grid one.
                if pair not in rels or (is_link and rels[pair][3] is None and not rels[pair][4]):
                    rels[pair] = (key, neigh, side, corner, is_link)

    for (key, neigh, side, corner, _is_link) in rels.values():
        this_corners = key_edges_world(key, key_1u, 5.0, hole_size,
                                       switch_border)[0]
        neigh_corners = key_edges_world(neigh, key_1u, 5.0, hole_size,
                                        switch_border)[0]
        (a0, a1), (n0, n1) = _FACING[side]

        if corner is None:
            # Full facing edge on this key -> full facing edge on neighbour.
            p_a0 = this_corners[a0]; p_a1 = this_corners[a1]
            p_n0 = neigh_corners[n0]; p_n1 = neigh_corners[n1]
            if _degenerate(p_a0, p_n0) and _degenerate(p_a1, p_n1):
                continue
            bridge_top = [p_a0, p_a1, p_n1, p_n0]
            if _polygon_scale([(p[0], p[1], 1.0) for p in bridge_top]) < 1.0:
                continue
            top.face(bridge_top)
        else:
            # Single named corner of THIS key -> the neighbour's full facing
            # edge, as a triangle [this.corner, neigh.n1, neigh.n0]. This fans
            # this.corner out to BOTH of the neighbour's facing corners; the
            # outer boundary edge (this.corner -> neighbour's far corner) then
            # walls straight down via build_shell's boundary stitching. The
            # corner-infill patch closes the notch on the cardinal side.
            pc = this_corners[corner]
            p_n0 = neigh_corners[n0]
            p_n1 = neigh_corners[n1]
            tri = [pc, p_n1, p_n0]
            if _polygon_scale([(p[0], p[1], 1.0) for p in tri]) >= 0.1:
                top.face(tri)

    # --- 3) Corner patches where 3-4 keys meet. Only FULL-EDGE links suppress
    # the patch (they span the junction). Single-corner links still want the
    # patch to close the notch between the link triangle and the cardinal
    # bridge, so those are excluded from the yield set.
    yield_pairs = _explicit_link_pairs(keys, full_edge_only=True)
    for patch_top in diagonal_corner_patches(keys_by_cr, key_1u, yield_pairs,
                                             hole_size, switch_border):
        if _polygon_scale([(p[0], p[1], 1.0) for p in patch_top]) < 0.1:
            continue
        top.face(patch_top)

    top.offset_normal = offset_normal
    return top, hole_vert_ids


def _skirt_profile(keylist_data):
    """
    Normalise the skirt's outer cross-section into a list of segments, each a
    dict {'frac': f, 'angle': deg or None, 'out': mm or None}.

    JSON `skirt_profile` is a list of steps, walked top-to-bottom from just
    below the plate down to `wall_base_z`:

        "skirt_profile": [
          {"fraction": 0.3, "angle": 5},    # 30% of the drop, 5deg outward
          {"fraction": 0.0, "out": 2.0},    # a horizontal ledge, 2mm out
          {"fraction": 0.7, "angle": -3}    # rest of the drop, 3deg INWARD
        ]

    `fraction` is a share of the LOCAL drop height (ztop - base_z), not an
    absolute mm — the perimeter's height varies around the board, so fractions
    keep every step in proportion and guarantee the sweep lands exactly on
    base_z. Fractions are normalised, so they need not sum to 1. A segment with
    fraction 0 is a horizontal step (a ledge); it must give `out`.

    Each segment sets its outward run either by `angle` (draft angle from
    vertical, positive = outward) or by an explicit `out` in mm. `out` wins if
    both are given.

    When `skirt_profile` is absent, falls back to the single-segment
    skirt_mode/skirt_angle/skirt_flare behaviour.
    """
    prof = keylist_data.get('skirt_profile')
    if not prof:
        mode = str(keylist_data.get('skirt_mode', 'angle')).lower()
        if mode == 'flare':
            prof = [{'fraction': 1.0, 'out': keylist_data.get('skirt_flare', 0.0)}]
        else:
            prof = [{'fraction': 1.0, 'angle': keylist_data.get('skirt_angle', 0.0)}]

    segs = []
    for s in prof:
        segs.append({
            'frac': float(s.get('fraction', 0.0)),
            'angle': (float(s['angle']) if 'angle' in s else None),
            'out': (float(s['out']) if 'out' in s else None),
        })

    total = sum(s['frac'] for s in segs)
    if total <= 1e-9:
        raise ValueError("skirt_profile: fractions sum to zero — at least one "
                         "segment must have a non-zero 'fraction'")
    for s in segs:
        s['frac'] /= total
        if s['out'] is None and s['angle'] is None:
            s['angle'] = 0.0
        if s['frac'] == 0.0 and s['out'] is None:
            raise ValueError("skirt_profile: a zero-fraction (horizontal) step "
                             "must specify 'out'")
    return segs


def _signed_area(pts):
    """Signed area of a polygon in XY (positive = CCW)."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def _segs_properly_cross(p, q, r, s):
    """True if segments pq and rs cross at an interior point of both."""
    d1 = _tri_area2(r, s, p)
    d2 = _tri_area2(r, s, q)
    d3 = _tri_area2(p, q, r)
    d4 = _tri_area2(p, q, s)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _point_in_poly(p, poly):
    """Ray-cast point-in-polygon test (XY)."""
    x, y = p[0], p[1]
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i][0], poly[i][1]
        x1, y1 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        if (y0 > y) != (y1 > y):
            xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xin:
                inside = not inside
    return inside


def _bridge_holes(outer, holes):
    """
    Merge circular/polygonal HOLES into an OUTER polygon by cutting a
    zero-width bridge to each, producing one simple polygon that ear clipping
    can triangulate.

    `outer` is CCW; each hole is reversed to CW so the merged traversal keeps
    the hole's interior on the outside of the material. Holes are bridged in
    order of decreasing max-x, which keeps each bridge clear of the ones still
    to come.
    """
    poly = list(outer)
    if _signed_area(poly) < 0:
        poly.reverse()

    holes = sorted(holes, key=lambda h: -max(p[0] for p in h))
    for hole in holes:
        h = list(hole)
        if _signed_area(h) > 0:
            h.reverse()   # holes traverse CW

        # Bridge from the hole's right-most vertex.
        mi = max(range(len(h)), key=lambda i: h[i][0])
        M = h[mi]

        # Pick the nearest polygon vertex that M can "see": the bridge must not
        # properly cross any polygon or hole edge, and must run through
        # material (its midpoint inside the polygon, outside every hole).
        best, best_d = None, None
        n = len(poly)
        for pi in range(n):
            P = poly[pi]
            ok = True
            for j in range(n):
                a, b = poly[j], poly[(j + 1) % n]
                if j == pi or (j + 1) % n == pi:
                    continue
                if _segs_properly_cross(M, P, a, b):
                    ok = False
                    break
            if ok:
                for j in range(len(h)):
                    a, b = h[j], h[(j + 1) % len(h)]
                    if j == mi or (j + 1) % len(h) == mi:
                        continue
                    if _segs_properly_cross(M, P, a, b):
                        ok = False
                        break
            if ok:
                mid = ((M[0] + P[0]) / 2.0, (M[1] + P[1]) / 2.0)
                if not _point_in_poly(mid, poly):
                    ok = False
                elif _point_in_poly(mid, h):
                    ok = False
            if ok:
                d = (M[0] - P[0]) ** 2 + (M[1] - P[1]) ** 2
                if best_d is None or d < best_d:
                    best, best_d = pi, d

        if best is None:
            raise ValueError("insert hole could not be bridged to the outline "
                             "(is it outside the baseplate, or overlapping?)")

        # Splice: ...P, hole[mi..end], hole[0..mi], P...
        poly = (poly[:best + 1]
                + h[mi:] + h[:mi + 1]
                + poly[best:])
    return poly


def _coincident(a, b, eps=1e-9):
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _triangulate_with_holes(outer, holes):
    """
    Triangulate a CCW outer polygon containing `holes` (lists of points).
    Returns (points, triangles) where `points` is the merged vertex list and
    each triangle is a CCW index triple into it.
    """
    if not holes:
        pts = list(outer)
        return pts, _ear_clip(pts)
    merged = _bridge_holes(outer, holes)
    return merged, _ear_clip(merged)


def _circle_pts(cx, cy, r, segments=32):
    """CCW circle as a list of (x, y)."""
    return [(cx + r * math.cos(2.0 * math.pi * i / segments),
             cy + r * math.sin(2.0 * math.pi * i / segments))
            for i in range(segments)]


def _tri_area2(a, b, c):
    """Twice the signed area of triangle abc in XY (positive = CCW)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_tri(p, a, b, c):
    """True if p lies inside (or on) triangle abc."""
    d1 = _tri_area2(p, a, b)
    d2 = _tri_area2(p, b, c)
    d3 = _tri_area2(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _ear_clip(pts):
    """
    Triangulate a SIMPLE polygon (concave allowed, no holes) given as an
    ordered list of (x, y). Returns a list of (i, j, k) index triples wound
    CCW. Pure Python — core.py must stay dependency-free for Blender.
    """
    n = len(pts)
    if n < 3:
        return []

    idx = list(range(n))
    # Work on a CCW traversal so emitted triangles come out CCW.
    area2 = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                for i in range(n))
    if area2 < 0:
        idx.reverse()

    tris = []
    guard = 0
    limit = 10 * n * n + 100
    while len(idx) > 2 and guard < limit:
        guard += 1
        clipped = False
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if _tri_area2(a, b, c) <= 1e-12:
                continue  # reflex or degenerate corner: not an ear
            # An ear must contain no other vertex. Bridged holes duplicate the
            # two bridge endpoints, so skip any vertex that coincides with a
            # corner of the candidate ear rather than testing it.
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = pts[j]
                if (_coincident(p, a) or _coincident(p, b) or _coincident(p, c)):
                    continue
                if _point_in_tri(p, a, b, c):
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                clipped = True
                break
        if not clipped:
            break

    if len(idx) > 2:
        raise ValueError("baseplate: could not triangulate the outline "
                         "(is it self-intersecting?)")
    return tris


def _skirt_outer_rings(keylist_data):
    """
    The skirt's OUTER footprint polygon(s) at `wall_base_z`, as ordered CCW
    lists of (x, y). One polygon per outer perimeter loop (disjoint key groups
    give more than one). Shares the sweep maths with build_shell, so the
    baseplate matches the case exactly.
    """
    keylist_data = resolve_keylist(keylist_data)
    if not keylist_data.get('skirt', False):
        raise ValueError("baseplate requires the fused-skirt wall method "
                         "(set \"skirt\": true)")

    thickness = keylist_data.get('thickness', 5.0)
    base_z = keylist_data.get('wall_base_z', 0.0)
    flange = keylist_data.get('skirt_flange', 0.0)
    segs = _skirt_profile(keylist_data)

    top, hole_ids = _build_top_surface(keylist_data)
    unit = top.unit_normals()
    override = getattr(top, 'offset_normal', {})

    # The skirt forces an aligned perimeter, so each perimeter vertex's XY is
    # its bottom-offset XY (same as build_shell uses).
    pts = list(top.points)
    for lp in _perimeter_loops(top, hole_ids):
        for vi in lp:
            u = override.get(vi, unit[vi])
            p = top.points[vi]
            pts[vi] = (p[0] - u[0] * thickness, p[1] - u[1] * thickness, p[2])

    loops = _perimeter_loops(top, hole_ids)
    if not loops:
        return []

    def _bbox_area(lp):
        xs = [pts[v][0] for v in lp]
        ys = [pts[v][1] for v in lp]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    areas = [_bbox_area(lp) for lp in loops]
    amax = max(areas)

    rings = []
    for lp, a in zip(loops, areas):
        if a < 0.5 * amax:
            continue  # interior hole: no skirt, no baseplate outline
        olp, nrms = _outward_normals_xy(pts, lp)
        ring = []
        for vi, (nx, ny) in zip(olp, nrms):
            p = pts[vi]
            drop = p[2] - base_z
            d = flange
            for s in segs:
                dz = s['frac'] * drop
                if s['out'] is not None:
                    d += s['out']
                else:
                    d += dz * math.tan(math.radians(s['angle']))
            ring.append((p[0] + nx * d, p[1] + ny * d))
        rings.append(ring)
    return rings


def build_baseplate(keylist_data):
    """
    Build the BASEPLATE: a flat bottom cover matching the fused skirt's outer
    footprint at `wall_base_z`, extruded DOWNWARD by `baseplate_thickness`
    (default 2mm). The case therefore rests on its top face, flush — there is
    no clearance gap, the two share the same outline exactly.

    Requires the fused-skirt wall method. Returns (vertices, faces) — one
    closed manifold per outer perimeter loop.

    JSON fields:
        baseplate_thickness : cover thickness in mm (default 2.0)
        wall_base_z         : the plane the case sits on (default 0)
    """
    keylist_data = resolve_keylist(keylist_data)
    t = keylist_data.get('baseplate_thickness', 2.0)
    base_z = keylist_data.get('wall_base_z', 0.0)
    if t <= 0:
        raise ValueError("baseplate_thickness must be > 0")

    inserts = insert_positions(keylist_data)
    segments = int(keylist_data.get('insert_hole_segments', 32))

    vertices = []
    faces = []

    for ring in _skirt_outer_rings(keylist_data):
        n = len(ring)
        if n < 3:
            continue

        # Screw clearance holes: coaxial with each insert's actual hole (which
        # may be offset from the disc centre by hole_x/hole_y), for those that
        # fall inside this ring.
        holes = []
        for ins in inserts:
            if ins['clearance_d'] <= 0:
                continue
            a = math.radians(ins['rot'])
            ca, sa = math.cos(a), math.sin(a)
            hx, hy = ins['hole_x'], ins['hole_y']
            wx = ins['x'] + hx * ca - hy * sa
            wy = ins['y'] + hx * sa + hy * ca
            if _point_in_poly((wx, wy), ring):
                holes.append(_circle_pts(wx, wy,
                                         ins['clearance_d'] / 2.0, segments))

        merged, tris = _triangulate_with_holes(ring, holes)

        # A single welded vertex pool per z-plane, so the caps and the side
        # walls share vertices (the bridged polygon repeats its bridge
        # endpoints, and hole points appear in both the cap and its wall).
        top_i, bot_i = {}, {}

        def vt(p):
            k = (round(p[0], 6), round(p[1], 6))
            if k not in top_i:
                top_i[k] = len(vertices)
                vertices.append((p[0], p[1], base_z))
            return top_i[k]

        def vb(p):
            k = (round(p[0], 6), round(p[1], 6))
            if k not in bot_i:
                bot_i[k] = len(vertices)
                vertices.append((p[0], p[1], base_z - t))
            return bot_i[k]

        # Top cap: CCW -> normal +Z (the case rests on this face).
        for (i, j, k) in tris:
            faces.append((vt(merged[i]), vt(merged[j]), vt(merged[k])))
        # Bottom cap: reversed -> normal -Z.
        for (i, j, k) in tris:
            faces.append((vb(merged[k]), vb(merged[j]), vb(merged[i])))

        # Outer side wall, outward-facing for the CCW ring.
        for i in range(n):
            j = (i + 1) % n
            faces.append((vt(ring[i]), vb(ring[i]), vb(ring[j]), vt(ring[j])))

        # Hole walls: traverse each hole CW so the same winding rule points the
        # normals into the hole (i.e. away from the material).
        for hole in holes:
            hw = list(hole)
            if _signed_area(hw) > 0:
                hw.reverse()
            m = len(hw)
            for i in range(m):
                j = (i + 1) % m
                faces.append((vt(hw[i]), vb(hw[i]), vb(hw[j]), vt(hw[j])))

    return vertices, faces


def build_baseplate_from_any(data):
    """build_baseplate accepting a keyboard or keylist JSON."""
    return build_baseplate(resolve_keylist(data))


def insert_positions(keylist_data):
    """
    Resolve every threaded-insert holder to world coordinates and parameters.

    Matches the original OpenSCAD pipeline: an insert is attached to a key and
    offset from that key's position by (x * u_width, y * u_height). The holder
    stands on the base plane (`wall_base_z`) — it does NOT follow the key's
    tilt — rising `height` mm.

    Per-insert fields (all optional except x/y):
        x, y         : disc-centre offset from the key centre, scaled by u size
        id           : hole diameter for the heat-set insert (4.0 -> Ø4)
        od           : disc diameter; radius = od/2 (8.0 -> Ø8 disc)
        height       : holder height above the base plane (4.2)
        rot          : holder rotation about Z, degrees (0)
        leg_0, leg_1, leg_2 : length of the left, centre and right legs, i.e.
                         each leg-tip's distance from the disc centre. The
                         holder outline is the hull of these three legs plus the
                         disc (see _insert_boss_outline). (5.0, 7.0, 5.0)
        hole_x, hole_y : Ø`id` hole-centre offset from the disc centre, mm
                         (0, 0). Clamped to keep the hole inside the disc.
        clearance_d  : baseplate screw clearance hole diameter (3.0)

    Returns a list of dicts with absolute 'x'/'y' plus the resolved parameters.
    """
    keylist_data = resolve_keylist(keylist_data)
    default_clear = keylist_data.get('insert_clearance_d', 3.0)
    out = []
    for key in keylist_data.get('keylist', []):
        ins = key.get('insert') or {}
        if ins.get('x') is None or ins.get('y') is None:
            continue
        u = key.get('u_width', 1)
        h = key.get('u_height', 1)
        out.append({
            'x': key['pos']['x'] + float(ins['x']) * u,
            'y': key['pos']['y'] + float(ins['y']) * h,
            'id': float(ins.get('id', 4.0)),
            'od': float(ins.get('od', 8.0)),
            'height': float(ins.get('height', 4.2)),
            'rot': float(ins.get('rot') or 0.0),
            'hole_x': float(ins.get('hole_x') or 0.0),
            'hole_y': float(ins.get('hole_y') or 0.0),
            'clearance_d': float(ins.get('clearance_d', default_clear)),
            'leg_0': float(ins.get('leg_0', 5.0)),
            'leg_1': float(ins.get('leg_1', 7.0)),
            'leg_2': float(ins.get('leg_2', 5.0)),
            'col': key['col'], 'row': key['row'],
        })
    return out


def _cross(o, a, b):
    """2D cross product OA x OB."""
    return (
        (a[0] - o[0]) * (b[1] - o[1])
        - (a[1] - o[1]) * (b[0] - o[0])
    )


def _convex_hull(points):
    """
    Andrew monotonic chain convex hull.
    Returns points CCW without repeating the first point.
    """
    pts = sorted(set(points))

    if len(pts) <= 1:
        return pts

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _circle_points(r, segments=64):
    """
    Polygon approximation of a circle.
    CCW order.
    """
    pts = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        pts.append((
            r * math.cos(a),
            r * math.sin(a)
        ))
    return pts


def _insert_boss_outline(ins, segments=64):
    """
    Cross-section of a threaded-insert holder: the "wishbone" from insert.scad.

    The holder is the UNION of two convex hulls that share the central disc and
    the centre-leg tip, giving a three-legged spider around the Ø`od` disc:

        union() {
            hull(circle(r), left,  top);   # left  leg + centre leg
            hull(circle(r), right, top);   # right leg + centre leg
        }

    where, in the local frame (before `rot` and translation):

        left  = (-r, leg_0)     # left  leg tip
        top   = ( 0, leg_1)     # centre leg tip
        right = ( r, leg_2)     # right leg tip

    so each leg's length is set independently by leg_0 / leg_1 / leg_2 (the tip
    distance from the disc centre along that leg). The three legs let the holder
    reach out to nearby walls/corners and fuse into them for rigidity.

    Placement guidance: the disc/junction is the anchor; the legs should reach
    toward the case walls so they weld into solid material. When choosing where
    to seat an insert, prioritise contact at the OUTER corners over the inner
    ones — a leg tying into an outer corner braces the holder far more than one
    reaching an inner corner, so it is worth positioning/orienting the insert to
    catch the outer walls first.

    Returns a CCW polygon (list of (x, y)) in the local frame.
    """
    r = ins["od"] / 2.0
    leg0 = ins["leg_0"]
    leg1 = ins["leg_1"]
    leg2 = ins["leg_2"]

    left = (-r, leg0)
    top = (0.0, leg1)
    right = (r, leg2)

    circle = _circle_points(r, segments)

    # The two OpenSCAD hulls. Both contain the same circular core and the shared
    # centre-leg tip, so their union is convex and equals the convex hull of all
    # the boundary points combined — which is what we build here.
    left_hull = _convex_hull(circle + [left, top])
    right_hull = _convex_hull(circle + [right, top])

    result = _convex_hull(left_hull + right_hull)
    if _signed_area(result) < 0:
        result.reverse()
    return result


def build_inserts(keylist_data):
    """
    Build the threaded-insert holders: for each insert, a prism whose outline is
    the three-leg wishbone (see _insert_boss_outline) drilled through the centre
    at Ø`id` for the heat-set insert's press-fit hole. Each holder stands on the
    base plane and rises `height` mm.

    Returns (vertices, faces) — one closed manifold per insert. These are the
    same bodies the original OpenSCAD unioned into the case.
    """
    keylist_data = resolve_keylist(keylist_data)
    base_z = keylist_data.get('wall_base_z', 0.0)
    segments = int(keylist_data.get('insert_hole_segments', 32))

    vertices = []
    faces = []
    for ins in insert_positions(keylist_data):
        outline = _insert_boss_outline(ins, segments)
        if len(outline) < 3:
            continue

        # Hole centre offset from the disc centre (local frame, before rot).
        # Clamp it so the Ø`id` hole stays inside the Ø`od` disc with a little
        # wall around it.
        rid = ins['id'] / 2.0
        r_disc = ins['od'] / 2.0
        hx, hy = ins['hole_x'], ins['hole_y']

        hd = math.hypot(hx, hy)
        max_hd = max(0.0, r_disc - rid - 0.01)
        if hd > max_hd and hd > 1e-9:
            hx, hy = hx / hd * max_hd, hy / hd * max_hd
        hole = _circle_pts(hx, hy, rid, segments)

        # Place: rotate the local cross-section by `rot`, then translate.
        a = math.radians(ins['rot'])
        ca, sa = math.cos(a), math.sin(a)

        def place(p):
            return (ins['x'] + p[0] * ca - p[1] * sa,
                    ins['y'] + p[0] * sa + p[1] * ca)

        outline_w = [place(p) for p in outline]
        hole_w = [place(p) for p in hole]
        if _signed_area(outline_w) < 0:
            outline_w.reverse()

        # If the hole doesn't sit strictly inside the holder outline, drop it —
        # a solid holder still prints, and it flags that the disc needs to be
        # bigger or the hole offset moved.
        if not all(_point_in_poly(p, outline_w) for p in hole_w):
            merged, tris = _triangulate_with_holes(outline_w, [])
            hole_w = None
        else:
            merged, tris = _triangulate_with_holes(outline_w, [hole_w])
        z0, z1 = base_z, base_z + ins['height']

        top_i, bot_i = {}, {}

        def vt(p):
            k = (round(p[0], 6), round(p[1], 6))
            if k not in top_i:
                top_i[k] = len(vertices)
                vertices.append((p[0], p[1], z1))
            return top_i[k]

        def vb(p):
            k = (round(p[0], 6), round(p[1], 6))
            if k not in bot_i:
                bot_i[k] = len(vertices)
                vertices.append((p[0], p[1], z0))
            return bot_i[k]

        for (i, j, k) in tris:                    # top cap, +Z
            faces.append((vt(merged[i]), vt(merged[j]), vt(merged[k])))
        for (i, j, k) in tris:                    # bottom cap, -Z
            faces.append((vb(merged[k]), vb(merged[j]), vb(merged[i])))

        m = len(outline_w)                        # outer wall
        for i in range(m):
            j = (i + 1) % m
            faces.append((vt(outline_w[i]), vb(outline_w[i]),
                          vb(outline_w[j]), vt(outline_w[j])))

        if hole_w is not None:
            hw = list(hole_w)                     # hole wall, facing the hole
            if _signed_area(hw) > 0:
                hw.reverse()
            m = len(hw)
            for i in range(m):
                j = (i + 1) % m
                faces.append((vt(hw[i]), vb(hw[i]), vb(hw[j]), vt(hw[j])))

    return vertices, faces


def build_inserts_from_any(data):
    """build_inserts accepting a keyboard or keylist JSON."""
    return build_inserts(resolve_keylist(data))


def _stitch_columns(faces, vertices, col_a, col_b):
    """
    Stitch two vertical vertex columns (each a list of vertex indices ordered
    bottom -> top) into a triangle strip forming the panel between them.

    The columns share their bottom z and their top z but may have different
    numbers of intermediate points (perimeter vertices can sit at different
    heights, so the constant-thickness inner face has a per-vertex ring count).
    We advance whichever side's next vertex is lower, so the triangles never
    cross and the panel stays watertight regardless of the counts.

    Winding matches the original inner-face quad (b, a, a_next, b_next): the
    panel faces inward/outward consistently with the rest of the skirt band.
    """
    ia = ib = 0
    za = [vertices[i][2] for i in col_a]
    zb = [vertices[i][2] for i in col_b]
    na, nb = len(col_a), len(col_b)
    while ia < na - 1 or ib < nb - 1:
        can_a = ia < na - 1
        can_b = ib < nb - 1
        # Advance the side whose next vertex is lower (or the only one that can).
        if can_a and (not can_b or za[ia + 1] <= zb[ib + 1]):
            faces.append((col_b[ib], col_a[ia], col_a[ia + 1]))
            ia += 1
        else:
            faces.append((col_b[ib], col_a[ia], col_b[ib + 1]))
            ib += 1


def build_shell(keylist_data):
    """
    Build a constant-thickness SHELL of the key plate (no perimeter walls yet).

    The top surface follows the tilted switch planes; the bottom surface is a
    uniform `thickness` offset PERPENDICULAR to the top (along shared averaged
    vertex normals), so adjacent tilted pieces keep a seamless underside with
    no troughs. Switch cutouts pass through the shell perpendicular to the top.

    Returns (vertices, faces) — a single closed manifold shell.
    """
    keylist_data = resolve_keylist(keylist_data)
    thickness = keylist_data.get('thickness', 5.0)
    # When True (default), the plate's outer perimeter edge is made VERTICAL so
    # the plate drops straight down into the wall recess. When False, the outer
    # edge stays PERPENDICULAR to the tilted top (the original constant-
    # thickness shell) — use this to print the plate standalone without walls.
    vertical_edges = keylist_data.get('vertical_edges', True)

    # A fused skirt REQUIRES an aligned (vertical) perimeter: the skirt's inner
    # face rises from the flat rim to the plate's bottom perimeter, so if the
    # bottom perimeter were offset outward (perpendicular edge) it would poke
    # through the skirt's outer face, giving a self-intersecting solid. Since
    # the plate's own edge band is not emitted when skirting, nothing is lost.
    if keylist_data.get('skirt', False):
        vertical_edges = True

    top, hole_ids = _build_top_surface(keylist_data)

    # --- Tent (side-to-side) and pitch (front-to-back) of the whole plate ----
    # Tilt the finished key plate as a rigid body BEFORE the walls/skirt are
    # built, so the walls still sweep straight down to `wall_base_z` from the
    # now-tilted perimeter and the case sits flat on the desk.
    #   tent_angle  : rotation about the Y axis (raises one side; ergonomic tent)
    #   pitch_angle : rotation about the X axis (raises the far/near edge)
    # Degrees. Rotation is about the plate's XY centre so it tilts in place; we
    # then lift the plate so its lowest point clears the base by `plate_lift`
    # (or wall_base_z), keeping every wall a positive height.
    tent_angle = float(keylist_data.get('tent_angle', 0.0) or 0.0)
    pitch_angle = float(keylist_data.get('pitch_angle', 0.0) or 0.0)
    _tilted = bool(tent_angle or pitch_angle)
    if _tilted:
        pts = top.points
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        def _tilt(p):
            q = _rot_xyz((p[0] - cx, p[1] - cy, p[2]),
                         pitch_angle, tent_angle, 0.0)
            return (q[0] + cx, q[1] + cy, q[2])

        top.points = [_tilt(p) for p in pts]
        # Rotate the accumulated face normals and any per-vertex offset normals
        # the same way, so the bottom offset still drops perpendicular to the
        # (now tilted) top rather than in a stale direction.
        top.normals = [list(_rot_xyz(tuple(nz), pitch_angle, tent_angle, 0.0))
                       for nz in top.normals]
        if getattr(top, 'offset_normal', None):
            top.offset_normal = {
                vi: _rot_xyz(nrm, pitch_angle, tent_angle, 0.0)
                for vi, nrm in top.offset_normal.items()}
        # The vertical lift (so the tilted plate clears the base) happens after
        # the bottom surface is built, since the plate UNDERSIDE is what must
        # stay above wall_base_z for the walls to have positive height.

    # --- Compute the offset bottom. Cell-corner vertices use their key's
    # plane normal (recorded in top.offset_normal) so the underside drops
    # cleanly; all other vertices use the area-averaged normal. ---
    unit = top.unit_normals()
    override = getattr(top, 'offset_normal', {})
    top_pts = list(top.points)
    bot_pts = []
    for vi, (p, ua) in enumerate(zip(top_pts, unit)):
        u = override.get(vi, ua)
        bot_pts.append((p[0] - u[0] * thickness,
                        p[1] - u[1] * thickness,
                        p[2] - u[2] * thickness))

    # If the plate was tented/pitched, lift the whole shell so its lowest point
    # (top OR bottom surface) clears the base by `plate_min_wall`, guaranteeing
    # every wall has positive height and drops cleanly to wall_base_z.
    if _tilted:
        base_z0 = float(keylist_data.get('wall_base_z', 0.0))
        min_clear = float(keylist_data.get('plate_min_wall', 1.0))
        min_z = min(min(p[2] for p in top.points),
                    min(p[2] for p in bot_pts))
        if min_z < base_z0 + min_clear:
            dz = (base_z0 + min_clear) - min_z
            top.points = [(p[0], p[1], p[2] + dz) for p in top.points]
            top_pts = [(p[0], p[1], p[2] + dz) for p in top_pts]
            bot_pts = [(p[0], p[1], p[2] + dz) for p in bot_pts]

    # --- Optionally make the OUTER perimeter edge vertical so the plate drops
    # straight into the wall recess. The perpendicular bottom offset pushes
    # each perimeter bottom vertex horizontally away from its top vertex; on a
    # tilted key that leaves the plate's outer edge slanted, which collides
    # with the wall's vertical inner face. We fix this by moving each outer-
    # perimeter TOP vertex horizontally to sit directly above its bottom
    # vertex (keeping its own z) — i.e. extending the top surface out to the
    # bottom footprint. The result: a vertical outer wall from the (moved)
    # top edge straight down to the bottom edge. Hole rims and interior
    # vertices are untouched (switch cutouts stay perpendicular).
    #
    # Disabled by vertical_edges=False, which keeps the original constant-
    # thickness perpendicular outer edge (for printing the plate on its own).
    if vertical_edges:
        for lp in _perimeter_loops(top, hole_ids):
            for vi in lp:
                _bz = bot_pts[vi][2]
                tx, ty, tz = top_pts[vi]
                bot_pts[vi] = (tx, ty, _bz)  # bottom now directly below top

    # Assemble the final shell mesh.
    vertices = list(top_pts) + list(bot_pts)
    n = len(top_pts)
    faces = []

    # --- Optional fused SKIRT walls ---------------------------------------
    # Instead of a separate recess frame (build_walls), the plate can grow its
    # own walls: sweep a skirt down from the outer perimeter to `wall_base_z`,
    # flaring outward, ending in a flat annular rim. The skirt is part of the
    # same closed solid, so there is nothing to mate and no clearance needed.
    #
    # Cross-section at each perimeter vertex, `d` measured outward along the
    # perimeter's XY normal:
    #     P0 = plate top perimeter              (d=0,                  z=ztop)
    #     P1 = outward flange                   (d=flange,             z=ztop)
    #     P2 = outer bottom                     (d=flange+flare,       z=base)
    #     P3 = inner bottom (flat rim)          (d=flange+flare-wt,    z=base)
    #     P4 = plate bottom perimeter           (d=0,                  z=zbot)
    # The plate's own outer edge band is NOT emitted — it becomes interior
    # material where the plate meets the skirt.
    #
    # `flare` depends on skirt_mode:
    #   'angle' : constant draft angle; flare = (ztop-base) * tan(angle), so the
    #             outward run varies with each vertex's height.
    #   'flare' : constant outward run `skirt_flare` everywhere, so the draft
    #             angle varies with height.
    skirt = keylist_data.get('skirt', False)
    wall_thickness = keylist_data.get('wall_thickness', 2.0)
    skirt_flange = keylist_data.get('skirt_flange', 0.0)
    skirt_mode = str(keylist_data.get('skirt_mode', 'angle')).lower()
    skirt_angle = keylist_data.get('skirt_angle', 0.0)
    skirt_flare = keylist_data.get('skirt_flare', 0.0)
    base_z = keylist_data.get('wall_base_z', 0.0)
    # Inner skirt face: False (default) = a single sloped panel from the rim up
    # to the plate underside, so the wall thickens as the outer face flares
    # away. True = the inner face parallels the outer profile at a constant
    # `wall_thickness` offset, blending to the plate edge at the very top.
    constant_thickness = keylist_data.get('constant_thickness_walls', False)

    # Outward XY normals for the OUTER perimeter loop(s) only. Interior holes
    # (e.g. enclosed by a full-edge diagonal link) and switch-cutout rims keep
    # the ordinary vertical band, so they stay closed.
    skirt_normals = {}
    if skirt:
        loops = _perimeter_loops(top, hole_ids)
        if loops:
            def _bbox_area(lp):
                xs = [top_pts[v][0] for v in lp]
                ys = [top_pts[v][1] for v in lp]
                return (max(xs) - min(xs)) * (max(ys) - min(ys))
            areas = [_bbox_area(lp) for lp in loops]
            amax = max(areas)
            for lp, a in zip(loops, areas):
                if a < 0.5 * amax:
                    continue  # interior hole: no skirt
                olp, nrms = _outward_normals_xy(top_pts, lp)
                for vi, nn in zip(olp, nrms):
                    skirt_normals[vi] = nn

    # Build the intermediate rings per perimeter vertex:
    #   rings[0]   = P1, the outward flange at z=ztop
    #   rings[1..] = one ring per profile segment endpoint, ending at base_z
    #   rim_ring   = the inner edge of the flat rim, at base_z
    # The inner skirt face then runs from rim_ring up to the plate's bottom
    # perimeter (P4). Its thickness therefore tapers from `wall_thickness` at
    # the rim to roughly the flange width where it meets the plate underside.
    segs = _skirt_profile(keylist_data) if skirt else []
    rings = [dict() for _ in range(len(segs) + 1)]
    rim_ring = {}
    # For each perimeter vertex, the inner-face column as vertex indices ordered
    # bottom (rim) -> top (plate underside). With constant_thickness off this is
    # just [rim, P4] (one sloped panel); on, it carries intermediate rings that
    # parallel the outer profile.
    inner_cols = {}
    for vi, (nx, ny) in skirt_normals.items():
        p = top_pts[vi]
        ztop = p[2]
        drop = ztop - base_z

        # Outer profile, recording (d, z) per ring so the inner face can mirror
        # its shape when constant thickness is requested.
        ring_dz = []
        d = skirt_flange
        z = ztop
        rings[0][vi] = len(vertices)
        vertices.append((p[0] + nx * d, p[1] + ny * d, z))
        ring_dz.append((d, z))

        for si, s in enumerate(segs):
            dz = s['frac'] * drop
            if s['out'] is not None:
                out = s['out']
            else:
                out = dz * math.tan(math.radians(s['angle']))
            d += out
            z -= dz
            # Land the final ring exactly on base_z (kills float drift).
            if si == len(segs) - 1:
                z = base_z
            rings[si + 1][vi] = len(vertices)
            vertices.append((p[0] + nx * d, p[1] + ny * d, z))
            ring_dz.append((d, z))

        rim_ring[vi] = len(vertices)
        d_bottom = ring_dz[-1][0]
        di = d_bottom - wall_thickness
        vertices.append((p[0] + nx * di, p[1] + ny * di, base_z))

        # Build the inner-face column (bottom -> top).
        col = [rim_ring[vi]]
        if constant_thickness:
            zbot = vertices[vi + n][2]   # plate underside z at this vertex

            # Smooth constant-thickness inner wall via Minkowski EROSION.
            #
            # We erode the wall cross-section inward by wall_thickness: the inner
            # face is the cavity-side boundary of the set of points at least
            # wall_thickness (perpendicular) from the outer profile. Computed as
            # the lower envelope of disks of radius wall_thickness centred on a
            # dense sampling of the outer polyline: at each height the inner d is
            # the smallest d that clears every disk.
            #
            # This follows the outer profile faithfully — including walls that
            # flare out in the MIDDLE (barrel/bulge shapes), which a convex hull
            # would wrongly cut straight across — while still smoothing steps,
            # never overhanging from step artifacts, and staying valid when a
            # step's run exceeds wall_thickness.
            wt = wall_thickness

            # Dense sample of the outer polyline (ring_dz is (d, z), top->bottom).
            opts = []
            for i in range(len(ring_dz) - 1):
                d0, z0 = ring_dz[i]
                d1, z1 = ring_dz[i + 1]
                seg_len = math.hypot(d1 - d0, z1 - z0)
                steps = max(1, int(seg_len / 0.15))
                for k in range(steps + 1):
                    t = k / steps
                    opts.append((d0 + (d1 - d0) * t, z0 + (z1 - z0) * t))

            # Lower envelope of radius-wt disks, sampled in z from base to plate.
            span = zbot - base_z
            if span > 1e-6:
                nz = max(2, int(span / 0.4))
                for zi in range(1, nz):
                    zc = base_z + span * zi / nz
                    best = None
                    for (pd, pz) in opts:
                        dz = zc - pz
                        if -wt < dz < wt:
                            left = pd - math.sqrt(wt * wt - dz * dz)
                            if best is None or left < best:
                                best = left
                    if best is None:
                        continue
                    d_in = max(best, 0.0)
                    idx = len(vertices)
                    vertices.append((p[0] + nx * d_in, p[1] + ny * d_in, zc))
                    col.append(idx)
        col.append(vi + n)          # P4, plate bottom perimeter
        inner_cols[vi] = col

    # Top faces keep their (CCW-from-above, normal-up) winding. Bottom faces
    # are the same loops shifted by n with REVERSED winding so their normals
    # point down/outward.
    for f in top.faces:
        faces.append(f)
        faces.append(tuple(reversed([i + n for i in f])))

    # Boundary walls: an edge used by exactly one top face lies on the outer
    # perimeter or a switch-cutout rim. We stitch each such edge to the bottom.
    # Crucially we keep the edge's DIRECTED orientation as it appears in its
    # top face, so the wall quad inherits a consistent outward winding:
    #   top face is CCW (normal up); its boundary edge (a->b) runs so the
    #   material is on the left. The outward wall is then (b, a, a+n, b+n).
    from collections import defaultdict
    edge_count = defaultdict(int)
    directed = {}
    for f in top.faces:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            e = frozenset((a, b))
            edge_count[e] += 1
            directed[e] = (a, b)

    for e, cnt in edge_count.items():
        if cnt != 1:
            continue  # interior edge shared by two top faces -> no wall
        a, b = directed[e]
        if skirt and a in rim_ring and b in rim_ring:
            # Skirt strip: P0 -> flange -> each profile segment -> flat rim ->
            # back up the inner face to the plate's bottom perimeter (P4).
            # Every quad inherits the plain band's outward winding.
            faces.append((b, a, rings[0][a], rings[0][b]))       # top flange ledge
            for i in range(len(rings) - 1):
                faces.append((rings[i][b], rings[i][a],
                              rings[i + 1][a], rings[i + 1][b]))  # outer segment
            last = rings[-1]
            faces.append((last[b], last[a], rim_ring[a], rim_ring[b]))  # flat rim
            # Inner face: stitch the two inner columns (rim -> ... -> P4). With
            # constant thickness off these are 2-point columns and this reduces
            # to the original single quad; on, they carry intermediate rings and
            # the stitch handles differing counts (from per-vertex heights).
            _stitch_columns(faces, vertices, inner_cols[a], inner_cols[b])
        else:
            # Ordinary vertical band (switch-cutout rims, interior holes, and
            # the whole perimeter when the skirt is off).
            faces.append((b, a, a + n, b + n))

    return vertices, faces


def _cell_ring(key, key_1u, hole_size=14.5, switch_border=1.5):
    """Outer cell ring (4 pts) in local frame, at local z=0. Sized to leave at
    least `switch_border` mm of flat plate around the switch cutout."""
    hw, hh = _cell_half_extents(key, key_1u, hole_size, switch_border)
    return [(-hw, hh, 0.0), (-hw, -hh, 0.0), (hw, -hh, 0.0), (hw, hh, 0.0)]


# (walls are defined above, before build_shell)


def _hole_ring(key, key_1u, hole_size):
    """Inner switch-cutout ring (4 pts) in local frame, at local z=0."""
    u = hole_size / 2.0
    return [(-u, u, 0.0), (-u, -u, 0.0), (u, -u, 0.0), (u, u, 0.0)]


def _perimeter_loops(top, hole_vert_ids):
    """
    From a built TopSurface, extract the OUTER perimeter as ordered vertex-
    index loops. Boundary edges (used by exactly one top face) that don't lie
    entirely on a switch-cutout rim are perimeter edges; we chain them into
    closed loops. Returns a list of loops, each a list of vertex indices.
    """
    from collections import defaultdict
    edge_count = defaultdict(int)
    for f in top.faces:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            edge_count[frozenset((a, b))] += 1

    # Keep boundary edges that aren't switch-cutout rims.
    perim_edges = []
    for e, cnt in edge_count.items():
        if cnt != 1:
            continue
        a, b = tuple(e)
        if a in hole_vert_ids and b in hole_vert_ids:
            continue  # switch hole rim, not the outer perimeter
        perim_edges.append((a, b))

    # Build adjacency and chain into loops.
    adj = defaultdict(list)
    for a, b in perim_edges:
        adj[a].append(b)
        adj[b].append(a)

    unused = set(map(frozenset, perim_edges))
    loops = []
    while unused:
        # Start a new loop from any remaining edge.
        start_edge = next(iter(unused))
        a, b = tuple(start_edge)
        loop = [a, b]
        unused.discard(start_edge)
        while True:
            cur = loop[-1]
            nxt = None
            for cand in adj[cur]:
                e = frozenset((cur, cand))
                if e in unused:
                    nxt = cand
                    unused.discard(e)
                    break
            if nxt is None:
                break
            if nxt == loop[0]:
                break  # closed
            loop.append(nxt)
        loops.append(loop)
    return loops


def _outward_normals_xy(points, loop):
    """
    For an ordered perimeter loop, return a unit OUTWARD XY normal per loop
    vertex. The loop is forced CCW (viewed from above) so 'outward' is the
    right-hand side of travel; the normal at a vertex averages its two
    adjacent edge normals.
    """
    n = len(loop)
    pts = [points[i] for i in loop]

    # Signed area (XY) to determine orientation; flip to CCW if needed.
    area2 = 0.0
    for i in range(n):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        area2 += x0 * y1 - x1 * y0
    if area2 < 0:
        loop = list(reversed(loop))
        pts = [points[i] for i in loop]

    def edge_out(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        # Right of travel for CCW loop = outward: (dy, -dx)
        ox, oy = dy, -dx
        m = (ox * ox + oy * oy) ** 0.5
        return (ox / m, oy / m) if m > 1e-12 else (0.0, 0.0)

    normals = []
    for i in range(n):
        prev_p = pts[(i - 1) % n]
        cur_p = pts[i]
        next_p = pts[(i + 1) % n]
        e0 = edge_out(prev_p, cur_p)
        e1 = edge_out(cur_p, next_p)
        ox, oy = e0[0] + e1[0], e0[1] + e1[1]
        m = (ox * ox + oy * oy) ** 0.5
        normals.append((ox / m, oy / m) if m > 1e-12 else e1)
    return loop, normals


def build_walls(keylist_data):
    """
    Build the perimeter WALLS as a separate object: a frame that the key plate
    rests into, dropping to z=0. Returns (vertices, faces) — a single closed
    manifold per perimeter loop.

    Cross-section, swept around the plate's outer perimeter (d = outward offset
    from the perimeter line, z = height):

        (+flange_offset, rim) ┌───────┐ (rim = plate_top_z + flange_z)
                              │       │
        (0, rim) ── recess ── ┤       │   ← recess above the seated plate
                       wall   │       │
        (0, ledge_z) ─┐       │       │   ← plate rests here (ledge top)
        (-lip, ledge) └──┐    │       │   ← inward ledge of width `plate_lip`
                         │    │       │
        (-lip, 0) ───────┴────┴───────┘ (+flange_offset, 0)   z=0 base

    JSON fields (all optional, sensible defaults):
        flange_offset : outward wall material width beyond the plate edge (mm)
        flange_z      : recess depth — how far the plate top sits below the
                        wall's top rim (mm)
        plate_lip     : width of the inward ledge the plate rests on (mm, ~1.5)
        plate_gap     : per-side clearance between plate and wall inner faces
                        for print tolerance (mm, default 0.25 -> 0.5 total)
        thickness     : plate thickness, used to place the ledge at the plate
                        bottom
        wall_base_z   : floor height the walls drop to (default 0)
    """
    keylist_data = resolve_keylist(keylist_data)
    thickness = keylist_data.get('thickness', 5.0)
    flange_offset = keylist_data.get('flange_offset', 0) or 0
    flange_z = keylist_data.get('flange_z', 0) or 0
    plate_lip = keylist_data.get('plate_lip', 1.5)
    base_z = keylist_data.get('wall_base_z', 0.0)
    # Printing-tolerance clearance: the wall's inner faces are pushed outward
    # by this much so the plate drops in with a gap on each side (total gap
    # around the plate = 2 * plate_gap). Default 0.25mm -> 0.5mm combined.
    plate_gap = keylist_data.get('plate_gap', 0.25)

    top, hole_ids = _build_top_surface(keylist_data)
    loops = _perimeter_loops(top, hole_ids)

    # Keep only OUTER perimeter loop(s); skip interior holes (e.g. the small
    # gap a diagonal linked-key bridge can enclose). We measure each loop's
    # XY bounding-box area and keep loops within 50% of the largest — this
    # keeps genuinely separate outer boundaries (disjoint key groups) while
    # dropping small interior holes.
    def _loop_bbox_area(lp):
        xs = [top.points[v][0] for v in lp]
        ys = [top.points[v][1] for v in lp]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    if loops:
        areas = [_loop_bbox_area(lp) for lp in loops]
        amax = max(areas)
        loops = [lp for lp, a in zip(loops, areas) if a >= 0.5 * amax]

    vertices = []
    faces = []

    def add_vert(p):
        vertices.append(p)
        return len(vertices) - 1

    # Per-vertex offset normals so the ledge can follow the plate's TRUE
    # underside (perpendicular offset), matching build_shell exactly. Cell-
    # corner perimeter vertices use their key's plane normal; anything else
    # falls back to the area-averaged normal.
    override = getattr(top, 'offset_normal', {})
    unit = top.unit_normals()

    for raw_loop in loops:
        loop, normals = _outward_normals_xy(top.points, raw_loop)
        n = len(loop)
        if n < 3:
            continue

        # For each perimeter vertex, build the 6-point cross-section ring.
        # Order around the section (CCW in the (d,z) plane) so the swept
        # prism has consistent outward faces:
        #   0 outer_bottom (+fo, base)
        #   1 outer_top    (+fo, rim)
        #   2 inner_top    (+gap, rim)      recess wall top
        #   3 ledge_top    (+gap, ledge)    plate rests, outer edge
        #   4 ledge_inner  (gap-lip, ledge) ledge inner edge
        #   5 inner_bottom (gap-lip, base)
        #
        # The recess wall (points 2,3) and the ledge (points 4,5) are all
        # shifted OUTWARD by `plate_gap` so the plate seats with a clearance
        # gap on every side for print tolerance. The ledge WIDTH (plate_lip)
        # is preserved by shifting both its edges together.
        #
        # The ledge height at each perimeter vertex is the plate's TRUE bottom
        # z there (top offset perpendicular by `thickness` along the plate's
        # offset normal), NOT ztop - thickness. On a tilted key the
        # perpendicular drop in z is only thickness*cos(tilt), so a flat
        # ztop-thickness ledge would sit too low and the sloped plate
        # underside wouldn't rest flush. Using the true bottom z makes the
        # ledge slope match the plate underside.
        rings = []
        for i, vi in enumerate(loop):
            p = top.points[vi]
            nx, ny = normals[i]
            u = override.get(vi, unit[vi])
            rim = p[2] + flange_z

            # The plate's true underside point at this perimeter vertex. Since
            # build_shell now makes the plate's OUTER edge VERTICAL (top moved
            # to sit directly above this bottom point), the plate's outer wall
            # is the vertical line through (bx,by). We therefore measure the
            # wall's recess/ledge offsets outward from THIS bottom point, so
            # the recess inner face sits `plate_gap` beyond the plate's actual
            # (vertical) outer edge — no clipping.
            bx = p[0] - u[0] * thickness
            by = p[1] - u[1] * thickness
            bz = p[2] - u[2] * thickness

            uz = u[2] if abs(u[2]) > 1e-6 else 1e-6

            def underside_z(x, y):
                # z on the plate underside plane (through bottom pt, normal u).
                return bz - (u[0] * (x - bx) + u[1] * (y - by)) / uz

            # Offsets are measured outward from the plate's vertical outer edge
            # at (bx,by).
            def at(offset, z):
                return (bx + nx * offset, by + ny * offset, z)

            def at_ledge(offset):
                x = bx + nx * offset
                y = by + ny * offset
                return (x, y, underside_z(x, y))

            inner = plate_gap                 # recess wall, gap beyond plate edge
            ledge_in = plate_gap - plate_lip  # ledge inner edge

            ring = [
                at(flange_offset, base_z),   # 0 outer_bottom
                at(flange_offset, rim),      # 1 outer_top
                at(inner, rim),              # 2 inner_top
                at_ledge(inner),             # 3 ledge_top (outer) on underside
                at_ledge(ledge_in),          # 4 ledge_inner   on underside
                at(ledge_in, base_z),        # 5 inner_bottom
            ]
            rings.append([add_vert(q) for q in ring])

        m = len(rings[0])  # 6

        # Sweep: connect ring i to ring i+1 (loop closes) with quads.
        for i in range(n):
            a = rings[i]
            b = rings[(i + 1) % n]
            for k in range(m):
                k2 = (k + 1) % m
                # Outward-facing quad around the tube.
                faces.append((a[k], b[k], b[k2], a[k2]))

        # No end caps needed — the loop is closed, so the swept cross-section
        # forms a complete torus-like solid frame.

    return vertices, faces


def build_walls_from_any(data):
    """build_walls accepting a keyboard or keylist JSON."""
    return build_walls(resolve_keylist(data))


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)
    verts, faces = build_shell_from_any(data)
    print(f"vertices: {len(verts)}  faces: {len(faces)}")
