"""Mould generation by mesh CSG — one printable set per design.

This replaces `mould_auto.py` and `mould_silicone.py`, which built moulds by
booleaning voxel occupancy grids. That approach was precise and it was fragile: it
needed 6-24 M voxels per solve, a few dozen full-grid arrays, minutes of runtime, and
it kept producing parts that verified beautifully and could not be opened — the
vessel's former failed its own release sweep to the end.

The geometry comes from two prior generators, rebuilt in Python on trimesh +
manifold booleans. Three ideas do the real work:

**A fused core, so there is nothing to get stuck** (`auto-mold-generator`). The part
stands in the middle of the tooling on a bottom flange; silicone fills the gap around
it. The elastomer is what demoulds, and it demoulds by stretching, which is the one
thing it is good at. The voxel path's whole release apparatus — sweeps,
discrimination controls, trapped-core detection — answers a question this geometry
does not ask.

**A box for the plastic, a hugging silhouette for the rubber.** The jacket and the
rigid block are axis-aligned boxes, as in `automated_3d_mold_generator`; the silicone
chamber is the part's own shadow offset by the skin thickness. Splitting it that way
is deliberate. A box face is two triangles where an offset silhouette must be
triangulated and comes out as a fan of long slivers from a single vertex — 628 cap
triangles on the vessel against 12 — and those slivers are what made the earlier
output look stretched and chewed. Buying the clean version in RUBBER would be
expensive; buying it in filament costs 1.02x on the tile, 1.18x on the vessel,
1.66x on the block, and the silicone bill does not move.

**Split by boolean, never by `slice_plane`.** Its cap is a fan from one vertex, so a
120 mm parting face came out as a star of enormous slivers. Intersecting with a
half-space box re-triangulates the cut properly.

WHAT IS KEPT FROM THE VOXEL PATH, because the geometry is not the point of this
project. A silicone face is a no-flux boundary: an enclosing skin scores **0.000**
cemented fraction, so the window lattice is still sized against transport rather than
chosen by eye (`size_windows`), and the tooling grows pillars across the gap so the
cast skin demoulds already perforated. A mould that cannot breathe is not simpler,
it is just wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .physics import drying as dry
from .physics import fields as pf
from .physics import oxygen as ox

#: Arc segments per quarter-circle when offsetting a silhouette. 16 keeps a 100 mm
#: outline within ~0.1 mm of a true round offset, which is under one FDM extrusion.
SEG = 16

#: Slices used to search for a parting plane. The search is over a plateau rather
#: than an argmax, so it does not need to be fine — 24 was enough on every typology.
PART_SLICES = 24


@dataclass(frozen=True)
class CastSpec:
    """Everything the generator cannot measure off the geometry.

    Lengths mm, angles deg. The defaults are a Shore 30A mould rubber cast in PETG
    tooling, which is what `docs/elastomer_summary.md` records parameters for.
    """

    goal: str = "silicone"          # "silicone" (2-pour tooling) or "rigid"
    wall: float = 6.0               # jacket wall thickness
    floor: float = 5.0              # tooling floor under the part
    silicone_t: float = 6.0         # rubber wall around the core = the skin
    draft_deg: float = 2.0          # release taper, applied away from the parting
    shrink_pct: float = 0.0         # enlarge the cavity to offset cure shrink
    clearance: float = 0.2          # print fit between jacket halves

    # breather lattice — the reason this is not a generic mould generator
    window_d: float = 0.0           # 0 -> 2.5 * d_max, safely inside the clog band
    window_spacing: float = 0.0     # 0 -> sized on the aeration requirement
    coverage_target: float = 0.85
    spacing_ladder: tuple = (48.0, 40.0, 34.0, 28.0, 24.0, 20.0, 17.0)

    # fasteners and fill
    key_count: int = 3
    key_d: float = 10.0
    spout_d: float = 14.0

    # mix and material
    d_max: float = 4.0
    jam_mult: float = 6.0
    rho_g_cm3: float = 1.15


# --------------------------------------------------------------------------
# Silhouettes and prisms
# --------------------------------------------------------------------------
def silhouette(mesh: trimesh.Trimesh, z: float | None = None,
               smooth: float = 1.5) -> Polygon:
    """The part's shadow on XY, or its cross-section at height `z`.

    A section is used for the flange rather than the full shadow, because the shadow
    fills in a tapered bottom: on a part whose base is chamfered, flanging to the
    shadow would bury the chamfer in tooling and the silicone would never take that
    shape. Falls back to the shadow when the section is empty.
    """
    if z is None:
        polys = mesh.projected(normal=[0, 0, 1]).polygons_full
    else:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            return silhouette(mesh, smooth=smooth)
        polys = sec.to_planar()[0].polygons_full
    if not len(polys):
        return silhouette(mesh, smooth=smooth) if z is not None else Polygon()
    u = unary_union(list(polys))
    if u.geom_type == "MultiPolygon":
        # A section can legitimately be several islands — the block's cores, a tile's
        # relief. Tooling is built on the outline that contains the part, so the
        # largest island wins and the rest are holes in it once offset. Taking the
        # union straight to `extrude_polygon` raises, which is how this surfaced.
        u = max(u.geoms, key=lambda g: g.area)
    # DE-STAIRCASE. The grammars mesh a voxel field, so this outline is the shadow of
    # a marching-cubes surface and carries a stair-step at every voxel — which is what
    # made the tooling rims look chewed. A morphological close at the voxel scale
    # removes the steps without moving the outline, and simplifying at a third of a
    # voxel drops the vertices they left behind. `smooth` is the pitch; at 2 mm this
    # takes a vessel outline from ~1400 vertices to ~90.
    if smooth > 0 and not u.is_empty:
        u = (u.buffer(smooth, resolution=SEG).buffer(-smooth, resolution=SEG)
              .simplify(smooth / 3.0))
        if u.geom_type == "MultiPolygon":
            u = max(u.geoms, key=lambda g: g.area)
    return u


def block(part: trimesh.Trimesh, offset: float, z0: float,
          z1: float) -> trimesh.Trimesh:
    """An axis-aligned box around the part, which is what a mould normally looks like.

    Used for the PLASTIC, while the rubber still hugs the part's silhouette. That
    split is the whole point: a box's faces are two triangles each, where an offset
    silhouette has to be triangulated and comes out as a fan of long slivers from a
    single vertex — measured, 628 cap triangles on the vessel against 12, and those
    slivers are what made every part look stretched and chewed. Paying for the clean
    version in RUBBER would be expensive; paying for it in filament is 1.02x on the
    tile, 1.18x on the vessel and 1.66x on the block.

    This is how `automated_3d_mold_generator` builds its mould, and it is the right
    call for the same reason.
    """
    lo, hi = part.bounds[0] - offset, part.bounds[1] + offset
    lo[2], hi[2] = z0, z1
    return trimesh.creation.box(
        extents=hi - lo,
        transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def prism(poly: Polygon, offset: float, z0: float, z1: float) -> trimesh.Trimesh:
    """Offset a 2D outline and extrude it between two heights."""
    if z1 <= z0:
        raise ValueError(f"prism height must be > 0, got {z1 - z0:.3f} mm")
    grown = poly.buffer(offset, resolution=SEG, join_style=1).simplify(0.05)
    m = trimesh.creation.extrude_polygon(grown, z1 - z0)
    m.apply_translation([0.0, 0.0, z0])
    return m


# --------------------------------------------------------------------------
# Parting: which way to pull, and where to cut
# --------------------------------------------------------------------------
def analyse_axis(mesh: trimesh.Trimesh, axis: int,
                 steps: int = PART_SLICES) -> dict:
    """Best parting plane along one axis, and how badly the part undercuts it.

    The plane goes at the CENTRE of the maximum-area plateau, not at the argmax. On a
    uniform prism every slice has the same area, so an argmax lands wherever the
    floating-point noise happens to peak — usually at an end, which splits the part
    into a lid and everything else.

    Undercut is the lateral protrusion of each slice past its inboard neighbour,
    accumulated moving away from the plane and normalised by the largest slice, so it
    is scale-free and comparable between axes. A slice that grows as you move outward
    is material the tooling would have to reach around.
    """
    m = mesh.copy()
    if axis != 2:                       # rotate the chosen axis onto z and sample there
        T = trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0] if axis == 0 else [1, 0, 0])
        m.apply_transform(T)
    lo, hi = float(m.bounds[0][2]), float(m.bounds[1][2])
    zs = [lo + (hi - lo) * i / steps for i in range(1, steps)]
    secs = [silhouette(m, z) for z in zs]
    areas = np.array([s.area if not s.is_empty else 0.0 for s in secs])
    if not areas.any():
        return {"axis": axis, "plane": 0.5 * (lo + hi), "undercut": np.inf}

    peak = int(areas.argmax())
    tol = max(1e-3, areas[peak] * 0.02)
    a, b = peak, peak
    while a > 0 and areas[a - 1] >= areas[peak] - tol:
        a -= 1
    while b < len(areas) - 1 and areas[b + 1] >= areas[peak] - tol:
        b += 1
    z_part = 0.5 * (zs[a] + zs[b])
    centre = (a + b) // 2

    undercut = 0.0
    for rng in (range(centre + 1, len(secs)), range(centre - 1, -1, -1)):
        prev = centre
        for i in rng:
            near, far = secs[prev], secs[i]
            if not near.is_empty and not far.is_empty:
                undercut += far.difference(near).area
            prev = i
    # the rotation that put this axis on z also flipped its sign
    plane = z_part if axis == 2 else -z_part
    return {"axis": axis, "plane": float(plane),
            "undercut": float(undercut / max(areas[peak], 1e-6)),
            "max_section_mm2": float(areas[peak])}


def choose_parting(mesh: trimesh.Trimesh) -> dict:
    """The axis that undercuts least. Searched, never assumed to be z."""
    cands = [analyse_axis(mesh, a) for a in (0, 1, 2)]
    best = min(cands, key=lambda r: r["undercut"])
    best["candidates"] = cands
    best["reason"] = (
        f"axis {best['axis']} undercuts {best['undercut']:.3f} against "
        + ", ".join(f"{c['undercut']:.3f}" for c in cands if c["axis"] != best["axis"]))
    return best


def apply_shrink(mesh: trimesh.Trimesh, pct: float) -> trimesh.Trimesh:
    """Scale about the centroid so the cast lands on size after it shrinks."""
    if abs(pct) < 1e-9:
        return mesh
    m = mesh.copy()
    c = m.centroid
    m.apply_translation(-c)
    m.apply_scale(1.0 + pct / 100.0)
    m.apply_translation(c)
    return m


def apply_draft(mesh: trimesh.Trimesh, axis: int, plane: float,
                deg: float) -> trimesh.Trimesh:
    """Taper the part away from the parting plane, both ways.

    Vertices on the plane do not move, so the plane survives for the bisection, and
    each half ends up widest at the parting line. The mesh is SUBDIVIDED first:
    warping only moves existing vertices, so a long flat face with corner vertices
    only would scale uniformly instead of tapering, and the draft would be reported
    but not present — the exact failure the voxel path documented at §2.
    """
    if deg <= 0:
        return mesh
    m = mesh.copy()
    span = float(m.bounds[1][axis] - m.bounds[0][axis])
    # Refinement is a means, not an end: it exists so a long flat face gains
    # intermediate vertices to taper. A marching-cubes mesh already has an edge every
    # voxel, so this normally does nothing — but on the tile's large flat faces it ran
    # four times and turned 56 k triangles into 224 k, which is where the 11 MB STLs
    # and the "why is this so complicated" came from.
    budget = min(4 * len(m.faces), 120_000)
    while m.edges_unique_length.max() > max(span / 12.0, 2.0):
        if len(m.faces) * 4 > budget:
            break
        m = m.subdivide()

    lat = [i for i in range(3) if i != axis]
    v = m.vertices.copy()
    half = max(np.abs(v[:, lat] - v[:, lat].mean(axis=0)).max(), 1e-6)
    d = np.abs(v[:, axis] - plane)
    k = 1.0 - (d * np.tan(np.radians(deg))) / half
    k = np.clip(k, 0.05, 1.0)[:, None]
    ctr = v[:, lat].mean(axis=0)
    v[:, lat] = ctr + (v[:, lat] - ctr) * k
    m.vertices = v
    return m


def bisect(solid: trimesh.Trimesh, axis: int, plane: float) -> tuple:
    """Cut a solid in two at `plane`, by INTERSECTING WITH A HALF-SPACE BOX.

    Not `slice_plane(cap=True)`, which was the single worst thing in the output.
    Its cap is a fan triangulation from one vertex, so a 120 mm parting face came
    out as a star of enormous slivers radiating from a point — that is the "stretched
    out" look, it is present on every part that gets split, and a slicer has to chew
    through degenerate triangles to print it. `automated_3d_mold_generator` splits by
    booleaning against boxes for the same reason; a boolean re-triangulates the cut
    face properly instead of fanning it.

    The box is oversized by the solid's own diagonal so it cannot clip anything but
    the intended side, whatever the part's extent.
    """
    lo_b, hi_b = solid.bounds
    pad = float(np.linalg.norm(hi_b - lo_b))
    out = []
    for sign in (-1, +1):
        lo = lo_b - pad
        hi = hi_b + pad
        if sign < 0:
            hi[axis] = plane
        else:
            lo[axis] = plane
        box = trimesh.creation.box(
            extents=hi - lo,
            transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
        out.append(trimesh.boolean.intersection([solid, box]))
    return out[0], out[1]


# --------------------------------------------------------------------------
# Breather windows, sized on the aeration requirement
# --------------------------------------------------------------------------
def cover_surrogate(occ: np.ndarray, src: np.ndarray, pitch: float,
                    L_eff: float) -> float:
    """Fraction of the body within the drained depth of an opening.

    The DRAINED-DEPTH criterion, not the reaction-diffusion solve. It is what the
    window ladder can afford to evaluate per candidate spacing, and it is stated
    rather than dressed up: the two are not interchangeable, and this one is only
    used to CHOOSE a spacing, never reported as the design's cemented fraction.
    """
    if not src.any():
        return 0.0
    depth = pf.depth_field(occ, src, pitch)
    d = depth[occ]
    return float(np.nanmean(d <= L_eff))


def window_lattice(shape, origin, pitch, axis: int, d: float,
                   spacing: float) -> np.ndarray:
    """Bores on a square grid, along the draw axis only.

    One family, not three. In the SKIN a bore is a cut and any direction works, but
    the tooling has to FORM it with a solid pillar, and a pillar across the pull
    shears through the cured rubber when the mould opens. Restricting to the draw
    axis costs open area and is the only version that can be built.
    """
    g = [origin[i] + np.arange(shape[i]) * pitch for i in range(3)]
    lat = [i for i in range(3) if i != axis]
    A, B = np.meshgrid(g[lat[0]], g[lat[1]], indexing="ij")
    u = np.mod(A / spacing + 0.31, 1.0) - 0.5
    v = np.mod(B / spacing + 0.53, 1.0) - 0.5
    disc = (u * u + v * v) * spacing ** 2 <= (d / 2) ** 2
    out = np.repeat(disc[:, :, None], shape[axis], axis=2)
    return np.moveaxis(out, 2, axis)


def size_windows(occ: np.ndarray, origin, pitch: float, axis: int,
                 spec: CastSpec, L_eff: float) -> dict:
    """Step the window pitch down until the body meets the coverage criterion.

    An enclosed skin is 0.000 cemented — not merely poor, but anoxic by construction
    — so this is what makes the silicone path viable at all. Coarsest acceptable
    spacing wins: every extra window is a pillar in the tooling and a hole in the
    rubber, and the criterion is a floor rather than something to overshoot.
    """
    from scipy import ndimage

    d_win = spec.window_d or 2.5 * spec.d_max
    ladder = ([spec.window_spacing] if spec.window_spacing
              else list(spec.spacing_ladder))
    surface = occ & ~ndimage.binary_erosion(occ)
    rows = []
    chosen = None
    for sp in ladder:
        bores = window_lattice(occ.shape, origin, pitch, axis, d_win, sp)
        src = bores & ~occ                       # air inside a bore = an opening
        cov = cover_surrogate(occ, src, pitch, L_eff)
        open_frac = float((bores & surface).sum() / max(surface.sum(), 1))
        rows.append({"spacing_mm": sp, "cover": cov, "open_area_frac": open_frac})
        if chosen is None and cov >= spec.coverage_target:
            chosen = rows[-1]
            break
    met = chosen is not None
    chosen = chosen or (rows[-1] if rows else {"spacing_mm": ladder[-1],
                                               "cover": 0.0, "open_area_frac": 0.0})
    return {"d_mm": d_win, "spacing_mm": chosen["spacing_mm"],
            "cover_surrogate": chosen["cover"],
            "open_area_frac": chosen["open_area_frac"],
            "meets_target": met, "ladder": rows,
            "limited_by": ("coverage met at the coarsest spacing that reaches it"
                           if met else
                           "the finest spacing on the ladder still misses the "
                           "target — the limit is drying depth, not open area, and "
                           "more windows will not fix it"),
            "note": "chosen on the drained-depth surrogate, not the field solve"}


def window_pillars(poly: Polygon, d: float, spacing: float,
                   z0: float, z1: float) -> trimesh.Trimesh | None:
    """Solid rods on the window grid, spanning the silicone gap.

    They do two jobs at once, which is why they earn their complexity: they cast the
    breather windows into the skin, and they hold the core at the silicone offset
    while the rubber is still liquid. A separate set of locating pins would leave a
    second set of holes to patch.

    Rods are vertical and the tooling is built with the draw on z, so they always run
    along the pull — a pillar across it shears through the cured rubber on opening.
    Kept only where the grid point lands on the part's own outline, so none of them
    stands in open space.
    """
    from shapely.geometry import Point

    lo, hi = poly.bounds[:2], poly.bounds[2:]
    us = np.arange(lo[0], hi[0] + spacing, spacing) + 0.31 * spacing
    vs = np.arange(lo[1], hi[1] + spacing, spacing) + 0.53 * spacing
    rods = [trimesh.creation.cylinder(radius=d / 2, height=z1 - z0, sections=24,
                                      transform=trimesh.transformations
                                      .translation_matrix([u, v, 0.5 * (z0 + z1)]))
            for u in us for v in vs if poly.contains(Point(u, v))]
    return trimesh.util.concatenate(rods) if rods else None


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def ring_point(poly: Polygon, angle_deg: float, inner: float,
               ring: float) -> tuple | None:
    """A point in the middle of the jacket ring, at this bearing from the centroid.

    PROJECTED onto the real outline, not placed at an equivalent-circle radius.
    `sqrt(area/pi)` is only the right radius for a disc: on the block's 390 x 190
    footprint it lands well outside the ring at the ends and inside it on the flanks,
    so a key put there floats free of the jacket — measured as 2-3 disconnected
    bodies per half before this. A ray from the centroid to the offset outline gives
    a point that is on the ring whatever the shape.
    """
    from shapely.geometry import LineString, Point

    c = poly.centroid
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    reach = 4.0 * max(poly.bounds[2] - poly.bounds[0],
                      poly.bounds[3] - poly.bounds[1]) + inner + ring
    ray = LineString([(c.x, c.y), (c.x + d[0] * reach, c.y + d[1] * reach)])
    hit = ray.intersection(poly.buffer(inner, resolution=SEG).exterior)
    if hit.is_empty:
        return None
    pts = [hit] if isinstance(hit, Point) else list(getattr(hit, "geoms", []))
    if not pts:
        return None
    p = max(pts, key=lambda q: (q.x - c.x) ** 2 + (q.y - c.y) ** 2)
    return float(p.x + d[0] * ring * 0.5), float(p.y + d[1] * ring * 0.5)


def frustum_keys(poly: Polygon, plane: float, inner: float, ring: float,
                 spec: CastSpec) -> tuple:
    """Tapered pegs on one jacket half, sockets on the other. Draw is z.

    Cones rather than cylinders so the halves self-centre as they close, and so print
    tolerance shows up as a small axial gap instead of a jam. At angles that are NOT
    rotationally symmetric: three at 0/140/250 deg mate one way only, which is what
    stops a half being clamped on 120 deg out with the parting line in the wrong
    place.
    """
    h = spec.key_d * 0.8
    pegs, sockets = [], []
    for ang in (0.0, 140.0, 250.0)[:spec.key_count]:
        xy = ring_point(poly, ang, inner, ring)
        if xy is None:
            continue
        for out, rad, extra in ((pegs, spec.key_d / 2, 0.0),
                                (sockets, spec.key_d / 2 + spec.clearance,
                                 spec.clearance)):
            # A FRUSTUM, not a cone. A sharp apex is a needle: it prints badly, it
            # locates nothing once the tip rounds over, and it reads as a spike in
            # every render. Cut at 60 % of the height, so the taper still self-centres
            # the halves as they close but lands on a flat.
            k = trimesh.creation.cone(radius=rad, height=h / 0.6, sections=24)
            k = k.slice_plane(plane_origin=[0, 0, h + extra],
                              plane_normal=[0, 0, -1], cap=True)
            k.apply_translation([xy[0], xy[1], plane])
            out.append(k)
    return (trimesh.util.concatenate(pegs) if pegs else None,
            trimesh.util.concatenate(sockets) if sockets else None)


def _to_draw_frame(mesh: trimesh.Trimesh, axis: int) -> trimesh.Trimesh:
    """Rotate so the chosen draw axis becomes +z.

    Everything downstream then assumes z, which is not laziness: the voxel path
    carried an axis argument through four functions and §7 of its design record is
    about the one that kept its old default. It is also the orientation a slicer
    wants, so the exported parts come out ready to place on a bed.
    """
    if axis == 2:
        return mesh.copy()
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [0, 1, 0] if axis == 0 else [1, 0, 0]))
    return m


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def build_silicone_tooling(part: trimesh.Trimesh, spec: CastSpec,
                           parting: dict, win: dict) -> dict:
    """Two-pour interlocking tooling: jacket A, jacket B, and the core.

    The sequence the parts are for:

      1. assemble core + jacket A, pour, cure, release
      2. assemble core + jacket B *against cured A*, pour, cure
      3. the two rubber halves interlock — A's sockets cast B's pegs

    Which is why the jacket is split and keyed rather than fused. A single fused box
    is simpler still and gives a one-piece glove that has to be cut off the master;
    on the vessel the one-piece hoop stretch is 163 % against a 62 % allowable, so
    that glove could not be removed intact anyway.
    """
    st, wall, floor = spec.silicone_t, spec.wall, spec.floor
    plane = parting["plane"]
    z_lo, z_hi = float(part.bounds[0][2]), float(part.bounds[1][2])

    # Core: the part, plus a flange taken from its BOTTOM SECTION so a chamfered
    # base stays on the core and the rubber takes that shape.
    base = silhouette(part, z_lo + min(0.2, 0.02 * (z_hi - z_lo)))
    flange_h = max(2.0, min(floor, 3.0))
    flange = prism(base, max(0.5 * st, 1.2), z_lo - flange_h, z_lo + 1.0)
    core = trimesh.boolean.union([part, flange])

    shadow = silhouette(part)
    chamber = prism(shadow, st, z_lo, z_hi + st)
    # The jacket gets a LID as well as a floor, and the pour goes through a spout in
    # it. Without one the upper half is an open ring: its pillars have nothing above
    # them to hang from and come off as loose rod — measured, 37 separate bodies in
    # one jacket half. A floor anchors the lower pillars; a lid anchors the upper.
    outer = block(part, st + wall, z_lo - flange_h - floor, z_hi + st + wall)

    # Pillars bridge the gap on the window grid: they form the skin's breather
    # windows and hold the core at the silicone offset while the rubber is liquid.
    pil = window_pillars(shadow, win["d_mm"], win["spacing_mm"],
                         z_lo - flange_h - floor, z_hi + st + wall)
    gap = trimesh.boolean.difference([chamber, core])
    if pil is not None:
        pil = trimesh.boolean.intersection([pil, outer])
        gap = trimesh.boolean.difference([gap, pil])

    jacket = trimesh.boolean.difference([outer, chamber])
    if pil is not None:
        jacket = trimesh.boolean.union([jacket, pil])
    # Fill port through the lid, on the axis, sized on viscous fill rather than on
    # jamming — the silicone pour carries no aggregate.
    spout = trimesh.creation.cylinder(
        radius=spec.spout_d / 2, height=4 * wall, sections=32,
        transform=trimesh.transformations.translation_matrix(
            [shadow.centroid.x, shadow.centroid.y, z_hi + st + wall]))
    jacket = trimesh.boolean.difference([jacket, spout])

    lo, hi = bisect(jacket, 2, plane)
    pegs, sockets = frustum_keys(shadow, plane, st, wall, spec)
    if pegs is not None:
        lo = trimesh.boolean.union([lo, pegs])
        hi = trimesh.boolean.difference([hi, sockets])

    v = float(gap.volume)
    return {
        "parts": {"jacket_a": lo, "jacket_b": hi, "core": core},
        "silicone_volume_mm3": v,
        "silicone_mass_g": v * spec.rho_g_cm3 / 1000.0,
        "n_pillars": 0 if pil is None else int(pil.body_count),
        "procedure": [
            "print jacket_a, jacket_b and core",
            "assemble core + jacket_a, pour silicone, cure, release",
            "assemble core + jacket_b against the cured half, pour, cure",
            "the two rubber halves interlock; cast the mix in them, cure open-faced",
        ],
    }


def build_rigid_mould(part: trimesh.Trimesh, spec: CastSpec,
                      parting: dict) -> dict:
    """A split rigid negative: pour the mix straight into it.

    No silicone, no second casting. Cheaper and faster, and it charges the draft
    against the cast's own section — which is why the silicone path exists.
    """
    wall, floor = spec.wall, spec.floor
    plane = parting["plane"]
    z_lo, z_hi = float(part.bounds[0][2]), float(part.bounds[1][2])
    shadow = silhouette(part)
    blk = block(part, wall, z_lo - floor, z_hi + wall)
    cavity = trimesh.boolean.difference([blk, part])
    lo, hi = bisect(cavity, 2, plane)
    pegs, sockets = frustum_keys(shadow, plane, 0.0, wall, spec)
    if pegs is not None:
        lo = trimesh.boolean.union([lo, pegs])
        hi = trimesh.boolean.difference([hi, sockets])
    return {
        "parts": {"lower": lo, "upper": hi},
        "silicone_volume_mm3": 0.0, "silicone_mass_g": 0.0, "n_pillars": 0,
        "procedure": [
            "print lower and upper",
            "cast the mix in the halves and CURE THEM OPEN-FACED — assembling early "
            "turns the parting face from an oxygen source into a sealed interface",
        ],
    }


def build_mould(geom, spec: CastSpec | None = None, *, pitch: float = 0.0,
                phys=None, cure_days: float = 28.0, rh_pct: float = 90.0) -> dict:
    """Generate a mould for one grammar design, and report what was decided.

    Meshes are CSG'd; the only voxel work left is the aeration solve, which needs an
    occupancy grid and is the reason this project generates moulds at all.
    """
    from .grammars import block as bl
    from .grammars import shell as sh
    from .grammars import tile as tl

    spec = spec or CastSpec()
    if phys is None:
        from .gui import engine as eng
        phys = eng.load_physics()

    mod = {"shell": sh, "block": bl, "tile": tl}[geom.typology]
    p = pitch or {"shell": 2.0, "block": 2.5, "tile": 1.6}[geom.typology]
    mesh, fld, origin, p = mod.build(geom, pitch=p, return_field=True)
    occ = fld <= 0.0

    parting = choose_parting(mesh)
    part = _to_draw_frame(mesh, parting["axis"])
    part = apply_shrink(part, spec.shrink_pct)
    part = apply_draft(part, 2, parting["plane"], spec.draft_deg)

    # transport scales, and the window pitch that meets coverage
    D_gas = ox.effective_diffusivity(phys.D_O2_gas[1], phys.phi[1], phys.sw[1],
                                     gas=True)
    L_gas = ox.analytic_penetration_depth(D_gas, phys.C_O2_gas[1],
                                          phys.R_O2_bulk[1])
    L_dry = dry.air_entry_depth(phys.E_evap[1], cure_days, phys.phi[1],
                                delta_saturation=phys.dS_air_entry[1], rh_pct=rh_pct)
    L_eff = float(min(L_gas, L_dry))
    win = size_windows(occ, origin, p, parting["axis"], spec, L_eff)

    built = (build_silicone_tooling(part, spec, parting, win)
             if spec.goal == "silicone" else build_rigid_mould(part, spec, parting))

    parts = built["parts"]
    report = {name: {"volume_cm3": round(float(m.volume) / 1000.0, 1),
                     "watertight": bool(m.is_watertight),
                     "bodies": int(m.body_count),
                     "bbox_mm": [round(float(v), 1) for v in m.extents],
                     "triangles": int(len(m.faces))}
              for name, m in parts.items()}
    plastic = sum(r["volume_cm3"] for r in report.values()
                  if not r.get("_cast")) if spec.goal == "rigid" else sum(
        report[n]["volume_cm3"] for n in report)

    return {
        "typology": geom.typology, "goal": spec.goal, "pitch": p,
        "parting": parting, "window": win, "L_eff_mm": L_eff, "L_dry_mm": L_dry,
        "spec": asdict(spec),
        "parts": parts, "report": report,
        "plastic_cm3": plastic,
        "silicone_volume_mm3": built["silicone_volume_mm3"],
        "silicone_mass_g": built["silicone_mass_g"],
        "n_pillars": built["n_pillars"],
        "procedure": built["procedure"],
        "checks": checks(report, win, spec),
    }


def checks(report: dict, win: dict, spec: CastSpec) -> list:
    """The handful of things that can actually be wrong here.

    Deliberately short. The voxel path ran volume balances, release sweeps and Euler
    counts because its booleans could silently misplace material on a grid; mesh CSG
    either produces a valid solid or raises. What is left is what the geometry cannot
    guarantee: that each part is one printable body, and that the mould breathes.
    """
    out = [{"check": "every part is a single printable body",
            "pass": all(r["bodies"] == 1 for r in report.values()),
            "detail": ", ".join(f"{n}: {r['bodies']}" for n, r in report.items())},
           {"check": "every part is watertight",
            "pass": all(r["watertight"] for r in report.values()), "detail": ""},
           {"check": "windows meet the aeration criterion",
            "pass": bool(win["meets_target"]),
            "detail": f"{win['cover_surrogate']:.3f} against "
                      f"{spec.coverage_target:.2f} at {win['spacing_mm']:.0f} mm "
                      f"pitch — {win['limited_by']}"},
           {"check": "window bore clears the clog band",
            "pass": bool(win["d_mm"] >= 2.0 * spec.d_max),
            "detail": f"{win['d_mm']:.0f} mm against 2 x d_max = "
                      f"{2 * spec.d_max:.0f} mm"}]
    return out
