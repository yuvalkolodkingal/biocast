"""Load capacity of a bio-cemented cast body, in kN, with its provenance attached.

WHY THIS IS A REPORTED NUMBER AND NOT A FIFTH SUBSCORE
------------------------------------------------------
The four subscores in `biocast.score` answer one question — will this body
solidify completely? — and they multiply because they are series requirements.
Capacity is a different kind of statement: it is what the object can carry ONCE
it has cemented, and it is conditional on the cementation having worked. Folding
it into the product would let a strong-but-uncastable geometry trade capacity
against castability, which is not a trade that exists: a starved mould does not
produce a weaker object, it produces a defective one.

So capacity is carried alongside the score, never gates feasibility, and is
ranked lexicographically behind broken hard rules (`gui.engine.rank_strength`).

WHY EVERY NUMBER HERE CARRIES A LABEL
-------------------------------------
The evidence is weak on purpose, and the weakness is structural rather than
something more sampling would fix:

  * UCS on this substrate class rests on ONE study. Fouladi et al. 2024
    (10.1007/s11440-024-02396-8) measured 0.341-0.724 MPa on washed recycled sand
    from demolition waste. That is the only measurement on construction waste and
    it is ~4x weaker than clean-sand MICP at comparable treatment effort.
  * The organism is wrong. Every UCS row in the retrieved set was measured with
    Sporosarcina pasteurii (ureolytic); this project uses Bacillus subtilis
    (non-ureolytic, oxidative, slower). NO numeric transfer factor exists — the
    source JSON says so in as many words — so a derating factor is ASSUMED,
    uniform 0.3-0.7, and labelled on every output.
  * There is no usable UCS-vs-CaCO3 curve. Pooled fits over the retrieved points
    give R^2 <= 0.01, and the Almajed 2019 pair differs 12x at the same 1.4 %
    carbonate. Carbonate content therefore enters as a hard GATE (Fu et al. 2023:
    below ~3 % the specimen is not self-supporting at all) and not as a trend.

The width of the resulting distribution is the honest answer, and it is wide.
`data/mechanics_params.json` -> `ucs_vs_caco3_fit.recommended_monte_carlo_treatment`
is the specification this module implements; it is quoted rather than paraphrased
in `PROVENANCE_NOTE` so the output cannot drift from the source.

THIS IS NOT A STRUCTURAL SIGN-OFF. Nothing here has been cast or tested. The
capacity is a literature UCS envelope multiplied by an analytic net section and
divided by an elastic notch factor — a way to rank geometries against each other,
in the same spirit as the score itself. ASTM C90 requires 13.8 MPa net-area
compressive strength for a loadbearing masonry unit; MICP on waste aggregate sits
5-20x below that, so a bio-cemented unit is not a structural CMU substitute.

Units
-----
Sections in mm^2, strengths in MPa, capacities reported in kN. MPa is N/mm^2
exactly, so `ucs * area` is already a force in newtons and the mm-at-the-boundary
convention costs no conversion factor — only the final /1000 into kN.
"""
from __future__ import annotations

import numpy as np

from ..params import BlockParams, Design, LitParams, ShellParams, TileParams
from .. import score as sc

# --------------------------------------------------------------------------
# Literature defaults. Used only when no LitParams is handed in; every one of
# them is also present in data/mechanics_params.json, and `_ucs_bounds` prefers
# the file so a corrected row propagates without a code change.
# --------------------------------------------------------------------------

#: Minimum CaCO3 for a self-supporting (unconfined) specimen, % by mass. Fu,
#: Saracho & Haigh 2023: "requires a minimum Ccc of circa 3%". Below it the body
#: is not weak — it does not stand up, so the capacity is 0 rather than small.
CACO3_FLOOR_PCT = 3.0

#: UCS envelope per substrate class, MPa, sampled LOG-uniform (the envelope spans
#: a decade for clean sand, and a log-uniform draw does not privilege the top of
#: it the way a uniform draw would).
#:
#: "rca"        Fouladi et al. 2024, washed recycled C&D sand — the actual
#:              substrate class here, and the only measurement on it.
#: "clean_sand" the pooled clean-sand band quoted in the source JSON's
#:              `recommended_monte_carlo_treatment`. It has no `parameters` row
#:              of its own (the nearest, `q_u`, is a wider 0.05-18 MPa pool
#:              across all substrates), so it stays a constant here with the
#:              quote recorded rather than being read from a row that means
#:              something else.
UCS_BOUNDS_MPa = {"rca": (0.34, 0.72), "clean_sand": (0.3, 12.0)}

#: Which `parameters` row backs each class, where one does.
UCS_SYMBOL = {"rca": "q_u,RCA", "clean_sand": None}

#: ASSUMED derating for the organism mismatch, uniform. There is no measured
#: transfer factor from S. pasteurii to B. subtilis; the source JSON recommends
#: exactly this range and exactly this treatment — "an explicit, clearly-labelled
#: ASSUMED derating factor with a wide range (suggest 0.3-0.7, uniform)".
DERATING_B_SUBTILIS = (0.3, 0.7)

#: ASTM C90 minimum net-area compressive strength for a loadbearing CMU, MPa.
#: Benchmark only — it is what the capacity is reported AGAINST, never a target.
C90_BENCHMARK_MPa = 13.8

SUBSTRATE_CLASSES = tuple(UCS_BOUNDS_MPa)

PROVENANCE_NOTE = (
    "UCS drawn log-uniform over the {lo:.2f}-{hi:.2f} MPa envelope for substrate "
    "class '{cls}' ({src}), then multiplied by an ASSUMED B. subtilis derating of "
    "{d_lo:g}-{d_hi:g} uniform — no numeric S. pasteurii-to-B. subtilis transfer "
    "factor exists. CaCO3 gate at {gate:g} % (MEASURED, Fu et al. 2023). Section "
    "and notch factor are analytic geometry, not a test. Not a structural "
    "sign-off: ASTM C90 asks {c90:g} MPa net area and MICP on waste aggregate is "
    "5-20x below it."
)


# --------------------------------------------------------------------------
# UCS sampling
# --------------------------------------------------------------------------
def _ucs_bounds(substrate_class: str, mec: LitParams | None) -> tuple[float, float]:
    """(low, high) MPa for a substrate class, from the retrieved rows when given."""
    if substrate_class not in UCS_BOUNDS_MPa:
        raise ValueError(f"unknown substrate_class {substrate_class!r}; "
                         f"expected one of {SUBSTRATE_CLASSES}")
    lo, hi = UCS_BOUNDS_MPa[substrate_class]
    sym = UCS_SYMBOL[substrate_class]
    if mec is not None and sym:
        try:
            lo, hi = mec.bounds(sym, (lo, hi))
        except Exception:
            pass
    return float(lo), float(hi)


def _caco3_floor(mec: LitParams | None) -> float:
    if mec is None:
        return CACO3_FLOOR_PCT
    try:
        return float(mec.get("Ccc_min", CACO3_FLOOR_PCT))
    except Exception:
        return CACO3_FLOOR_PCT


def sample_ucs(rng: np.random.Generator, substrate_class: str, caco3_pct: float,
               *, mec: LitParams | None = None) -> float:
    """One draw of unconfined compressive strength, MPa.

    Three steps, in the order the source JSON prescribes:

    1. the carbonate gate. Below ~3 % CaCO3 the specimen cannot stand without
       confinement, so this returns 0.0 — a discontinuity, deliberately, because
       that is what the measurement reports. It is not a soft penalty.
    2. a LOG-uniform draw over the substrate class's measured envelope. Not a
       fitted curve: every pooled UCS-vs-CaCO3 fit in the retrieved set has
       R^2 <= 0.01, and the same carbonate content has produced strengths 12x
       apart, so the envelope IS the model.
    3. the ASSUMED organism derating, uniform 0.3-0.7, applied on top.
    """
    if caco3_pct < _caco3_floor(mec):
        return 0.0
    lo, hi = _ucs_bounds(substrate_class, mec)
    if hi <= lo:
        ucs = float(lo)
    else:
        ucs = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return ucs * float(rng.uniform(*DERATING_B_SUBTILIS))


# --------------------------------------------------------------------------
# Stress concentration
# --------------------------------------------------------------------------
def notch_of(geom) -> tuple[float, float]:
    """(notch depth, root radius) in mm, for the Inglis stress-concentration factor.

    A shell gets no notch depth: its rim is a change of section, not a groove cut
    into a face, and the grammar gives it nothing to measure a depth against. A
    rim step therefore has to be supplied by the load case when one is being
    modelled — `validate_paper.py` case E does exactly that.

    `gui.engine.notch_of` is this function; it lives here so the scorer and the
    capacity model cannot disagree about what the notch is.
    """
    if isinstance(geom, ShellParams):
        return 0.0, max(geom.fillet_r, 1e-6)
    depth = float(getattr(geom, "groove_depth", 0.0) or 0.0)
    if depth <= 0:
        return 0.0, max(geom.fillet_r, 1e-6)
    width = float(getattr(geom, "groove_width", 0.0) or 0.0)
    root = min(geom.fillet_r, width / 2 if width > 0 else geom.fillet_r, depth)
    return depth, max(root, 1e-6)


# --------------------------------------------------------------------------
# Critical section, analytic per grammar
# --------------------------------------------------------------------------
#: Angular quadrature for the shell's superellipse cross-sections, and height
#: samples along its axis. Both are 1D, so this is microseconds.
_N_THETA = 720
_N_Z = 241

#: In-plane sampling cell for the tile, mm. The relief is millimetres wide, so a
#: 0.5 mm cell resolves it an order of magnitude finer than the voxel pitch the
#: check compares against.
_TILE_CELL_MM = 0.5

#: How far the analytic section may sit from the voxel count before the geometry
#: model is considered to have lost track of what the grammar builds. Reported,
#: never enforced — see `section_check`.
SECTION_CHECK_TOL = 0.15


def _superellipse_profile(p: ShellParams):
    """Reference polar profile of |x/a|^n + |y/b|^n = 1, with its integrals.

    Every horizontal section of the ovoid is this ONE profile scaled by a single
    factor k(z) = taper(z) * (1 - |z/c|^n)^(1/n), because the taper multiplies
    both semi-axes equally. So the area and perimeter integrals are done once and
    scaled by k^2 and k, rather than re-integrated at every height.

    Returns (rho, area, perimeter, inradius) for k = 1.
    """
    th = np.linspace(0.0, 2.0 * np.pi, _N_THETA, endpoint=False)
    dth = 2.0 * np.pi / _N_THETA
    rho = (np.abs(np.cos(th) / p.a) ** p.n
           + np.abs(np.sin(th) / p.b) ** p.n) ** (-1.0 / p.n)
    # circular difference: the profile is periodic, so np.gradient's one-sided
    # end rule would put a spurious kink at theta = 0
    drho = (np.roll(rho, -1) - np.roll(rho, 1)) / (2.0 * dth)
    area = float(0.5 * np.sum(rho ** 2) * dth)
    perim = float(np.sum(np.sqrt(rho ** 2 + drho ** 2)) * dth)
    return rho, area, perim, float(rho.min())


def _shell_scale(p: ShellParams, z: np.ndarray) -> np.ndarray:
    """k(z): the single factor every horizontal section of the ovoid is scaled by."""
    zn = np.clip(z / p.c, -1.0, 1.0)
    taper = np.clip(1.0 - p.ovoid * (zn + 1.0) / 2.0, 0.15, None)
    s = np.clip(1.0 - np.abs(zn) ** p.n, 0.0, None)
    return taper * s ** (1.0 / p.n)


def _shell_net_area(p: ShellParams, k, prof) -> np.ndarray:
    """Net material area of the sections at scale `k`, mm^2.

    The section is the outer superellipse minus the hole, and the hole is
    whichever of two things is larger: the cavity, or the aperture bore.

    The cavity is the 3D inward offset of the outer surface by `wall`
    (`sdf.shell_of` works on a true distance field), and its in-plane area comes
    from Steiner's formula for the inner parallel body, A - P*w + pi*w^2, guarded
    by the section's own inradius — without that guard the formula returns
    pi*(R-w)^2 for a section too small to hollow at all, i.e. it re-opens a
    cavity at the poles where there is none. The 3D offset is never wider than
    the in-plane one, so this slightly OVERstates the cavity and understates the
    section: conservative, and exact where the wall runs vertical.

    The bore is a true cylinder subtraction spanning the whole body, so it is
    integrated in polar form as the circle of radius `aperture_r` CLIPPED to the
    section, rather than assumed to fit inside it.
    """
    rho, area_u, perim_u, rho_min = prof
    k = np.atleast_1d(np.asarray(k, float))
    a_out = area_u * k ** 2
    inradius = rho_min * k

    a_cav = np.where(inradius > p.wall,
                     np.clip(a_out - perim_u * k * p.wall + np.pi * p.wall ** 2,
                             0.0, None), 0.0)
    a_cav = np.minimum(a_cav, a_out)

    if p.aperture_r > 0:
        r_lim = np.minimum(k[:, None] * rho[None, :], p.aperture_r)
        a_bore = 0.5 * np.sum(r_lim ** 2, axis=1) * (2.0 * np.pi / _N_THETA)
    else:
        a_bore = np.zeros_like(a_out)

    return np.clip(a_out - np.maximum(a_cav, a_bore), 0.0, None)


def _shell_neck(p: ShellParams):
    """(scale, height) of the smallest full-thickness ring, or (None, None).

    WITH AN APERTURE, THE NECK IS WHERE THE BORE LEAVES EXACTLY ONE WALL
    THICKNESS. The bore is a through-cylinder of fixed radius while the body
    tapers around it, so above some height the ring alongside it is thinner than
    the wall the designer asked for, and at the rim it thins to nothing. Below
    that height the hole is the cavity again and the ring is a full `wall` thick
    however wide the section gets. So the named section is the last full-thickness
    ring: the height whose inradius is `aperture_r + wall`.

    WITHOUT ONE, it is the height at which the cavity closes — inradius equal to
    `wall`. Below it the section is a full-thickness annulus that only grows;
    above it a solid cap that only shrinks toward a point that carries nothing.

    Not a scan for the pointwise minimum, deliberately. That minimum is always
    zero — the rim tapers to a knife edge, and a closed form's poles close to a
    point — and any value near it is set by exactly where the body ends, which
    moves by a voxel when the pitch changes. A level crossing of a smooth
    function does not move. Since section area depends on height only through k,
    and k is pinned by the inradius condition, the ring above the equator and the
    one below it have the SAME area; only the upper height is returned, for the
    grid check to aim at.

    Returns (None, None) for a solid form, which has no ring at all and falls
    back on the bearing-band minimum.
    """
    rho_min = _superellipse_profile(p)[3]
    z = np.linspace(-p.c, p.c, _N_Z)
    k = _shell_scale(p, z)
    k_max = float(k.max())

    if p.aperture_r <= 0 and p.wall >= k_max * rho_min:
        return None, None                      # no cavity and no bore: a solid body
    k_neck = (p.aperture_r + p.wall) / rho_min
    # A bore wide enough that no full-thickness ring survives anywhere leaves the
    # whole vessel a tapering lip. There is then no neck to name and the widest
    # ring is reported instead, which OVERSTATES it — but a bore that wide is
    # rejected by the jamming and minimum-section rules long before capacity is
    # the interesting number.
    k_neck = min(k_neck, k_max)
    upper = z >= z[int(np.argmax(k))]
    # k falls monotonically from its peak to the top pole, so the crossing is unique
    z_neck = float(np.interp(-k_neck, -k[upper], z[upper]))
    return float(k_neck), z_neck


def _bearing_band(z: np.ndarray, area: np.ndarray, lip: float) -> np.ndarray:
    """Planes the load has to cross, excluding the rounded lip at each end.

    A closed ovoid has no neck to name, and its pointwise minimum section is at a
    pole, which carries nothing: that is where load is INTRODUCED, and the
    grammar rounds it over `fillet_r` precisely so there is no edge there. So the
    first section that has to carry the full load sits one fillet radius in from
    each extreme — the same rule `_block_section` applies to the rounded top
    arris of a masonry unit.
    """
    live = np.flatnonzero(area > 0)
    if live.size == 0:
        return np.zeros_like(area, dtype=bool)
    z_lo, z_hi = z[live[0]], z[live[-1]]
    lip = min(max(lip, 0.0), 0.45 * (z_hi - z_lo))
    band = (z >= z_lo + lip) & (z <= z_hi - lip) & (area > 0)
    return band if band.any() else (area > 0)


def _shell_section(p: ShellParams) -> float:
    """Smallest load-carrying net section of a hollow ovoid, mm^2."""
    prof = _superellipse_profile(p)
    k_neck, _ = _shell_neck(p)
    if k_neck is not None:
        return float(_shell_net_area(p, k_neck, prof)[0])
    z = np.linspace(-p.c, p.c, _N_Z)
    a_net = _shell_net_area(p, _shell_scale(p, z), prof)
    band = _bearing_band(z, a_net, p.fillet_r)
    return float(a_net[band].min()) if band.any() else 0.0


def _block_section(p: BlockParams) -> float:
    """Net horizontal area of a hollow-core unit, mm^2 — ASTM C90's own measure.

    C90 states unit strength on the NET area (face shells plus webs), which is
    exactly the horizontal section through the cores, so the same number serves
    the standard and the capacity model.

    Taken at z = H/2 - fillet_r, the highest plane at full width. The cores are
    drafted for release and so are widest at the top, which makes that plane the
    smallest net section in the prismatic part of the block; above it the rounded
    top arris cuts the footprint down, but that is the bearing face — a chamfer
    detail at the load introduction, not a section the load has to cross.
    """
    from ..grammars.block import core_centers, core_dims

    core_w, core_d = core_dims(p)
    r = max(float(p.fillet_r), 0.0)
    corner = (4.0 - np.pi) * r ** 2                  # rounded rectangle correction

    grow = np.tan(np.deg2rad(max(p.core_taper, 0.0))) * max(p.H - r, 0.0)
    cw = max(core_w + 2.0 * grow, 0.0)
    cd = min(max(core_d + 2.0 * grow, 0.0), p.W)     # a core cannot exceed the block

    # The draft widens every core by the same amount, so at enough draft adjacent
    # cores MERGE and the webs between them are gone. Summing n * cw * cd would
    # then subtract the overlaps twice and understate the section by ~20 % at 5
    # deg on a three-core unit — so the x-extents are merged first, exactly as the
    # SDF's successive subtractions do.
    spans: list[list[float]] = []
    for xc in core_centers(p):
        lo, hi = max(xc - cw / 2, -p.L / 2), min(xc + cw / 2, p.L / 2)
        if hi <= lo:
            continue
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])

    area = p.L * p.W - corner
    for lo, hi in spans:
        area -= max((hi - lo) * cd - corner, 0.0)    # each merged void has 4 corners
    if p.groove_count > 0 and p.groove_depth > 0 and p.groove_width > 0:
        # one groove per long face, both faces
        area -= 2.0 * p.groove_count * p.groove_width * p.groove_depth
    return float(max(area, 0.0))


def _tile_section(p: TileParams, pitch: float = 0.0) -> float:
    """Net area of a relief tile at the groove-root plane, mm^2.

    The root plane is where the notch tip sits, so it is the plane the Inglis Kt
    belongs to: net-section stress times Kt at the root is the whole model. Above
    it the section is inside the relief, which is a drainage and slip surface
    rather than a bearing one.

    Sampled rather than derived per pattern, and sampled with the grammar's OWN
    `_pattern_field`, so the flower and radial motifs are handled exactly and a
    change to a pattern cannot leave a hand-derived area formula behind. The
    tile's rounded perimeter is evaluated at the same height — with a 40 mm tile,
    an 8 mm edge radius and a 3 mm relief, the root plane is inside the rounded
    band and the footprint there is ~3 % under L*W.
    """
    from ..grammars.tile import _pattern_field

    r_edge = max(min(p.fillet_r, p.t / 2.0 - pitch, min(p.L, p.W) / 4.0), 0.0)
    grooved = p.groove_depth > 0 and p.groove_width > 0
    # with no relief the weakest plane is simply mid-thickness, at full width
    z = (p.t / 2.0 - p.groove_depth) if grooved else 0.0

    az = max(abs(z) - (p.t / 2.0 - r_edge), 0.0)
    rr = float(np.sqrt(max(r_edge ** 2 - az ** 2, 0.0)))
    hx, hy = p.L / 2.0 - r_edge + rr, p.W / 2.0 - r_edge + rr

    cell = min(_TILE_CELL_MM, max(pitch, _TILE_CELL_MM) / 2.0) if pitch else _TILE_CELL_MM
    xs = np.arange(-p.L / 2.0 - cell, p.L / 2.0 + cell, cell)
    ys = np.arange(-p.W / 2.0 - cell, p.W / 2.0 + cell, cell)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    dx, dy = np.abs(X) - (hx - rr), np.abs(Y) - (hy - rr)
    foot = (np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
            + np.minimum(np.maximum(dx, dy), 0.0) - rr) <= 0.0
    if grooved:
        foot &= _pattern_field(X, Y, p) >= 0.0
    return float(foot.sum() * cell ** 2)


def critical_section_mm2(geom, diag: dict) -> float:
    """Cross-sectional area at the weakest section, mm^2, analytic per grammar.

    `diag` is only read for its voxel pitch (the tile's perimeter radius is
    clamped against it in the grammar); the area itself never comes off the grid,
    so this stays exact where the discretisation is not. `section_check` compares
    the two.
    """
    pitch = float(diag.get("_grid", {}).get("pitch", 0.0) or 0.0) if diag else 0.0
    if isinstance(geom, ShellParams):
        return _shell_section(geom)
    if isinstance(geom, BlockParams):
        return _block_section(geom)
    if isinstance(geom, TileParams):
        return _tile_section(geom, pitch)
    raise TypeError(type(geom))


def section_check(geom, diag: dict) -> dict:
    """Cross-check the analytic section against the occupancy grid.

    Cheap because the grid is already built and the comparison is one sum per
    plane. It is REPORTED, never enforced: the grid is a discretisation at 1.6-3
    mm pitch and the analytic value is not, so a few per cent of disagreement is
    the voxel edge, not an error. Anything past `SECTION_CHECK_TOL` means the
    analytic model has lost track of what the grammar builds, which is worth
    seeing. Two known ways it does, both on the shell:

    * A LARGE APERTURE FILLET EATS THE NECK, and the analytic annulus does not
      model it. At the neck the cavity wall and the bore are coincident by
      construction, so `op_round_subtract`'s blend joins two nearly tangent
      surfaces and takes a bite off the ring's inner edge that grows fast with
      `fillet_r / wall`. Measured on a 6 mm wall with a 35 mm bore: 1.5 % out at
      fillet_r = 2 mm, 36 % at 8 mm, and at 14 mm the built ring is essentially
      gone (21 mm^2) while the analytic still reports 1486. The analytic reads
      OPTIMISTIC here, so the check failing is the finding.
    * A CLOSED SHELL'S CAVITY CLOSES NEAR A POLE, where a 2-3 mm voxel is coarse
      against a ring a few millimetres across. That one is discretisation and it
      converges — 32 % at 2.0 mm pitch, 4 % at 0.6 mm.

    Each typology is compared on the plane its analytic value describes, since
    comparing a rim annulus against a mid-body section would tell us nothing.
    """
    analytic = critical_section_mm2(geom, diag)
    grid = (diag or {}).get("_grid")
    if not grid:
        return {"analytic_mm2": analytic, "voxel_mm2": None, "rel_diff": None,
                "grid_agrees": None, "plane": "no occupancy grid supplied"}

    occ, pitch, origin = grid["occ"], float(grid["pitch"]), np.asarray(grid["origin"])
    areas = occ.sum(axis=(0, 1)).astype(float) * pitch ** 2
    z = origin[2] + np.arange(areas.size) * pitch

    if isinstance(geom, ShellParams):
        _, z_neck = _shell_neck(geom)
        if z_neck is None:
            band = _bearing_band(z, areas, geom.fillet_r)
            vox = float(areas[band].min()) if band.any() else 0.0
            note = "smallest plane a fillet radius inside each end, as the analytic is"
        else:
            idx = int(np.argmin(np.abs(z - z_neck)))
            vox = float(areas[idx])
            note = f"the neck plane the analytic names, z = {z[idx]:.1f} mm"
    elif isinstance(geom, BlockParams):
        band = (np.abs(z) <= geom.H / 2.0 - geom.fillet_r) & (areas > 0)
        vox = float(areas[band].min()) if band.any() else 0.0
        note = "smallest plane at full width (below the rounded top arris)"
    elif isinstance(geom, TileParams):
        if geom.groove_depth > 0 and geom.groove_width > 0:
            root = geom.t / 2.0 - geom.groove_depth
            cand = np.where((z >= root) & (areas > 0))[0]
            idx = int(cand[0]) if cand.size else int(np.argmax(areas))
            note = f"first plane at or above the groove root, z = {z[idx]:.1f} mm"
        else:
            idx = int(np.argmin(np.abs(z)))
            note = "mid-thickness plane"
        vox = float(areas[idx])
    else:
        raise TypeError(type(geom))

    rel = abs(analytic - vox) / vox if vox > 0 else None
    return {"analytic_mm2": analytic, "voxel_mm2": vox, "rel_diff": rel,
            "grid_agrees": None if rel is None else bool(rel <= SECTION_CHECK_TOL),
            "plane": note}


# --------------------------------------------------------------------------
# Top-level capacity estimate
# --------------------------------------------------------------------------
def load_capacity(design: Design, diag: dict, *, phys: LitParams | None = None,
                  n_mc: int = 400, seed: int = 0,
                  notch_depth_mm: float | None = None,
                  root_radius_mm: float | None = None) -> dict:
    """Estimated load capacity of one design, kN, with its interval and provenance.

    capacity = UCS x A_net / Kt, per draw, with UCS from `sample_ucs` and Kt the
    Inglis factor for the geometry's own notch. Dividing by Kt rather than
    multiplying the stress is the net-section form: the section carries the load,
    the notch multiplies the local stress at the root, so the load that first
    reaches UCS somewhere is smaller by exactly that factor. The source JSON
    warns that Inglis OVERESTIMATES Kt by 30 % or more against FE for rough
    roots, which makes this the conservative side — the right side to err on for
    a brittle cast object, and the same convention `score.s_structural` uses.

    `phys` is the MECHANICS `LitParams` (data/mechanics_params.json), not
    `score.PhysicsInputs`. The capacity model reads its bounds straight from the
    retrieved rows rather than from a (low, mode, high) triangle, because on this
    substrate there is a single study and a triangular prior would invent a mode
    the measurement does not have.

    `notch_depth_mm` / `root_radius_mm` default to the geometry's own notch via
    `notch_of`. They are exposed for the same reason `score.score_design` exposes
    them: a shell's rim step is a property of the load case, not of the grammar.

    Capacity NEVER marks a design infeasible. It is reported next to the score,
    and only breaks ties behind the hard rules when the search is asked for
    strength.
    """
    rng = np.random.default_rng(seed)
    area = critical_section_mm2(design.geom, diag)
    check = section_check(design.geom, diag)

    nd, rr = notch_of(design.geom)
    nd = nd if notch_depth_mm is None else float(notch_depth_mm)
    rr = rr if root_radius_mm is None else float(root_radius_mm)
    kt = sc.kt_inglis(nd, rr)

    cls = getattr(design.mix, "substrate_class", "rca")
    caco3 = float(getattr(design.mix, "caco3_achieved_pct", 0.0))
    ucs = np.array([sample_ucs(rng, cls, caco3, mec=phys) for _ in range(int(n_mc))])

    # Kt = inf for a zero root radius: the capacity is then 0, which is the
    # honest reading of a square-cut root in a brittle cast, not a divide error.
    cap_kN = (ucs * area / kt / 1000.0) if np.isfinite(kt) else np.zeros_like(ucs)

    lo, hi = _ucs_bounds(cls, phys)
    ucs_nom = float(np.median(ucs))
    return {
        "capacity_kN": float(np.median(cap_kN)),
        "capacity_lo_kN": float(np.percentile(cap_kN, 5)),
        "capacity_hi_kN": float(np.percentile(cap_kN, 95)),
        "capacity_mean_kN": float(cap_kN.mean()),
        "ucs_nom_MPa": ucs_nom,
        "ucs_lo_MPa": float(np.percentile(ucs, 5)),
        "ucs_hi_MPa": float(np.percentile(ucs, 95)),
        "critical_section_mm2": float(area),
        "critical_section_voxel_mm2": check["voxel_mm2"],
        "critical_section_rel_diff": check["rel_diff"],
        "critical_section_agrees": check["grid_agrees"],
        "critical_section_plane": check["plane"],
        "kt_used": float(kt),
        "notch_depth_mm": float(nd),
        "notch_root_mm": float(rr),
        "substrate_class": cls,
        "caco3_achieved_pct": caco3,
        "caco3_floor_pct": _caco3_floor(phys),
        "c90_benchmark_MPa": C90_BENCHMARK_MPa,
        "c90_ratio": (C90_BENCHMARK_MPa / ucs_nom) if ucs_nom > 0 else float("inf"),
        "n_mc_strength": int(n_mc),
        "strength_provenance": provenance_text(cls, phys),
    }


def provenance_text(substrate_class: str, mec: LitParams | None = None) -> str:
    """One sentence recording where every factor in the capacity came from."""
    lo, hi = _ucs_bounds(substrate_class, mec)
    src = "MEASURED, single study, S. pasteurii on demolition-waste sand"
    if mec is not None and UCS_SYMBOL.get(substrate_class):
        p = mec.provenance(UCS_SYMBOL[substrate_class])
        bits = [b for b in (p.get("evidence_class"), p.get("organism"),
                            p.get("source_doi")) if b]
        if bits:
            src = ", ".join(str(b) for b in bits)
    return PROVENANCE_NOTE.format(lo=lo, hi=hi, cls=substrate_class, src=src,
                                  d_lo=DERATING_B_SUBTILIS[0],
                                  d_hi=DERATING_B_SUBTILIS[1],
                                  gate=_caco3_floor(mec), c90=C90_BENCHMARK_MPa)
