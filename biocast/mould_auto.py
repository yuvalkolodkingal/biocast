"""Automatic mould generation — one driver for any grammar.

`mould.py` proved the geometry machinery on two hand-tuned typologies: the parting
plane, the draft sweeps, the attributed volume balance and the release test are all
verified there and are reused unchanged. What it did *not* do is decide anything on
its own. Every gate angle, drain radius, flange span, key ring and mould wall was a
literal in `build_shell_mould` / `build_block_mould`, which is why the tile grammar
has no mould at all: adding one meant writing a third driver by hand.

This module replaces those literals with measurements. Given any grammar's field it
derives the parting axis and plane, the flange outline, the mould wall, the draft
angle, the feature layout and the core strategy, then hands the result to
`mould.assemble`. The five decisions below are the ones that were previously
hand-made, and each is now computed with its own failure mode guarded.

1. PARTING AXIS AND PLANE ARE SEARCHED, NOT ASSUMED.
   `mould.widest_section` locates the widest cross-section along a *given* axis; it
   cannot tell you which axis to part on, and for the block the hand driver simply
   knew. Here every axis is scored on the quantity that actually matters —
   RE-ENTRANT VOLUME, the material that has to scrape past the mould wall on the way
   out — with draw depth as the tie-break, because draft relief is proportional to
   draw depth. A prism ties on width along all three axes but not on draw depth, so
   the tie-break is what picks the short axis.

2. THE FLANGE OUTLINE IS THE FORM'S OWN SILHOUETTE, NOT A CIRCLE OR A RECTANGLE.
   `mould.ring_positions` / `ring_positions_rect` are the two cases someone wrote
   down, and choosing the wrong one is a silent failure: a circular ring sized for
   the block's long side put every key 54 mm outside the flange, where the
   intersection with the half deleted them, and every volume check still closed
   because a feature that was never there cannot unbalance anything. Here features
   are placed by RAY-CASTING the actual dilated silhouette (`project_to_outline`),
   so a superellipse, a rounded rectangle and a circle are all handled by the same
   code and a feature can never land in air.

3. CHIRALITY IS TESTED AGAINST THE OUTLINE'S MEASURED SYMMETRY GROUP.
   The existing pair of tests hard-codes two groups: the full O(2) for a circle,
   {identity, 180 deg, two mirrors} for a rectangle. Both are special cases of "the
   symmetries the flange actually has", which `detect_symmetries` measures off the
   silhouette mask. This matters because a pattern chiral on a circle can become
   mirror-symmetric once projected onto a rectangle — (0, 118, 242) deg did exactly
   that on the 390 x 190 block flange — and a mould that mates flipped casts a step
   at the parting line.

4. THE MOULD WALL IS SOLVED FROM THE DEFLECTION TARGET, NOT PICKED.
   `mould.wall_deflection` evaluates a thickness; the wall was then chosen by hand
   and the block's 20 mm came out marginal (0.117 mm against a 0.10 mm target).
   `auto_wall` inverts the Kirchhoff plate relation for the thinnest wall that meets
   the target on the measured span, so the marginal case cannot pass unnoticed.

5. AN INTEGRAL BOSS IS ADMITTED ONLY WHEN THE RELEASE CONDITION ALLOWS ONE.
   The kinematic condition for a fixed core inside a body withdrawn toward +z is
   B(z) <= min over z' <= z of R_hole(z'), and a running minimum from below is
   non-increasing. The vessel fails it (a 45 mm cavity behind a Ø32 bore) and needs
   loose cores; the block passes it and gets hourglass bosses. `decide_cores`
   evaluates the condition on the grid instead of relying on the caller to know
   which typology they have.

Nothing here re-implements the field machinery. `slice_sdf2d`'s half-voxel
correction, `cone_sweep`'s complement routing and uniform-slice reset, and
`assemble`'s attribution are imported from `mould.py`, so the traps documented there
stay fixed in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy import ndimage

from . import mould
from .grammars import sdf

BIG = mould.BIG


# --------------------------------------------------------------------------
# Automation inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AutoSpec:
    """Process and material inputs the automation needs. Lengths mm, angles deg.

    These are the quantities that cannot be measured off the geometry: the
    aggregate the mix is made from, the material the mould is made of, and the
    tolerance the caster is willing to accept.
    """

    # granular flow — sets the floor on aggregate passages and the ceiling on drains
    d_max: float = 4.0
    jam_mult: float = 6.0          # aggregate passages: clear >= jam_mult * d_max
    clog_mult: float = 3.0         # liquid drains: clear <= clog_mult * d_max

    # mould material and load (defaults: PETG, wet mix, tamped)
    E_GPa: float = 2.0
    nu: float = 0.4
    rho_mix: float = 1890.0
    compaction_factor: float = 10.0
    deflect_target_mm: float = 0.10
    wall_min: float = 12.0
    wall_max: float = 40.0
    wall_round: float = 1.0        # round the solved wall up to this increment

    # draft: 0.0 means "choose from the draw depth"
    draft_deg: float = 0.0
    draft_min_deg: float = 1.0
    draft_max_deg: float = 3.0
    draft_bias: str = "erode"      # "erode" preserves nominal AT the parting plane

    # flange and fasteners
    flange_frac: float = 0.28      # flange width as a fraction of the mould wall span
    flange_min: float = 20.0
    key_count: int = 3
    key_clear: float = 0.30
    bolt_count: int = 0            # 0 -> derived from the flange perimeter
    bolt_d: float = 8.5
    bolt_pitch_max: float = 140.0  # max spacing between clamp bolts

    # apertures
    gate_count: int = 0            # 0 -> derived from the parting-face perimeter
    drain_spacing: float = 90.0    # target drain spacing on the mould floor

    pitch: float = 0.0             # 0 -> grammar default


# --------------------------------------------------------------------------
# 1. Parting axis and plane
# --------------------------------------------------------------------------
def reentrant_volume(form: np.ndarray, pitch: float, axis: int, k_part: int) -> dict:
    """Volume that cannot leave a two-part cavity parted at slice `k_part`.

    A half releases by a straight pull only if its silhouette never widens as you
    move away from the parting plane. The undercut is measured as the material
    outside the running-maximum silhouette accumulated from the parting plane
    outward — which is the swept envelope the cavity must contain — so it is zero
    exactly when the pull is clean.
    """
    f = np.moveaxis(form, axis, -1)
    n = f.shape[-1]
    total = 0
    for rng in (range(k_part, n), range(k_part - 1, -1, -1)):
        env = None
        for k in rng:
            s = f[..., k]
            if env is None:
                env = s.copy()
                continue
            total += int((s & ~env).sum())   # material wider than everything inboard
            env |= s
    return {"reentrant_mm3": float(total * pitch ** 3),
            "reentrant_voxels": int(total)}


def analyse_parting(occ: np.ndarray, origin, pitch: float, *,
                    axes=(0, 1, 2)) -> dict:
    """Choose the parting axis and plane by measurement.

    Scored on re-entrant volume first (the material that would scrape), draw depth
    second (draft relief is proportional to it). Reported for every candidate so the
    choice is auditable rather than asserted.
    """
    form = mould.fill_slices(occ, axis=2)
    o = np.asarray(origin, float)
    cands = []
    for ax in axes:
        formax = mould.fill_slices(occ, axis=ax)
        ws = mould.widest_section(occ, origin, pitch, axis=ax)
        # `widest_section` returns the FIRST tied slice, which for a prism is an
        # extreme face: the block then reads a 182 mm draw (its whole height) and
        # the tile 33.6 mm, when parting mid-extent halves both. Within the tied
        # plateau every plane is equally wide, so the plane is moved to the tied
        # slice nearest mid-extent — the draw-depth tie-break, applied where it
        # belongs rather than only between axes.
        area = ws["area_profile"]
        nz = np.flatnonzero(area > 0)
        tied = np.flatnonzero(area >= 0.999 * area[ws["index"]])
        mid = 0.5 * (nz.min() + nz.max())
        k = int(tied[np.argmin(np.abs(tied - mid))])
        coord = float(o[ax] + k * pitch)
        re = reentrant_volume(formax, pitch, ax, k)
        extent = ws["extent_hi"] - ws["extent_lo"]
        draw = max(coord - ws["extent_lo"], ws["extent_hi"] - coord)
        cands.append({"axis": ax, "coord": coord, "index": k,
                      "area_mm2": ws["area_mm2"], "r_equiv_mm": ws["r_equiv_mm"],
                      "extent_mm": extent, "draw_depth_mm": draw,
                      "is_prismatic": ws["is_prismatic"],
                      "plateau_width_mm": ws["plateau_width_mm"],
                      **re, "widest": ws})
    # normalise the undercut so a hairline voxel artefact does not outvote a real
    # halving of the draw depth
    vol = float(form.sum() * pitch ** 3)
    for c in cands:
        c["reentrant_frac"] = c["reentrant_mm3"] / max(vol, 1e-9)
    # Third key breaks a genuine tie toward the highest axis: the block's W and H
    # axes tie on both undercut and draw, and parting across z is the convention the
    # hand-written driver used (and the one that puts the cores across the plane).
    best = min(cands, key=lambda c: (round(c["reentrant_frac"], 4),
                                     round(c["draw_depth_mm"], 3), -c["axis"]))
    return {"axis": best["axis"], "coord": best["coord"], "index": best["index"],
            "chosen": best, "candidates": cands, "form_mm3": vol,
            "reason": ("minimum re-entrant volume"
                       if len({round(c["reentrant_frac"], 4) for c in cands}) > 1
                       else "all axes release cleanly; minimum draw depth")}


# --------------------------------------------------------------------------
# 2. Flange outline and generic feature placement
# --------------------------------------------------------------------------
def outline_at(occ: np.ndarray, axis: int, k: int) -> np.ndarray:
    """Filled 2D silhouette of the form at slice k, normal to `axis`."""
    s = np.moveaxis(occ, axis, -1)[..., k]
    return ndimage.binary_fill_holes(s)


def outline_metrics(mask: np.ndarray, pitch: float, origin2) -> dict:
    """Centroid, spans and how close the outline is to its own bounding box."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        raise ValueError("empty parting-plane silhouette")
    o = np.asarray(origin2, float)
    pts = o + idx * pitch
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    area = float(mask.sum()) * pitch ** 2
    box = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    return {"centroid": tuple(pts.mean(axis=0)), "lo": tuple(lo), "hi": tuple(hi),
            "half_x": float((hi[0] - lo[0]) / 2), "half_y": float((hi[1] - lo[1]) / 2),
            "area_mm2": area, "box_fill_frac": float(area / max(box, 1e-9)),
            "r_equiv_mm": float(np.sqrt(area / np.pi)),
            "aspect": float((hi[0] - lo[0]) / max(hi[1] - lo[1], 1e-9))}


def dilate_outline(mask: np.ndarray, pitch: float, grow_mm: float) -> np.ndarray:
    """Grow a silhouette by a true distance (the flange), not by voxel steps."""
    if grow_mm <= 0:
        return mask.copy()
    d_out = ndimage.distance_transform_edt(~mask, sampling=pitch)
    return mask | (d_out <= grow_mm)


def project_to_outline(mask: np.ndarray, pitch: float, origin2, centre,
                       angles_deg, frac: float) -> list:
    """Place points at a fraction of the way to the outline edge, per direction.

    Generic replacement for the circular/rectangular ring pair: a ray is cast from
    the centroid and the last inside sample is found, so the point lands on material
    for ANY outline shape. `frac` < 1 keeps it inboard of the edge.
    """
    o = np.asarray(origin2, float)
    c = np.asarray(centre, float)
    ny, nx = mask.shape
    out = []
    for a in angles_deg:
        u = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        t_hit, t = 0.0, 0.0
        span = pitch * max(nx, ny) * 1.5
        while t <= span:
            p = c + u * t
            i, j = np.round((p - o) / pitch).astype(int)
            if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and mask[i, j]:
                t_hit = t
            t += 0.5 * pitch
        out.append(tuple(c + u * (t_hit * frac)))
    return out


def detect_symmetries(mask: np.ndarray, pitch: float, centre, origin2, *,
                      iou_min: float = 0.985) -> list:
    """Measure which rotations and mirrors map the outline onto itself.

    Replaces the hard-coded assumption of a symmetry group. Returned entries are
    ("rot", angle) or ("mirror", axis_angle); the identity is omitted.
    """
    o = np.asarray(origin2, float)
    c = np.asarray(centre, float)
    idx = np.argwhere(mask)
    pts = o + idx * pitch - c

    def occupied(q):
        ij = np.round((q + c - o) / pitch).astype(int)
        ok = ((ij[:, 0] >= 0) & (ij[:, 0] < mask.shape[0])
              & (ij[:, 1] >= 0) & (ij[:, 1] < mask.shape[1]))
        hit = np.zeros(len(q), bool)
        hit[ok] = mask[ij[ok, 0], ij[ok, 1]]
        return hit

    syms = []
    for k in (2, 3, 4, 6, 8):
        ang = 360.0 / k
        ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        q = pts @ np.array([[ca, -sa], [sa, ca]]).T
        if occupied(q).mean() >= iou_min:
            syms.append(("rot", ang))
    for ang in np.arange(0.0, 180.0, 5.0):
        t = np.deg2rad(2 * ang)
        M = np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])
        if occupied(pts @ M.T).mean() >= iou_min:
            syms.append(("mirror", float(ang)))
    return syms


def keys_chiral_auto(pts, syms, *, tol: float = 1.0) -> tuple:
    """True when no measured symmetry of the flange maps the key set onto itself.

    Generalises `mould.keys_are_chiral` (circle) and `keys_are_chiral_xy`
    (rectangle) to the group actually measured on the outline.
    """
    P = np.asarray(pts, float)
    if len(P) < 2:
        return False, "fewer than two keys"
    ctr = P.mean(axis=0)
    Q0 = P - ctr

    def maps_onto(Q):
        for q in Q:
            if np.min(np.linalg.norm(Q0 - q, axis=1)) > tol:
                return False
        return True

    for kind, ang in syms:
        if kind == "rot":
            for m in range(1, int(round(360.0 / ang))):
                t = np.deg2rad(ang * m)
                R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                if maps_onto(Q0 @ R.T):
                    return False, f"invariant under {ang * m:.0f} deg rotation"
        else:
            t = np.deg2rad(2 * ang)
            M = np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])
            if maps_onto(Q0 @ M.T):
                return False, f"mirror symmetric about {ang:.0f} deg"
    return True, "chiral"


def chiral_angles(n: int, syms, *, tries: int = 4000, seed: int = 0) -> list:
    """Search for a key layout that is chiral on THIS outline.

    A fixed literal tuple cannot be right for every outline — the block's
    (0, 118, 242) was chiral on a circle and mirror-symmetric on its own flange —
    so the angles are searched against the measured group and the first chiral set
    with a good minimum separation is taken.
    """
    rng = np.random.default_rng(seed)
    best, best_sep = None, -1.0
    for _ in range(tries):
        a = np.sort(rng.uniform(0, 360, n))
        pts = np.stack([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))], axis=1)
        ok, _ = keys_chiral_auto(pts, syms, tol=0.05)
        if not ok:
            continue
        gaps = np.diff(np.concatenate([a, a[:1] + 360.0]))
        if gaps.min() > best_sep:
            best, best_sep = list(a), gaps.min()
        if best_sep > 360.0 / n * 0.55:
            break
    if best is None:                       # fully symmetric outline: offset one key
        base = [i * 360.0 / n for i in range(n)]
        base[-1] += 360.0 / (3.0 * n)
        return base
    return best


# --------------------------------------------------------------------------
# 3. Mould wall, draft
# --------------------------------------------------------------------------
def auto_wall(span_mm: float, depth_mm: float, spec: AutoSpec) -> dict:
    """Thinnest mould wall meeting the deflection target on the measured span.

    Inverts the Kirchhoff plate relation `w = 0.00406 q a^4 / D` with
    `D = E t^3 / (12(1 - nu^2))`, rather than evaluating a guessed thickness.

    `span_mm` must be the SHORT side of the panel. The 0.00406 coefficient is the
    square-plate case, and for a rectangle the short span governs — a two-way plate
    carries load on its stiff direction. Feeding the long side of the block's
    390 x 190 face asks for a 55 mm wall against the ~26 mm the panel needs, which
    then clamps at wall_max and reports the target as unmet on a wall that is in fact
    adequate. Being wrong in the safe direction still costs: it prints a mould twice
    as heavy and reports a false failure.
    """
    pres = mould.mix_pressure(depth_mm, rho_kg_m3=spec.rho_mix,
                             compaction_factor=spec.compaction_factor)
    q = pres["p_design_kPa"] * 1e3
    E = spec.E_GPa * 1e9
    w = spec.deflect_target_mm * 1e-3
    a = span_mm * 1e-3
    D_req = 0.00406 * q * a ** 4 / max(w, 1e-12)
    t = (D_req * 12 * (1 - spec.nu ** 2) / E) ** (1.0 / 3.0) * 1e3
    t_r = float(np.ceil(t / spec.wall_round) * spec.wall_round)
    t_use = float(min(max(t_r, spec.wall_min), spec.wall_max))
    chk = mould.wall_deflection(pres["p_design_kPa"], span_mm, t_use,
                                E_GPa=spec.E_GPa, nu=spec.nu)
    return {"span_mm": span_mm, "t_required_mm": float(t), "t_mm": t_use,
            "clamped_at_max": bool(t_r > spec.wall_max),
            "deflection_mm": chk["w_center_mm"],
            "meets_target": bool(chk["w_center_mm"] <= spec.deflect_target_mm),
            **pres}


def auto_flange(key_base_d: float, spec: AutoSpec) -> dict:
    """Flange width from what it must HOUSE, not from the object's size.

    A flange scaled as a fraction of the span gives the block a 132 mm flange — more
    than triple the 40 mm its keys and bolts need, adding roughly half the block's
    own footprint again in printed material. The flange's job is to carry a key
    socket and a bolt side by side with wall between them, so that is what sets it:

        key diameter + bolt diameter + three ligaments

    with a floor from the spec. Independent of object scale, which is correct — an
    M8 bolt needs the same landing on a tile as on a block.
    """
    lig = max(6.0, 1.5 * spec.bolt_d)
    need = key_base_d + spec.bolt_d + 3.0 * lig
    w = float(max(spec.flange_min, need))
    return {"flange_mm": w, "required_mm": float(need), "ligament_mm": float(lig),
            "governed_by": "spec floor" if w > need else "key + bolt + ligaments"}


def auto_key_size(flange_mm: float, d_max: float) -> dict:
    """Registration key sized to the flange, above the FDM detail floor.

    Keys take the clamping shear, so they scale with the flange that carries them
    rather than being fixed. The base diameter is held above ~4 mm (where extruded
    detail degrades) and above the aggregate size, since a key smaller than a
    fragment cannot survive slurry between the faces.
    """
    base = float(max(12.0, 3.0 * d_max, min(0.42 * flange_mm, 26.0)))
    return {"key_base_d": base, "key_top_d": float(base * 0.68),
            "key_h": float(max(8.0, 0.5 * base))}


def auto_draft(draw_depth_mm: float, spec: AutoSpec) -> dict:
    """Draft angle from the draw depth, bounded by the spec.

    Relief costs `tan(theta) * draw`, so a deep draw wants the small end of the
    range and a shallow one can afford the large end. A fragile green body wants as
    much draft as the section budget will pay for, which is why the default upper
    bound is 3 deg rather than the 1 deg mould-release minimum.
    """
    if spec.draft_deg > 0:
        th = spec.draft_deg
        why = "explicit"
    else:
        # target ~2.5 mm of relief, bounded
        th = float(np.degrees(np.arctan(2.5 / max(draw_depth_mm, 1.0))))
        th = float(min(max(th, spec.draft_min_deg), spec.draft_max_deg))
        why = "2.5 mm relief target, bounded by draft_min/max"
    return {"draft_deg": th, "reason": why,
            "relief_per_face_mm": float(np.tan(np.deg2rad(th)) * draw_depth_mm),
            "draw_depth_mm": draw_depth_mm}


# --------------------------------------------------------------------------
# 4. Core strategy from the release condition
# --------------------------------------------------------------------------
def decide_cores(form: np.ndarray, obj: np.ndarray, axis: int, k_part: int,
                 pitch: float, *, tol_mm: float = 1.0) -> dict:
    """Integral boss or loose core, decided on the kinematic release condition.

    A boss is anchored at its half's OUTER face and reaches the parting plane, and
    the cast body lifts off it along the draw. So the boss must be non-increasing in
    width from the outer face toward the parting plane — equivalently, scanning from
    the parting plane OUTWARD the hollow must never narrow. A constriction on that
    path is the disqualifying feature: the vessel's Ø32 top bore sits outboard of a
    45 mm cavity, so a boss wide enough to form the cavity cannot pass back out
    through the bore, and a loose core is required. The block's cores only widen
    toward the outer faces, which is exactly the verified hourglass boss.

    Getting the sense of this backwards is not a benign error: it reports the
    block's working integral boss as needing loose cores, which prints two extra
    parts and leaves the cast cores unformed by the mould that was meant to make
    them.

    The *presence* of a constriction is the wrong test, though, because every
    drafted boss narrows: the sweep itself removes `tan(theta) * draw` on the way
    out, so a form whose hollow tapers by less than the draft was going to lose that
    material anyway. The block's 2 deg core taper produces an 8 mm constriction on a
    124 mm hollow and still releases cleanly (verified, 120 steps). What decides the
    strategy is therefore the VOLUME the boss cannot reach: the running-minimum
    envelope is built explicitly and its volume compared against the hollow it is
    meant to form. The vessel's Ø32 bore pins the envelope far inside a 45 mm cavity
    and leaves most of it unformed — a 32 mm wall where 26 mm was designed — while
    the block's envelope fills nearly all of its core. That is a measured fraction,
    not a threshold on a shape feature.
    """
    hollow = form & ~obj
    if not hollow.any():
        return {"needed": False, "strategy": "none", "reason": "form is solid"}
    h = np.moveaxis(hollow, axis, -1)
    n = h.shape[-1]
    out = {}
    for side, rng in (("lower", range(k_part - 1, -1, -1)),
                      ("upper", range(k_part, n))):
        ks = [k for k in rng if h[..., k].any()]
        if not ks:
            out[side] = {"hollow_mm3": 0.0, "boss_mm3": 0.0, "formed_frac": 1.0}
            continue
        # The admissible boss at slice k is the intersection of the hollow over every
        # slice from the outer face up to k — the running minimum as a SET, which is
        # what the release condition constrains, rather than a scalar width.
        env = np.ones_like(h[..., ks[0]], dtype=bool)
        boss_v = hollow_v = 0
        for k in reversed(ks):              # from the outer face inward
            env &= h[..., k]
            boss_v += int(env.sum())
            hollow_v += int(h[..., k].sum())
        out[side] = {"hollow_mm3": float(hollow_v * pitch ** 3),
                     "boss_mm3": float(boss_v * pitch ** 3),
                     "formed_frac": float(boss_v / max(hollow_v, 1))}
    formed = min(v["formed_frac"] for v in out.values())
    needs_loose = formed < 0.85
    return {"needed": True,
            "strategy": "loose_core" if needs_loose else "integral_boss",
            "formed_frac": float(formed),
            "reason": (f"a withdrawable boss reaches only {formed*100:.1f} % of the "
                       "designed hollow, so the rest would cast solid"
                       if needs_loose else
                       f"a withdrawable boss forms {formed*100:.1f} % of the hollow, "
                       "so an integral boss clears"),
            "per_half": out}


# --------------------------------------------------------------------------
# 5. Grammar dispatch
# --------------------------------------------------------------------------
def build_field(geom, pitch: float = 0.0):
    """Field for any grammar parameter object, with that grammar's own default pitch."""
    from .grammars import shell as g_shell, block as g_block, tile as g_tile
    from .params import ShellParams, BlockParams, TileParams
    table = {ShellParams: (g_shell, 1.25), BlockParams: (g_block, 2.0),
             TileParams: (g_tile, 1.2)}
    for cls, (mod, dflt) in table.items():
        if isinstance(geom, cls):
            p = pitch or dflt
            m, d, o, _ = mod.build(geom, pitch=p, return_field=True)
            return {"mesh": m, "field": d, "origin": o, "pitch": p,
                    "grammar": cls.__name__}
    raise TypeError(f"no grammar registered for {type(geom).__name__}")


# --------------------------------------------------------------------------
# 6. Frame handling — everything downstream of `analyse_parting` assumes z
# --------------------------------------------------------------------------
# `mould.assemble`, `cup_flange_block` and `cone_sweep`'s default all take the
# parting normal to be +z: `assemble` slices the block with `below[:, :, :k_part]`
# and builds the registration frusta with `frustum_field`, which is a z-axis
# primitive. `analyse_parting`, by contrast, is free to return axis 0 or 1 — and
# does, for any form whose shortest draw is not along z.
#
# Rather than teach four functions an axis argument (and risk one of them keeping
# the old default silently), the grids are PERMUTED into a z-normal frame, the
# whole mould is generated there, and the permutation is recorded so a caller can
# map back. The permutation is the cyclic one (axis -> z), whose determinant is
# +1, so handedness is preserved and an exported STL is not mirrored. All parts,
# `origin` and `obj_mesh` are returned in the SAME frame, so `mould.occ_to_mesh`
# output registers with `obj_mesh` without further work.
def frame_order(axis: int) -> tuple:
    """Cyclic permutation sending `axis` to position 2 (z). det = +1."""
    return ((axis + 1) % 3, (axis + 2) % 3, axis)


def to_z_frame(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.transpose(arr, frame_order(axis))


def mesh_to_z_frame(m, axis: int):
    """Same permutation applied to a mesh, so it registers with the grids."""
    order = list(frame_order(axis))
    out = m.copy()
    out.vertices = np.asarray(m.vertices, float)[:, order]
    import trimesh as _tm
    _tm.repair.fix_normals(out)
    return out


def _snap_lo(desired: float, ref: float, pitch: float) -> float:
    """Largest grid coordinate <= desired that lies on the source field's lattice.

    Keeps `resample_field` on integer indices, so the transfer of the object onto
    the (larger) mould grid is a pure shift rather than a trilinear blur, and the
    parting plane lands exactly on a node — otherwise `k_part` rounds and the
    plane moves by up to half a voxel, which is enough to put the flange band and
    the cavity's nominal section on different slices.
    """
    return float(ref - np.ceil((ref - desired) / pitch) * pitch)


# --------------------------------------------------------------------------
# 7. Measurement of the GENERATED mould (realised, not requested)
# --------------------------------------------------------------------------
def measure_draft(field: np.ndarray, pitch: float, k_part: int, side: str, *,
                  pct: float = 5.0, xy_min: float = 0.5,
                  cap_cut_deg: float = 75.0) -> dict:
    """Realised draft on one half, read off the SWEPT FIELD's own gradient.

    The requested angle is an input; what the mould has is an output, and the two
    differ wherever the form's own taper already exceeds theta (the sweep is then
    the identity).

    It must be measured on the field, not on the thresholded occupancy, and that is
    the same trap as `slice_sdf2d`'s half-voxel correction one level up. The
    per-slice step is `tan(theta) * pitch` — 0.052 mm at 1.49 deg and 2.0 mm pitch —
    so the whole 2.5 mm of relief across a 96 mm draw is barely one voxel, and the
    occupancy boundary does not move over any window short enough to be local.
    Measured on the block, an area/perimeter recession over a 12 mm window returned
    0.000 deg (median 0.000) on the lower half and 0.072 deg (median 0.993, min
    0.034) on the upper — i.e. it under-reports the 1.49 deg request by one to two
    orders of magnitude and is exactly zero on one side, which is indistinguishable
    from the sweep having silently degraded to the identity. The asymmetry is itself
    discretisation: the two halves quantise the same sub-voxel step differently.
    Meanwhile the block's auto cavity reproduces the hand-tuned baseline's slice-area
    profile to the voxel (38 096 -> 33 552 mm2), so the draft is present and it is
    the measure, not the mould, that was failing.

    `cone_sweep` returns the field before thresholding, and the recursion
    `G[k] = min(d[k], G[k-1] - tan(theta) * pitch)` writes the step into it exactly.
    Because `slice_sdf2d` restores |grad_xy| = 1, the local wall angle at the cavity
    surface is

        theta_local = arctan(|dG/dz| / |grad_xy G|)

    sampled on the |G| <= pitch band. `xy_min` drops samples where the in-plane
    gradient collapses — the cup floor and the poles, where the surface closes off
    instead of sliding and draft is not defined; this is also the guard against the
    axis blow-up that makes a normalised 3D field unusable in the interior. Reported
    at the `pct` percentile, because the tight spot is what governs release.
    """
    g = np.asarray(field, float)
    d0, d1, d2 = np.gradient(g, pitch)
    gxy = np.sqrt(d0 ** 2 + d1 ** 2)
    ang = np.degrees(np.arctan2(np.abs(d2), np.maximum(gxy, 1e-12)))
    nz = g.shape[2]
    zs = (slice(1, k_part) if side == "lower" else slice(k_part, nz - 1))
    band = (np.abs(g) <= pitch) & (gxy >= xy_min)
    m = np.zeros_like(band)
    m[:, :, zs] = band[:, :, zs]
    m &= ang <= cap_cut_deg
    if not m.any():
        return {"draft_deg": float("nan"), "n": 0}
    v = ang[m]
    return {"draft_deg": float(np.percentile(v, pct)),
            "draft_median_deg": float(np.median(v)),
            "draft_p50_p95_deg": (float(np.percentile(v, 50)),
                                  float(np.percentile(v, 95))),
            "n": int(v.size)}


def measure_wall(cavity: np.ndarray, block_occ: np.ndarray, pitch: float) -> dict:
    """Realised mould wall: cavity surface to the outside of the mould block.

    Measured on the envelope BEFORE the halves are cut, so bolt and drain bores do
    not masquerade as thin walls — they are holes by design, and the question here
    is whether the wall between the mix and the atmosphere is the thickness the
    deflection calculation assumed.
    """
    d_out = ndimage.distance_transform_edt(block_occ, sampling=pitch)
    surf = ndimage.binary_dilation(cavity) & ~cavity & block_occ
    if not surf.any():
        return {"min_mm": float("nan"), "median_mm": float("nan"), "n": 0}
    v = d_out[surf]
    return {"min_mm": float(v.min()), "median_mm": float(np.median(v)),
            "p5_mm": float(np.percentile(v, 5)), "n": int(v.size)}


def solid_runs(occ: np.ndarray, k: int, pitch: float, *, axis: int,
               through: int | None = None) -> dict:
    """Solid run lengths along one in-plane axis of slice `k`.

    The section measure the masonry standards actually use: face shell and web are
    run lengths across the unit, not medial-axis thicknesses. Taking them on the
    same grid for the nominal body and the as-cast cavity is what separates the
    draft cost from geometry the design already had.
    """
    s = occ[..., k]
    line = s[:, through] if axis == 0 else s[through, :]
    runs, n = [], 0
    for val in np.append(line, False):
        if val:
            n += 1
        elif n:
            runs.append(n * pitch)
            n = 0
    return {"runs_mm": [float(r) for r in runs],
            "min_mm": float(min(runs)) if runs else 0.0,
            "max_mm": float(max(runs)) if runs else 0.0,
            "n_runs": len(runs)}


def outline_perimeter(mask: np.ndarray, pitch: float) -> float:
    """Perimeter of a voxelised outline by marching squares, not voxel counting.

    Counting boundary voxels overestimates a curved perimeter by up to 4/pi (27 %)
    and it is the perimeter that sets the bolt and gate COUNTS, so the error would
    propagate straight into the number of holes drilled through the flange.
    """
    from skimage import measure as _measure
    total = 0.0
    for c in _measure.find_contours(mask.astype(float), 0.5):
        total += float(np.abs(np.diff(c, axis=0)).__pow__(2).sum(axis=1).__pow__(0.5).sum())
    return total * pitch


# --------------------------------------------------------------------------
# 8. The automatic rigid mould
# --------------------------------------------------------------------------
def build_auto_mould(geom, spec: AutoSpec | None = None, *,
                     mould_kind: str = "rigid") -> dict:
    """Generate a split rigid (FDM) mould for ANY grammar, with no hand tuning.

    Same return shape as `mould.build_shell_mould` — occupancy grids per part plus
    `origin`, `pitch`, `balance`, `obj_mesh` — so `mould.occ_to_mesh` and existing
    callers keep working. Everything that was a literal in the two hand-written
    drivers is now measured off the field:

    parting axis/plane   `analyse_parting`  (re-entrant volume, draw-depth tie-break)
    mould wall           `auto_wall` on the SHORT span of the parting outline
    flange               `auto_flange` (what it must house, not the object's size)
    keys                 `auto_key_size` + `chiral_angles` against `detect_symmetries`
    draft                `auto_draft` from the draw depth
    envelope             the parting silhouette dilated by (wall + flange)
    feature counts       measured perimeter / area, not literals
    cores                `decide_cores` on the kinematic release condition

    Two conventions are inherited from `mould.py` deliberately.

    ERODE AWAY FROM THE PARTING PLANE. Both halves are drafted by eroding the
    solid as you move away from the plane, which is identically "the cavity widens
    toward the opening" and therefore releases, while holding the NOMINAL section
    at the parting plane and putting the relief on the outer faces. The
    alternative (dilate away) also releases but oversizes the cast body away from
    the plane, and for a masonry unit that is a dimensional-tolerance failure
    rather than a section loss.

    THE CAVITY IS CUT FROM THE FORM WHEN CORES ARE LOOSE, FROM THE OBJECT WHEN
    THEY ARE NOT. This is not a detail. Eroding the OBJECT of a hollow vessel
    moves its outer surface in and its inner surface out at once, so the wall
    thins by TWICE the relief — 8.6 mm on a 26 mm wall at the vessel's 98 mm draw,
    which lands below the 6 x d_max = 24 mm jamming floor and would break the very
    design the baseline validated. When `decide_cores` returns `loose_core` the
    halves are cut from the slice-filled FORM and the cavity is formed by separate
    cores eroded from the hollow, so both surfaces move the same way and the wall
    is preserved.
    """
    if mould_kind != "rigid":
        raise NotImplementedError(
            f"mould_kind={mould_kind!r}: only the rigid (FDM) path is implemented "
            "here; an elastomer skin needs its own release model, and its mould "
            "face is a no-flux boundary either way")
    spec = spec or AutoSpec()
    rec: dict = {"spec": spec, "mould_kind": mould_kind}

    # ---- 1. field, occupancy, parting -----------------------------------
    fb = build_field(geom, spec.pitch)
    pitch = fb["pitch"]
    d_obj, o_obj, obj_mesh0 = fb["field"], np.asarray(fb["origin"], float), fb["mesh"]
    occ0 = d_obj <= 0.0
    part = analyse_parting(occ0, o_obj, pitch)
    axis = part["axis"]
    rec["grammar"] = fb["grammar"]
    rec["parting_analysis"] = part

    # ---- 2. into a z-normal frame ---------------------------------------
    order = frame_order(axis)
    d_z = to_z_frame(d_obj, axis)
    o_z = o_obj[list(order)]
    parting = part["coord"]                      # coordinate along `axis` == new z
    occ_z = d_z <= 0.0
    form_z = mould.fill_slices(occ_z, axis=2)
    k0 = int(round((parting - o_z[2]) / pitch))

    # ---- 3. outline measurements at the parting plane -------------------
    sil = ndimage.binary_fill_holes(form_z[..., k0])
    o2 = o_z[:2]
    met = outline_metrics(sil, pitch, o2)
    centre = met["centroid"]
    span_short = 2.0 * min(met["half_x"], met["half_y"])
    span_long = 2.0 * max(met["half_x"], met["half_y"])
    z_lo = float(o_z[2] + np.flatnonzero(occ_z.any(axis=(0, 1))).min() * pitch)
    z_hi = float(o_z[2] + np.flatnonzero(occ_z.any(axis=(0, 1))).max() * pitch)
    draw = max(parting - z_lo, z_hi - parting)

    # ---- 4. sizing: wall, flange, keys, draft ---------------------------
    wl = auto_wall(span_short, draw, spec)
    # Single pass, deliberately. `auto_key_size`'s `0.42 * flange` term is a CEILING
    # (the key must fit the flange), not a driver: feeding it back into
    # `auto_flange` — which sizes the flange as key + bolt + three ligaments — makes
    # the pair diverge to a fixed point at flange 72.8 / key 26.0 mm, i.e. a flange
    # a third wider than anything the fasteners need. So the key is sized from the
    # spec floor and the flange is sized to house THAT key; the ceiling then holds
    # by construction and is checked below.
    ky = auto_key_size(spec.flange_min, spec.d_max)
    fl = auto_flange(ky["key_base_d"], spec)
    dr = auto_draft(draw, spec)
    wall, flange = wl["t_mm"], fl["flange_mm"]
    lig = fl["ligament_mm"]
    flange_t = float(max(ky["key_h"] + lig,
                         spec.jam_mult * spec.d_max / 2 + spec.d_max / 2 + lig))
    rec.update(wall=wl, flange=fl, key=ky, draft=dr, outline=met,
               span_short_mm=span_short, span_long_mm=span_long,
               draw_depth_mm=draw, flange_t_mm=flange_t)

    # ---- 5. mould grid, aligned to the object's own lattice -------------
    pad = wall + flange + 4 * pitch
    lo = np.array([_snap_lo(met["lo"][0] - pad, o_z[0], pitch),
                   _snap_lo(met["lo"][1] - pad, o_z[1], pitch),
                   _snap_lo(z_lo - wall - 2 * pitch, o_z[2], pitch)])
    hi = np.array([met["hi"][0] + pad, met["hi"][1] + pad, z_hi + wall + 2 * pitch])
    X, Y, Z, origin, sh = sdf.make_grid(lo, hi, pitch)
    obj = mould.resample_field(d_z, o_z, pitch, X, Y, Z) <= 0.0
    form = mould.resample_field(
        sdf.signed_distance_from_binary(form_z, pitch), o_z, pitch, X, Y, Z) <= 0.0
    k_part = int(round((parting - origin[2]) / pitch))
    below = np.zeros(sh, bool)
    below[:, :, :k_part] = True

    # ---- 6. conformal envelope: silhouette dilated by (wall + flange) ---
    # `cup_flange_block` offers a cylinder or a box. A superellipse is neither, and
    # forcing it into either one either buries the form in surplus material or
    # clips it. Dilating the measured silhouette by a true distance gives a
    # conformal cup for any outline, at the cost of being a voxel construction
    # rather than an analytic field — which is fine, because everything downstream
    # of here is occupancy anyway.
    sil_g = ndimage.binary_fill_holes(form[..., k_part])
    env_cup = dilate_outline(sil_g, pitch, wall)
    env_fl = dilate_outline(sil_g, pitch, wall + flange)
    z_cup_lo, z_cup_hi = z_lo - wall, z_hi + wall
    block_occ = ((env_cup[:, :, None] & (np.abs(Z - (z_cup_hi + z_cup_lo) / 2)
                                        <= (z_cup_hi - z_cup_lo) / 2))
                 | (env_fl[:, :, None] & (np.abs(Z - parting) <= flange_t)))
    rec["envelope"] = {"cup_area_mm2": float(env_cup.sum()) * pitch ** 2,
                       "flange_area_mm2": float(env_fl.sum()) * pitch ** 2,
                       "flange_perimeter_mm": outline_perimeter(env_fl, pitch),
                       "z_lo": z_cup_lo, "z_hi": z_cup_hi}

    # ---- 7. core strategy, then the cavity source ----------------------
    cores = decide_cores(form, obj, 2, k_part, pitch)
    rec["cores"] = cores
    src = form if cores["strategy"] == "loose_core" else obj

    sgn_up = (spec.draft_bias == "erode")
    mode = "erode" if spec.draft_bias == "erode" else "dilate"
    # keep the FIELDS: the realised draft is sub-voxel and is only recoverable from
    # the swept field, never from the thresholded occupancy (see `measure_draft`)
    g_lo = mould.cone_sweep(src & below, pitch, axis=2, up=not sgn_up,
                            mode=mode, draft_deg=dr["draft_deg"])
    g_up = mould.cone_sweep(src & ~below, pitch, axis=2, up=sgn_up,
                            mode=mode, draft_deg=dr["draft_deg"])
    cav_lo, cav_up = g_lo <= 0, g_up <= 0
    cavity = (cav_lo & below) | (cav_up & ~below)

    # ---- 8. features, counted from measurements ------------------------
    per_fl = rec["envelope"]["flange_perimeter_mm"]
    gate_clear = spec.jam_mult * spec.d_max + 0.5 * spec.d_max   # floor + half a grain
    drain_d = float(max(4.0, 1.5 * spec.d_max))                  # mid-retention band
    # gates: spaced so at least ten gate-widths of intact parting-line seal remain
    n_gate = int(spec.gate_count or max(2, int(per_fl // (10.0 * gate_clear))))
    # drains: a per-unit-AREA duty on the mould floor, not a perimeter one
    n_drain = int(max(4, round(met["area_mm2"] / spec.drain_spacing ** 2)))
    n_bolt = int(spec.bolt_count or max(4, int(np.ceil(per_fl / spec.bolt_pitch_max))))

    # Every mask from here on lives on the MOULD grid, whose origin is `lo`, not the
    # object field's `o_z`. Ray-casting a mould-grid mask with the object's origin is
    # a silent failure of exactly the kind `project_to_outline` exists to prevent:
    # the rays start from a point offset by (lo - o_z), the "last inside sample"
    # comes back near the grid centre, and the keys land partly in air where
    # `assemble`'s intersection deletes them — measured 1462.9 of 1938.4 mm3
    # nominal, with every volume check still closing. The centroid itself is a WORLD
    # point and so is grid-independent; only the origin has to be the mask's own.
    og = origin[:2]
    syms = detect_symmetries(env_fl, pitch, centre, og)
    key_ang = chiral_angles(spec.key_count, syms)
    rec["symmetries"] = syms

    def _band_pts(angles, offset):
        """Place at a true distance outboard of the cup edge, per direction."""
        p_cup = project_to_outline(env_cup, pitch, og, centre, angles, 1.0)
        out = []
        for a, pc in zip(angles, p_cup):
            u = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
            t_cup = float(np.linalg.norm(np.asarray(pc) - np.asarray(centre)))
            out.append(tuple(np.asarray(centre) + u * (t_cup + offset)))
        return out

    key_xy = _band_pts(key_ang, lig + ky["key_base_d"] / 2)
    bolt_ang = [(i + 0.5) * 360.0 / n_bolt for i in range(n_bolt)]
    bolt_xy = _band_pts(bolt_ang, flange - lig - spec.bolt_d / 2)
    chiral = keys_chiral_auto([(np.asarray(p) - np.asarray(centre)) for p in key_xy],
                              syms)
    # Drains must open into the CAST BODY's floor, so they are placed on the object's
    # own footprint at the parting plane (not the silhouette, which includes the
    # cores) and by maximum edge clearance, not at a fixed fraction of the radius —
    # see `place_by_clearance` for the two ways the fractional placement failed.
    foot = obj[..., k_part]
    dp = place_by_clearance(foot, pitch, og, centre, n_drain,
                            need_clear_mm=drain_d / 2 + 2.0 * pitch,
                            min_sep_mm=0.6 * spec.drain_spacing)
    drain_xy = dp["points"]
    rec["drain_placement"] = dp
    n_drain = len(drain_xy)
    gate_ang = [(i + 0.25) * 360.0 / n_gate for i in range(n_gate)]
    reach = max(np.linalg.norm(np.asarray(p) - np.asarray(centre))
                for p in project_to_outline(env_fl, pitch, og, centre,
                                            np.arange(0, 360, 5.0), 1.0)) + 4 * pitch

    def _gate(ang):
        ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        cx, cy = centre

        def f(Xa, Ya, Za):
            u = (Xa - cx) * ca + (Ya - cy) * sa
            w = -(Xa - cx) * sa + (Ya - cy) * ca
            return np.maximum.reduce([
                np.sqrt(w ** 2 + (Za - parting) ** 2) - gate_clear / 2,
                -u, u - reach])
        return f

    def _drain(pt):
        return lambda Xa, Ya, Za: sdf.sd_cylinder_z(
            Xa, Ya, Za, drain_d / 2, 1e4, center=(pt[0], pt[1], 0.0))

    mp = mould.MouldParams(
        wall=wall, draft_deg=dr["draft_deg"], flange=flange, flange_t=flange_t,
        key_count=spec.key_count, key_base_d=ky["key_base_d"],
        key_top_d=ky["key_top_d"], key_h=ky["key_h"], key_clear=spec.key_clear,
        key_angles_deg=tuple(key_ang), bolt_d=spec.bolt_d, bolt_count=n_bolt,
        drain_d=drain_d, drain_count=n_drain)

    A = mould.assemble(obj_occ=obj, cavity_occ=cavity, block_occ=block_occ,
                       X=X, Y=Y, Z=Z, origin=origin, pitch=pitch, parting=parting,
                       mp=mp, r_ref=met["r_equiv_mm"] + wall,
                       gate_specs=[_gate(a) for a in gate_ang],
                       drain_specs=[_drain(p) for p in drain_xy],
                       square=False, key_xy=key_xy, bolt_xy=bolt_xy)

    # ---- 9. cores, honouring the decision -----------------------------
    parts = ["lower", "upper"]
    hollow = form & ~obj
    if cores["strategy"] == "loose_core":
        A["core_lo"] = (mould.cone_sweep(hollow & below, pitch, axis=2,
                                         up=False, mode="erode",
                                         draft_deg=dr["draft_deg"]) <= 0) & below
        A["core_up"] = (mould.cone_sweep(hollow & ~below, pitch, axis=2,
                                         up=True, mode="erode",
                                         draft_deg=dr["draft_deg"]) <= 0) & ~below
        parts += ["core_lo", "core_up"]
        A["as_cast"] = A["cavity"] & ~(A["core_lo"] | A["core_up"])
    else:
        # integral bosses: they are already part of `lower`/`upper` because the
        # cavity was cut from the OBJECT, so the hollow is mould material
        A["as_cast"] = A["cavity"]

    A.update(obj=obj, form=form, block=block_occ, cavity_full=cavity, hollow=hollow,
             cav_field_lo=g_lo, cav_field_up=g_up,
             below=below, origin=origin, pitch=pitch, parting=parting,
             obj_mesh=mesh_to_z_frame(obj_mesh0, axis), mp=mp,
             axis=axis, frame_order=order, k_part=k_part, part_names=parts,
             gate_clear_mm=gate_clear, drain_d_mm=drain_d,
             drain_xy=drain_xy, gate_angles_deg=gate_ang, key_angles_deg=key_ang,
             n_gate=n_gate, n_drain=n_drain, n_bolt=n_bolt,
             keys_chiral_auto=chiral, decisions=rec)
    return A


def count_through_holes(A: dict) -> dict:
    """Count the through-holes of each mould half by COMPONENT, not by intent.

    The genus check is only worth running if the expected number is counted off the
    generated geometry rather than copied from the requested feature counts: a
    feature that landed in air, merged with its neighbour, or failed to break
    through would then be invisible, because the same literal would appear on both
    sides of the comparison. So each feature mask is labelled and a component is
    counted only when it reaches BOTH the cavity and the outside of the mould
    block — i.e. when it is genuinely a tunnel.

    Which features raise the genus and which do not follows from the halves being
    open CUPS, not balls:

    bolt   the flange is closed on both faces, so a bolt hole is a tunnel: +1 each.
    drain  the cavity is a dent in the cup, and a drain turns dent + channel into a
           single tunnel from the parting face to the outer floor: +1 in the half it
           is cut into (`assemble` cuts drains into the LOWER half only).
    gate   cut at the parting plane and open along its whole length to the parting
           FACE, so it is a surface groove, not a handle: +0.

    A component is attributed to a half by an ENCLOSURE test — is its cross-section
    a hole in that half's own material, in some slice — not by adjacency. Adjacency
    is what a one-voxel dilation tests, and it double-counts: `L0` and `U0` are
    disjoint but share a face, so dilating a lower-half drain reaches upper-half
    material and the drain is counted in both halves. Measured on the block that
    inflated the expected upper genus from 12 to 17 and produced a spurious Euler
    mismatch (-22 measured against -32 "expected") on a mould that was in fact
    correct — the check would have condemned good geometry.
    """
    lower, upper, block = A["lower"], A["upper"], A["block"]
    out = {}
    for nm, feat in (("bolts", A["bolts"]), ("drains", A["drains"])):
        lab, n = ndimage.label(feat & block)
        cnt = {"lower": 0, "upper": 0}
        for i in range(1, n + 1):
            comp = lab == i
            ks = np.flatnonzero(comp.any(axis=(0, 1)))
            for side, part in (("lower", lower), ("upper", upper)):
                enclosed = False
                for k in ks:
                    cs = comp[..., k]
                    if not cs.any():
                        continue
                    ps = part[..., k]
                    holes = ndimage.binary_fill_holes(ps) & ~ps
                    if (cs & ~holes).sum() == 0:      # entirely a hole in this half
                        enclosed = True
                        break
                cnt[side] += int(enclosed)
        out[nm] = cnt
    out["gates_counted"] = int(ndimage.label(A["gates"] & block)[1])
    out["expected_genus"] = {
        "lower": out["bolts"]["lower"] + out["drains"]["lower"],
        "upper": out["bolts"]["upper"] + out["drains"]["upper"]}
    return out


def auto_apertures(A: dict, spec: AutoSpec) -> list:
    """Inventory every passage of the generated mould, each with its own class.

    The classes carry OPPOSITE tests (`mould.check_apertures`), so the inventory is
    where a sizing error becomes visible: an aggregate passage below 6 x d_max
    starves, and a "drain" above 3 x d_max is a hole the mix falls out of.

    The narrowest cast section is included as an aggregate aperture because that is
    the passage the mix must actually fill — for the block it is the drafted web,
    not the feed gate, and it is measured on the as-cast body rather than taken from
    the nominal parameters.
    """
    from .physics.section import min_section
    pitch = A["pitch"]
    k = A["k_part"]
    # feed opening at the parting plane: largest inscribed circle of the as-cast
    # outline there — the bridging aperture the mix has to pass on the way in
    sl = A["as_cast"][..., k]
    feed = 2.0 * float(ndimage.distance_transform_edt(sl, sampling=pitch).max()) \
        if sl.any() else 0.0
    sec = min_section(A["as_cast"], pitch)
    aps = [
        mould.Aperture("feed/vent gate", "aggregate", A["gate_clear_mm"],
                       f"x{A['n_gate']}, radial at the parting plane"),
        mould.Aperture("open parting face (inscribed)", "aggregate", feed,
                       "largest inscribed circle of the as-cast parting section"),
        mould.Aperture("narrowest as-cast section", "aggregate",
                       sec["min_section_p5_mm"],
                       "p5 of local thickness on the medial ridge, measured "
                       "post-draft"),
        mould.Aperture("liquid drain", "liquid", A["drain_d_mm"],
                       f"x{A['n_drain']}, bridges by design to retain aggregate"),
    ]
    if "core_lo" in A:
        for nm in ("core_lo", "core_up"):
            c = A[nm]
            if not c.any():
                continue
            ks = np.flatnonzero(c.any(axis=(0, 1)))
            kk = ks.max() if nm == "core_lo" else ks.min()   # at the parting plane
            bore = 2.0 * float(ndimage.distance_transform_edt(
                c[..., kk], sampling=pitch).max())
            aps.append(mould.Aperture(f"{nm} bore at parting plane", "aggregate",
                                      bore, "cast by the loose core"))
    return aps


def place_by_clearance(mask: np.ndarray, pitch: float, origin2, centre, n: int, *,
                       need_clear_mm: float, min_sep_mm: float = 0.0,
                       sectors_from_deg: float = 15.0) -> dict:
    """Place `n` points on `mask`, one per angular sector, at maximum edge clearance.

    `project_to_outline` is the right tool for a feature that belongs on the OUTLINE
    (keys, bolts, gates): it ray-casts the real silhouette so the point cannot land
    in air. It is the wrong tool for a feature that must land in the INTERIOR of a
    possibly non-convex footprint, and that failure is silent in a different way
    than the one it was written to prevent.

    Measured on the block: drains placed at 0.55 of the way to the outline edge put
    8 of 9 holes in the CORE VOIDS. Those are mould material (the integral bosses),
    not cast body, so each "drain" ran from the outer floor up through a boss and
    opened at the boss's top face — communicating with the atmosphere above the
    parting plane rather than with the cast body it was supposed to drain, and
    casting a peg of mix into the hole in the process. Every volume check still
    closed, because the void was correctly attributed to `drains`; the balance
    cannot tell a hole that drains from a hole that does not. On the vessel the same
    placement left only 3.8 mm of edge clearance for a 6.0 mm drain, i.e. the bore
    broke the footprint edge.

    Here the footprint's own distance-to-edge field is maximised within each angular
    sector, which puts every drain on material with a recorded clearance, and any
    sector that cannot host one is REPORTED rather than silently dropped.

    `min_sep_mm` is required, not optional garnish. Unconstrained maximisation of a
    distance-to-edge field COLLAPSES on a convex footprint: the maximum is the
    incentre, and it is the maximum in every angular sector at once, so all n points
    land within a voxel of the same place. Measured on the tile — a solid 200 x 200
    footprint — five requested drains merged into ONE hole, and the failure was
    visible only because the topology count disagreed with the requested number (1
    counted against 5 placed). Enforcing a separation greedily, best clearance
    first, keeps the drains distributed and preserves the per-unit-area duty the
    count was derived from.
    """
    d_in = ndimage.distance_transform_edt(mask, sampling=pitch)
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return {"points": [], "clearance_mm": [], "skipped_sectors": list(range(n)),
                "n_requested": n}
    o = np.asarray(origin2, float)
    c = np.asarray(centre, float)
    pts_w = o + idx * pitch
    ang = np.degrees(np.arctan2(pts_w[:, 1] - c[1], pts_w[:, 0] - c[0])) % 360.0
    clear = d_in[idx[:, 0], idx[:, 1]]
    out, cl, skipped = [], [], []
    width = 360.0 / n
    for s in range(n):
        a0 = (sectors_from_deg + s * width) % 360.0
        sel = np.flatnonzero((((ang - a0) % 360.0) < width)
                             & (clear >= need_clear_mm))
        placed = False
        for j in sel[np.argsort(-clear[sel])]:          # best clearance first
            p = pts_w[j, :2]
            if out and min(float(np.linalg.norm(p - np.asarray(q)))
                           for q in out) < min_sep_mm:
                continue
            out.append((float(p[0]), float(p[1])))
            cl.append(float(clear[j]))
            placed = True
            break
        if not placed:
            skipped.append(s)
    return {"points": out, "clearance_mm": cl, "skipped_sectors": skipped,
            "n_requested": n, "n_placed": len(out),
            "min_clearance_mm": float(min(cl)) if cl else float("nan")}
