"""Elastomeric (silicone) mould generator: conformal skin + rigid mother mould.

This is the second automatic mould path. `mould_auto` generates a RIGID two-part
negative; this module generates a compliant skin against the cast surface and a
rigid two-part jacket against the skin's outer face. All of the field machinery —
`slice_sdf2d`'s half-voxel correction, `cone_sweep`'s complement routing and
uniform-slice reset, `release_sweep`, `check_apertures`, `occ_to_mesh` — is
imported from `mould.py`, and the parting search, wall solve, flange sizing and
core decision come from `mould_auto.py`. Nothing is re-implemented here.

Four things genuinely differ from the rigid path, and every one of them is a
consequence of mechanism rather than of preference.

1. THE CAST OBJECT NEEDS NO DRAFT, SO THE DRAFT BUDGET COMES BACK.
   A rigid cavity releases by a straight pull, so it must be swept open at the
   draft angle and the cast section loses `tan(theta) * draw` per face. On the
   block that is 1.5 deg over a 95 mm draw = 2.49 mm/face, which is exactly why
   the verified rigid design carries a 30 mm nominal web where the standard
   minimum is 25 mm: 25 + 2 x 2.49 = 29.98. A skin releases by STRETCHING, so
   the sweep is not needed on the cast surface at all and that 4.98 mm returns
   to the designer. `section_budget` computes the return per typology from the
   realised draft of the rigid path rather than quoting the nominal angle.

   What the elastomer does NOT buy is breathability — see 4.

2. THE SKIN IS A CONFORMAL OFFSET OF THE WHOLE CAST SURFACE, INCLUDING CORES.
   `skin_sets` offsets the object's own surface outward by `skin_t` on a TRUE
   distance field, so the layer lands on every cast face: the outside, the walls
   of a hollow core, the inside of a vessel cavity. Offsetting the FILLED
   silhouette instead would skin only the outside and leave the core walls to be
   formed by rigid boss material pressed straight against the green body, which
   both re-imposes the draft the skin was supposed to remove and puts a no-flux
   rigid face where the geometry most needs air.

   The offset is taken on a corrected field. `sdf.signed_distance_from_binary`
   returns `outside - inside` from the two-sided EDT, and both branches measure
   to the nearest voxel CENTRE of the opposite class, so every magnitude is half
   a voxel too large; thresholding it at `skin_t` delivers `skin_t - pitch/2`.
   `true_sdf` applies the half-voxel shift, and the realised thickness is then
   MEASURED back off the generated occupancy rather than reported as requested.
   (The related but distinct |grad| = 2 trap bites level-set recursions whose
   per-slice step is sub-voxel; that one is handled inside `mould.slice_sdf2d`
   and is why the jacket draft goes through `mould.cone_sweep`.)

   Thickness is a real trade, not a preference: thicker is stiffer, more durable
   and more expensive, and a worse vapour barrier. 6 mm is the default because
   it is about the thinnest section that a two-part RTV pour fills reliably
   around a detailed pattern without tearing on demould, and because the barrier
   penalty (4) is already saturated at 6 mm — going thinner does not buy
   breathability back, so there is nothing to trade against durability.

3. THE JACKET TAKES THE DRAFT AND THE PRESSURE. The skin's outer face is smooth
   (it has already absorbed the object's detail), so the jacket cavity is swept
   from THAT surface, not from the cast. The skin is ~1.1 MPa (Shore 30A) against
   ~2000 MPa for the printed jacket, a factor of ~1800, so the skin transmits the
   mix pressure rather than carrying it: the jacket is sized on the deflection
   target with `mould_auto.auto_wall` and the target still governs, because a
   bulging jacket lets the skin bulge with it and the cast wall thickness is the
   parameter the drying and oxygen models are most sensitive to. Perforating the
   jacket for breather windows removes bending stiffness, so the solved wall is
   scaled by (1 - f_open)^(-1/3) — the thickness that restores D = E t^3/(12(1-nu^2)).

   Release ORDER is not a convention, it is kinematics: the jacket must come off
   FIRST. Inside the jacket the skin has nowhere to go — the jacket cavity is
   generated against the skin's own outer face, so the radial clearance available
   for the skin to stretch over anything is the draft relief and nothing more.
   `verify_release` reports the jacket-first sweep, the skin's straight-pull
   sweep, and the confinement clearance that makes skin-first impossible.

4. A SILICONE FACE IS A NO-FLUX BOUNDARY, SO OPEN AREA MUST BE GENERATED.
   PDMS is famously oxygen-permeable (350-800 Barrer), which invites the
   conclusion that a silicone mould breathes. It does not, because what MICP needs
   is DRAINED PORES and the comparison that decides cementation is against air,
   not against water. `barrier_diagnostics` computes both sides: the drained-pore
   equivalent permeability D_eff/(RT) is ~2.6e-10 mol m/(m2 s Pa) against
   ~1.7e-13 for silicone at 520 Barrer, so a 6 mm skin carries a few hundred times
   the diffusive resistance of the 26 mm drained wall behind it. Against SATURATED
   pores the same skin is transparent (~1e-3 of their resistance), which is
   precisely why the intuition misleads. The vapour side compounds it: silicone is
   the most vapour-permeable common elastomer and a 6 mm section still passes only
   ~1 % of the free evaporation rate, which collapses the air-entry depth from
   ~21 mm to a fraction of a millimetre — and since oxygen only travels far
   through drained pores, throttling drying throttles aeration too.

   So the skin and jacket are perforated with an ALIGNED breather lattice, and it
   is sized on the aeration requirement rather than by eye (`size_windows`). A
   breather is not a passage the mix flows through, so the 6 x d_max jamming floor
   does not apply to it; the correct test is the INVERSE one used for the rigid
   path's drains — a window below 3 x d_max always clogs (Vani 2022), so bridging
   IS the damming mechanism and the window is self-sealing against aggregate while
   passing gas and spent cementation solution freely. The one aperture the mix
   really does flow through is the fill gate, and that is checked against the
   6 x d_max floor.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import numpy as np
import trimesh
from scipy import ndimage

from . import mould, mould_auto
from .mould_auto import AutoSpec
from .grammars import sdf
from .physics import oxygen as ox, drying as dry, fields as pf

BIG = mould.BIG

# 1 Barrer = 1e-10 cm3(STP) cm / (cm2 s cmHg) -> mol m / (m2 s Pa)
BARRER = 1e-10 * (1.0 / 22414.0) * 1e-2 * 1e4 / 1333.2239
R_GAS = 8.314462618            # J/(mol K)


# --------------------------------------------------------------------------
# Elastomer inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SiliconeSpec:
    """Elastomer and jacket inputs. Lengths mm, angles deg.

    Provenance is mixed and is flagged where it matters:
    - shore_a / elong_break_pct / rho: CONVENTION, typical platinum-cure RTV
      datasheet values for a Shore 30A mould rubber (250 % is the low end of the
      250-600 % range those datasheets quote, chosen conservatively).
    - o2_barrer / wvtr: CONVENTION for PDMS-family elastomers, and NOT measured on
      the specific rubber a caster would buy.
    - strain_sf: ASSUMED engineering safety factor on a cyclically loaded rubber.
    """

    skin_t: float = 6.0             # conformal skin thickness
    shore_a: float = 30.0
    elong_break_pct: float = 250.0
    strain_sf: float = 4.0
    rho_g_cm3: float = 1.15         # 1.1-1.2 typical for filled RTV

    # MEASURED, not a chosen midpoint: 600 Barrer for O2 and 23000 Barrer for water
    # on RTV 615 films (Blume et al. 1991), the pair recorded in
    # data/elastomer_params.json as P_O2,sil and P_H2O,sil. An earlier default of
    # 520 Barrer was a hand-picked mid-range value and reported 210x where the
    # provenance-tagged file derives 294x, so the design record and the generator
    # disagreed on the headline number. Both sit inside the 350-800 Barrer envelope
    # and both say "sealed", but a report that contradicts its own parameter file is
    # a defect regardless of whether the conclusion survives.
    o2_barrer: float = 600.0        # PDMS envelope 350-800; 600 measured on RTV 615
    h2o_barrer: float = 23000.0    # same source, same films
    wvtr_g_m2_day_at_50um: float = 2000.0
    E_free_evap_mm_day: float = 1.5  # stage-1/2 transition rate the drying model uses

    # breather lattice
    window_d: float = 0.0           # 0 -> 2.5 * d_max (safely inside the clog band)
    window_spacing: float = 0.0     # 0 -> sized on the aeration requirement
    coverage_target: float = 0.85   # cemented-fraction criterion the sizing must meet
    spacing_ladder: tuple = (48.0, 40.0, 34.0, 28.0, 24.0, 20.0, 17.0)

    # jacket
    jacket_E_GPa: float = 2.0       # printed PETG
    perforation_knockdown: bool = True

    # pour shell (the former the skin itself is cast in)
    pour_clear: float = 0.0         # extra clearance on the skin's outer face
    spout_d: float = 14.0
    vent_d: float = 8.0

    # THE FORMER GETS ITS OWN WALL RULE, and it is the single biggest lever on how
    # much plastic this workflow costs. Two inherited constraints were governing it
    # and neither one is about this part.
    #
    # `AutoSpec.wall_min = 12.0` is a floor for a MOULD carrying a tamped 1890 kg/m3
    # mix. The former carries an untamped rubber head at 0.5-2.3 kPa, and the plate
    # solve asks for 3.7 mm on the vessel — so the floor, not the physics, was setting
    # the thickness, and it set it three times too high. The floor that does apply is
    # printability: 3 mm is comfortably more than the 4-6 perimeters an FDM wall wants.
    #
    # `AutoSpec.deflect_target_mm = 0.10` exists because a mould that bulges casts an
    # out-of-tolerance SECTION, and section is what the drying and oxygen models are
    # most sensitive to. That argument does not transfer: the former's deflection lands
    # on the skin's OUTER face, which mates with the jacket through 6 mm of rubber. The
    # cast body's geometry is set by the pattern, which is rigid and does not care.
    # Half a millimetre there is invisible in the cast, and t scales as the cube root
    # of the target, so accepting 0.5 mm instead of 0.10 mm takes another 1.7x off.
    #
    # Together: vessel former 12.0 -> 3.0 mm wall, about a quarter of the filament.
    pour_deflect_mm: float = 0.50
    pour_wall_min: float = 3.0
    # …but the parting land is thickened back up locally. A 3 mm rim cannot host a
    # tongue and groove, and it is nothing to clamp on. Local thickening buys both for
    # a few cm3 in a narrow band, where thickening the whole shell to suit would cost
    # the saving above twice over.
    pour_rim_min: float = 8.0

    release_steps: int = 120


def shore_a_to_E_MPa(shore_a: float) -> dict:
    """Two published Shore A -> Young's modulus correlations, both reported.

    They disagree by a few percent in the 30A range and by much more at the ends,
    so quoting one alone would overstate what is known. ASTM D2240-derived
    exponential: E = exp(0.0235 SA - 0.6403) MPa. Gent (1958):
    E = 0.0981 (56 + 7.62336 SA) / (0.137505 (254 - 2.54 SA)) MPa.
    """
    e_astm = float(np.exp(0.0235 * shore_a - 0.6403))
    e_gent = float(0.0981 * (56.0 + 7.62336 * shore_a)
                   / (0.137505 * (254.0 - 2.54 * shore_a)))
    return {"E_astm_MPa": e_astm, "E_gent_MPa": e_gent,
            "E_MPa": float(0.5 * (e_astm + e_gent))}


#: Henry solubility of O2 in water at 30 C on a partial-pressure basis,
#: mol/(m3 Pa): 0.234 mol/m3 dissolved at p_O2 = 0.2095 * 101325 Pa.
S_O2_WATER = 1.102e-5


def _p_sat_water(T_K: float) -> float:
    """Saturation vapour pressure of water (Pa), Tetens over liquid.

    Only used away from the 30 C cure temperature, where the steam-table value
    4246 Pa is used directly. Tetens reproduces that to within ~0.3 % at 30 C, which
    is checked in the test below rather than asserted.
    """
    t_c = T_K - 273.15
    return 610.78 * float(np.exp(17.27 * t_c / (t_c + 237.3)))


def barrier_diagnostics(spec: SiliconeSpec, *, wall_mm: float,
                        D_eff_gas_m2s: float, D_eff_sat_m2s: float,
                        T_K: float = 303.15, L_dry_free_mm: float = 21.0,
                        rh_pct: float = 90.0, cure_days_ref: float = 28.0,
                        phi_ref: float = 0.40, dS_ref: float = 0.5) -> dict:
    """Is the skin a barrier? Compare it to the pore network behind it.

    Permeability of a porous medium to a gas, expressed in the same units as a
    polymer permeability, is D_eff/(R T): flux = P dp / t. That is the comparison
    that decides cementation, and it is the one the "PDMS is permeable" intuition
    skips.

    The two phases need DIFFERENT constitutive relations, and using one for both is
    a real error rather than a simplification. In gas-filled pores the concentration
    paired with a partial-pressure driving force is `C = p/(RT)`, so `P = D_eff/(RT)`.
    In water-filled pores it is Henry's law, `C = S p`, so `P = D_eff * S` — and
    `S_O2_water` is about 1.1e-5 mol/(m3 Pa) against `1/(RT)` ~ 4.0e-4, a factor of
    36 apart. Applying the gas relation to the saturated case therefore overstates
    saturated-pore permeability by ~36x and makes the skin look far more limiting
    against wet pores than it is, which weakens the very asymmetry that explains why
    the permeability intuition misleads.
    """
    P_sil = spec.o2_barrer * BARRER
    P_gas = D_eff_gas_m2s / (R_GAS * T_K)
    P_sat = D_eff_sat_m2s * S_O2_WATER
    t_sk, t_w = spec.skin_t * 1e-3, wall_mm * 1e-3
    Rs = t_sk / P_sil
    Rg = t_w / P_gas
    Rw = t_w / P_sat
    # WVTR from the MEASURED water permeability at the ACTUAL cure driving force,
    # not by inverse-thickness scaling of a datasheet figure. The widely repeated
    # "2000 g/(m2 day) at 50 um" has no retrievable source behind it and carries no
    # test temperature or RH gradient (recorded as WVTR_tds, ASSUMED, in
    # data/elastomer_params.json), and a supplier WVTR is quoted at full gradient
    # and 38 C — worth a factor of ~15 on the driving force alone. Here the pore air
    # inside is saturated and the chamber outside is at rh_pct, so:
    #     flux = P_H2O * dp / t,  dp = p_sat(T) * (1 - RH)
    P_h2o = spec.h2o_barrer * BARRER
    p_sat = 4246.0 if abs(T_K - 303.15) < 1.0 else _p_sat_water(T_K)
    dp = p_sat * (1.0 - rh_pct / 100.0)
    wvtr = float(P_h2o * dp / t_sk * 18.015 * 86400.0)   # mol/(m2 s) -> g/(m2 day)
    free = 1500.0                       # g/m2/day free-water evaporation reference
    frac = wvtr / free
    return {
        "P_silicone_mol_m_per_m2_s_Pa": P_sil,
        "P_drained_pore_mol_m_per_m2_s_Pa": P_gas,
        "P_saturated_pore_mol_m_per_m2_s_Pa": P_sat,
        "R_skin_over_R_drained_wall": float(Rs / Rg),
        "R_skin_over_R_saturated_wall": float(Rs / Rw),
        "skin_wvtr_g_m2_day": float(wvtr),
        "wvtr_frac_of_free_evap": float(frac),
        # Substitute the silicone-limited flux into the drying relation itself,
        # L_dry = E t / (phi dS), rather than scaling the open-face L_dry by the flux
        # ratio. The open-face value already carries its own (1 - RH) factor, so the
        # ratio form applies that factor twice — the flux is ALREADY the
        # RH-driven rate. That double-count reads 10x low (0.012 mm against 0.119 mm)
        # and would disagree with data/elastomer_params.json's L_dry,skin row.
        "L_dry_behind_skin_mm": float(
            (wvtr / 1000.0) * cure_days_ref / (phi_ref * dS_ref)),
        "verdict": ("silicone face is effectively no-flux for both O2 and vapour; "
                    "only genuinely open area is atmosphere"),
    }


# --------------------------------------------------------------------------
# Field helpers
# --------------------------------------------------------------------------
def true_sdf(vol: np.ndarray, pitch: float) -> np.ndarray:
    """Signed distance with the half-voxel interface correction (negative inside).

    `sdf.signed_distance_from_binary` measures to the nearest voxel CENTRE of the
    opposite class, so it is half a voxel too large on both branches and an offset
    threshold at `t` delivers `t - pitch/2`. Shifting the interface half a voxel
    off each centre makes the requested offset the delivered one, which is what
    lets the skin thickness be specified rather than calibrated.
    """
    vol = np.asarray(vol, bool)
    h = 0.5 * pitch
    inside = ndimage.distance_transform_edt(vol, sampling=pitch)
    outside = ndimage.distance_transform_edt(~vol, sampling=pitch)
    return np.where(vol, -(inside - h), outside - h)


def measure_offset(inner: np.ndarray, layer: np.ndarray, pitch: float) -> dict:
    """Measure the realised normal thickness of `layer` grown off `inner`.

    Reported rather than assumed: the requested offset, the discretisation and the
    later boolean cuts can all move it, and a skin thinner than asked is a vapour
    barrier that tears.

    Measured at the layer's OUTER FACE, on the same corrected metric the offset was
    taken with. Two mistakes are easy here and both inflate the answer by half a
    voxel each: measuring with the raw EDT (which reads to voxel centres) and then
    also adding a half-voxel for the outer voxel face. Using `true_sdf` — whose zero
    level already sits on the material boundary — and adding the outer half-voxel
    once gives a realised thickness that lands within one voxel of the request, and
    a systematic offset in the answer would otherwise look like a real geometry
    error.
    """
    if not layer.any():
        return {"t_max_mm": 0.0, "t_p50_mm": 0.0, "t_min_mm": 0.0, "n_voxels": 0}
    d = true_sdf(np.asarray(inner, bool), pitch)
    solid = np.asarray(inner, bool) | np.asarray(layer, bool)
    face = solid & ~ndimage.binary_erosion(solid)
    face &= layer                          # only the layer's own outer face
    v = (d[face] if face.any() else d[layer]) + 0.5 * pitch
    return {"t_max_mm": float(v.max()), "t_p50_mm": float(np.percentile(v, 50)),
            "t_mean_mm": float(v.mean()), "t_min_mm": float(v.min()),
            "t_p05_mm": float(np.percentile(v, 5)),
            "n_voxels": int(np.asarray(layer, bool).sum()),
            "measured_on": "layer outer face, true_sdf metric + half voxel"}


def skin_sets(obj: np.ndarray, form: np.ndarray, pitch: float,
              skin_t: float) -> dict:
    """Conformal skin on EVERY cast face, split into outer skin and core lining.

    The offset is taken on the OBJECT, not on the filled silhouette, so core walls
    and cavity walls get skinned too (see module docstring, point 2).
    """
    d = true_sdf(obj, pitch)
    layer = (d > 0.0) & (d <= skin_t)
    outer = obj | layer                     # the solid the jacket is built against
    return {"skin_all": layer,
            "skin_out": layer & ~form,      # outside the silhouette
            "skin_core": layer & form,      # inside a core / cavity
            "outer": outer, "d_obj": d}


def surface_normals(d_field: np.ndarray, pitch: float) -> np.ndarray:
    """Unit outward normal field from a signed-distance field, shape (3, ...).

    On a true SDF the gradient IS the outward normal and it is defined off the
    surface too, which is what lets a bore be tested for normal alignment at every
    voxel it passes through rather than only where it meets material.
    """
    g = np.gradient(d_field, pitch)
    n = np.stack(g)
    mag = np.sqrt((n ** 2).sum(axis=0))
    return n / np.maximum(mag, 1e-9)


def window_lattice(X, Y, Z, *, d: float, spacing: float,
                   axes=(0, 1, 2), phase=(0.31, 0.53, 0.17),
                   normals: np.ndarray | None = None,
                   normal_min: float = 0.55) -> np.ndarray:
    """Periodic lattice of straight bores, kept only where they PIERCE the surface.

    Axis-aligned bores are chosen over surface-normal ones for a good reason — a
    cylinder's minimum cross-section is its own diameter whatever it pierces, so the
    aperture clearance is exactly `d` on a flat face and no less on a curved one —
    and the bores are shared between skin and jacket by construction, which is what
    "aligned" has to mean: a window in the skin with jacket behind it is not a
    window.

    But an axis-aligned bore is only a window where it runs roughly ALONG THE
    SURFACE NORMAL. Where it runs tangentially it does not pierce the skin, it cuts
    a slot ALONG it, and on a 6 mm skin that severs it. This is not a cosmetic
    problem and it is close to invisible: measured on the block, an unrestricted
    three-axis lattice cut the skin into 54 DISCONNECTED FRAGMENTS, and because
    `mould.occ_to_mesh` keeps only the largest body the exported STL silently dropped
    the other 53 — watertight, winding-consistent, single-body, and missing 4.1 % of
    the occupancy volume (1936923 of 2020438 mm3 retained). Every mesh check passed.
    Note how small the volume signal is: the fragments are thin slivers, so a
    volume-ratio check would not have caught this either. Only the component count
    does.

    So a bore family is kept only where |n_axis| >= `normal_min` (0.55 ~ 57 deg of
    the normal). On a box face exactly one family survives, which is correct; on a
    sphere the surviving family rotates with position, which is also correct.
    `normals=None` restores the unrestricted lattice — used only for the sizing
    ladder's upper bound, never for geometry.
    """
    r = d / 2.0
    coords = (X, Y, Z)
    lat = np.zeros(X.shape, bool)
    for ax in axes:
        o = [i for i in range(3) if i != ax]
        A, B = coords[o[0]], coords[o[1]]
        u = np.mod(A / spacing + phase[ax], 1.0) - 0.5
        v = np.mod(B / spacing + phase[(ax + 1) % 3], 1.0) - 0.5
        bore = (u * u + v * v) * spacing ** 2 <= r * r
        if normals is not None:
            bore &= np.abs(normals[ax]) >= normal_min
        lat |= bore
    return lat


# --------------------------------------------------------------------------
# 1. Release mechanics of an elastomeric skin
# --------------------------------------------------------------------------
def undercut_strain(form: np.ndarray, pitch: float, axis: int, k_part: int, *,
                    elong_break_pct: float = 250.0, sf: float = 4.0,
                    min_voxels: int = 8) -> dict:
    """Strain required to stretch a skin over each re-entrant feature.

    A lip protruding `u` laterally beyond everything inboard of it forces the skin
    band spanning `s` along the pull to take the chord path
    `s sqrt(1 + (2u/s)^2)`, so

        eps = sqrt(1 + (2u/s)^2) - 1.

    `u` is measured as the in-plane distance from the protruding material to the
    running-maximum envelope at its own slice (a per-slice 2D EDT), so it is the
    real lateral excursion rather than a nominal dimension. Two spans are reported
    for every feature: `s_feature`, the feature's own axial extent, and `s_free`,
    the axial run of skin from the feature to the outboard end of the form, which
    is the length actually free to elongate. The CONSERVATIVE `s_feature` drives
    the verdict; `s_free` is reported because it is the physically available span
    and the gap between them is the size of the conservatism.
    """
    f = np.moveaxis(np.asarray(form, bool), axis, -1)
    n = f.shape[-1]
    allow = elong_break_pct / max(sf, 1e-9)
    feats = []
    for side, rng in (("lower", list(range(k_part - 1, -1, -1))),
                      ("upper", list(range(k_part, n)))):
        if len(rng) < 2:
            continue
        env = None
        re = np.zeros(f.shape, bool)
        env_stack = {}
        for k in rng:
            s = f[..., k]
            if env is None:
                env = s.copy()
                continue
            env_stack[k] = env.copy()
            re[..., k] = s & ~env
            env |= s
        if not re.any():
            continue
        lbl, nlab = ndimage.label(re)
        end_k = rng[-1]
        for i in range(1, nlab + 1):
            comp = lbl == i
            nv = int(comp.sum())
            if nv < min_voxels:
                continue
            ks = np.unique(np.argwhere(comp)[:, 2])
            u = 0.0
            for k in ks:
                dd = ndimage.distance_transform_edt(~env_stack[k], sampling=pitch)
                u = max(u, float(dd[comp[..., k]].max()))
            s_feat = float((ks.max() - ks.min() + 1) * pitch)
            s_free = float(abs(end_k - (ks.min() if side == "upper" else ks.max()))
                           * pitch + s_feat)
            eps_c = float(np.sqrt(1.0 + (2.0 * u / max(s_feat, pitch)) ** 2) - 1.0)
            eps_f = float(np.sqrt(1.0 + (2.0 * u / max(s_free, pitch)) ** 2) - 1.0)
            feats.append({"side": side, "voxels": nv,
                          "volume_mm3": float(nv * pitch ** 3),
                          "u_mm": float(u), "s_feature_mm": s_feat,
                          "s_free_mm": s_free,
                          "eps_conservative_pct": 100.0 * eps_c,
                          "eps_free_span_pct": 100.0 * eps_f})
    worst = max(feats, key=lambda r: r["eps_conservative_pct"]) if feats else None
    return {"features": feats, "n_features": len(feats), "worst": worst,
            "worst_eps_pct": float(worst["eps_conservative_pct"]) if worst else 0.0,
            "allowable_eps_pct": float(allow),
            "elong_break_pct": elong_break_pct, "safety_factor": sf,
            "admissible_u_over_s": float(np.sqrt((1 + allow / 100.0) ** 2 - 1) / 2.0),
            "ok": bool((worst is None) or worst["eps_conservative_pct"] <= allow),
            "note": ("no re-entrant feature on this parting: the measured parting "
                     "plane already gives a clean straight pull, so the elastomer's "
                     "undercut tolerance is spare capacity"
                     if not feats else "re-entrant features present")}


def hoop_strain_one_piece(form: np.ndarray, obj: np.ndarray, pitch: float,
                          axis: int, *, elong_break_pct: float = 250.0,
                          sf: float = 4.0) -> dict:
    """Can a ONE-PIECE skin come off, or must it be parted/slit?

    Peeling a closed glove skin off in one piece means passing its own opening over
    the body's widest section, so the governing strain is hoop, not chord:

        eps_hoop = sqrt(A_max / A_open) - 1

    (linear scale ratio from cross-sectional areas). A body with no opening at all
    on the pull side gives A_open = 0 and the honest answer is infinity: the skin
    cannot be removed in one piece and must be parted. This is the check that
    decides whether the skin is shipped as one part or two, and it is why an ovoid
    vessel's OUTER skin is parted while the lining of its cavity is not.
    """
    allow = elong_break_pct / max(sf, 1e-9)
    fa = np.moveaxis(np.asarray(form, bool), axis, -1)
    area = fa.sum(axis=(0, 1)) * pitch ** 2
    nz = np.flatnonzero(area > 0)
    out = {}
    for side, k_end in (("pull_from_high", nz.max()), ("pull_from_low", nz.min())):
        # the opening the skin must pass over the girth: the free cross-section at
        # the outboard end that is NOT cast material
        sl = np.moveaxis(np.asarray(obj, bool), axis, -1)[..., k_end]
        a_open = float((fa[..., k_end] & ~sl).sum()) * pitch ** 2
        a_max = float(area.max())
        eps = np.inf if a_open <= 0 else float(np.sqrt(a_max / a_open) - 1.0) * 100.0
        out[side] = {"A_open_mm2": a_open, "A_max_mm2": a_max, "eps_hoop_pct": eps,
                     "ok": bool(eps <= allow)}
    best = min(out.values(), key=lambda r: r["eps_hoop_pct"])
    return {"per_direction": out, "best_eps_hoop_pct": best["eps_hoop_pct"],
            "allowable_eps_pct": float(allow),
            "one_piece_ok": bool(best["ok"]),
            "verdict": ("one-piece skin releases over the girth"
                        if best["ok"] else
                        "one-piece removal exceeds allowable hoop strain: "
                        "skin must be parted or slit at the parting plane")}


def cavity_lining_strain(form: np.ndarray, obj: np.ndarray, pitch: float,
                         axis: int, *, elong_break_pct: float = 250.0,
                         sf: float = 4.0) -> dict:
    """Hoop strain to withdraw the CAVITY lining through the cast's own aperture.

    Separate from `hoop_strain_one_piece` because it asks the opposite question: the
    outer skin has to pass over the body, the lining has to come out through a hole
    in it. For the vessel this is the constriction that forced a LOOSE CORE in the
    rigid path (`mould_auto.decide_cores`, 64.7 % formed) — the same constriction an
    elastomeric lining simply squeezes through, which is the one place the skin
    changes the part count rather than just the section budget.
    """
    allow = elong_break_pct / max(sf, 1e-9)
    hollow = np.asarray(form, bool) & ~np.asarray(obj, bool)
    if not hollow.any():
        return {"applies": False, "note": "solid body: no cavity lining"}
    h = np.moveaxis(hollow, axis, -1)
    w = h.sum(axis=(0, 1)) * pitch ** 2
    nz = np.flatnonzero(w > 0)
    a_max = float(w.max())
    # the aperture is the narrowest open cross-section on the way out, taken as the
    # smallest non-zero hollow section outboard of the widest one
    k_max = int(nz[np.argmax(w[nz])])
    out = {}
    for side, ks in (("high", [k for k in nz if k >= k_max]),
                     ("low", [k for k in nz if k <= k_max])):
        a_open = float(min(w[k] for k in ks)) if ks else 0.0
        eps = np.inf if a_open <= 0 else float(np.sqrt(a_max / a_open) - 1.0) * 100.0
        out[side] = {"A_open_mm2": a_open, "A_max_mm2": a_max,
                     "eps_hoop_pct": eps, "ok": bool(eps <= allow)}
    best = min(out.values(), key=lambda r: r["eps_hoop_pct"])
    return {"applies": True, "per_direction": out,
            "eps_hoop_pct": best["eps_hoop_pct"], "allowable_eps_pct": float(allow),
            "ok": bool(best["ok"]),
            "verdict": ("elastomeric lining squeezes out through the aperture in "
                        "one piece" if best["ok"] else
                        "lining cannot pass its own aperture: loose rigid core "
                        "still required")}


def section_budget(draw_depth_mm: float, draft_deg: float, *,
                   nominal_web_mm: float = 25.0,
                   nominal_face_shell_mm: float = 31.75) -> dict:
    """The section a zero-draft skin returns, from the REALISED rigid-path draft.

    Relief is `tan(theta) * draw` per face, and an internal partition loses it on
    both faces. This is the quantity that made the verified rigid block carry a
    30 mm web against a 25 mm minimum.
    """
    relief = float(np.tan(np.deg2rad(draft_deg)) * draw_depth_mm)
    return {"draw_depth_mm": float(draw_depth_mm), "draft_deg": float(draft_deg),
            "relief_per_face_mm": relief,
            "web_rigid_mm": float(nominal_web_mm + 2 * relief),
            "web_silicone_mm": float(nominal_web_mm),
            "web_returned_mm": float(2 * relief),
            "face_shell_rigid_mm": float(nominal_face_shell_mm + relief),
            "face_shell_silicone_mm": float(nominal_face_shell_mm),
            "face_shell_returned_mm": relief}


# --------------------------------------------------------------------------
# 2. Aeration: size the breather lattice on the requirement
# --------------------------------------------------------------------------
def geodesic_depth(obj: np.ndarray, src: np.ndarray, pitch: float) -> np.ndarray:
    """Distance from each cast voxel to the nearest atmosphere, AROUND the mould.

    A plain EDT is the wrong metric for a moulded body and fails in the direction
    that flatters the design: it measures straight-line distance to the nearest
    source regardless of what lies between, so with a fully enclosing skin it
    happily measures through 20 mm of jacket to the air outside and reports a
    cemented fraction of 8 % where the field solve correctly gives 0. The path has
    to stay inside the cast (or in open air), so the distance is geodesic on the
    domain `obj | src` with the mould as an obstacle.
    """
    from skimage.graph import MCP_Geometric
    dom = np.asarray(obj, bool) | np.asarray(src, bool)
    if not (dom.any() and np.asarray(src, bool).any()):
        return np.full(obj.shape, np.inf)
    cost = np.where(dom, 1.0, np.inf)
    mcp = MCP_Geometric(cost, sampling=(pitch,) * 3, fully_connected=True)
    seeds = [tuple(i) for i in np.argwhere(np.asarray(src, bool))]
    d, _ = mcp.find_costs(seeds)
    return np.where(np.isfinite(d), d, np.inf)


def _cover_surrogate(obj: np.ndarray, src: np.ndarray, pitch: float,
                     L_eff: float, *, mould_occ=None) -> float:
    """Fraction of the cast within one penetration depth of atmosphere.

    `mould_occ` truthy selects the GEODESIC metric (paths confined to cast + open
    air), which is what a moulded body needs. `None` selects the plain EDT and is
    only correct for a bare, fully exposed body.
    """
    if not np.asarray(src, bool).any():
        return 0.0
    d = (geodesic_depth(obj, src, pitch) if mould_occ is not None
         else ndimage.distance_transform_edt(~src, sampling=(pitch,) * 3))
    return ox.cemented_fraction_from_depth(np.where(obj, d, np.nan), obj, L_eff)


def size_windows(obj: np.ndarray, form: np.ndarray, X, Y, Z, pitch: float,
                 spec: SiliconeSpec, *, d_max: float, L_eff_mm: float,
                 vent: np.ndarray | None = None,
                 normals: np.ndarray | None = None) -> dict:
    """Choose the breather spacing from the aeration requirement, not by eye.

    Scans a ladder of spacings coarse-to-fine with the cheap distance surrogate
    (`cemented_fraction_from_depth`, which is what the literature depth rows
    directly support) and stops at the LARGEST spacing that meets the coverage
    target, because open area is not free: it costs jacket stiffness and cast
    surface finish. If no spacing on the ladder meets the target the finest is
    returned with `met=False`, so the shortfall is reported rather than hidden by
    an ever-finer search.
    """
    d_win = spec.window_d or 2.5 * d_max
    rows = []
    chosen = None
    for s in spec.spacing_ladder:
        if s <= d_win * 1.35:            # ligament would fall below ~35 % of the bore
            continue
        lat = window_lattice(X, Y, Z, d=d_win, spacing=s, normals=normals)
        # The sources are the BORES ONLY, and the metric is geodesic. Both matter.
        # Letting outside-connected air be a source at sizing time is the mistake the
        # whole module exists to avoid — the skin covers that air — and scoring with a
        # straight-line EDT lets the path run out through the skin anyway. Sizing on
        # the optimistic metric picks a spacing that then misses the target when the
        # real geometry is checked (measured on the tile: 0.90 predicted, 0.77
        # realised at 48 mm), which is a window pattern that has to be re-cut after
        # the jacket is printed.
        src = lat & ~obj
        if vent is not None:
            src = src | (vent & ~obj)
        cov = _cover_surrogate(obj, src, pitch, L_eff_mm, mould_occ=True)
        f_face = float(np.pi * d_win ** 2 / (4.0 * s ** 2))
        rows.append({"spacing_mm": s, "window_d_mm": d_win,
                     "ligament_mm": float(s - d_win),
                     "face_open_frac_geom": f_face,
                     "cover_surrogate": cov})
        if chosen is None and cov >= spec.coverage_target:
            chosen = rows[-1]
    if chosen is None and rows:
        chosen = rows[-1]
    return {"ladder": rows, "chosen": chosen,
            "met": bool(chosen is not None
                        and chosen["cover_surrogate"] >= spec.coverage_target),
            "window_d_mm": d_win, "L_eff_mm": L_eff_mm,
            "target": spec.coverage_target}


def aeration_case(obj: np.ndarray, mould_occ: np.ndarray, pitch: float, *,
                  L_dry_mm: float, L_gas_mm: float, D_eff_m2s: float,
                  C0_mol_m3: float, R_mol_m3_s: float,
                  parting_axis: int | None = None, parting_index: int | None = None,
                  open_parting_face: bool = False, label: str = "") -> dict:
    """Cemented fraction for one boundary condition, mould faces treated no-flux.

    The domain of the oxygen solve is the DRAINED subdomain — within `L_dry` of an
    open surface — because oxygen only travels usefully through pores evaporation
    has already emptied. Both the field solve and the cheap depth surrogate are
    returned; they answer slightly different questions and disagreeing is
    informative rather than an error.
    """
    em = pf.exposure_mask_in_mould(obj, mould_occ, parting_axis=parting_axis,
                                  parting_index=parting_index,
                                  open_parting_face=open_parting_face)
    src = em["src"]
    out = {"label": label, "open_area_frac": em["open_area_frac"],
           "open_faces": em["open_faces"], "sealed_faces": em["sealed_faces"],
           "L_dry_mm": L_dry_mm, "L_gas_mm": L_gas_mm,
           "L_eff_mm": float(min(L_dry_mm, L_gas_mm))}
    if not src.any():
        out.update({"cemented_frac_field": 0.0, "cemented_frac_surrogate": 0.0,
                    "drained_frac": 0.0, "resolution_warning": None,
                    "note": "no open area: fully enclosed, no atmosphere boundary"})
        return out
    # geodesic, not Euclidean: the drying and oxygen paths both run through the cast
    # and out through open area, never through the mould wall (see `geodesic_depth`)
    dist = geodesic_depth(obj, src, pitch)
    drained = obj & (dist <= L_dry_mm)
    out["drained_frac"] = float(drained.sum() / max(obj.sum(), 1))
    out["cemented_frac_surrogate"] = ox.cemented_fraction_from_depth(
        np.where(obj, dist, np.nan), obj, out["L_eff_mm"])
    if drained.sum() == 0:
        out.update({"cemented_frac_field": 0.0, "resolution_warning": None,
                    "note": "nothing drains within the cure: anoxic by construction"})
        return out
    r = ox.solve_oxygen(drained, src, pitch, D_eff_m2s=D_eff_m2s,
                        C0_mol_m3=C0_mol_m3, R_mol_m3_s=R_mol_m3_s)
    oxy = np.zeros_like(obj)
    oxy[drained] = r["oxygenated"][drained]
    out.update({"cemented_frac_field": float(oxy.sum() / max(obj.sum(), 1)),
                "resolution_warning": r.get("resolution_warning"),
                "analytic_depth_mm": r["analytic_depth_mm"], "note": ""})
    return out


# --------------------------------------------------------------------------
# 3. Release verification, in order
# --------------------------------------------------------------------------
def verify_release(parts: dict, pitch: float, axis: int, *, steps: int = 120,
                   skin_free_clear_mm: float = 0.0) -> dict:
    """Jacket off the skin FIRST, then the skin off the cast. Order is kinematic.

    Three tests, and the third is the one that makes the order a result rather
    than an assertion:

    (a) each jacket half clears the skin-clad body by a straight pull along the
        draw (it is drafted against the skin's outer face, so it should);
    (b) the skin's own straight pull off the cast, which is allowed to FAIL — the
        skin releases by stretching, not by translation, and `undercut_strain` is
        the test that governs it;
    (c) a DISCRIMINATION CONTROL that must INTERFERE: the same jacket half with the
        cavity pinched inward over a band of slices, i.e. a burr left at the parting
        line. Without it a pass in (a) is not evidence of anything, since an empty
        or mis-signed set clears trivially. Note that neither an undrafted cavity
        nor a reversed sweep works as a control — a swept cavity contains the body
        at every slice by construction, so for a monotone body no draft sense makes
        an axial pull interfere. The sweep is sensitive to re-entrancy, so the
        control has to contain some.

    Skin-first is refuted by measurement, not by argument: the clearance between
    the skin's outer face and the jacket's inner face is the draft relief and
    nothing more, so the lateral excursion the skin would need to stretch over any
    feature is unavailable while the jacket is on.
    """
    out = {}
    for name in ("jacket_lower", "jacket_upper"):
        if name not in parts:
            continue
        up = name.endswith("upper")
        out[name] = mould.release_sweep(parts[name], parts["outer_body"], axis=axis,
                                        up=not up, steps=steps)
    if "skin_all" in parts and "obj" in parts:
        out["skin_straight_pull"] = mould.release_sweep(
            parts["skin_all"], parts["obj"], axis=axis, up=True, steps=steps)
    if "control_undrafted" in parts:
        out["control_undrafted"] = mould.release_sweep(
            parts["control_undrafted"], parts["outer_body"], axis=axis, up=True,
            steps=steps)
    jacket_ok = all(out[k]["clears"] for k in out if k.startswith("jacket"))
    ctrl = out.get("control_undrafted")
    return {"sweeps": out, "jacket_clears": bool(jacket_ok),
            "control_interferes": bool(ctrl is not None and not ctrl["clears"]),
            "skin_confinement_clear_mm": float(skin_free_clear_mm),
            "order_ok": bool(jacket_ok and (ctrl is None or not ctrl["clears"])),
            "order": ["remove jacket halves from the skin-clad cast",
                      "peel the skin off the cast",
                      "withdraw the cavity lining through the aperture"],
            "why_order_matters": (
                "the jacket cavity is generated against the skin's own outer face, "
                f"so the skin has only {skin_free_clear_mm:.2f} mm of lateral "
                "clearance while jacketed and cannot stretch over anything; "
                "skin-first is kinematically unavailable")}


# --------------------------------------------------------------------------
# 4. Assembly with attributed volume balance
# --------------------------------------------------------------------------
def balance(envelope: np.ndarray, solids: dict, voids: dict,
            pitch: float) -> dict:
    """Attributed partition check, same philosophy as `mould.assemble`.

    The residual is computed as `envelope - all solids` and then attributed by
    intersecting it with each NAMED void. A void the named features cannot explain
    lands in `unattributed_mm3`, which is the number that catches a mis-signed
    boolean or a feature that cut through a wall — a hand-tallied balance cannot,
    because a wrong boolean moves both sides of the tally at once.
    """
    v = pitch ** 3
    union = np.zeros_like(envelope)
    overlap = {}
    for k, s in solids.items():
        ov = int((union & s).sum())
        if ov:
            overlap[k] = float(ov * v)
        union |= s
    residual = envelope & ~union
    attributed = np.zeros_like(residual)
    attr = {}
    for k, f in voids.items():
        hit = residual & f & ~attributed
        attr[k + "_mm3"] = float(hit.sum() * v)
        attributed |= hit
    unattr = residual & ~attributed
    out = {"envelope_mm3": float(envelope.sum() * v),
           **{k + "_mm3": float(s.sum() * v) for k, s in solids.items()},
           "residual_void_mm3": float(residual.sum() * v), **attr,
           "unattributed_mm3": float(unattr.sum() * v),
           "overlaps_mm3": overlap,
           "outside_envelope_mm3": float((union & ~envelope).sum() * v)}
    out["exact"] = bool(out["unattributed_mm3"] == 0.0 and not overlap
                        and out["outside_envelope_mm3"] == 0.0)
    return out


def build_silicone_mould(geom, spec: SiliconeSpec | None = None,
                         auto: AutoSpec | None = None, pitch: float = 0.0,
                         *, phys=None, cure_days: float = 28.0,
                         rh_pct: float = 90.0, verbose: bool = False) -> dict:
    """Full elastomeric mould for any grammar: skin, jacket, cores, pour shell.

    Sequence, each step measured rather than assumed:
      1. object field from the grammar; parting axis/plane from `analyse_parting`
      2. conformal skin by true-SDF offset of the CAST SURFACE
      3. jacket wall from `auto_wall` on the SHORT span of the skin-clad body,
         scaled for perforation
      4. breather lattice sized on the aeration requirement
      5. jacket cavity swept from the skin's outer face at the drafted angle
      6. attributed volume balance, release sweeps in order, aperture classes
    """
    spec = spec or SiliconeSpec()
    auto = auto or AutoSpec()
    if phys is None:
        from .gui import engine as eng
        phys = eng.load_physics()

    # ---- 1. object, form, parting ---------------------------------------
    bf = mould_auto.build_field(geom, pitch or auto.pitch)
    pitch = bf["pitch"]
    d_obj0, o_obj = bf["field"], bf["origin"]
    occ0 = d_obj0 <= 0.0
    form0 = mould.fill_slices(occ0, axis=2)
    part = mould_auto.analyse_parting(occ0, o_obj, pitch)
    axis, parting = part["axis"], part["coord"]
    ch = part["chosen"]
    draw = ch["draw_depth_mm"]
    draft = mould_auto.auto_draft(draw, auto)

    # ---- transport scales ----------------------------------------------
    D_gas = ox.effective_diffusivity(phys.D_O2_gas[1], phys.phi[1], phys.sw[1],
                                     gas=True)
    D_sat = ox.effective_diffusivity(1.998e-9, phys.phi[1], phys.sw[1], gas=False)
    L_gas = ox.analytic_penetration_depth(D_gas, phys.C_O2_gas[1], phys.R_O2_bulk[1])
    L_dry = dry.air_entry_depth(phys.E_evap[1], cure_days, phys.phi[1],
                                delta_saturation=phys.dS_air_entry[1],
                                rh_pct=rh_pct)
    L_eff = float(min(L_gas, L_dry))

    # ---- 2. skin on the object's own grid (for measurement) -------------
    sk0 = skin_sets(occ0, form0, pitch, spec.skin_t)
    skin_meas = measure_offset(occ0, sk0["skin_all"], pitch)

    # release mechanics, measured on the object grid
    k_part0 = int(round((parting - np.asarray(o_obj, float)[axis]) / pitch))
    uc = undercut_strain(form0, pitch, axis, k_part0,
                         elong_break_pct=spec.elong_break_pct, sf=spec.strain_sf)
    hoop = hoop_strain_one_piece(form0, occ0, pitch, axis,
                                 elong_break_pct=spec.elong_break_pct,
                                 sf=spec.strain_sf)
    lining = cavity_lining_strain(form0, occ0, pitch, axis,
                                  elong_break_pct=spec.elong_break_pct,
                                  sf=spec.strain_sf)
    cores = mould_auto.decide_cores(form0, occ0, axis, k_part0, pitch)

    # ---- 3. jacket wall from the skin-clad short span -------------------
    om = mould_auto.outline_metrics(
        mould_auto.outline_at(sk0["outer"], axis, k_part0), pitch,
        np.delete(np.asarray(o_obj, float), axis))
    span_short = 2.0 * min(om["half_x"], om["half_y"])
    wall = mould_auto.auto_wall(span_short, draw, auto)
    key = mould_auto.auto_key_size(
        mould_auto.auto_flange(24.0, auto)["flange_mm"], auto.d_max)
    fl = mould_auto.auto_flange(key["key_base_d"], auto)
    flange_w = fl["flange_mm"]

    # ---- 4. window sizing (on the object grid: bores pierce the jacket
    #         whatever its thickness, so the sizing does not depend on it) ----
    d_win = spec.window_d or 2.5 * auto.d_max
    Xa, Ya, Za = np.meshgrid(*[np.asarray(o_obj, float)[i]
                               + np.arange(occ0.shape[i]) * pitch
                               for i in range(3)], indexing="ij")
    vent0 = None
    if lining.get("applies"):
        # axial vent through the cavity lining, out through the cast's aperture:
        # without it the bores inside a closed cavity are isolated pockets
        hollow0 = form0 & ~occ0
        hm = mould_auto.outline_metrics(
            mould_auto.outline_at(hollow0, axis, int(np.argmax(
                np.moveaxis(hollow0, axis, -1).sum(axis=(0, 1))))), pitch,
            np.delete(np.asarray(o_obj, float), axis))
        cx, cy = hm["centroid"]
        r_vent = max(3.0, min(0.5 * d_win, 6.0))
        vent0 = (np.sqrt((Xa - cx) ** 2 + (Ya - cy) ** 2) <= r_vent)
    nrm0 = surface_normals(sk0["d_obj"], pitch)
    win = size_windows(occ0, form0, Xa, Ya, Za, pitch, spec, d_max=auto.d_max,
                       L_eff_mm=L_eff, vent=vent0, normals=nrm0)
    spacing = spec.window_spacing or win["chosen"]["spacing_mm"]
    f_geom = float(np.pi * d_win ** 2 / (4.0 * spacing ** 2))
    t_jacket = wall["t_mm"]
    if spec.perforation_knockdown:
        t_jacket = float(np.ceil(wall["t_mm"] * (1.0 - f_geom) ** (-1.0 / 3.0)))

    # ---- 5. padded grid, jacket envelope --------------------------------
    lo = np.asarray(o_obj, float) - (spec.skin_t + t_jacket + flange_w + 3 * pitch)
    hi = (np.asarray(o_obj, float) + (np.asarray(occ0.shape) - 1) * pitch
          + (spec.skin_t + t_jacket + flange_w + 3 * pitch))
    X, Y, Z, origin, sh = sdf.make_grid(lo, hi, pitch)
    obj = mould.resample_field(d_obj0, o_obj, pitch, X, Y, Z) <= 0.0
    form = mould.fill_slices(obj, axis=2)
    sk = skin_sets(obj, form, pitch, spec.skin_t)
    outer = sk["outer"]

    k_part = int(round((parting - origin[axis]) / pitch))
    below = np.zeros(sh, bool)
    sl = [slice(None)] * 3
    sl[axis] = slice(0, k_part)
    below[tuple(sl)] = True

    # jacket envelope: cup + parting flange, on the skin-clad silhouette
    om2 = mould_auto.outline_metrics(mould_auto.outline_at(outer, axis, k_part),
                                     pitch, np.delete(origin, axis))
    square = om2["box_fill_frac"] > 0.90
    zb = np.moveaxis(outer, axis, -1)
    nzk = np.flatnonzero(zb.any(axis=(0, 1)))
    z_lo = origin[axis] + nzk.min() * pitch - t_jacket
    z_hi = origin[axis] + nzk.max() * pitch + t_jacket
    cx, cy = om2["centroid"]
    Xc, Yc = (X - cx, Y - cy) if axis == 2 else (X, Y)
    envelope = mould.cup_flange_block(
        Xc, Yc, Z, r_cup=om2["r_equiv_mm"] + t_jacket, z_lo=z_lo, z_hi=z_hi,
        parting=parting, flange=flange_w, flange_t=max(18.0, 0.9 * t_jacket),
        square=square, half_x=om2["half_x"] + t_jacket,
        half_y=om2["half_y"] + t_jacket) <= 0.0
    envelope |= outer                     # never clip the body itself

    # jacket cavity: drafted sweep of the SKIN's outer face
    cav_lo = mould.cone_sweep(outer & below, pitch, axis=axis, up=True,
                              mode="dilate", draft_deg=draft["draft_deg"]) <= 0
    cav_up = mould.cone_sweep(outer & ~below, pitch, axis=axis, up=False,
                              mode="dilate", draft_deg=draft["draft_deg"]) <= 0
    cavity = ((cav_lo & below) | (cav_up & ~below)) & envelope
    cavity |= outer

    # core bodies: everything inside the silhouette that is neither cast nor skin
    core_body = form & ~obj & ~sk["skin_all"]

    # ---- features -------------------------------------------------------
    nrm = surface_normals(sk["d_obj"], pitch)
    lat = window_lattice(X, Y, Z, d=d_win, spacing=spacing, normals=nrm)
    r_ref = (max(om2["half_x"], om2["half_y"]) if square else om2["r_equiv_mm"])
    key_ang = mould_auto.chiral_angles(auto.key_count, mould_auto.detect_symmetries(
        mould_auto.outline_at(outer, axis, k_part), pitch, (cx, cy),
        np.delete(origin, axis)))
    dil = mould_auto.dilate_outline(mould_auto.outline_at(outer, axis, k_part),
                                    pitch, flange_w)
    key_xy = mould_auto.project_to_outline(dil, pitch, np.delete(origin, axis),
                                           (cx, cy), key_ang, 0.80)
    n_bolt = auto.bolt_count or max(4, int(np.ceil(
        2 * np.pi * (r_ref + flange_w) / auto.bolt_pitch_max)))
    bolt_xy = mould_auto.project_to_outline(
        dil, pitch, np.delete(origin, axis), (cx, cy),
        [(i + 0.5) * 360.0 / n_bolt for i in range(n_bolt)], 0.93)

    keys = np.zeros(sh, bool)
    sockets = np.zeros(sh, bool)
    bolts = np.zeros(sh, bool)
    # The key is rooted BELOW the parting plane, by two voxels, and is not clipped to
    # the upper side. A key nominally starting exactly at z = parting does not
    # survive discretisation: `k_part` is a rounded index, so the slice at the
    # parting plane can sit a fraction of a voxel BELOW `parting`, the frustum's
    # `z0 - Z > 0` test then excludes it, and the key's lowest voxel lands one slice
    # above the lower half's top face. The result is a registration key floating in
    # air with a one-voxel gap under it — measured on the shell as three separate
    # 4529 mm3 bodies, which is exactly the failure mode `mould.ring_positions_rect`
    # documents (a key that was never attached cannot unbalance anything, so every
    # volume check still closed). Rooting it in the lower half's material makes the
    # attachment independent of where the rounding falls.
    z_root = parting - 2.0 * pitch
    for (kx, ky) in key_xy:
        keys |= mould.frustum_field(X, Y, Z, r_base=key["key_base_d"] / 2,
                                    r_top=key["key_top_d"] / 2, z0=z_root,
                                    z1=parting + key["key_h"],
                                    center=(kx, ky)) <= 0
        sockets |= mould.frustum_field(
            X, Y, Z, r_base=key["key_base_d"] / 2 + auto.key_clear,
            r_top=key["key_top_d"] / 2 + auto.key_clear, z0=z_root,
            z1=parting + key["key_h"] + auto.key_clear, center=(kx, ky)) <= 0
    for (bx, by) in bolt_xy:
        bolts |= sdf.sd_cylinder_z(X, Y, Z, auto.bolt_d / 2, 1e4,
                                   center=(bx, by, parting)) <= 0
    keys &= envelope                # never a key floating outside the mould block
    sockets &= ~below

    # fill gate: the one aperture the MIX flows through, so the 6 x d_max floor
    gate_d = auto.jam_mult * auto.d_max
    gate = (np.sqrt((Y - cy) ** 2 + (Z - parting) ** 2) <= gate_d / 2) & (X >= cx)
    gate &= ~below

    # axial vent through the core body, so bores inside a closed cavity connect out
    vent = np.zeros(sh, bool)
    if lining.get("applies") and core_body.any():
        r_vent = max(3.0, min(0.5 * d_win, 6.0))
        vent = (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) <= r_vent) & (form | envelope)

    keepout = np.zeros(sh, bool)
    for (kx, ky) in list(key_xy) + list(bolt_xy):
        keepout |= (np.sqrt((X - kx) ** 2 + (Y - ky) ** 2)
                    <= 0.5 * key["key_base_d"] + d_win)

    # ---- 6. parts, with the window spacing REFINED on the real geometry ----
    #
    # The ladder in `size_windows` scores a bare lattice, and the assembled mould is
    # not that: bores are removed where they would break into a key socket or a bolt
    # hole, and the aeration is then measured with the real skin, core and jacket in
    # place. On the tile the bare ladder promised 0.889 coverage at 40 mm spacing and
    # the assembled geometry delivered 0.748 — enough of a gap to ship a jacket that
    # has to be re-drilled. So the ladder result is treated as a starting estimate
    # and the spacing is stepped down until the ASSEMBLED geometry meets the target.
    # Cheap to do: the windows are boolean cuts on sets that are already computed.
    def _despeckle(m, min_mm3=200.0):
        """Drop islands below a printable size.

        A bore grazing the flange edge leaves 3-14 mm3 crumbs — a few voxels each.
        They are not design features, they are debris that would print as loose
        specks and, more importantly, they make an otherwise sound part read as
        fragmented and so mask the real fragmentation the connectivity guard exists
        to catch. Removed by size, and the removed volume is accounted for in the
        balance rather than discarded silently.
        """
        lab, n = ndimage.label(m)
        if n <= 1:
            return m, 0.0
        sizes = np.bincount(lab.ravel())
        keep = np.flatnonzero(sizes * pitch ** 3 >= min_mm3)
        keep = keep[keep > 0]
        out = np.isin(lab, keep) if len(keep) else np.zeros_like(m)
        return out, float((m & ~out).sum() * pitch ** 3)

    def _assemble(sp):
        lat_s = window_lattice(X, Y, Z, d=d_win, spacing=sp, normals=nrm)
        win_s = lat_s & ~keepout & ~gate
        jr = envelope & ~cavity & ~bolts
        raw = {
            "jacket_lower": (jr & below & ~win_s) | (keys & ~cavity & ~win_s),
            "jacket_upper": jr & ~below & ~sockets & ~gate & ~win_s,
            "skin_all": sk["skin_all"] & ~win_s & ~gate & ~vent,
            "skin_out": sk["skin_out"] & ~win_s & ~gate & ~vent,
            "skin_core": sk["skin_core"] & ~win_s & ~vent,
            "core": core_body & ~win_s & ~vent}
        out = {"spacing": sp, "windows": win_s}
        swept = 0.0
        for k, v in raw.items():
            cleaned, lost = _despeckle(v)
            out[k] = cleaned
            swept += lost
        out["swept_mm3"] = swept
        return out

    ladder = [s for s in ([spacing] if spec.window_spacing else spec.spacing_ladder)
              if s >= spacing - 1e-9 or spec.window_spacing]
    ladder = ([spacing] if spec.window_spacing
              else [s for s in spec.spacing_ladder if s <= spacing + 1e-9])
    # FRAGMENTATION IS A HARD CONSTRAINT AND IT BINDS BEFORE AERATION DOES.
    #
    # Chasing the coverage target alone drives the spacing down until the bores
    # overlap enough to cut a part into pieces: measured on the shell, the loop ran
    # to 20 mm spacing and left the jacket in 40 disconnected bodies and the skin in
    # 12. Those parts cannot be printed or fitted, and no mesh check rejects them
    # (`occ_to_mesh` keeps the largest body). So each candidate spacing is tested for
    # connectivity too and a fragmenting one is REFUSED, even when it is the only
    # spacing that would have met the coverage target.
    #
    # Connectivity is counted per LOGICAL part, which is not the same as per emitted
    # set. `skin_all` is legitimately several pieces — the outer skin and the lining
    # of each hollow core are separate parts, demoulded differently (the outer skin
    # peels off, the lining squeezes out through the aperture) — so counting bodies
    # on their union would condemn a correct design. `skin_out` must be one piece,
    # and `skin_core` may have one piece per core.
    n_cores_expected = max(1, int(ndimage.label(core_body)[1])) if core_body.any() else 0

    def _connectivity(cand):
        rows = {}
        rows["skin_out"] = int(ndimage.label(cand["skin_out"])[1])
        rows["jacket_lower"] = int(ndimage.label(cand["jacket_lower"])[1])
        rows["jacket_upper"] = int(ndimage.label(cand["jacket_upper"])[1])
        rows["skin_core"] = int(ndimage.label(cand["skin_core"])[1]) \
            if cand["skin_core"].any() else 0
        ok = (rows["skin_out"] <= 1 and rows["jacket_lower"] <= 1
              and rows["jacket_upper"] <= 1
              and rows["skin_core"] <= max(n_cores_expected, 1))
        return rows, bool(ok)

    refine = []
    A = None
    for sp in ladder:
        cand = _assemble(sp)
        conn, conn_ok = _connectivity(cand)
        m_occ = (cand["jacket_lower"] | cand["jacket_upper"] | cand["skin_all"]
                 | cand["core"])
        em = pf.exposure_mask_in_mould(obj, m_occ)
        cov = _cover_surrogate(obj, em["src"], pitch, L_eff, mould_occ=True)
        refine.append({"spacing_mm": sp, "open_area_frac": em["open_area_frac"],
                       "cover_surrogate_assembled": cov, "bodies": conn,
                       "connected": conn_ok,
                       "accepted": bool(conn_ok)})
        if not conn_ok:
            break                       # finer spacings only fragment further
        A = cand
        if cov >= spec.coverage_target:
            break
    if A is None:                       # even the coarsest spacing fragments
        A = _assemble(ladder[0])
        refine[0]["accepted"] = False
    spacing = A["spacing"]
    f_geom = float(np.pi * d_win ** 2 / (4.0 * spacing ** 2))
    windows = A["windows"]
    jacket_lower, jacket_upper = A["jacket_lower"], A["jacket_upper"]
    skin_all, skin_out, skin_core, core = (A["skin_all"], A["skin_out"],
                                           A["skin_core"], A["core"])
    # The envelope was already cut at the pre-refinement thickness, so the
    # perforation knockdown is re-evaluated and REPORTED rather than silently
    # reassigned — quietly changing the number after the geometry exists would make
    # the reported wall a fiction. If refinement chose a finer spacing than the
    # estimate, the built wall can be thinner than the perforated panel needs, and
    # that must surface as a failure the caster sees rather than as a nicer number.
    t_req = (float(np.ceil(wall["t_mm"] * (1.0 - f_geom) ** (-1.0 / 3.0)))
             if spec.perforation_knockdown else wall["t_mm"])
    jacket_wall = {"t_solved_mm": wall["t_mm"], "t_built_mm": t_jacket,
                   "t_required_perforated_mm": t_req,
                   "knockdown_factor": float((1.0 - f_geom) ** (-1.0 / 3.0)),
                   "adequate": bool(t_jacket >= t_req)}

    # WHO OWNS THE CORE BACKING. Behind the elastomeric lining of a hollow core sits
    # rigid material, and it belongs to exactly one part. The jacket envelope minus
    # the swept cavity claims it by default, which double-counts it against a
    # separately emitted `core` piece — measured as a 4.11e6 mm3 overlap on the block
    # and 1.69e4 mm3 on the shell, and it would have printed two parts occupying the
    # same space. `decide_cores` already answers the question on the kinematic
    # release condition, so the answer is taken from there rather than guessed:
    #
    #   integral_boss -> the backing IS the jacket (split at the parting plane), no
    #                    separate part; this is the block, 95.9 % of its hollow formed
    #                    by a withdrawable boss.
    #   loose_core    -> a separate piece, subtracted from the jacket; this is the
    #                    vessel, whose bore pins a withdrawable boss to 64.7 %.
    #
    # Note what the elastomer does and does not change here. The LINING passes back
    # out through the vessel's bore by stretching (`cavity_lining_strain`), but the
    # rigid backing behind it still cannot, so the part count is unchanged and the
    # skin buys section budget rather than parts on this typology.
    core_integral = cores.get("strategy") == "integral_boss"
    if core_integral:
        core = np.zeros(sh, bool)
    else:
        jacket_lower = jacket_lower & ~core
        jacket_upper = jacket_upper & ~core

    parts = {"jacket_lower": jacket_lower, "jacket_upper": jacket_upper,
             "skin_all": skin_all, "skin_out": skin_out, "skin_core": skin_core,
             "core": core, "obj": obj, "outer_body": outer, "form": form,
             "envelope": envelope, "below": below, "windows": windows,
             "cavity": cavity, "vent": vent, "gate": gate}

    solids = {"jacket_lower": jacket_lower, "jacket_upper": jacket_upper,
              "skin": skin_all, "core": core, "cast": obj}
    # `swept_slivers` is a named void, not slack in the balance: the despeckler
    # removes material, so without an explicit entry that volume would land in
    # `unattributed_mm3` and be indistinguishable from a mis-signed boolean.
    swept = envelope & ~(jacket_lower | jacket_upper | skin_all | core | obj)
    voids = {"windows": windows, "gate": gate, "vent": vent, "bolts": bolts,
             "key_clearance": sockets & ~keys,
             "draft_relief": cavity & ~outer,
             "cast_hollow": form & ~obj,
             "swept_slivers": swept}
    bal = balance(envelope, solids, voids, pitch)

    # DISCRIMINATION CONTROL — a mould that MUST fail the release test.
    #
    # Two obvious controls are both useless here, and understanding why is the point.
    # An undrafted cavity passes: straight walls clear a straight pull. Reversing the
    # sweep sense ALSO passes, and for a more interesting reason — a swept cavity
    # contains the body at every slice by construction, so for a monotone body no
    # draft sense can make an axial translation interfere. The release sweep is only
    # sensitive to genuine RE-ENTRANCY, so a control that exercises it has to contain
    # some.
    #
    # This one is the failure a caster actually gets: a burr left at the parting
    # line, modelled as the cavity pinched inward by two voxels over a band of slices
    # inside the lower half. Nothing about it is visible in a render, it is
    # watertight, and it is unopenable. If `release_sweep` does not flag it, a pass
    # on the real jacket is not evidence of anything.
    # The band must lie inside the cavity's MEASURED axial extent. Placing it at a
    # fraction of the parting index instead put it 10 slices below the cavity on the
    # tile, where there was nothing to pinch: the control then cleared and read as a
    # passing release test on a mould that had never been perturbed. A control that
    # silently does nothing is worse than no control.
    band = np.zeros(sh, bool)
    cav_lo_side = cavity & below
    kc = np.flatnonzero(np.moveaxis(cav_lo_side, axis, -1).any(axis=(0, 1)))
    if len(kc) >= 3:
        kb0 = int(kc.min() + max(1, int(round(0.55 * (kc.max() - kc.min())))))
        kb1 = min(kb0 + max(2, int(round(3.0 / pitch))), int(kc.max()))
        sl_b = [slice(None)] * 3
        sl_b[axis] = slice(kb0, max(kb1, kb0 + 2))
        band[tuple(sl_b)] = True
    pinch = ndimage.binary_erosion(
        cavity, structure=ndimage.generate_binary_structure(3, 1),
        iterations=2) | ~band
    parts["control_undrafted"] = envelope & below & ~(cavity & pinch)
    rel = verify_release(parts, pitch, axis, steps=spec.release_steps,
                         skin_free_clear_mm=0.0)

    # ---- aeration comparison -------------------------------------------
    mould_windowed = jacket_lower | jacket_upper | skin_all | core
    mould_enclosed = (envelope & ~obj)      # nothing open at all
    cases = {
        "enclosed_skin": aeration_case(
            obj, mould_enclosed, pitch, L_dry_mm=L_dry, L_gas_mm=L_gas,
            D_eff_m2s=D_gas, C0_mol_m3=phys.C_O2_gas[1],
            R_mol_m3_s=phys.R_O2_bulk[1], label="fully enclosed silicone skin"),
        "windowed": aeration_case(
            obj, mould_windowed, pitch, L_dry_mm=L_dry, L_gas_mm=L_gas,
            D_eff_m2s=D_gas, C0_mol_m3=phys.C_O2_gas[1],
            R_mol_m3_s=phys.R_O2_bulk[1], label="windowed skin + jacket"),
        "rigid_open_parting": aeration_case(
            obj, mould_enclosed, pitch, L_dry_mm=L_dry, L_gas_mm=L_gas,
            D_eff_m2s=D_gas, C0_mol_m3=phys.C_O2_gas[1],
            R_mol_m3_s=phys.R_O2_bulk[1], parting_axis=axis,
            parting_index=k_part, open_parting_face=True,
            label="rigid split mould, parting face open (paper Fig. 6)"),
    }

    # ---- apertures ------------------------------------------------------
    aps = [mould.Aperture("breather_window", "liquid", d_win,
                          "self-damming: below 3 x d_max the mix always bridges "
                          "(Vani 2022), so it retains aggregate while passing gas "
                          "and spent cementation solution"),
           mould.Aperture("fill_gate", "aggregate", gate_d,
                          "the one passage the wet mix flows through")]
    ap = mould.check_apertures(aps, auto.d_max, jam_mult=auto.jam_mult,
                              certain_clog_mult=auto.clog_mult)
    pour_pass = {"spout_d_mm": spec.spout_d, "vent_d_mm": spec.vent_d,
                 "class": "uncured elastomer, no granular phase",
                 "jamming_applies": False,
                 "pourable": bool(spec.spout_d >= 10.0 and spec.vent_d >= 5.0),
                 "note": ("the jamming criterion is a statement about a granular "
                          "suspension; the silicone pour has no aggregate, so the "
                          "spout is sized on viscous fill and air escape instead")}

    # ---- silicone quantity ---------------------------------------------
    v_sil = float(skin_all.sum() * pitch ** 3)
    v_core_sil = float(skin_core.sum() * pitch ** 3)
    mass = v_sil * spec.rho_g_cm3 / 1000.0

    res = {
        "typology": bf["grammar"], "pitch": pitch, "origin": origin,
        "axis": axis, "parting": parting, "k_part": k_part, "below": below,
        "parting_analysis": part, "draft": draft, "wall": wall,
        "jacket_wall_mm": t_jacket, "jacket_wall": jacket_wall,
        "flange_mm": flange_w, "key": key,
        "key_xy": key_xy, "bolt_xy": bolt_xy, "square": square,
        "skin_measured": skin_meas, "skin_t_requested": spec.skin_t,
        "undercut": uc, "hoop": hoop, "lining": lining, "cores": cores,
        "window": {"d_mm": d_win, "spacing_mm": spacing,
                   "face_open_frac_geom": f_geom, "sizing": win,
                   "refinement": refine,
                   "spacing_estimate_mm": win["chosen"]["spacing_mm"],
                   "accepted_rows": [r for r in refine if r["accepted"]],
                   "fragmented_at_mm": next((r["spacing_mm"] for r in refine
                                             if not r["connected"]), None),
                   "cover_at_chosen": float(
                       [r for r in refine if r["accepted"]][-1]
                       ["cover_surrogate_assembled"]) if any(
                           r["accepted"] for r in refine) else 0.0,
                   "met_assembled": bool(any(
                       r["accepted"]
                       and r["cover_surrogate_assembled"] >= spec.coverage_target
                       for r in refine)),
                   "limited_by": ("fragmentation: a finer window pitch would cut the "
                                  "skin or jacket into disconnected pieces"
                                  if any(not r["connected"] for r in refine)
                                  else "coverage target met while every part stayed "
                                       "connected")},
        "balance": bal, "release": rel, "aeration": cases, "apertures": ap,
        "pour_apertures": pour_pass,
        "silicone_volume_mm3": v_sil, "silicone_core_volume_mm3": v_core_sil,
        "silicone_mass_g": mass,
        "section_budget": section_budget(draw, draft["draft_deg"]),
        # The reference wall is the vessel's 26 mm — the section the design record's
        # ratio is quoted against — so the number stays comparable across typologies
        # instead of moving with each body's own drained depth.
        "barrier": barrier_diagnostics(
            spec, wall_mm=26.0, D_eff_gas_m2s=D_gas, D_eff_sat_m2s=D_sat,
            L_dry_free_mm=L_dry, rh_pct=rh_pct),
        "transport": {"D_eff_gas_m2s": D_gas, "D_eff_sat_m2s": D_sat,
                      "L_gas_mm": L_gas, "L_dry_mm": L_dry, "L_eff_mm": L_eff,
                      "cure_days": cure_days, "rh_pct": rh_pct},
        "elastomer": shore_a_to_E_MPa(spec.shore_a),
        "spec": asdict(spec), "auto": asdict(auto),
        "_parts": parts, "_obj_mesh": bf["mesh"],
    }
    res["pour_shell"] = build_pour_shell(res, spec, auto)
    return res


def build_pour_shell(res: dict, spec: SiliconeSpec,
                     auto: AutoSpec) -> dict:
    """The printed former the SKIN itself is poured in. A skin that cannot be
    manufactured is not a design.

    Cavity = the skin's outer face (so the former shapes the outside of the skin);
    core = the master pattern, which is the printed object itself. The silicone
    fills the gap between them. Two halves, because the pattern has to be seated
    and the cured skin lifted out; no draft is applied to the cavity, because what
    it releases from is the cured SKIN and that stretches — the one place in this
    module where the flexible-part argument works in the mould's favour rather
    than the cast's.

    TWO THINGS THE FORMER HAS TO DO THAT A HOLLOW BLOCK DOES NOT.

    1. **Form the breather windows.** The skin is the only thing between the cast and
       the atmosphere, and §11 of the design record puts the enclosed-skin cemented
       fraction at exactly 0.000 — windows are not a refinement, they are the whole
       reason the skin cements anything. But `windows` is a boolean cut applied to the
       skin OCCUPANCY, and the earlier former was built from `outer`, which is the
       un-windowed offset body. So it cast an unperforated skin: the geometry the
       aeration solve reported and the geometry the former produced were different
       objects, and the difference was the one that decides whether the design works.
       The window bores are therefore carried into the former as PILLARS spanning the
       gap, and the skin demoulds already perforated.

    2. **Hold the pattern.** The gap only exists while the pattern sits at exactly
       `skin_t` from the former's cavity wall, and nothing was holding it there — a
       loose pattern floats or sinks in the mix and the skin comes out wedge-shaped on
       one side and torn on the other. The pillars do this too, and they are the right
       feature for it: they already run from the former's outer wall to the pattern's
       surface, they are distributed over the whole body at the window pitch, and the
       marks they leave are the windows the skin needs anyway. A dedicated set of
       locating pins would leave a second set of holes to be patched.

    Pillars are taken on the OUTER skin only (`~form`). A bore through the lining of a
    hollow core would land inside the silhouette with no former material to attach to
    — a floating pillar, which prints as loose debris and fragments the part.
    """
    p = res["_parts"]
    pitch, origin, axis = res["pitch"], res["origin"], res["axis"]
    below, k_part = p["below"], res["k_part"]
    outer, obj, form = p["outer_body"], p["obj"], p["form"]
    windows, gate = p["windows"], p["gate"]
    one_piece = bool(res["hoop"]["one_piece_ok"])
    d_out = true_sdf(outer, pitch)
    X, Y, Z = np.meshgrid(*[origin[i] + np.arange(outer.shape[i]) * pitch
                            for i in range(3)], indexing="ij")

    # ---- wall SOLVED for the silicone head, not halved off the jacket's -----
    #
    # The old `max(8.0, 0.5 * jacket_wall)` was an unchecked literal inheriting a load
    # case that is not this one: the jacket is sized for a tamped 1890 kg/m3 mix at a
    # compaction factor of 10, and this part holds an untamped 1150 kg/m3 rubber head
    # at static pressure — about a sixteenth of the design pressure. Halving a wall
    # multiplies its deflection by eight, so the two errors happened to cancel on the
    # three shipped typologies and the result was reported as a number rather than as
    # a check. Worse, the GUI's deflection slider moved it the wrong way: raising the
    # target thinned the jacket and thinned this with it, while the load stayed put.
    #
    # The floor and the tolerance are the former's own too — see `SiliconeSpec`. Both
    # inherited values were about a mould carrying a tamped mix, and between them they
    # were setting this wall at four times what it needs.
    om = mould_auto.outline_metrics(
        mould_auto.outline_at(outer, axis, k_part), pitch,
        np.delete(np.asarray(origin, float), axis))
    kk = np.flatnonzero(np.moveaxis(outer, axis, -1).any(axis=(0, 1)))
    head_mm = float((kk.max() - kk.min() + 1) * pitch) if len(kk) else 0.0
    wall = mould_auto.auto_wall(
        2.0 * min(om["half_x"], om["half_y"]), head_mm,
        replace(auto, rho_mix=spec.rho_g_cm3 * 1000.0, compaction_factor=1.0,
                deflect_target_mm=spec.pour_deflect_mm,
                wall_min=spec.pour_wall_min))
    shellwall = wall["t_mm"]
    rim_t = max(shellwall, spec.pour_rim_min)
    rim_h = max(4.0, 3.0 * pitch)
    n_rim = max(1, int(round(rim_h / pitch)))
    sl_r = [slice(None)] * 3
    sl_r[axis] = slice(max(0, k_part - n_rim), k_part + n_rim)
    rim_zone = np.zeros(outer.shape, bool)
    rim_zone[tuple(sl_r)] = True
    envelope = (d_out <= shellwall) | ((d_out <= rim_t) & rim_zone)
    # THE FORMER IS PURELY EXTERNAL — `~form` is what makes it openable at all.
    #
    # Without it, `envelope & ~outer` lines the INSIDE of a hollow body too: on the
    # vessel that is a shellwall-thick plug sitting in the bore, connected to the rest
    # of the former only through a 16 mm aperture that the cavity behind it is several
    # times wider than. It is a classic trapped core — the release sweep flagged it and
    # nothing else would have, because it prints, meshes watertight and balances
    # exactly. It also cannot be dug out afterwards without destroying the skin.
    #
    # So the former shapes the outer skin and nothing else. The core lining is cast
    # against the separately printed `core` part, which the silicone path already
    # emits, and `core_lining_windows_unformed` continues to say that its windows are
    # hand-punched. This is also the second-largest plastic saving in the part.
    block_raw = envelope & ~outer & ~form
    # Cutting a hollow body's interior away can in principle leave the bowl beside
    # loose offcuts, so the bowl is taken as the one connected body and anything else
    # is named in the balance rather than dropped quietly. On all three shipped
    # typologies `orphan` measures 0 mm3 — this is a guard, not a fix, and the
    # fragmentation actually observed on the vessel came from the tongue instead (see
    # the tongue's own note). Kept because a silent extra body here would print as
    # debris and would not show up in any mesh check.
    _lab_b, _n_b = ndimage.label(block_raw, structure=ndimage.generate_binary_structure(3, 1))
    if _n_b > 1:
        _sz = np.bincount(_lab_b.ravel())
        _sz[0] = 0
        block = _lab_b == int(_sz.argmax())
        orphan = block_raw & ~block
    else:
        block, orphan = block_raw, np.zeros_like(block_raw)
    gap = outer & ~obj & ~form               # the OUTER skin, which is all it forms

    # ---- window pillars, on the OUTER skin, DRAW-AXIS FAMILY ONLY -----------
    #
    # `window_lattice` emits three axis-aligned bore families and the skin keeps all
    # three, because on the skin and the jacket a bore is a CUT. In the former it is
    # solid material, and a peg running across the draw cannot come out of the hole it
    # made: opening the halves shears it through the cured rubber. Measured with all
    # three families carried into the former, ~80 % of the vessel's pillar volume ran
    # transverse and `pour_release` interfered on both halves at step one — the
    # tool's own check catching a former that would have been printed, poured, cured
    # for a day and then found to be welded shut.
    #
    # So the former forms only the bores aligned with the pull. That is a real loss of
    # open area, not a free fix, and it is measured rather than absorbed: the rest of
    # the window set has to be punched by hand to the supplied skin STL, and
    # `open_area_formed_frac` says how much of it that is. The fill gate is a
    # transverse cylinder for the same reason and is likewise not formed — one hole,
    # cut by hand.
    d_win = float(res["window"]["d_mm"])
    d_obj = true_sdf(obj, pitch)
    axial = window_lattice(X, Y, Z, d=d_win, spacing=float(res["window"]["spacing_mm"]),
                           axes=(axis,), normals=surface_normals(d_obj, pitch))
    designed = (windows | gate) & gap        # `gap` is already outer-skin only
    raw = windows & axial & gap
    if spec.pour_clear > 0:
        # optional relief so the pillar tips stop short of the pattern. Default 0.0
        # on purpose: a pillar that does not touch leaves a film of rubber across the
        # window, and a window that does not go through is not a window. The cost of
        # 0.0 is a zero-allowance stack against print tolerance, which the MANIFEST
        # states rather than hides.
        raw = raw & (d_obj > spec.pour_clear)
    pillars, pil = span_pillars(raw, block, obj, pitch, designed=designed,
                                min_mm3=0.25 * np.pi / 4 * d_win ** 2 * spec.skin_t)

    # ---- parting membrane ---------------------------------------------------
    #
    # `hoop_strain_one_piece` decides whether the cured skin comes off as a single
    # glove, and on the vessel it does not: 163.5 % against a 62.5 % allowable, so
    # `export` and `_bundle_parts` both ship skin_lower + skin_upper. The former was
    # casting one continuous bag across the parting plane regardless — two STLs
    # describing pieces the supplied former could not make, and a pattern sealed
    # inside a closed rubber shell with no open area, which by this module's own
    # relation needs infinite strain to extract. A shim spanning the gap at the parting
    # plane casts the two halves the release analysis assumed.
    #
    # IT IS ITS OWN PART, not a feature of either half, and the release sweep is what
    # forced that. Attached to the upper half it sits directly UNDER the upper skin, so
    # lifting that half drives it up into the rubber; attached to the lower half it
    # sits directly OVER the lower skin, with the mirror-image problem. A disc between
    # two skin halves blocks a straight pull of whichever half carries it, which is why
    # the real process pours one half against a removable wall and then the other. So
    # it is emitted as `parting_plate`, lifted out at the parting plane once the upper
    # half is off — an annulus around the pattern, which clears because the parting
    # plane is at or near the widest section and the body narrows above it. That is a
    # straight pull, so it is swept like everything else rather than asserted.
    membrane = np.zeros_like(gap)
    if not one_piece:
        t_mem = max(1, int(round(max(2.0, 2.0 * pitch) / pitch)))
        sl = [slice(None)] * 3
        sl[axis] = slice(k_part, k_part + t_mem)
        band = np.zeros_like(gap)
        band[tuple(sl)] = True
        membrane = gap & band

    cavity = gap & ~pillars & ~membrane      # the silicone that actually gets cast

    # Spout and vent are PLACED, then verified to reach the cavity. A bore that
    # misses the gap is a former that cannot be filled, and it is invisible in a
    # render: the shell still prints, still passes a watertightness check, and the
    # silicone simply will not go in. So each bore is aimed at a measured point on
    # the cavity, and the realised open cross-section at the cavity is reported.
    #
    # THE BORES ARE CLIPPED TO ONE HALF. They were unbounded cylinders along the
    # working axis subtracted from BOTH halves, which put a matching hole in the
    # shell's floor directly under each one: measured on the tile, 1116 mm3 of
    # material removed below the lowest cavity slice in the spout column and nothing
    # left under it — a clean through-hole that pours the rubber onto the bench. The
    # part still printed, still meshed watertight, and still balanced exactly, because
    # `balance` was told the whole cylinder was a legitimate void.
    #
    # Clipping also makes the pour work when the skin is parted: the membrane seals
    # the two cavities from each other, so each needs its own fill route, and each
    # is fed from ITS OWN outer face (the lower half is inverted to pour it).
    o = [i for i in range(3) if i != axis]
    coords = (X, Y, Z)
    A, B = coords[o[0]], coords[o[1]]

    def _bore(ca, cb, d, side):
        return (np.sqrt((A - ca) ** 2 + (B - cb) ** 2) <= d / 2) & side

    def _aim(target, cands, d, side):
        """Pick the candidate whose bore actually intersects the cavity.

        The cavity CENTROID is the obvious aim point and it is wrong for any annular
        skin: the vessel's skin is a shell, so its centroid sits in the middle of the
        bore where there is no cavity at all, and the spout then opens into thin air.
        Measured on the shell: 0 mm3 of spout/cavity intersection. Candidates are
        therefore ranked by the open cross-section they actually deliver, and the
        target's own voxels are always among them so a hit is guaranteed to exist.
        """
        best, best_v = cands[0], -1.0
        for c in cands:
            v = float((_bore(c[0], c[1], d, side) & target).sum())
            if v > best_v:
                best, best_v = c, v
        return (float(best[0]), float(best[1])), best_v

    def _cands(target):
        idx = np.argwhere(target)
        if not len(idx):
            return [(0.0, 0.0)]
        pts = np.asarray(origin, float) + idx * pitch
        # the plan centroid (right for a solid pattern), plus a sample of real
        # cavity voxels near the top of the pour, where a spout belongs
        top = idx[idx[:, axis] >= np.percentile(idx[:, axis], 92)]
        step = max(1, len(top) // 64)
        out = [(float(pts[:, o[0]].mean()), float(pts[:, o[1]].mean()))]
        out += [(float(origin[o[0]] + i[o[0]] * pitch),
                 float(origin[o[1]] + i[o[1]] * pitch)) for i in top[::step]]
        return out

    # One fill route per CAVITY BODY. With a parting membrane there are two, sealed
    # from each other, and a single spout would fill one and starve the other.
    lab_c, n_c = ndimage.label(cavity)
    spout = np.zeros_like(cavity)
    ventv = np.zeros_like(cavity)
    side_union = np.zeros_like(cavity)   # every half a bore is allowed to touch
    routes = []
    for i in range(1, max(n_c, 1) + 1):
        comp = (lab_c == i) if n_c else cavity
        if not comp.any():
            continue
        # feed each body from the face of the half it mostly lives in
        up_side = int((comp & ~below).sum()) >= int((comp & below).sum())
        side = ~below if up_side else below
        cands = _cands(comp)
        (sx, sy), s_hit = _aim(comp, cands, spec.spout_d, side)
        (vx, vy), v_hit = _aim(comp, [c for c in cands if c != (sx, sy)] or cands,
                               spec.vent_d, side)
        sp, vt = _bore(sx, sy, spec.spout_d, side), _bore(vx, vy, spec.vent_d, side)
        spout |= sp
        ventv |= vt
        side_union |= side
        routes.append({
            "body": i, "half": "upper" if up_side else "lower",
            "volume_mm3": float(comp.sum() * pitch ** 3),
            "spout_xy": (sx, sy), "vent_xy": (vx, vy),
            "spout_open_mm3": float((sp & comp).sum() * pitch ** 3),
            "vent_open_mm3": float((vt & comp).sum() * pitch ** 3),
            "fed": bool((sp & comp).any() and (vt & comp).any())})

    # ---- tongue and groove, so the halves cannot shear ----------------------
    #
    # The former had no keys, no flange and no clamping feature of any kind, while the
    # jacket built from the same measurements gets three chiral keys and nine bolts.
    # Every millimetre the halves shear comes straight off the skin thickness on one
    # side and adds it on the other — on the very dimension the module insists must be
    # delivered rather than calibrated.
    #
    # An annular tongue rather than discrete keys, on purpose: it follows the body's
    # own outline, so it cannot land on a bore or a thin section, it cannot fragment,
    # and it doubles as the gasket land that stops uncured rubber flashing out of the
    # butt joint. It lives strictly inside the rim (0.30-0.65 of the wall), so it can
    # never intrude on the cavity.
    t_h = max(3.0, 2.0 * pitch)
    sl = [slice(None)] * 3
    sl[axis] = slice(k_part, k_part + max(1, int(round(t_h / pitch))))
    rim_band = np.zeros_like(gap)
    rim_band[tuple(sl)] = True
    # `~form` for the same reason the block has it, and it is not optional: `d_out` is
    # a distance from the skin-clad body, so a band of it exists INSIDE a hollow bore
    # as well as outside. Without the exclusion the vessel got a second tongue ring
    # sitting in its own bore — 14 loose fragments at 20-25 mm radius, all `in_form`
    # 1.00, which both fragmented the lower half and interfered with the release,
    # because they lie exactly where the cast body has to come out.
    tongue = ((d_out > 0.30 * rim_t) & (d_out <= 0.65 * rim_t)
              & rim_band & envelope & ~form)
    groove = ((d_out > 0.30 * rim_t - auto.key_clear)
              & (d_out <= 0.65 * rim_t + auto.key_clear)
              & rim_band & envelope & ~form)

    solid = block | pillars
    lower = ((solid & below) | tongue) & ~spout & ~ventv & ~membrane
    upper = (solid & ~below & ~groove) & ~spout & ~ventv & ~membrane
    plate = membrane & ~spout & ~ventv
    bal = balance(envelope, {"lower": lower, "upper": upper, "parting_plate": plate,
                             "pattern": obj, "skin_cavity": cavity},
                  {"spout": spout & ~obj, "vent": ventv & ~obj,
                   "registration_clearance": groove & ~tongue,
                   "orphan_shell": orphan,
                   "cast_hollow": form & ~obj}, pitch)

    # Does the shell stay CLOSED under every bore? The balance cannot tell — it was
    # handed the whole cylinder as a named void, so a bore that exits the far side is
    # attributed and reads exact, which is how an unbounded cylinder put a matching
    # hole in the floor under every spout and still reported `exact: True`. Measured
    # directly instead: material the bores took out of the half they do not belong to.
    holed = float((block & (spout | ventv) & ~side_union).sum() * pitch ** 3)
    reach = {"spout_reaches_cavity_mm3": float((spout & cavity).sum() * pitch ** 3),
             "vent_reaches_cavity_mm3": float((ventv & cavity).sum() * pitch ** 3),
             "cavity_bodies": int(n_c), "routes": routes,
             "wrong_side_removal_mm3": holed}
    reach["ok"] = bool(routes and all(r["fed"] for r in routes) and holed == 0.0)

    return {"lower": lower, "upper": upper, "pattern": obj, "parting_plate": plate,
            "balance": bal, "wall": wall, "wall_mm": shellwall, "rim_mm": rim_t,
            "plastic_cm3": float((lower | upper | plate).sum() * pitch ** 3 / 1000.0),
            "orphan_shell_mm3": float(orphan.sum() * pitch ** 3),
            "pattern_cm3": float(obj.sum() * pitch ** 3 / 1000.0),
            "spout_xy": routes[0]["spout_xy"] if routes else (0.0, 0.0),
            "vent_xy": routes[0]["vent_xy"] if routes else (0.0, 0.0),
            "reach": reach,
            "cavity_mm3": float(cavity.sum() * pitch ** 3),
            "membrane_mm3": float(membrane.sum() * pitch ** 3),
            "skin_parted_by_former": bool(not one_piece),
            "cavity_matches_skin": cavity_vs_skin(cavity, p, pitch),
            "registration": {"tongue_mm3": float(tongue.sum() * pitch ** 3),
                             "clearance_mm": auto.key_clear,
                             "height_mm": t_h,
                             "note": "annular tongue on the lower half, matching "
                                     "groove in the upper; also the gasket land"},
            "pillars": pillar_report(pil, lower, upper),
            "release": pour_release(lower, upper, plate, obj, obj | cavity, outer,
                                    axis, spec.release_steps),
            "procedure": (
                "clamp both halves shut around the pattern and pour through the "
                "spout" if one_piece else
                "the skin is parted at the membrane, so the two cavities are sealed "
                "from each other: each half is poured through its own spout, from "
                "its own outer face (invert the lower half to pour it)"),
            "note": ("cavity is the skin's outer face minus the window and gate "
                     "pillars and the parting membrane; the core is the printed "
                     "master pattern, which the pillars hold at the skin offset. No "
                     "draft, because what the former releases from is the cured skin "
                     "and that stretches")}


def cavity_vs_skin(cavity: np.ndarray, parts: dict, pitch: float) -> dict:
    """Does the former cast the skin the aeration solve was run on?

    These were different objects and the difference was the one that decides the
    design. `skin_all` carries the window, gate and vent cuts; the former was built
    off `outer_body`, which is the offset body BEFORE any of them, so the rubber that
    came out was the fully enclosing skin — the boundary condition this module scores
    at cemented fraction 0.000 — while the tool reported the windowed skin's 0.861.
    The mass quoted to the caster was short by the same 30 %.

    Compared on the OUTER skin only. The core lining is cast in the same pour but its
    bores have no former (a pillar inside the silhouette has nothing to attach to), so
    it is reported separately rather than folded into a number that would hide it.
    """
    v = pitch ** 3
    outer_skin = parts["skin_out"]
    cav_out = cavity & ~parts["form"]
    a, b = float(cav_out.sum() * v), float(outer_skin.sum() * v)
    return {"cavity_outer_mm3": a, "skin_out_mm3": b,
            "ratio": float(a / b) if b else 0.0,
            "core_lining_mm3": float(parts["skin_core"].sum() * v),
            "core_lining_windows_unformed": bool(parts["skin_core"].any()),
            "ok": bool(b > 0 and abs(a / b - 1.0) <= 0.05),
            "note": ("the former's outer cavity against the windowed skin the "
                     "aeration solve used; within 5 % means the rubber that demoulds "
                     "is the geometry that was scored")}


def pour_release(lower: np.ndarray, upper: np.ndarray, plate: np.ndarray,
                 pattern: np.ndarray, cast: np.ndarray,
                 solid_skin: np.ndarray, axis: int, steps: int) -> dict:
    """Can each former half be opened off the cured skin?

    The module's standard is that an untested release proves nothing, and the former
    was the one part never swept — its "no draft needed because the cured skin flexes"
    note was an assertion rather than a result.

    THE SWEEP TARGET IS `pattern | cavity`, NOT `outer_body`. What sits in the former
    when it is opened is the pattern plus the skin *as cast*, and that skin has a hole
    wherever a pillar stood. Sweeping against `outer_body` — the solid offset body —
    asks the pillars to pass through rubber that does not exist there, so every former
    with working pillars reports an interference. Measured on the vessel: `outer_body`
    interferes on both halves at step 1, `pattern | cavity` clears both.

    That distinction also supplies the DISCRIMINATION CONTROL this module demands, and
    for once it is not contrived: sweeping the same half against the *unperforated*
    skin MUST interfere, because the pillars occupy exactly the volume the windows
    remove. A former whose pillars are absent or too short passes both sweeps, and the
    control is what tells the two cases apart.

    A pass is a statement about a RIGID pull. The cured skin can also stretch, which
    only helps, so a clear result is conservative and an interference is real.
    """
    lo = mould.release_sweep(lower, cast, axis=axis, up=True, steps=steps)
    up = mould.release_sweep(upper, cast, axis=axis, up=False, steps=steps)
    c_lo = mould.release_sweep(lower, solid_skin, axis=axis, up=True, steps=steps)
    c_up = mould.release_sweep(upper, solid_skin, axis=axis, up=False, steps=steps)
    ctrl = bool(not c_lo["clears"] or not c_up["clears"])
    # The plate is lifted off the PATTERN once the upper half and its skin are away,
    # so it is swept against the bare pattern. It clears when the parting plane sits
    # at or above the widest section — which is where `analyse_parting` puts it, but
    # that is a consequence to be measured rather than an assumption to lean on.
    pl = (mould.release_sweep(plate, pattern, axis=axis, up=False, steps=steps)
          if plate.any() else {"clears": True, "first_interference_step": None,
                               "max_interfering_voxels": 0, "steps_tested": 0,
                               "note": "no parting plate — the skin is one piece"})
    return {"lower": lo, "upper": up, "parting_plate": pl,
            "control_unperforated_skin": {"lower": c_lo, "upper": c_up},
            "control_interferes": ctrl,
            "clears": bool(lo["clears"] and up["clears"] and pl["clears"]),
            "ok": bool(lo["clears"] and up["clears"] and pl["clears"] and ctrl),
            "order": ["lift the upper former half off",
                      "peel the upper skin off the pattern",
                      "lift the parting plate off over the pattern",
                      "lift the lower former half off",
                      "peel the lower skin off the pattern"],
            "note": ("straight pull of each former half off the pattern plus the "
                     "as-cast skin, and of the parting plate off the pattern; the "
                     "control against an unperforated skin must interfere, or the "
                     "pillars are not there")}


def span_pillars(raw: np.ndarray, block: np.ndarray, obj: np.ndarray, pitch: float,
                 *, min_mm3: float, designed: np.ndarray | None = None) -> tuple:
    """Keep only the bores that actually BRIDGE former wall to pattern.

    A pillar has to do two things and a bore does neither by construction. It must be
    rooted in the former's own wall, or it prints as a loose peg lying in the cavity;
    and it must land on the pattern, or it leaves a blind pocket, so the skin comes
    out dimpled where a window should be and the aeration the solve reported never
    materialises. Neither failure is visible in a render.

    They are common, not hypothetical. `window_lattice` keeps a bore family only where
    it runs within ~57 deg of the surface normal, and the accepted set is further cut
    by the key/bolt keepout and the fill gate — so a bore can survive as a handful of
    isolated voxels in the middle of the gap, touching nothing. Measured on the vessel
    before this filter: 114 pillar bodies, of which 46 reached no wall and 34 reached
    no pattern, and the resulting former exported as 23 lower and 25 upper STLs, all
    but one of each a sub-cubic-centimetre speck.

    Rejected bores are returned to the cavity, so the silicone simply fills them and
    the skin has no window there. That is a real loss of open area, so it is counted
    rather than absorbed: `formed_frac` is what the caster actually gets against what
    the aeration solve assumed.
    """
    v = pitch ** 3
    struct = ndimage.generate_binary_structure(3, 1)
    des_mm3 = float(designed.sum() * v) if designed is not None else 0.0
    lab, n = ndimage.label(raw, structure=struct)
    if n == 0:
        return raw, {"n": 0, "n_raw": 0, "dropped": 0, "volume_mm3": 0.0,
                     "dropped_mm3": 0.0, "formed_frac": 0.0,
                     "formed_vol_frac": 0.0, "designed_mm3": des_mm3,
                     "open_area_formed_frac": 0.0, "ok": False,
                     "halves_connected": True,
                     "note": "NO window pillars — the former would cast an "
                             "unperforated skin, which the aeration solve puts at "
                             "0.000 cemented fraction"}
    sizes = np.bincount(lab.ravel())
    rooted = set(np.unique(lab[ndimage.binary_dilation(block, struct) & raw])) - {0}
    landed = set(np.unique(lab[ndimage.binary_dilation(obj, struct) & raw])) - {0}
    keep = [i for i in range(1, n + 1)
            if i in rooted and i in landed and sizes[i] * v >= min_mm3]
    out = np.isin(lab, keep) if keep else np.zeros_like(raw)
    kept_mm3 = float(out.sum() * v)
    lost_mm3 = float((raw & ~out).sum() * v)
    return out, {
        "n": len(keep), "n_raw": int(n), "dropped": int(n - len(keep)),
        "volume_mm3": kept_mm3, "dropped_mm3": lost_mm3,
        "formed_frac": float(len(keep) / n),
        # BY VOLUME is the number to read. Most rejects are slivers a bore leaves
        # where it grazes the gap, so the count fraction understates badly: on the
        # vessel 78 of 114 bodies are dropped (0.32 by count) and they carry 11 % of
        # the bore volume (0.89 formed). What the caster loses is open area, which
        # scales with volume, not with how many pieces the labeller found.
        "formed_vol_frac": float(kept_mm3 / max(kept_mm3 + lost_mm3, 1e-9)),
        # …and this is what the CASTER loses. `formed_vol_frac` is the yield of the
        # bore family the former is allowed to carry; `open_area_formed_frac` is that
        # family against every window the skin was designed with, so it counts the
        # transverse bores the former cannot form at all. The remainder has to be
        # punched by hand to the supplied skin STL.
        "designed_mm3": des_mm3,
        "open_area_formed_frac": float(kept_mm3 / des_mm3) if des_mm3 else 0.0,
        "min_printable_mm3": float(min_mm3),
        "unrooted": int(n - len(rooted)), "missed_pattern": int(n - len(landed)),
        "ok": bool(kept_mm3 > 0 and kept_mm3 / max(kept_mm3 + lost_mm3, 1e-9) >= 0.75),
        "note": ("each kept pillar spans the skin gap — rooted in the former's wall, "
                 "landing on the pattern — so it locates the pattern at the skin "
                 "offset and forms one breather window. Dropped bores are not formed: "
                 "the skin has no window there")}


def pillar_report(pil: dict, lower: np.ndarray, upper: np.ndarray) -> dict:
    """`span_pillars`'s findings plus the one thing only the finished halves can say.

    Adding material to a part can only ever join it, but the split into halves is
    taken afterwards — so a fragmented half means something in the pillar set, the
    membrane or the bores left debris, and that is the check the mesh route cannot
    make (`occ_to_mesh` keeps the largest body and reports the rest as watertight).
    """
    struct = ndimage.generate_binary_structure(3, 1)
    n_lo = int(ndimage.label(lower, structure=struct)[1])
    n_up = int(ndimage.label(upper, structure=struct)[1])
    return {**pil, "bodies_lower": n_lo, "bodies_upper": n_up,
            "halves_connected": bool(n_lo <= 1 and n_up <= 1),
            "ok": bool(pil["ok"] and n_lo <= 1 and n_up <= 1)}


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def part_topology(occ: np.ndarray, mesh_euler: int, pitch: float, *,
                  expect_bodies: int = 1) -> dict:
    """Voxel Betti numbers vs mesh genus — two independent routes to one integer.

    For a 3D solid chi = b0 - b1 + b2 (components, tunnels, enclosed cavities), so
    the tunnel count comes off the voxel grid without touching the mesh; a
    single-body closed surface has genus (2 - chi_mesh)/2, which must equal b1. When
    they disagree the mesh is not a faithful record of the set.

    `b0` against its EXPECTED value is the check that matters most, and it is the one
    no mesh test performs. Expected is not always 1: a hollow-core block's cavity
    lining is legitimately one piece per core (measured on the 2-core block: two
    linings of 368953 and 365078 mm3 centred at x = -86.5 and +86.2 mm), and each is
    demoulded separately. Condemning that as fragmentation would reject a correct
    design, so the caller passes `expect_bodies`.
    `mould.occ_to_mesh(largest=True)` keeps the biggest body, so a fragmented part
    exports as a watertight, winding-consistent, single-body STL with pieces missing.
    Measured: an unrestricted bore lattice cut the block's skin into 54 pieces, and
    the export reported every mesh check as passing while silently dropping 53 of
    them. The volume signal was tiny — 1936923 of 2020438 mm3 retained, so only
    4.1 % went missing — because the lost fragments are thin slivers. That is why
    a volume-ratio tolerance would not have caught it, and why `fragment_loss_frac`
    is reported
    alongside `bodies_voxel` but the PASS condition is `b0 == 1`: the volume signal
    is too small to trip a tolerance, and the component count is unambiguous.
    """
    from skimage.measure import euler_number as vox_euler
    b0 = int(ndimage.label(occ)[1])
    lbl, n = ndimage.label(np.pad(~occ, 1, constant_values=True))
    b2 = int(n - 1)
    chi = int(vox_euler(occ, connectivity=3))
    b1 = b0 + b2 - chi
    genus = int((2 - mesh_euler) // 2)
    sizes = np.bincount(ndimage.label(occ)[0].ravel())[1:]
    frag = float(1.0 - sizes.max() / max(sizes.sum(), 1)) if len(sizes) else 0.0
    # The mesh genus is only comparable to b1 for a SINGLE body — `occ_to_mesh` keeps
    # the largest, so on a legitimately multi-piece part the two numbers describe
    # different objects and the comparison is skipped rather than failed.
    single = b0 == 1
    return {"bodies_voxel": b0, "expect_bodies": int(expect_bodies),
            "tunnels_b1": b1, "cavities_b2": b2,
            "chi_voxel": chi, "genus_mesh": genus,
            "fragment_loss_frac": frag,
            "bodies_as_expected": bool(b0 == expect_bodies),
            "genus_matches_tunnels": (bool(b1 == genus) if single else None),
            "ok": bool(b0 == expect_bodies and (b1 == genus if single else True))}


def export(res: dict, out_dir, prefix: str) -> dict:
    """Mesh and write every part. Only 'obj_mesh' is a Trimesh in these dicts —
    everything else is a voxel OCCUPANCY grid, so each part goes through
    `mould.occ_to_mesh`. Iterating for objects that have `.export` writes the cast
    object and silently nothing else.

    Every part is topology-checked against its own voxel set (`part_topology`)
    BEFORE its mesh numbers are believed, because the mesh route cannot see
    fragmentation.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p, pitch, origin = res["_parts"], res["pitch"], res["origin"]
    want = {"jacket_lower": p["jacket_lower"], "jacket_upper": p["jacket_upper"]}
    if res["hoop"]["one_piece_ok"]:
        want["skin"] = p["skin_all"]
    else:
        want["skin_lower"] = p["skin_all"] & p["below"]
        want["skin_upper"] = p["skin_all"] & ~p["below"]
    if p["core"].any():
        want["core"] = p["core"]
    want["pour_shell_lower"] = res["pour_shell"]["lower"]
    want["pour_shell_upper"] = res["pour_shell"]["upper"]
    want["parting_plate"] = res["pour_shell"]["parting_plate"]
    # The MASTER PATTERN, which the former is useless without. The former's cavity is
    # the skin's outer face and the silicone fills the gap between it and the pattern;
    # with no pattern in the box there is no gap, and the pour yields a solid rubber
    # copy of the whole body instead of a mould. It was generated all along (`obj`,
    # the grammar's own field) and was the one part never written out.
    want["pattern"] = res["_parts"]["obj"]

    # One cavity lining per hollow core, so its expected component count is the
    # number of cores. Emitted as its own part because it demoulds differently from
    # the outer skin (squeezed out through the cast's aperture, not peeled off).
    expect = {}
    if p["skin_core"].any():
        n_lin = int(ndimage.label(p["skin_core"])[1])
        want["skin_core_lining"] = p["skin_core"]
        expect["skin_core_lining"] = n_lin
        if "skin" in want:
            want["skin"] = p["skin_out"]
        else:
            want["skin_lower"] = p["skin_out"] & p["below"]
            want["skin_upper"] = p["skin_out"] & ~p["below"]
    rows = {}
    for name, occ in want.items():
        if not occ.any():
            rows[name] = {"skipped": True, "reason": "empty set"}
            continue
        n_exp = expect.get(name, 1)
        # A legitimately multi-piece part must be written as ONE FILE PER PIECE.
        # `mould.occ_to_mesh(largest=True)` keeps the biggest body, so meshing the
        # union of the block's two cavity linings would silently ship one of them —
        # the same class of loss the fragmentation check exists to catch, arriving
        # this time through a correct design rather than a broken one.
        lab, nlab = ndimage.label(occ)
        if n_exp > 1 and nlab > 1:
            sizes = np.bincount(lab.ravel())
            order = sorted(range(1, nlab + 1), key=lambda i: -sizes[i])
            files, meshes = [], []
            for j, i in enumerate(order, start=1):
                mj = mould.occ_to_mesh(lab == i, origin, pitch)
                fj = out_dir / f"{prefix}_{name}_{j}.stl"
                mj.export(fj)
                files.append(str(fj))
                meshes.append(mould.mesh_report(mj))
            topo = part_topology(occ, meshes[0]["euler_number"], pitch,
                                expect_bodies=n_exp)
            v_mesh = float(sum(m_["volume_mm3"] for m_ in meshes))
            rows[name] = {
                "files": files, "n_pieces": len(files), "pieces": meshes,
                "watertight": all(m_["watertight"] for m_ in meshes),
                "winding_consistent": all(m_["winding_consistent"] for m_ in meshes),
                "volume_mm3": v_mesh, "topology": topo,
                "occ_mm3": float(occ.sum() * pitch ** 3),
                "mesh_vs_occ_volume_frac": float(
                    v_mesh / max(occ.sum() * pitch ** 3, 1e-9)),
                "ok": bool(all(m_["watertight"] and m_["winding_consistent"]
                               for m_ in meshes) and topo["ok"]
                           and len(files) == n_exp)}
            continue
        m = mould.occ_to_mesh(occ, origin, pitch)
        f = out_dir / f"{prefix}_{name}.stl"
        m.export(f)
        mr = mould.mesh_report(m)
        topo = part_topology(occ, mr["euler_number"], pitch, expect_bodies=n_exp)
        rows[name] = {"file": str(f), "n_pieces": 1, **mr, "topology": topo,
                      "occ_mm3": float(occ.sum() * pitch ** 3),
                      "mesh_vs_occ_volume_frac": float(mr["volume_mm3"]
                                                       / max(occ.sum() * pitch ** 3, 1e-9)),
                      "ok": bool(mr["watertight"] and mr["winding_consistent"]
                                 and topo["ok"])}
    return rows
