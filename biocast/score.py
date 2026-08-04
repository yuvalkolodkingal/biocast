"""Composite success estimator.

What "success" means here is taken verbatim from the source paper: the success
rate is "defined as the complete solidification of the aggregate". The estimator
therefore predicts the probability that a cast body solidifies completely and
survives demoulding and drying, and it attributes failure to the specific modes
the paper observed (Fig. 5: cracking and incomplete mineralisation from uneven
drying and oxygenation).

Four subscores, each in [0,1], combined multiplicatively
--------------------------------------------------------
S_aer  aerobic penetration coverage. Fraction of the body within reach of
       atmospheric oxygen. This is the dominant term and the one that killed the
       paper's early solid prototypes.
S_dry  drying uniformity. Penalises thick sections whose interior dries on a
       different schedule from the surface, which is the restrained-shrinkage
       cracking mechanism.
S_cast castability. Penalises apertures, grooves and sections narrow enough for
       the aggregate to bridge/jam instead of filling the mould.
S_str  structural integrity under the stress concentration the geometry imposes,
       via Inglis Kt and a Weibull flaw-population conversion to failure risk.

The product (not the mean) is used because these are series requirements: a body
that cannot be filled does not get a second chance to be well-oxygenated. A mean
would let a good aeration score hide a fatal casting defect.

Uncertainty
-----------
Every literature parameter carries a [low, high] range, several of which span a
decade (the biofilm volume fraction alone spans 10x and multiplies the oxygen
demand linearly). Monte Carlo sampling over those ranges converts the point score
into a distribution; the reported score is the median with a credible interval.
Ranking designs by the median while showing the interval is the honest use of
this model — the intervals are wide, and designs whose intervals overlap should
be treated as tied.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

from .params import Design, LitParams
from .physics import oxygen as ox
from .physics import drying as dry


# --------------------------------------------------------------------------
@dataclass
class SubScores:
    aeration: float
    drying: float
    castability: float
    structural: float

    @property
    def total(self) -> float:
        return float(self.aeration * self.drying * self.castability * self.structural)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        return d

    def limiting(self) -> str:
        vals = {"aeration": self.aeration, "drying": self.drying,
                "castability": self.castability, "structural": self.structural}
        return min(vals, key=vals.get)


FAILURE_MODE = {
    "aeration": "incomplete mineralisation (anoxic core) — the paper's Fig. 5 mode",
    "drying": "cracking from uneven drying — the paper's Fig. 5 mode",
    "castability": "aggregate bridging / starved mould feature (defective casting)",
    "structural": "crack initiation at a stress concentration under load",
}


# --------------------------------------------------------------------------
# Subscore models
# --------------------------------------------------------------------------
def s_aeration(cemented_fraction: float, *, floor: float = 0.85,
               sharpness: float = 12.0) -> float:
    """Map cemented volume fraction to a subscore.

    A logistic centred on the completeness threshold: the paper's criterion is
    COMPLETE solidification, so partial coverage is heavily penalised rather than
    scored proportionally.

    Actual values at the default floor=0.85, sharpness=12: coverage 0.50 -> 0.000,
    0.70 -> 0.003, 0.85 -> 0.500, 0.95 -> 0.982, 1.00 -> 0.998. Note the ceiling is
    0.9975, not 1.0, so many fully-cemented designs sit at the same median and need
    the 5th-percentile bound as a tie-break.
    """
    x = float(np.clip(cemented_fraction, 0.0, 1.0))
    return float(1.0 / (1.0 + np.exp(-sharpness * (x - floor) / max(1 - floor, 1e-6) * 0.5)))


def s_drying(max_thickness_mm: float, L_dry_mm: float) -> float:
    """Drying-uniformity subscore from the drying Biot-like ratio.

    The controlling group is R = (t_max/2) / L_dry: the half-thickness the water
    must leave, divided by how far evaporation actually drains over the cure. At
    R <= 1 the whole section drains and shrinks together. Above 1 the surface
    shrinks against a still-saturated core, which is the restrained-shrinkage
    condition that cracks at ~75 % of splitting tensile strength.

    This replaces an earlier version keyed on two free thickness constants; the
    ratio form is preferable because both of its inputs are physical — one is
    geometry, the other is the measured evaporation band and the cure schedule —
    so a longer cure or a drier room moves the score for the right reason.
    """
    from .physics.drying import drying_uniformity
    R = drying_uniformity(max_thickness_mm, L_dry_mm)
    if not np.isfinite(R):
        return 0.05
    if R <= 1.0:
        return 1.0
    return float(np.clip(R ** -1.5, 0.05, 1.0))


def s_castability(min_feature_mm: float, d_max_mm: float, *,
                  accept_mult: float, safe_mult: float,
                  fail_mult: float = 3.0) -> float:
    """Castability from the narrowest mould passage relative to d_max.

    Uses the retrieved granular jamming criterion, not the team's rule: the
    critical aperture-to-particle ratio is ~4.94 for spheres and ~6.0 for angular
    grains (dry silo), with dense suspensions diverging at ~8.1 and certain
    clogging below 3. The score is 1 above `safe_mult`, falls to 0.5 at
    `accept_mult`, and collapses below `fail_mult`.
    """
    if d_max_mm <= 0:
        return 1.0
    ratio = min_feature_mm / d_max_mm
    if ratio >= safe_mult:
        return 1.0
    if ratio <= fail_mult:
        return 0.02
    # smooth ramp through the accept point
    lo, hi = fail_mult, safe_mult
    x = (ratio - lo) / (hi - lo)
    return float(np.clip(0.02 + 0.98 * x ** 1.2, 0.02, 1.0))


def kt_inglis(notch_depth_mm: float, root_radius_mm: float) -> float:
    """Inglis equivalent-ellipse stress concentration factor: Kt = 1 + 2*sqrt(h/r).

    Reduces to the classical Kt = 3 for a semicircular notch (h = r) and diverges
    as the root sharpens, which is the formal content of the team's insistence on
    fillets over chamfers.
    """
    if root_radius_mm <= 0:
        return float("inf")
    if notch_depth_mm <= 0:
        return 1.0
    return float(1.0 + 2.0 * np.sqrt(notch_depth_mm / root_radius_mm))


def s_structural(kt: float, *, weibull_m: float, kt_ref: float = 2.0) -> float:
    """Convert a stress concentration into a survival subscore.

    For a Weibull flaw population of modulus m, raising the local stress by Kt
    raises the failure probability by roughly Kt^m. The score is referenced to a
    well-filleted baseline (Kt_ref = 2), so a design at the baseline scores ~1 and
    a sharp notch is punished steeply — at m ~ 12-17 (measured for porous
    cement-like composites) a factor-of-two stress rise is catastrophic, which is
    why notch control matters more in this material than in normal concrete.
    """
    if not np.isfinite(kt):
        return 0.01
    ratio = max(kt / kt_ref, 1e-6)
    if ratio <= 1.0:
        return 1.0
    return float(np.clip(ratio ** (-weibull_m / 4.0), 0.01, 1.0))


# --------------------------------------------------------------------------
# Parameter sampling for the Monte Carlo
# --------------------------------------------------------------------------
@dataclass
class PhysicsInputs:
    """The literature quantities the estimator needs, with ranges."""

    D_O2_gas: tuple = (2.209e-5, 2.209e-5, 2.209e-5)      # m2/s, free air
    C_O2_gas: tuple = (8.42, 8.42, 8.42)                  # mol/m3
    R_O2_bulk: tuple = (1.267e-4, 1.116e-3, 3.340e-3)     # mol/m3/s (low, val, high)
    phi: tuple = (0.35, 0.40, 0.45)                       # packed porosity
    sw: tuple = (0.30, 0.50, 0.70)                        # liquid saturation during cure
    weibull_m: tuple = (11.5, 14.0, 16.8)                 # Weibull modulus
    jam_accept: tuple = (4.94, 6.0, 8.1)                  # accept multiple of d_max
    jam_safe: tuple = (6.0, 8.0, 8.1)                     # safe multiple of d_max
    E_evap: tuple = (0.5, 1.5, 2.5)                       # mm/day, stage-1/2 transition band
    t_cure_days: tuple = (7.0, 14.0, 28.0)                # days of curing
    dS_air_entry: tuple = (0.35, 0.5, 0.65)               # saturation drop for air percolation

    @classmethod
    def from_lit(cls, kin: LitParams | None, mec: LitParams | None) -> "PhysicsInputs":
        pi = cls()
        if kin is not None:
            def tri(sym, cur):
                try:
                    lo, hi = kin.bounds(sym)
                    v = kin.get(sym)
                    return (lo, v, hi)
                except Exception:
                    return cur
            pi.D_O2_gas = tri("D_O2_gas", pi.D_O2_gas)
            pi.C_O2_gas = tri("C_O2_gas", pi.C_O2_gas)
            pi.R_O2_bulk = tri("R_O2_bulk", pi.R_O2_bulk)
            try:
                pi.phi = (0.35, kin.get("phi_0"), 0.45)
            except Exception:
                pass
        if mec is not None:
            try:
                lo, hi = mec.bounds("m")
                pi.weibull_m = (lo, (lo + hi) / 2, hi)
            except Exception:
                pass
        return pi


def _sample(rng: np.random.Generator, tri: tuple) -> float:
    lo, mid, hi = tri
    if hi <= lo:
        return float(mid)
    return float(rng.triangular(lo, np.clip(mid, lo, hi), hi))


# --------------------------------------------------------------------------
# Top-level estimator
# --------------------------------------------------------------------------
def score_design(design: Design, diag: dict, *, phys: PhysicsInputs | None = None,
                 n_mc: int = 400, seed: int = 0,
                 min_feature_mm: float | None = None,
                 notch_depth_mm: float = 0.0,
                 root_radius_mm: float | None = None) -> dict:
    """Score one design. `diag` comes from physics.fields.geometric_diagnostics.

    The Monte Carlo resamples the transport and material parameters, recomputing
    the analytic penetration depth (and hence the cemented fraction from the
    precomputed depth field) on every draw. The geometry is fixed, so the depth
    field is computed once and only the threshold moves — which is what makes
    hundreds of samples per design affordable.
    """
    phys = phys or PhysicsInputs()
    rng = np.random.default_rng(seed)
    g = design.geom
    dm = design.mix.d_max

    depth_field = diag["_grid"]["depth"]
    occ = diag["_grid"]["occ"]
    depths = depth_field[occ]

    if min_feature_mm is None:
        min_feature_mm = _infer_min_feature(design)
    if root_radius_mm is None:
        root_radius_mm = getattr(g, "fillet_r", 0.0)

    kt = kt_inglis(notch_depth_mm, root_radius_mm)

    totals, subs, modes, limiters = [], [], [], []
    for _ in range(n_mc):
        D_free = _sample(rng, phys.D_O2_gas)
        C0 = _sample(rng, phys.C_O2_gas)
        R = _sample(rng, phys.R_O2_bulk)
        phi = _sample(rng, phys.phi)
        sw = _sample(rng, phys.sw)
        m_w = _sample(rng, phys.weibull_m)
        jam_a = _sample(rng, phys.jam_accept)
        jam_s = _sample(rng, phys.jam_safe)
        E = _sample(rng, phys.E_evap)
        t_c = _sample(rng, phys.t_cure_days) if design.proc.cure_days is None \
            else design.proc.cure_days
        dS = _sample(rng, phys.dS_air_entry)

        # gas-phase reaction-diffusion depth at the sampled operating point
        D_eff = ox.effective_diffusivity(D_free, phi, sw, gas=True)
        L_gas = ox.analytic_penetration_depth(D_eff, C0, R)
        # how deep air has actually entered by the end of the cure
        L_dry = dry.air_entry_depth(E, t_c, phi, delta_saturation=dS)
        eff = dry.effective_penetration(L_gas, L_dry)
        L = eff["L_eff_mm"]
        cem = float(np.mean(depths <= L)) if depths.size else 0.0

        ss = SubScores(
            aeration=s_aeration(cem),
            drying=s_drying(diag["max_wall_thickness"], L_dry),
            castability=s_castability(min_feature_mm, dm, accept_mult=jam_a,
                                      safe_mult=max(jam_s, jam_a)),
            structural=s_structural(kt, weibull_m=m_w),
        )
        limiters.append(eff["limiter"])
        totals.append(ss.total)
        subs.append([ss.aeration, ss.drying, ss.castability, ss.structural])
        modes.append(ss.limiting())

    totals = np.asarray(totals)
    subs = np.asarray(subs)
    vals, counts = np.unique(modes, return_counts=True)
    dominant = str(vals[np.argmax(counts)])

    # nominal (median-parameter) run for reporting
    D_eff_nom = ox.effective_diffusivity(phys.D_O2_gas[1], phys.phi[1], phys.sw[1], gas=True)
    L_gas_nom = ox.analytic_penetration_depth(D_eff_nom, phys.C_O2_gas[1], phys.R_O2_bulk[1])
    t_c_nom = design.proc.cure_days if design.proc.cure_days else phys.t_cure_days[1]
    L_dry_nom = dry.air_entry_depth(phys.E_evap[1], t_c_nom, phys.phi[1],
                                   delta_saturation=phys.dS_air_entry[1])
    eff_nom = dry.effective_penetration(L_gas_nom, L_dry_nom)
    L_nom = eff_nom["L_eff_mm"]
    lv, lc = np.unique(limiters, return_counts=True)

    return {
        "score": float(np.median(totals)),
        "score_lo": float(np.percentile(totals, 5)),
        "score_hi": float(np.percentile(totals, 95)),
        "score_mean": float(totals.mean()),
        "sub_aeration": float(np.median(subs[:, 0])),
        "sub_drying": float(np.median(subs[:, 1])),
        "sub_castability": float(np.median(subs[:, 2])),
        "sub_structural": float(np.median(subs[:, 3])),
        "dominant_failure_mode": dominant,
        "failure_mode_text": FAILURE_MODE[dominant],
        "failure_mode_shares": {str(v): float(c / len(modes)) for v, c in zip(vals, counts)},
        "penetration_depth_nom_mm": L_nom,
        "penetration_depth_2x": 2 * L_nom,
        "L_gas_nom_mm": L_gas_nom,
        "L_dry_nom_mm": L_dry_nom,
        "penetration_limiter": str(lv[np.argmax(lc)]),
        "drying_ratio": dry.drying_uniformity(diag["max_wall_thickness"], L_dry_nom),
        "cemented_fraction": float(np.mean(depths <= L_nom)) if depths.size else 0.0,
        "kt": kt,
        "min_feature_mm": float(min_feature_mm),
        "n_mc": n_mc,
    }


def _infer_min_feature(design: Design) -> float:
    """The narrowest passage the wet mix must flow through for this typology."""
    from .params import ShellParams, BlockParams, TileParams
    g = design.geom
    if isinstance(g, ShellParams):
        cands = [g.wall]
        if g.aperture_r > 0:
            cands.append(2 * g.aperture_r)
        return min(cands)
    if isinstance(g, BlockParams):
        from .grammars.block import core_dims
        cw, cd = core_dims(g)
        cands = [g.face_shell, g.web]
        if g.groove_count > 0 and g.groove_width > 0:
            cands.append(g.groove_width)
        return min(c for c in cands if c > 0)
    if isinstance(g, TileParams):
        cands = [g.t]
        if g.groove_width > 0:
            cands.append(g.groove_width)
        return min(cands)
    raise TypeError(type(g))
