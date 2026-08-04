"""Hard-constraint rule set for castable, bio-cementable geometry.

Each rule is a predicate returning a `Verdict`. Rules are tagged by origin:

  TEAM  - the project's own stated design rules (notes of 19/07/2026)
  LIT   - threshold taken from retrieved literature / standards
  STD   - masonry or paving standard
  GEOM  - self-consistency of the parameter set (a rule that prevents
          nonsensical geometry rather than a physical failure)

The checker is deliberately conservative: `severity="fail"` means the design is
rejected before mould printing, which is exactly the trial-and-error saving the
team asked for. `severity="warn"` means castable but suboptimal.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable

from .params import Design, ShellParams, BlockParams, TileParams


@dataclass
class Verdict:
    rule: str
    origin: str
    passed: bool
    severity: str          # "fail" | "warn"
    value: float | None
    limit: float | None
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Thresholds. Defaults are the team's numbers; `lit` overrides where a
# retrieved source gives a better-grounded value.
# --------------------------------------------------------------------------
@dataclass
class Thresholds:
    fillet_mult_min: float = 1.5      # r >= 1.5 * d_max        (TEAM)
    fillet_mult_good: float = 2.0     # r >= 2.0 * d_max        (TEAM, preferred)
    groove_w_mult_min: float = 2.0    # w >= 2 * d_max          (TEAM)
    groove_w_mult_good: float = 3.0   # w >= 3 * d_max          (TEAM, safe)
    jam_ratio: float = 4.0            # aperture/particle below which granular flow jams (LIT)
    groove_depth_frac_block: float = 1.0 / 3.0   # h <= t/3     (TEAM)
    groove_depth_frac_tile: float = 1.0 / 4.0    # h <= t/4     (TEAM, thick tile)
    min_section_mult: float = 3.0     # min wall thickness >= 3 * d_max (LIT/ASSUMED)
    face_shell_min: float = 32.0      # mm, ASTM C90-type minimum for the 200 module (STD)
    web_min: float = 25.0             # mm (STD)
    void_frac_lo: float = 0.40        # CMU void fraction band (TEAM/STD)
    void_frac_hi: float = 0.50
    joint_min: float = 3.0            # mm, minimum joint for feed/O2 passage (TEAM, unquantified)
    penetration_min_frac: float = 0.85  # fraction of body volume that must be cementable
    sa_vol_min: float = 0.030         # mm^-1, minimum surface-area-to-volume for aerobic growth
    relief_frac_tile_max: float = 0.10  # Panot relief < 10 % of thickness (TEAM/STD)
    draft_min_deg: float = 1.0        # mould release draft

    @classmethod
    def from_lit(cls, lit) -> "Thresholds":
        """Override defaults with retrieved values where available."""
        t = cls()
        # granular jamming / bridging aperture ratio
        for key in ("jam_ratio", "W_over_d_jam", "aperture_ratio_jamming",
                    "bridging_ratio", "D_over_d_jam"):
            if lit is not None and key in lit:
                t.jam_ratio = lit.get(key)
                break
        for key in ("min_section_over_dmax", "clear_spacing_over_dmax"):
            if lit is not None and key in lit:
                t.min_section_mult = lit.get(key)
                break
        for key in ("face_shell_min_mm", "face_shell_thickness_min"):
            if lit is not None and key in lit:
                t.face_shell_min = lit.get(key)
                break
        for key in ("web_min_mm", "web_thickness_min"):
            if lit is not None and key in lit:
                t.web_min = lit.get(key)
                break
        return t


# --------------------------------------------------------------------------
# Rule implementations
# --------------------------------------------------------------------------
def _v(rule, origin, passed, severity, value, limit, msg) -> Verdict:
    return Verdict(rule, origin, bool(passed), severity, value, limit, msg)


def _fillet_rules(d: Design, th: Thresholds) -> list[Verdict]:
    g, dm = d.geom, d.mix.d_max
    r = getattr(g, "fillet_r", 0.0)
    hard, good = th.fillet_mult_min * dm, th.fillet_mult_good * dm
    out = [
        _v("fillet_radius_min", "TEAM", r >= hard - 1e-9, "fail", r, hard,
           f"fillet r={r:.1f} mm must be >= {th.fillet_mult_min}x d_max = {hard:.1f} mm; "
           "sharp corners initiate cracks in brittle bio-cement"),
        _v("fillet_radius_preferred", "TEAM", r >= good - 1e-9, "warn", r, good,
           f"fillet r={r:.1f} mm below preferred {th.fillet_mult_good}x d_max = {good:.1f} mm"),
    ]
    return out


def _section_rules(d: Design, th: Thresholds) -> list[Verdict]:
    g, dm = d.geom, d.mix.d_max
    lim = th.min_section_mult * dm
    out = []
    if isinstance(g, ShellParams):
        out.append(_v("min_section_thickness", "LIT", g.wall >= lim, "fail", g.wall, lim,
                      f"shell wall {g.wall:.1f} mm must be >= {th.min_section_mult}x d_max "
                      f"= {lim:.1f} mm to pack aggregate without arching"))
        # the fillet must fit inside the wall
        out.append(_v("fillet_fits_wall", "GEOM", g.fillet_r <= g.wall, "warn",
                      g.fillet_r, g.wall,
                      f"fillet r={g.fillet_r:.1f} mm exceeds wall {g.wall:.1f} mm"))
        if g.aperture_r > 0:
            out.append(_v("aperture_not_jamming", "LIT",
                          2 * g.aperture_r >= th.jam_ratio * dm, "fail",
                          2 * g.aperture_r, th.jam_ratio * dm,
                          f"aperture width {2*g.aperture_r:.1f} mm must exceed jamming limit "
                          f"{th.jam_ratio}x d_max = {th.jam_ratio*dm:.1f} mm"))
            out.append(_v("aperture_within_body", "GEOM", g.aperture_r < 0.9 * min(g.a, g.b),
                          "fail", g.aperture_r, 0.9 * min(g.a, g.b),
                          "aperture radius must be smaller than the body cross-section"))
            out.append(_v("cavity_exists", "GEOM", g.wall < min(g.a, g.b, g.c), "fail",
                          g.wall, min(g.a, g.b, g.c),
                          "wall thickness leaves no internal cavity"))
    elif isinstance(g, BlockParams):
        out += [
            _v("face_shell_min", "STD", g.face_shell >= th.face_shell_min, "fail",
               g.face_shell, th.face_shell_min,
               f"face shell {g.face_shell:.1f} mm below standard minimum "
               f"{th.face_shell_min:.1f} mm"),
            _v("web_min", "STD", g.web >= th.web_min, "fail", g.web, th.web_min,
               f"web {g.web:.1f} mm below standard minimum {th.web_min:.1f} mm"),
            _v("web_over_dmax", "LIT", g.web >= lim, "fail", g.web, lim,
               f"web {g.web:.1f} mm must be >= {th.min_section_mult}x d_max = {lim:.1f} mm"),
        ]
        # cores must have positive width
        core_w = (g.L - 2 * g.face_shell - (g.n_cores - 1) * g.web) / max(g.n_cores, 1)
        core_d = g.W - 2 * g.face_shell
        out += [
            _v("core_width_positive", "GEOM", core_w > 2 * g.fillet_r, "fail", core_w,
               2 * g.fillet_r,
               f"core width {core_w:.1f} mm too small for fillet r={g.fillet_r:.1f} mm"),
            _v("core_depth_positive", "GEOM", core_d > 2 * g.fillet_r, "fail", core_d,
               2 * g.fillet_r,
               f"core depth {core_d:.1f} mm too small for fillet r={g.fillet_r:.1f} mm"),
            _v("core_not_jamming", "LIT", min(core_w, core_d) >= th.jam_ratio * dm, "fail",
               min(core_w, core_d), th.jam_ratio * dm,
               "core opening below granular jamming limit"),
            _v("draft_for_release", "GEOM", g.core_taper >= th.draft_min_deg, "warn",
               g.core_taper, th.draft_min_deg,
               f"core draft {g.core_taper:.1f} deg below {th.draft_min_deg} deg; "
               "hard to demould a green bio-cement body"),
        ]
    elif isinstance(g, TileParams):
        out.append(_v("min_section_thickness", "LIT", g.t >= lim, "fail", g.t, lim,
                      f"tile thickness {g.t:.1f} mm must be >= {th.min_section_mult}x d_max "
                      f"= {lim:.1f} mm"))
        out.append(_v("joint_min", "TEAM", g.joint >= th.joint_min, "fail", g.joint,
                      th.joint_min,
                      f"joint {g.joint:.1f} mm below {th.joint_min:.1f} mm; feed solution and "
                      "oxygen must reach the tile edges in a composition"))
    return out


def _groove_rules(d: Design, th: Thresholds) -> list[Verdict]:
    g, dm = d.geom, d.mix.d_max
    out: list[Verdict] = []
    if isinstance(g, TileParams):
        depth, width, thick = g.groove_depth, g.groove_width, g.t
        frac = th.groove_depth_frac_tile if g.thick_tile else th.groove_depth_frac_block
        tag = "t/4 (thick tile)" if g.thick_tile else "t/3"
    elif isinstance(g, BlockParams):
        if g.groove_count == 0 or g.groove_depth <= 0:
            return out
        depth, width, thick = g.groove_depth, g.groove_width, g.W
        frac, tag = th.groove_depth_frac_block, "t/3 (block)"
    else:
        return out

    if depth <= 0:
        return out

    lim_d = frac * thick
    out.append(_v("groove_depth_max", "TEAM", depth <= lim_d + 1e-9, "fail", depth, lim_d,
                  f"groove depth {depth:.1f} mm exceeds {tag} = {lim_d:.1f} mm; "
                  "the unit risks snapping through the grooved plane"))
    lim_w = th.groove_w_mult_min * dm
    out.append(_v("groove_width_min", "TEAM", width >= lim_w - 1e-9, "fail", width, lim_w,
                  f"groove width {width:.1f} mm must be >= {th.groove_w_mult_min}x d_max "
                  f"= {lim_w:.1f} mm; narrower and aggregate bridges the opening, starving "
                  "the groove of sand and bacteria"))
    lim_wg = th.groove_w_mult_good * dm
    out.append(_v("groove_width_safe", "TEAM", width >= lim_wg - 1e-9, "warn", width, lim_wg,
                  f"groove width {width:.1f} mm below safe {th.groove_w_mult_good}x d_max "
                  f"= {lim_wg:.1f} mm"))
    out.append(_v("groove_jamming", "LIT", width >= th.jam_ratio * dm, "fail", width,
                  th.jam_ratio * dm,
                  f"groove width {width:.1f} mm below granular jamming limit "
                  f"{th.jam_ratio}x d_max = {th.jam_ratio*dm:.1f} mm"))
    # fillet must physically fit in the groove
    out.append(_v("groove_fillet_fits", "GEOM", 2 * g.fillet_r <= width + 1e-9, "warn",
                  2 * g.fillet_r, width,
                  f"fillet diameter {2*g.fillet_r:.1f} mm wider than groove {width:.1f} mm; "
                  "the groove profile degenerates to a shallow dish"))
    if isinstance(g, TileParams):
        out.append(_v("relief_fraction", "STD", depth <= th.relief_frac_tile_max * thick + 1e-9,
                      "warn", depth / thick, th.relief_frac_tile_max,
                      f"relief is {100*depth/thick:.1f} % of thickness; Panot practice keeps it "
                      f"below {100*th.relief_frac_tile_max:.0f} % (drainage, not stiffening)"))
        if g.groove_pitch > 0:
            out.append(_v("groove_pitch_gt_width", "GEOM", g.groove_pitch > width, "fail",
                          g.groove_pitch, width,
                          "groove pitch must exceed groove width or the relief merges"))
    return out


def _void_rules(d: Design, th: Thresholds) -> list[Verdict]:
    g = d.geom
    out: list[Verdict] = []
    if isinstance(g, BlockParams):
        core_w = (g.L - 2 * g.face_shell - (g.n_cores - 1) * g.web) / max(g.n_cores, 1)
        core_d = g.W - 2 * g.face_shell
        if core_w > 0 and core_d > 0:
            vf = (g.n_cores * core_w * core_d * g.H) / (g.L * g.W * g.H)
            ok = th.void_frac_lo - 0.08 <= vf <= th.void_frac_hi + 0.10
            out.append(_v("void_fraction_band", "STD", ok, "warn", vf,
                          (th.void_frac_lo + th.void_frac_hi) / 2,
                          f"void fraction {100*vf:.1f} % outside the "
                          f"{100*th.void_frac_lo:.0f}-{100*th.void_frac_hi:.0f} % CMU band"))
    return out


def _aeration_rules(d: Design, th: Thresholds, diag: dict | None) -> list[Verdict]:
    """Rules that need the computed geometric/physics diagnostics."""
    out: list[Verdict] = []
    if not diag:
        return out
    sav = diag.get("sa_to_vol")
    if sav is not None:
        out.append(_v("surface_to_volume_min", "LIT", sav >= th.sa_vol_min, "fail", sav,
                      th.sa_vol_min,
                      f"surface-area-to-volume {sav:.4f} 1/mm below the aerobic floor "
                      f"{th.sa_vol_min:.4f} 1/mm; B. subtilis cannot respire in the core "
                      "(the failure mode of the paper's early solid prototypes)"))
    cov = diag.get("cemented_fraction")
    if cov is not None:
        out.append(_v("penetration_coverage", "LIT", cov >= th.penetration_min_frac, "fail",
                      cov, th.penetration_min_frac,
                      f"only {100*cov:.1f} % of the body reaches the cementation threshold; "
                      f"need >= {100*th.penetration_min_frac:.0f} % for complete solidification"))
    tmax = diag.get("max_wall_thickness")
    if tmax is not None:
        lim = diag.get("penetration_depth_2x")
        if lim:
            out.append(_v("thickness_vs_penetration", "LIT", tmax <= lim, "warn", tmax, lim,
                          f"maximum local thickness {tmax:.1f} mm exceeds twice the oxygen "
                          f"penetration depth ({lim:.1f} mm); expect an uncemented core"))
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
RULE_GROUPS: tuple[Callable, ...] = (_fillet_rules, _section_rules, _groove_rules, _void_rules)


def check(d: Design, th: Thresholds | None = None, diag: dict | None = None) -> list[Verdict]:
    """Run every applicable rule. `diag` supplies computed field diagnostics."""
    th = th or Thresholds()
    out: list[Verdict] = []
    for grp in RULE_GROUPS:
        out += grp(d, th)
    out += _aeration_rules(d, th, diag)
    return out


def summarise(verdicts: Iterable[Verdict]) -> dict:
    vs = list(verdicts)
    fails = [v for v in vs if not v.passed and v.severity == "fail"]
    warns = [v for v in vs if not v.passed and v.severity == "warn"]
    return {
        "feasible": len(fails) == 0,
        "n_rules": len(vs),
        "n_fail": len(fails),
        "n_warn": len(warns),
        "failed_rules": [v.rule for v in fails],
        "warned_rules": [v.rule for v in warns],
        "messages": [v.message for v in fails],
    }
