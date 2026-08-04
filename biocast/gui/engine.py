"""Engine adapter for the GUI: one call in, one verdict dict out.

Design intent: the GUI must contain NO physics. Everything numeric here routes to
the validated package, so the interface and the batch sweep cannot drift apart.

Two things this layer adds on top of the raw package calls, both of which the
sweep learned the hard way and any interactive user needs by default:

1. MEASURED section, not nominal. `score._infer_min_feature` reads the narrowest
   passage off the nominal parameters, which overstates it by up to 4.5x when the
   aperture bore eats into the shell wall. We measure it from the occupancy grid
   (`physics.section.min_section`) and feed that in, so castability is scored on
   the passage that actually exists.
2. Literature jamming threshold. `Thresholds()` ships jam_ratio=4.0; the retrieved
   granular-flow literature says 4.94 (spheres) to 6.0 (angular grains). The GUI
   defaults to 6.0 and exposes it, because that choice changes which designs are
   rejected and the user should see it rather than inherit it silently.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..params import (BlockParams, Design, LitParams, Mix, Process, ShellParams,
                      TileParams)
from ..grammars import block as bl
from ..grammars import shell as sh
from ..grammars import tile as tl
from ..physics import fields as fl
from ..physics import section as sec
from ..physics import drying as dry
from ..physics import oxygen as ox
from .. import constraints as cons
from .. import score as sc

#: recommended jamming multiple (angular grains, Zuriguel 2005) — see module docstring
JAM_RATIO_LIT = 6.0

#: Voxel pitch per typology, chosen for interactive latency. Compared with the
#: batch sweep's pitches (shell 2.0, block 2.5, tile 1.2 mm): shell is IDENTICAL,
#: block and tile are coarser (3.0 and 1.6 mm). So shell scores here match the
#: sweep exactly, while block and tile carry slightly more discretisation error in
#: the measured section and wall-thickness fields. The field solve is skipped by
#: default and only run on demand (`with_field=True`).
PITCH = {"shell": 2.0, "block": 3.0, "tile": 1.6}

_HERE = Path(__file__).resolve().parents[2]


#: Where the literature JSONs may live, relative to the project root. `data/` is the
#: repository layout; the bare root is the flat working-directory layout. Both are
#: searched because a silent fallback to package defaults is nearly invisible: several
#: default tuples coincide with the retrieved values, so a wrong path does not change
#: those numbers and only shows up in the ones that differ.
_PARAM_DIRS = ("data", "")


def _find_param_file(explicit: str, filename: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for sub in _PARAM_DIRS:
        cand = (_HERE / sub / filename) if sub else (_HERE / filename)
        if cand.exists():
            return cand
    return None


@lru_cache(maxsize=1)
def load_physics(kin_path: str = "", mec_path: str = "") -> sc.PhysicsInputs:
    """Literature parameters. Raises if neither file is found, rather than silently
    falling back — see `_PARAM_DIRS` for why a silent fallback is dangerous here."""
    kp = _find_param_file(kin_path, "micp_kinetics_params.json")
    mp = _find_param_file(mec_path, "mechanics_params.json")
    if kp is None and mp is None:
        raise FileNotFoundError(
            "no literature parameter files found. Expected "
            "micp_kinetics_params.json and mechanics_params.json in "
            f"{_HERE / 'data'} or {_HERE}. Without them the estimator would run on "
            "package defaults, which silently coincide with several retrieved values "
            "and would make the discrepancy hard to notice.")
    try:
        kin = LitParams(str(kp)) if kp else None
    except Exception:
        kin = None
    try:
        mec = LitParams(str(mp)) if mp else None
    except Exception:
        mec = None
    return sc.PhysicsInputs.from_lit(kin, mec)


def make_geom(typology: str, **kw):
    """Build the right params dataclass, ignoring keys it does not define."""
    cls = {"shell": ShellParams, "block": BlockParams, "tile": TileParams}[typology]
    fields = {f for f in cls.__dataclass_fields__ if f != "typology"}
    return cls(**{k: v for k, v in kw.items() if k in fields})


def build_mesh(geom, pitch: float | None = None):
    """Mesh + generating field. Always uses return_field=True: trimesh's .fill()
    floods internal cavities (27 % volume overestimate) and makes every hollow
    design read as solid."""
    mod = {"shell": sh, "block": bl, "tile": tl}[geom.typology]
    p = pitch if pitch is not None else PITCH[geom.typology]
    return mod.build(geom, pitch=p, return_field=True)


def notch_of(geom) -> tuple[float, float]:
    """(notch depth, root radius) for the Inglis stress-concentration factor."""
    if isinstance(geom, ShellParams):
        return 0.0, max(geom.fillet_r, 1e-6)
    depth = float(getattr(geom, "groove_depth", 0.0) or 0.0)
    if depth <= 0:
        return 0.0, max(geom.fillet_r, 1e-6)
    width = float(getattr(geom, "groove_width", 0.0) or 0.0)
    root = min(geom.fillet_r, width / 2 if width > 0 else geom.fillet_r, depth)
    return depth, max(root, 1e-6)


def evaluate(typology: str, geom_kw: dict, mix_kw: dict, proc_kw: dict, *,
             jam_ratio: float = JAM_RATIO_LIT, n_mc: int = 300,
             pitch: float | None = None, with_field: bool = False) -> dict:
    """Score one design end to end.

    Returns a flat dict: geometry diagnostics, measured section, all four
    subscores, total score with its 5-95 % interval, constraint verdicts, and
    (optionally) the solved oxygen field for cross-section display.
    """
    geom = make_geom(typology, **geom_kw)
    mix = Mix(**{k: v for k, v in mix_kw.items() if k in Mix.__dataclass_fields__})
    proc = Process(**{k: v for k, v in proc_kw.items()
                      if k in Process.__dataclass_fields__})
    design = Design(geom=geom, mix=mix, proc=proc, name=typology)

    mesh, fld, origin, pitch_used = build_mesh(geom, pitch)
    parting_axis = 2 if proc.split_mould else None
    diag = fl.geometric_diagnostics(
        mesh, pitch_used, parting_axis=parting_axis,
        parting_frac=0.5 if proc.split_mould else None,
        field=(fld, origin, pitch_used))

    occ = diag["_grid"]["occ"]
    ms = sec.min_section(occ, pitch_used)

    nominal = sc._infer_min_feature(design)
    measured = ms["min_section_p5_mm"]
    min_feature = min(nominal, measured) if measured > 0 else nominal

    notch_d, root_r = notch_of(geom)
    phys = load_physics()
    res = sc.score_design(design, diag, phys=phys, n_mc=n_mc,
                          min_feature_mm=min_feature,
                          notch_depth_mm=notch_d, root_radius_mm=root_r)

    th = cons.Thresholds(jam_ratio=jam_ratio)
    # `_aeration_rules` reads `cemented_fraction` and `penetration_depth_2x` from the
    # diag dict, but those are produced by the SCORER, not by geometric_diagnostics.
    # Passing the raw geometric diag makes `penetration_coverage` — the rule that
    # encodes the paper's own failure mode — silently vanish from the verdict list,
    # so a design with a 39 % anoxic core reports "feasible". Merge them first.
    diag_for_rules = dict(diag)
    diag_for_rules.update({
        "cemented_fraction": res["cemented_fraction"],
        "penetration_depth_2x": res["penetration_depth_2x"],
    })
    verdicts = cons.check(design, th, diag=diag_for_rules)
    got = {v.rule for v in verdicts}
    for required in ("penetration_coverage", "surface_to_volume_min"):
        assert required in got, f"rule {required} did not run — diag key missing"
    # re-check the jamming rule against the MEASURED section, which is the number
    # that decides whether the mould can actually be filled
    jam_limit = jam_ratio * mix.d_max
    verdicts.append(cons.Verdict(
        rule="measured_section_not_jamming", origin="LIT",
        passed=bool(measured >= jam_limit), severity="fail",
        value=float(measured), limit=float(jam_limit),
        message=(f"measured narrowest section {measured:.1f} mm vs "
                 f"{jam_ratio:g} x d_max = {jam_limit:.1f} mm")))
    summary = cons.summarise(verdicts)

    out = dict(res)
    out.update({
        "typology": typology,
        "volume_mm3": float(mesh.volume),
        "area_mm2": float(mesh.area),
        "sa_to_vol": diag["sa_to_vol"],
        "max_wall_thickness_mm": diag["max_wall_thickness"],
        "mean_wall_thickness_mm": diag["mean_wall_thickness"],
        "min_section_measured_mm": measured,
        "min_section_absolute_mm": ms["min_section_min_mm"],
        "min_feature_nominal_mm": nominal,
        "min_feature_used_mm": min_feature,
        "section_over_dmax": measured / mix.d_max if mix.d_max else float("nan"),
        "watertight": bool(mesh.is_watertight),
        "euler": int(mesh.euler_number),
        "voxel_pitch_mm": pitch_used,
        "feasible": summary["feasible"],
        "n_fail": summary["n_fail"],
        "n_warn": summary["n_warn"],
        "failed_rules": summary["failed_rules"],
        "warned_rules": summary["warned_rules"],
        "verdicts": [v.__dict__ for v in verdicts],
        "jam_ratio_used": jam_ratio,
        "_mesh": mesh,
        "_diag": diag,
    })

    if with_field:
        out["_oxygen"] = solve_field(diag, phys, proc)
    return out


def solve_field(diag: dict, phys: sc.PhysicsInputs, proc: Process) -> dict:
    """Run the reaction-diffusion solve on the DRAINED subdomain.

    Oxygen only travels usefully through drained pores, so the solve domain is the
    part of the body evaporation has reached (depth <= L_dry); everything deeper is
    saturated and anoxic by construction.
    """
    g = diag["_grid"]
    occ, src, depth, pitch = g["occ"], g["src"], g["depth"], g["pitch"]

    t_cure = proc.cure_days or phys.t_cure_days[1]
    L_dry = dry.air_entry_depth(phys.E_evap[1], t_cure, phys.phi[1],
                               delta_saturation=phys.dS_air_entry[1],
                               rh_pct=proc.rh_pct)
    drained = occ & (np.nan_to_num(depth, nan=np.inf) <= L_dry)
    D_eff = ox.effective_diffusivity(phys.D_O2_gas[1], phys.phi[1], phys.sw[1], gas=True)

    if drained.sum() == 0:
        return {"C": np.zeros_like(occ, dtype=float), "occ": occ, "drained": drained,
                "L_dry_mm": L_dry, "oxygenated_fraction": 0.0, "pitch": pitch,
                "anoxic_fraction": 1.0}

    # `src` is the AIR region acting as Dirichlet atmosphere — it is disjoint from
    # `occ`, so it must NOT be intersected with the drained solid (that yields an
    # empty boundary and a uniformly zero field that looks like a 100 % anoxic body).
    # The drained solid is the solve DOMAIN; the air next to it is the boundary.
    r = ox.solve_oxygen(drained, src, pitch, D_eff_m2s=D_eff,
                        C0_mol_m3=phys.C_O2_gas[1], R_mol_m3_s=phys.R_O2_bulk[1])
    C = np.zeros_like(occ, dtype=float)
    C[drained] = r["C"][drained]
    oxy = (C > 0) & occ
    return {"C": C, "occ": occ, "drained": drained, "L_dry_mm": L_dry,
            "pitch": pitch, "oxygenated_fraction": float(oxy.sum() / occ.sum()),
            "anoxic_fraction": float(1.0 - oxy.sum() / occ.sum()),
            "resolution_warning": r.get("resolution_warning")}


def feasible_window(d_max: float, cure_days: float, rh_pct: float,
                    jam_ratio: float = JAM_RATIO_LIT) -> dict:
    """The castability floor and drying ceiling on section thickness.

    Two limits act on the same dimension from opposite directions. When the floor
    exceeds the ceiling there is NO feasible section thickness and the design must
    change process, not geometry.
    """
    phys = load_physics()
    L_dry = dry.air_entry_depth(phys.E_evap[1], cure_days, phys.phi[1],
                               delta_saturation=phys.dS_air_entry[1], rh_pct=rh_pct)
    floor = jam_ratio * d_max
    ceiling = 2.0 * L_dry
    return {"floor_mm": float(floor), "ceiling_mm": float(ceiling),
            "L_dry_mm": float(L_dry), "open": bool(floor <= ceiling),
            "width_mm": float(ceiling - floor)}


def min_cure_to_open(d_max: float, rh_pct: float, jam_ratio: float = JAM_RATIO_LIT,
                     t_hi: float = 120.0) -> float | None:
    """Shortest cure (days) that opens the window at this d_max and RH."""
    for t in np.arange(1.0, t_hi + 1.0, 1.0):
        if feasible_window(d_max, float(t), rh_pct, jam_ratio)["open"]:
            return float(t)
    return None


def max_dmax_to_open(cure_days: float, rh_pct: float,
                     jam_ratio: float = JAM_RATIO_LIT) -> float:
    """Largest aggregate size that still leaves the window open (sieve target)."""
    w = feasible_window(1.0, cure_days, rh_pct, jam_ratio)
    return float(w["ceiling_mm"] / jam_ratio)
