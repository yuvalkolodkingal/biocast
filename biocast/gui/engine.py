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


#: Voxel budget for a mould solve, in millions of voxels. Peak RSS measured on this
#: machine scales close to linearly with grid size for the silicone path — 1.1 GB at
#: 2.1 M voxels, 3.1 GB at 6.2 M — because the driver holds a few dozen full-grid
#: boolean and float arrays plus the oxygen solve's sparse operator. The tile's
#: grammar pitch of 1.2 mm is 43 M voxels, which extrapolates past 12 GB and did in
#: fact OOM-kill a 15 GB kernel during development.
#:
#: 6 M is chosen to leave headroom on a 16 GB container while staying fine enough to
#: resolve the features that matter: at pitch 2.0 the tile's windowed cemented
#: fraction is 0.885 against 0.879 at pitch 3.0 — the physics is nearly pitch-
#: independent here, so the cost of coarsening is small and the cost of not
#: coarsening is a killed process with no result at all.
MOULD_VOXEL_BUDGET_M = 6.0

#: The rigid path is much lighter (1.25 GB at grammar pitch for all three
#: typologies) because it carries no field solve, so it gets a larger budget and
#: normally runs at the grammar's own pitch.
MOULD_VOXEL_BUDGET_RIGID_M = 24.0


#: Envelope volume actually ALLOCATED by each driver, in units of 1e6 mm^3, so that
#: `n_voxels_M(p) = ENVELOPE_K[kind][typology] / p**3`.
#:
#: These are MEASURED off each driver's own grid, not computed from a bounding box.
#: An earlier version of this function used a hand-written per-typology span dict,
#: and it was wrong in the direction that defeats the purpose: it put the tile
#: silicone envelope at 380x380x160 mm where the driver actually allocates ~450 mm
#: cubed, so it chose a pitch whose real grid was ~10.7 M voxels against a stated
#: 6 M budget. A guard calibrated on a guess is worse than no guard, because it
#: reports safety it has not checked.
#:
#: Measurement (grid shape read from the driver's return, K = n_M * p^3):
#:   rigid    p=3.0  shell (89,89,62)=0.49M  block (190,124,80)=1.88M  tile (123,123,26)=0.39M
#:   silicone p=4.0  shell (93,93,104)=0.90M  block (167,117,117)=2.29M
#:   silicone p=3.0  tile  (149,149,96)=2.13M
#: The silicone envelope is several times the rigid one because it must contain the
#: jacket and the pour shell as well as the cast body.
ENVELOPE_K = {
    "rigid":    {"shell": 13.3, "block": 50.9, "tile": 10.6},
    "silicone": {"shell": 57.6, "block": 146.3, "tile": 57.6},
}

#: Peak RSS runs about 0.5 GB per million voxels for the silicone path (measured:
#: 2.1 M -> 1.12 GB, 6.2 M -> 3.08 GB, 10.7 M -> 4.29 GB including STL bundling), so
#: the voxel budget is really a memory budget in disguise.
GB_PER_M_VOXEL = 0.5


def mould_pitch_for(typology: str, kind: str, *,
                    budget_M: float | None = None) -> dict:
    """Coarsest-acceptable pitch: the grammar's own unless the grid blows the budget.

    A mould must be generated finer than the interactive scoring pitch — a 0.30 mm
    key clearance and a 10 mm breather lattice cannot be represented on a 3 mm voxel
    — but "as fine as the grammar" is not affordable for the silicone path on a
    hosted container. Rather than silently coarsening or silently dying, this
    returns the pitch AND the reason, which the caller surfaces.

    Grid size is estimated from the MEASURED envelope constant for this
    (kind, typology), and `generate_mould` checks the realised grid against the
    estimate afterwards so a bad constant shows up as a reported mismatch rather
    than as an out-of-memory kill.
    """
    grammar_pitch = {"shell": 1.25, "block": 2.0, "tile": 1.2}[typology]
    K = ENVELOPE_K[kind][typology]
    budget = budget_M if budget_M is not None else (
        MOULD_VOXEL_BUDGET_RIGID_M if kind == "rigid" else MOULD_VOXEL_BUDGET_M)

    def n_voxels(p):
        return K / p ** 3

    p = grammar_pitch
    if n_voxels(p) <= budget:
        return {"pitch": p, "grid_M": n_voxels(p), "budget_M": budget,
                "est_peak_GB": n_voxels(p) * GB_PER_M_VOXEL, "coarsened": False,
                "reason": (f"grammar pitch {p} mm needs ~{n_voxels(p):.1f} M voxels, "
                           f"inside the {budget:.0f} M budget")}
    p_need = (K / budget) ** (1.0 / 3.0)
    p = float(np.ceil(p_need / 0.25) * 0.25)
    return {"pitch": p, "grid_M": n_voxels(p), "budget_M": budget,
            "est_peak_GB": n_voxels(p) * GB_PER_M_VOXEL, "coarsened": True,
            "reason": (f"coarsened {grammar_pitch} -> {p} mm: the grammar pitch needs "
                       f"~{n_voxels(grammar_pitch):.0f} M voxels "
                       f"(~{n_voxels(grammar_pitch)*GB_PER_M_VOXEL:.0f} GB) against a "
                       f"{budget:.0f} M budget")}


def generate_mould(typology: str, geom_kw: dict, mix_kw: dict, proc_kw: dict, *,
                   kind: str = "rigid", skin_t: float = 6.0,
                   deflect_target_mm: float = 0.10,
                   draft_deg: float = 0.0, pitch: float | None = None) -> dict:
    """Generate a mould for the current design and report its verification.

    `kind="rigid"` -> `mould_auto.build_auto_mould` (printed FDM negative).
    `kind="silicone"` -> `mould_silicone.build_silicone_mould` (elastomer skin plus
    rigid jacket, breather lattice, pour shell).

    The mould pitch is the GRAMMAR's default, not `PITCH[typology]`. The interactive
    scoring pitches are coarsened for latency (block 3.0 mm, tile 1.6 mm), and a
    mould carries features — a 0.30 mm key clearance, drains at 2.5 x d_max, a
    breather lattice — that a 3 mm voxel cannot represent. A mould generated at the
    scoring pitch would verify against its own discretisation rather than against
    the geometry that gets printed.

    Returns the raw driver dict plus a flat `summary` the GUI can table directly.
    Nothing here decides anything: the drivers measure, and this only reshapes.
    """
    from .. import mould_auto as mauto
    from .. import mould

    geom = make_geom(typology, **geom_kw)
    mix = Mix(**{k: v for k, v in mix_kw.items() if k in Mix.__dataclass_fields__})
    proc = Process(**{k: v for k, v in proc_kw.items()
                      if k in Process.__dataclass_fields__})

    # An explicit pitch is honoured; otherwise choose the finest that fits the voxel
    # budget. Left unbounded, the silicone path needs >12 GB at the tile's grammar
    # pitch and is killed by the container before it can report anything.
    pchoice = ({"pitch": pitch, "grid_M": None, "coarsened": False,
                "reason": "caller-specified pitch"} if pitch else
               mould_pitch_for(typology, kind))
    spec = mauto.AutoSpec(d_max=mix.d_max, jam_mult=JAM_RATIO_LIT,
                          deflect_target_mm=deflect_target_mm,
                          draft_deg=draft_deg, pitch=pchoice["pitch"])

    if kind == "rigid":
        res = mauto.build_auto_mould(geom, spec)
        dec = res["decisions"]
        summary = {
            "kind": "rigid",
            "parting_axis": res["axis"],
            "parting_coord_mm": res["parting"],
            "parting_reason": dec["parting_analysis"]["reason"],
            "core_strategy": dec["cores"]["strategy"],
            "core_reason": dec["cores"]["reason"],
            "mould_wall_mm": dec["wall"]["t_mm"],
            "wall_deflection_mm": dec["wall"]["deflection_mm"],
            "wall_meets_target": dec["wall"]["meets_target"],
            "flange_mm": dec["flange"]["flange_mm"],
            "draft_requested_deg": dec["draft"]["draft_deg"],
            "relief_per_face_mm": dec["draft"]["relief_per_face_mm"],
            "keys_chiral": res["keys_chiral_auto"][0],
            "n_gate": res["n_gate"], "n_drain": res["n_drain"],
            "n_bolt": res["n_bolt"],
            "unattributed_mm3": res["balance"]["unattributed_mm3"],
            "balance_exact": res["balance"]["exact"],
            "parts": list(res["part_names"]),
        }
        # `auto_apertures` and `count_through_holes` are separate calls in the
        # rigid driver rather than fields on its return, so run them here — a
        # summary that reported None for "apertures pass" would read as "not
        # checked" when the check is one call away.
        aps = mauto.auto_apertures(res, spec)
        chk = mould.check_apertures(aps, spec.d_max, jam_mult=spec.jam_mult,
                                    certain_clog_mult=spec.clog_mult)
        summary["apertures_pass"] = chk["all_passed"]
        summary["apertures"] = chk["apertures"]
        summary["through_holes"] = mauto.count_through_holes(res)
    elif kind == "silicone":
        from .. import mould_silicone as msil
        sspec = msil.SiliconeSpec(skin_t=skin_t)
        res = msil.build_silicone_mould(geom, sspec, spec,
                                        cure_days=proc.cure_days, rh_pct=proc.rh_pct)
        aer, win, bar = res["aeration"], res["window"], res["barrier"]
        summary = {
            "kind": "silicone",
            "parting_axis": res["axis"],
            "parting_coord_mm": res["parting"],
            "skin_t_mm": res["skin_t_requested"],
            "skin_t_realised_mm": res["skin_measured"]["t_p50_mm"],
            "silicone_volume_mm3": res["silicone_volume_mm3"],
            "silicone_mass_g": res["silicone_mass_g"],
            "jacket_wall_mm": res["jacket_wall_mm"],
            "jacket_adequate": res["jacket_wall"]["adequate"],
            "window_d_mm": win["d_mm"], "window_spacing_mm": win["spacing_mm"],
            "open_area_frac": aer["windowed"]["open_area_frac"],
            "cemented_frac_enclosed": aer["enclosed_skin"]["cemented_frac_field"],
            "cemented_frac_windowed": aer["windowed"]["cemented_frac_field"],
            "cemented_frac_rigid_baseline":
                aer["rigid_open_parting"]["cemented_frac_field"],
            "meets_coverage": win["met_assembled"],
            "coverage_limited_by": win["limited_by"],
            # the finding that decides whether silicone is viable at all
            "R_skin_over_R_drained_wall": bar["R_skin_over_R_drained_wall"],
            "R_skin_over_R_saturated_wall": bar["R_skin_over_R_saturated_wall"],
            "wvtr_frac_of_free_evap": bar["wvtr_frac_of_free_evap"],
            "L_dry_behind_skin_mm": bar["L_dry_behind_skin_mm"],
            "barrier_verdict": bar["verdict"],
            "worst_undercut_strain_pct": res["undercut"]["worst_eps_pct"],
            "allowable_strain_pct": res["undercut"]["allowable_eps_pct"],
            "undercuts_ok": res["undercut"]["ok"],
            "release_order_ok": res["release"]["order_ok"],
            "release_control_interferes": res["release"]["control_interferes"],
            "web_returned_mm": res["section_budget"]["web_returned_mm"],
            "unattributed_mm3": res["balance"].get("unattributed_mm3",
                                                   res["balance"].get("residual_void_mm3")),
            "balance_exact": res["balance"]["exact"],
            # NOT the keys of `_parts`: that dict also carries the working sets the
            # driver needs (`windows`, `cavity`, `vent`, `gate`, `form`, `envelope`)
            # and an exclusion list has to be kept in step with it by hand. It was
            # not — the table listed `windows` and `gate` as parts to print while
            # omitting the pour shell, which is the one thing a caster must print
            # first. Read off the bundler instead, so the table cannot disagree with
            # the zip it describes.
            "parts": None,      # filled below, once `pour_shell` is on `res`
            "apertures_pass": res["apertures"]["all_passed"],
            "apertures": res["apertures"]["apertures"],
            "pour_shell_pourable": res["pour_apertures"]["pourable"],
            "spout_d_mm": res["pour_apertures"]["spout_d_mm"],
            "vent_d_mm": res["pour_apertures"]["vent_d_mm"],
            "pour_shell_reaches_cavity": res["pour_shell"]["reach"]["ok"],
            "pour_shell_wall_mm": res["pour_shell"]["wall_mm"],
            "pour_shell_rim_mm": res["pour_shell"]["rim_mm"],
            "pour_shell_plastic_cm3": res["pour_shell"]["plastic_cm3"],
            "pour_shell_orphan_mm3": res["pour_shell"]["orphan_shell_mm3"],
            "pattern_plastic_cm3": res["pour_shell"]["pattern_cm3"],
            "pour_shell_wall_ok": res["pour_shell"]["wall"]["meets_target"],
            "pour_shell_deflection_mm": res["pour_shell"]["wall"]["deflection_mm"],
            "pour_shell_balance_exact": res["pour_shell"]["balance"]["exact"],
            "pour_shell_release_ok": res["pour_shell"]["release"]["ok"],
            "pour_shell_control_interferes":
                res["pour_shell"]["release"]["control_interferes"],
            "pour_cavity_cm3": res["pour_shell"]["cavity_mm3"] / 1000.0,
            "cavity_matches_skin": res["pour_shell"]["cavity_matches_skin"],
            "skin_parted_by_former": res["pour_shell"]["skin_parted_by_former"],
            "pour_procedure": res["pour_shell"]["procedure"],
            "pillars": res["pour_shell"]["pillars"],
        }
    else:
        raise ValueError(f"kind must be 'rigid' or 'silicone', got {kind!r}")

    res["summary"] = summary          # `_bundle_parts` dispatches on summary["kind"]
    summary["parts"] = list(_bundle_parts(res))
    summary["pitch_mm"] = res["pitch"]
    summary["pitch_coarsened"] = pchoice["coarsened"]
    summary["pitch_reason"] = pchoice["reason"]

    # Check the ESTIMATE against the grid the driver actually allocated. An envelope
    # constant that drifts (a grammar gains a feature, a flange rule changes) would
    # otherwise show up as an out-of-memory kill with no diagnosis; here it shows up
    # as a number the caller can read. The realised K is reported so the constant
    # above can be corrected from a run rather than re-derived by hand.
    grid = (res["lower"] if kind == "rigid" else res["_parts"]["envelope"]).shape
    n_real = float(np.prod(grid)) / 1e6
    summary["grid_M_realised"] = n_real
    summary["grid_M_estimated"] = pchoice.get("grid_M")
    summary["envelope_K_realised"] = n_real * res["pitch"] ** 3
    if pchoice.get("grid_M"):
        summary["grid_estimate_ratio"] = n_real / pchoice["grid_M"]
        summary["over_budget"] = bool(
            pchoice.get("budget_M") and n_real > 1.15 * pchoice["budget_M"])
    res["summary"] = summary
    return res


#: Which generated parts are PRINTED, and which are cast in the printed ones. The
#: distinction is the whole point of the silicone workflow and is easy to get wrong
#: from the filenames alone: `skin` is the silicone itself — a cast product, not a
#: printable part — while `pour_shell_*` is the negative you print in order to cast
#: that skin. Handing someone the skin STL to print gives them a rigid plastic copy
#: of the flexible part, which fits the jacket and releases nothing.
PART_ROLE = {
    # rigid path
    "lower": "print", "upper": "print", "core_lo": "print", "core_up": "print",
    # silicone path
    "jacket_lower": "print", "jacket_upper": "print",
    "pour_shell_lower": "print", "pour_shell_upper": "print",
    "pattern": "print", "parting_plate": "print",
    "skin": "cast_silicone", "skin_lower": "cast_silicone",
    "skin_upper": "cast_silicone", "skin_core_lining": "cast_silicone",
    "core": "print",
}


#: Directory each role is written to inside the zip. The folder IS the instruction:
#: a flat archive of eight STLs gives a printer no way to tell that two of them are a
#: former for casting a third, and printing the skin is a silent failure — the rigid
#: copy fits the jacket perfectly and releases nothing.
ROLE_DIR = {"print": "1_print_these", "cast_silicone": "2_cast_these_in_silicone"}

#: Smallest connected component worth writing as its own STL. Below this a "piece" is
#: debris a boolean shed, not a part: a 2 mm speck is unprintable, unfindable in a
#: slicer, and dilutes the file list the fabrication manifest depends on.
MIN_PRINTABLE_MM3 = 200.0


def mould_stl_bundle(res: dict, *, prefix: str, roles=("print", "cast_silicone")) -> tuple:
    """Mesh the requested parts and return `(zip_bytes, manifest)`.

    Parts are voxel occupancy grids — only `obj_mesh` is a Trimesh — so each goes
    through `mould.occ_to_mesh`; iterating the dict for objects with `.export` writes
    the cast object and silently nothing else.

    `roles` selects by `PART_ROLE`, and the selected parts are filed into one
    directory per role (`ROLE_DIR`) rather than dumped flat, because the printed and
    the cast families are not interchangeable and the archive is the only thing that
    travels to whoever fabricates it. A genuinely multi-piece part (the block's two
    cavity linings) is written one file per connected component: `occ_to_mesh` keeps
    the largest body, so meshing their union would silently ship one of the two.
    """
    import io
    import zipfile
    from scipy import ndimage as _ndi
    from .. import mould

    parts, pitch, origin = _bundle_parts(res), res["pitch"], res["origin"]
    manifest, debris, buf = [], [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, occ in parts.items():
            role = PART_ROLE.get(name, "print")
            if role not in roles or occ is None or not occ.any():
                continue
            lab, nlab = _ndi.label(occ)
            # `np.bincount(lab.ravel())` inside the sort key recomputes a full
            # histogram of a multi-million-voxel array per comparison; hoisted.
            sizes = np.bincount(lab.ravel()) if nlab > 1 else None
            if nlab == 1:
                pieces, dropped = [(occ, "")], []
            else:
                order = sorted(range(1, nlab + 1), key=lambda i: -sizes[i])
                # A part that is legitimately several pieces (the block's two cavity
                # linings) must be written one file per piece. But a part that has
                # SHED debris must not: a few stray voxels are not a component
                # somebody prints, and writing them turned one former into 25 STLs,
                # 24 of them under a cubic centimetre. The threshold is printability,
                # and what falls below it is reported in the manifest rather than
                # dropped quietly.
                keep = [i for i in order
                        if sizes[i] * pitch ** 3 >= MIN_PRINTABLE_MM3] or order[:1]
                # one survivor IS the part, so it keeps the bare name — a lone
                # `_1` suffix would suggest a second file that does not exist
                pieces = ([(lab == keep[0], "")] if len(keep) == 1 else
                          [(lab == i, f"_{j}") for j, i in enumerate(keep, start=1)])
                dropped = [i for i in order if i not in keep]
                if dropped:
                    debris.append((name, len(dropped),
                                   float(sum(sizes[i] for i in dropped) * pitch ** 3)))
            for sub, suffix in pieces:
                m = mould.occ_to_mesh(sub, origin, pitch)
                fn = f"{ROLE_DIR.get(role, role)}/{prefix}_{name}{suffix}.stl"
                z.writestr(fn, m.export(file_type="stl"))
                manifest.append({
                    "file": fn, "role": role, "part": name,
                    "volume_cm3": round(float(m.volume) / 1000.0, 1),
                    "watertight": bool(m.is_watertight),
                    "bbox_mm": [round(float(v), 1) for v in m.extents],
                })
        z.writestr("MANIFEST.txt", _bundle_readme(res, manifest, prefix, debris))
    return buf.getvalue(), manifest


def jsonable(o):
    """Strip numpy scalars and arrays so a summary survives json.dumps and, more
    importantly, so nothing that reaches `mould_record`'s return still holds a
    reference to a full voxel grid."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return f"<array shape={o.shape}>"
    return o


def mould_record(typology: str, geom_kw: dict, mix_kw: dict, proc_kw: dict, *,
                 kind: str = "rigid", skin_t: float = 6.0,
                 deflect_target_mm: float = 0.10,
                 progress=None) -> dict:
    """Generate, verify, mesh and bundle in ONE call, returning only plain data.

    Two problems this solves, both of which made the STL download unreachable in the
    GUI rather than merely slow.

    1. **Meshing cannot be deferred behind a second button in Streamlit.** Every
       widget interaction re-runs the whole script, so a "Prepare STLs" button in a
       tab whose body is guarded by `if st.button("Generate")` runs on a pass where
       the generate button reads False — and the tab, results and all, is gone before
       the mesher is reached. Doing both in one action is what makes the download
       exist at all. It costs ~10 s on top of a 15 s rigid solve and is lost in the
       noise of a 2-3 minute silicone one.
    2. **The driver's return cannot be kept across re-runs.** `res` holds a few dozen
       full voxel grids — 75-200 MB for a silicone solve — and parking that in
       `st.session_state` multiplies it by the number of open sessions. Everything
       the interface renders is derived here and the grids are dropped on return.

    Returns: summary (json-able), the rigid path's boundary-condition comparison,
    the zip bytes, the manifest, and the parts/roles table.
    """
    if progress:
        progress(f"solving the {kind} mould…")
    res = generate_mould(typology, geom_kw, mix_kw, proc_kw, kind=kind,
                         skin_t=skin_t, deflect_target_mm=deflect_target_mm)
    s = res["summary"]

    aeration = None
    if kind == "rigid":
        if progress:
            progress("comparing boundary conditions…")
        aeration = mould_aeration(proc_kw, mould_res=res)

    if progress:
        progress("meshing every part — marching cubes, the expensive step…")
    prefix = f"{kind}_{typology}"
    blob, manifest = mould_stl_bundle(res, prefix=prefix)

    parts = [{"part": p, "role": PART_ROLE.get(p, "print")} for p in s["parts"]]

    # One flag over the former's own checks. Printing a full-size master pattern, a
    # 12 mm-walled former and a jacket is many hours and several hundred grams of
    # filament, and a failing row ninth in a nine-row table above a primary-styled
    # download button is not a warning anybody reads. The archive is still offered —
    # the parts are real and a caster may want them anyway — but the reason travels
    # with it.
    blockers = []
    if kind == "silicone":
        for ok, why in (
                (s["pour_shell_release_ok"],
                 "the former's halves do not open off the cured skin"),
                (s["pillars"]["ok"],
                 "the window pillars do not bridge the former wall to the pattern"),
                (s["pour_shell_reaches_cavity"],
                 "the spout or vent does not reach every cavity the pour has to fill"),
                (s["cavity_matches_skin"]["ok"],
                 "the former would cast a skin that is not the one the aeration "
                 "solve scored"),
                (s["pour_shell_balance_exact"],
                 "the former's volume balance does not close")):
            if not ok:
                blockers.append(why)
    return {
        "kind": kind, "typology": typology,
        "summary": jsonable(s),
        "aeration": jsonable(aeration),
        "zip": blob, "manifest": manifest, "parts": parts,
        "manufacturable": not blockers, "blockers": blockers,
        "zip_name": f"mould_{prefix}.zip",
        "n_files": len(manifest),
        "n_print": sum(1 for m in manifest if m["role"] == "print"),
        "n_cast": sum(1 for m in manifest if m["role"] == "cast_silicone"),
    }


def _bundle_parts(res: dict) -> dict:
    """The named, exportable parts of either driver's return, on one grid."""
    if res["summary"]["kind"] == "rigid":
        return {n: res[n] for n in res["part_names"]}
    p = res["_parts"]
    out = {"pattern": p["obj"],
           "pour_shell_lower": res["pour_shell"]["lower"],
           "pour_shell_upper": res["pour_shell"]["upper"],
           "parting_plate": res["pour_shell"]["parting_plate"],
           "jacket_lower": p["jacket_lower"], "jacket_upper": p["jacket_upper"]}
    # The skin is split exactly when the one-piece hoop stretch exceeds the
    # allowable — the same decision `mould_silicone.export` makes, read off the
    # record rather than re-derived, so a bundle cannot disagree with the STL set.
    skin = p["skin_out"] if p["skin_core"].any() else p["skin_all"]
    if res["hoop"]["one_piece_ok"]:
        out["skin"] = skin
    else:
        out["skin_lower"] = skin & p["below"]
        out["skin_upper"] = skin & ~p["below"]
    if p["skin_core"].any():
        out["skin_core_lining"] = p["skin_core"]
    if p["core"].any():
        out["core"] = p["core"]
    # An empty set is not a part. The bundler already skips it, so leaving it here
    # would put a row in the on-screen parts table and a line in the fabrication
    # manifest for a file that is not in the zip.
    return {k: v for k, v in out.items() if v is not None and v.any()}


def _bundle_readme(res: dict, manifest: list, prefix: str, debris: list = ()) -> str:
    """Fabrication order in the archive, because the parts do not explain themselves.

    Someone opening a zip of eight STLs cannot tell that two of them are a former
    for casting a third, and printing the skin instead of pouring it is a silent
    failure — the plastic part fits the jacket perfectly and releases nothing.
    """
    s = res["summary"]
    L = [f"{prefix} — generated by biocast mould_auto / mould_silicone", ""]
    if s["kind"] == "rigid":
        L += ["RIGID SPLIT MOULD. Print every file here.", "",
              f"  parting plane   {s['parting_coord_mm']:.1f} mm on axis "
              f"{s['parting_axis']}",
              f"  mould wall      {s['mould_wall_mm']:.0f} mm "
              f"({s['wall_deflection_mm']:.3f} mm deflection at the design pressure)",
              f"  draft           {s['draft_requested_deg']:.2f} deg, "
              f"{s['relief_per_face_mm']:.2f} mm relief per face",
              f"  cores           {s['core_strategy']} — {s['core_reason']}", "",
              "CURE THE HALVES OPEN-FACED. A mould face is a no-flux boundary for",
              "oxygen; assembling the halves early converts the parting face from an",
              "oxygen source into a sealed interface and reproduces the solid-cast",
              "failure this geometry exists to avoid."]
    else:
        pil = s["pillars"]
        L += ["SILICONE MOULD. This is TWO CASTINGS, not one: you print a former,",
              "cast the rubber mould in it, then cast the bio-concrete in the rubber.",
              f"Files in {ROLE_DIR['print']}/ are printed. Files in",
              f"{ROLE_DIR['cast_silicone']}/ are what comes OUT of the pour — they are",
              "there for fit checking and rubber estimation. Printing the skin gives a",
              "rigid copy that fits the jacket and releases nothing.", "",
              "  1. PRINT  pattern",
              "     A solid positive of the body. Sacrificial master, not part of the",
              "     mould. Without it in the box the pour has no gap to fill and you",
              "     get a solid rubber copy of the body instead of a mould.",
              "  2. PRINT  pour_shell_lower + pour_shell_upper",
              "     The former. Seat the pattern on the window pillars, clamp shut,",
              f"     and pour {s['silicone_mass_g']:.0f} g of silicone in through the",
              f"     {s['spout_d_mm']:.0f} mm spout; air leaves by the "
              f"{s['vent_d_mm']:.0f} mm vent.",
              f"     {pil['n']} pillars span the {s['skin_t_mm']:.0f} mm gap. They do",
              "     two jobs: they hold the pattern at the skin offset, and they form",
              "     the breather windows, so the skin demoulds already perforated.",
              "     The pillar tips touch the pattern with no allowance, so print",
              f"     tolerance lands directly on the {s['skin_t_mm']:.0f} mm skin.",
              "  3. DEMOULD the skin. That cured rubber IS the mould.",
              "     skin / skin_lower + skin_upper (+ skin_core_lining)",
              "     PUNCH THE REMAINING WINDOWS. The former forms only the bores",
              "     running along the draw — a peg across the pull shears through",
              "     the rubber when the halves open — which is "
              f"{pil['open_area_formed_frac']*100:.0f} % of the designed",
              "     window area. Punch the rest, and the fill gate, to skin*.stl."
              + ("" if not s["cavity_matches_skin"]["core_lining_windows_unformed"]
                 else "\n     The hollow core's lining is cast solid for the same"
                      "\n     reason; its windows are hand-punched too."),
              "  4. PRINT  jacket_lower + jacket_upper",
              "     The rigid jacket that carries mix pressure — silicone at ~1.1 MPa",
              "     cannot hold a section. Draft lives here, not on the cast body.",
              "  5. CAST the mix in the skin, inside the jacket. Cure open-faced.", "",
              f"  breather windows  {s['window_d_mm']:.0f} mm at "
              f"{s['window_spacing_mm']:.0f} mm pitch, "
              f"{s['open_area_frac']*100:.1f} % open area", "",
              "THE SKIN IS A SEALED FACE, NOT A BREATHABLE ONE.",
              f"  It carries ~{s['R_skin_over_R_drained_wall']:.0f}x the oxygen",
              "  resistance of the drained pore network behind it and passes",
              f"  {s['wvtr_frac_of_free_evap']*100:.2f} % of free evaporation. The",
              "  windows are what aerate the cast, so keep skin and jacket windows",
              "  ALIGNED on assembly — a window with jacket behind it is not a window.",
              f"  Enclosed: {s['cemented_frac_enclosed']:.3f} cemented. "
              f"Windowed: {s['cemented_frac_windowed']:.3f}.",
              "", "DISASSEMBLY ORDER: jacket off the skin first, then peel the skin",
              "off the cast. The reverse tears the skin."]
    L += ["", "FILES", *[f"  {m['file']:52s} {m['role']:14s} "
                         f"{m['volume_cm3']:9.1f} cm3  "
                         f"{'x'.join(f'{v:.0f}' for v in m['bbox_mm'])} mm"
                         for m in manifest]]
    if debris:
        L += ["", f"NOT WRITTEN — below the {MIN_PRINTABLE_MM3:.0f} mm3 printable "
                  "floor, reported rather than dropped silently:",
              *[f"  {n:24s} {k} fragment(s), {v:.1f} mm3 total"
                for n, k, v in debris]]
    L += ["", "Not a structural sign-off. Nothing here has been cast; verification is",
          "geometric and transport-based. See docs/mould_auto_notes.md."]
    return "\n".join(L)


def mould_aeration(proc_kw: dict, *, mould_res: dict) -> dict:
    """Cemented fraction with a RIGID mould in place, as a sealed-face boundary.

    The silicone driver computes this comparison itself (`res["aeration"]`, three
    labelled cases with the full field solve) because window sizing depends on it.
    The rigid path has no equivalent, and it needs one for the same reason: a rigid
    mould face is no-flux too, so a closed rigid mould cements no better than a
    closed silicone one. This supplies that comparison for `build_auto_mould`
    output, using the cheap drained-depth criterion rather than the field solve —
    which is stated in the return, because the two are not interchangeable.

    `evaluate` scores the demoulded body: `fields.exposure_mask` treats every air
    voxel connected to the grid boundary as atmosphere, which is right for a body
    out of its mould and wrong for one still in it. In the mould the path from the
    cast surface to the air runs through mould material, and both PETG and silicone
    are no-flux on the timescale that matters — silicone despite being highly
    oxygen-permeable in absolute terms, because the comparison that decides
    cementation is against DRAINED PORES, which beat PDMS by about two and a half
    orders of magnitude, and because a skin that throttles evaporation throttles
    the drained network oxygen needs.

    So this reports the same body under three boundary conditions — demoulded, fully
    enclosed, and enclosed with only genuinely open area (breather windows, or an
    open parting face on a split mould) — because the SPREAD is the design
    information. A fully enclosed mould reproduces the source paper's Fig. 5
    failure regardless of how good the geometry is.
    """
    phys = load_physics()
    proc = Process(**{k: v for k, v in proc_kw.items()
                      if k in Process.__dataclass_fields__})

    occ = mould_res["obj"]
    pitch = mould_res["pitch"]
    # Parts are returned in the z-normal working frame, so the parting normal is
    # axis 2 HERE regardless of which object axis it came from; `axis` in the record
    # is the axis in the object's own frame and must not be used to index these.
    k_part = mould_res["k_part"]
    mould_occ = mould_res["lower"] | mould_res["upper"]

    L_dry = dry.air_entry_depth(phys.E_evap[1], proc.cure_days or 28.0, phys.phi[1],
                                delta_saturation=phys.dS_air_entry[1],
                                rh_pct=proc.rh_pct)
    out = {"L_dry_mm": float(L_dry)}
    cases = {
        "demoulded": fl.exposure_mask(occ),
        "enclosed": fl.exposure_mask_in_mould(occ, mould_occ)["src"],
        "open_faces_only": fl.exposure_mask_in_mould(
            occ, mould_occ, parting_axis=2, parting_index=k_part,
            open_parting_face=True)["src"],
    }
    for name, src in cases.items():
        depth = fl.depth_field(occ, src, pitch)
        d = depth[occ]
        out[name] = {
            "cemented_fraction": float(np.nanmean(d <= L_dry)),
            "depth_max_mm": float(np.nanmax(d)),
            "src_voxels": int(src.sum()),
        }
    out["note"] = ("cemented fraction here is the DRAINED-DEPTH criterion "
                   "(depth <= L_dry), not the field solve; it is the cheap "
                   "comparison across boundary conditions")
    return out


def rank_design(r: dict) -> tuple:
    """Sort key for candidate designs, best last (use `reverse=True` to list best first).

    Lexicographic and feasibility-first. A design that breaks a hard rule cannot be
    cast at all, so no score should be able to buy its way past one that can — and on
    score alone the ordering really does invert. See `search_shapes` for the measured
    case: an 18 mm-walled vessel breaking two rules outranked a 27 mm one breaking one.
    """
    return (-int(r.get("n_fail", 0)), float(r.get("score_lo", 0.0)),
            float(r.get("score", 0.0)))


def search_shapes(typology: str, space: dict, choices: dict, derive,
                  mix_kw: dict, proc_kw: dict, *, jam_ratio: float = JAM_RATIO_LIT,
                  n_random: int = 24, n_refine: int = 2, seed: int = 0,
                  n_mc: int = 150, start: dict | None = None,
                  progress=None) -> list:
    """Find the design most likely to cement, by sampling then refining.

    Random search alone is a poor optimiser in eight dimensions, and it was what the
    Explore tab did: 24 draws over a box that big lands nowhere near a peak, and
    doubling the draws buys almost nothing. What it is good at is finding a BASIN,
    because the scoring surface here is not smooth — `feasible` is a step, and the
    dominant failure mode switches discontinuously as the section crosses the jamming
    floor or the drying ceiling.

    So: sample to find the basin, then walk downhill inside it. The refinement is a
    COMPASS SEARCH — try each parameter up and down at the current step, accept the
    first improvement, and halve the step only when a whole sweep fails to find one.
    It needs no gradient (there isn't one to have) and cannot cross a feasibility
    boundary blindly, because every trial point is scored by the same `evaluate` as
    everything else.

    HOW MUCH THE REFINEMENT ACTUALLY BUYS: on the evidence so far, nothing. Measured
    across all three typologies at 24 samples and 2 levels, the refined design equalled
    the best sampled one every time — the sampling finds the design and the compass
    search only confirms it is a local optimum for single-parameter moves. Dropping the
    first step from 25 % of each range to 10 % did not change that either. It is kept
    because confirming a local optimum is worth something, and because it is the only
    stage that can improve on a hand-tuned `start`; it is not kept because it has been
    shown to find better designs. Spend the budget on samples first.

    `start` is scored first and seeds the refinement if it beats every random draw, so
    a search can never return something worse than the design already on the sliders.

    RANKING IS LEXICOGRAPHIC: fewest broken hard rules first, then `score_lo`, then
    the median. Feasibility has to lead, and ranking on score alone gets it backwards
    in a way that is easy to miss. Measured on the vessel at d_max = 4 mm: an 18 mm
    wall scores `score_lo` 0.207 while breaking TWO rules (the section and the aperture
    are both under the 24 mm jamming floor); a 27 mm wall breaks only the aperture rule
    and scores `score_lo` 0.000 with a median of 0.296. On score alone the search
    parks on the 18 mm design and every refinement step is rejected — which is exactly
    what it did, twice, gaining nothing across 52 evaluations. `-n_fail` gives it the
    gradient it was missing: 2 broken rules to 1 to none.

    The 5th percentile leads the tiebreak rather than the median because the intervals
    here are wide and driven mostly by an assumed biofilm volume fraction, so ranking
    on the median promotes whichever design happens to have the widest interval. It
    collapses to 0.000 for plenty of sound designs, and the median then does the
    discriminating — which is the honest outcome, not a workaround.

    Returns every candidate scored, best first.
    """
    rng = np.random.default_rng(int(seed))
    names = list(space)
    rows, seen = [], set()

    def _score(g):
        key = tuple(round(float(g[k]), 4) if isinstance(g[k], (int, float)) else g[k]
                    for k in sorted(g))
        if key in seen:
            return None
        seen.add(key)
        try:
            r = evaluate(typology, derive(typology, g), mix_kw, proc_kw,
                         jam_ratio=jam_ratio, n_mc=n_mc)
        except Exception:
            return None
        row = {**g, "score": r["score"], "score_lo": r["score_lo"],
               "score_hi": r["score_hi"], "feasible": r["feasible"],
               "n_fail": r["n_fail"], "limiting": r["dominant_failure_mode"],
               "section_mm": r["min_section_measured_mm"],
               "volume_cm3": r["volume_mm3"] / 1000.0,
               "cemented_fraction": r["cemented_fraction"],
               "failed": ", ".join(r["failed_rules"])}
        rows.append(row)
        return row

    all_names = names + list(choices)
    budget = int(n_refine) * len(all_names) * 4     # ceiling, not a target
    total = int(n_random) + budget + (1 if start else 0)
    done = 0

    def _tick():
        nonlocal done
        done += 1
        if progress:
            progress(min(done / max(total, 1), 1.0), done, total)

    # Only if it covers every searched parameter: a partial start would be scored with
    # dataclass defaults filling the gaps, and could then be picked as `best` — whose
    # missing keys the refinement's `cur` would immediately fail on.
    if start and all(k in start for k in all_names):
        _score({k: start[k] for k in all_names})
        _tick()
    for _ in range(int(n_random)):
        g = {k: float(rng.uniform(s["lo"], s["hi"])) for k, s in space.items()}
        for k, opts in choices.items():
            g[k] = opts[int(rng.integers(len(opts)))]
        _score(g)
        _tick()

    if not rows:
        return []
    best = max(rows, key=rank_design)
    cur = {k: best[k] for k in all_names}

    # The first step is 10 % of each range, not 25 %. A quarter of the range is a leap,
    # not a probe: 24 random draws already land somewhere reasonable, and from there
    # every quarter-range move was rejected on all three typologies, so the whole
    # budget went on shrinking rather than on improving.
    used, frac, frac_min = 0, 0.10, 0.10 / (2 ** (int(n_refine) + 2))
    while frac >= frac_min and used < budget:
        improved = False
        for k in choices:                     # discrete: enumerate, do not perturb
            for opt in choices[k]:
                if opt == cur[k] or used >= budget:
                    continue
                r = _score({**cur, k: opt})
                used += 1
                _tick()
                if r and rank_design(r) > rank_design(best):
                    best, cur = r, {kk: r[kk] for kk in all_names}
                    improved = True
        for k in names:
            s = space[k]
            step = frac * (s["hi"] - s["lo"])
            for sign in (+1, -1):
                if used >= budget:
                    break
                trial = dict(cur)
                trial[k] = float(np.clip(cur[k] + sign * step, s["lo"], s["hi"]))
                r = _score(trial)
                used += 1
                _tick()
                if r and rank_design(r) > rank_design(best):
                    best, cur = r, {kk: r[kk] for kk in all_names}
                    improved = True
                    break                     # move on; this direction paid off
        if not improved:
            frac /= 2.0                       # only shrink when a full sweep fails

    rows.sort(key=rank_design, reverse=True)
    return rows


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
