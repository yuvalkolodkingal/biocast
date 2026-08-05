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

    spec = mauto.AutoSpec(d_max=mix.d_max, jam_mult=JAM_RATIO_LIT,
                          deflect_target_mm=deflect_target_mm,
                          draft_deg=draft_deg, pitch=pitch or 0.0)

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
            "parts": [k for k in res["_parts"]
                      if k not in ("below", "obj", "form", "envelope",
                                   "control_undrafted", "outer_body")],
            "apertures_pass": res["apertures"]["all_passed"],
            "apertures": res["apertures"]["apertures"],
            "pour_shell_pourable": res["pour_apertures"]["pourable"],
        }
    else:
        raise ValueError(f"kind must be 'rigid' or 'silicone', got {kind!r}")

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
    "skin": "cast_silicone", "skin_lower": "cast_silicone",
    "skin_upper": "cast_silicone", "skin_core_lining": "cast_silicone",
    "core": "print",
}


def mould_stl_bundle(res: dict, *, prefix: str, roles=("print",)) -> tuple:
    """Mesh the requested parts and return `(zip_bytes, manifest)`.

    Meshing happens here rather than at generation time because it is the expensive
    step and most sessions only want the numbers. Parts are voxel occupancy grids —
    only `obj_mesh` is a Trimesh — so each goes through `mould.occ_to_mesh`;
    iterating the dict for objects with `.export` writes the cast object and
    silently nothing else.

    `roles` selects by `PART_ROLE`, so the default hands back exactly the parts a
    printer should see. A genuinely multi-piece part (the block's two cavity
    linings) is written one file per connected component: `occ_to_mesh` keeps the
    largest body, so meshing their union would silently ship one of the two.
    """
    import io
    import zipfile
    from scipy import ndimage as _ndi
    from .. import mould

    parts, pitch, origin = _bundle_parts(res), res["pitch"], res["origin"]
    manifest, buf = [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, occ in parts.items():
            role = PART_ROLE.get(name, "print")
            if role not in roles or occ is None or not occ.any():
                continue
            lab, nlab = _ndi.label(occ)
            pieces = ([(occ, "")] if nlab == 1 else
                      [(lab == i, f"_{j}") for j, i in enumerate(
                          sorted(range(1, nlab + 1),
                                 key=lambda i: -np.bincount(lab.ravel())[i]),
                          start=1)])
            for sub, suffix in pieces:
                m = mould.occ_to_mesh(sub, origin, pitch)
                fn = f"{prefix}_{name}{suffix}.stl"
                z.writestr(fn, m.export(file_type="stl"))
                manifest.append({
                    "file": fn, "role": role,
                    "volume_cm3": round(float(m.volume) / 1000.0, 1),
                    "watertight": bool(m.is_watertight),
                    "bbox_mm": [round(float(v), 1) for v in m.extents],
                })
        z.writestr("MANIFEST.txt", _bundle_readme(res, manifest, prefix))
    return buf.getvalue(), manifest


def _bundle_parts(res: dict) -> dict:
    """The named, exportable parts of either driver's return, on one grid."""
    if res["summary"]["kind"] == "rigid":
        return {n: res[n] for n in res["part_names"]}
    p = res["_parts"]
    out = {"jacket_lower": p["jacket_lower"], "jacket_upper": p["jacket_upper"],
           "pour_shell_lower": res["pour_shell"]["lower"],
           "pour_shell_upper": res["pour_shell"]["upper"]}
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
    return out


def _bundle_readme(res: dict, manifest: list, prefix: str) -> str:
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
        L += ["SILICONE MOULD. Two of these part families are PRINTED and one is",
              "CAST IN SILICONE — printing the skin gives a rigid copy that fits the",
              "jacket and releases nothing.", "",
              "  1. PRINT  pour_shell_lower + pour_shell_upper",
              "     The former. Clamp it shut and pour silicone into the cavity to",
              "     make the skin. This is the mould for the mould.",
              "  2. CAST   skin / skin_lower + skin_upper (+ skin_core_lining)",
              f"     {s['silicone_mass_g']:.0f} g of rubber at "
              f"{s['skin_t_mm']:.0f} mm nominal. Do not print these.",
              "  3. PRINT  jacket_lower + jacket_upper",
              "     The rigid jacket that carries mix pressure — silicone at ~1.1 MPa",
              "     cannot hold a section. Draft lives here, not on the cast body.", "",
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
    L += ["", "FILES", *[f"  {m['file']:44s} {m['role']:14s} "
                         f"{m['volume_cm3']:9.1f} cm3  "
                         f"{'x'.join(f'{v:.0f}' for v in m['bbox_mm'])} mm"
                         for m in manifest]]
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
