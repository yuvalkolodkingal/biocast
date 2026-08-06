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


#: What each generated part is for, and whether it is PRINTED or comes OUT of the
#: pour. The distinction is the whole point of the silicone workflow: `core` is the
#: master the rubber is cast around, and handing someone a rigid copy of the skin
#: would fit the jacket and release nothing.
PART_ROLE = {
    # silicone tooling, all printed — the skin is the thing you POUR, not a file
    "jacket_a": "print", "jacket_b": "print", "core": "print",
    # rigid split negative
    "lower": "print", "upper": "print",
}

PART_NOTE = {
    "jacket_a": "first pour: clamp on the core, fill, cure, release",
    "jacket_b": "second pour: clamp on the core against cured half A",
    "core": "the master the rubber is cast around — sacrificial, not part of the mould",
    "lower": "mould half; cure it OPEN-FACED",
    "upper": "mould half; cure it OPEN-FACED",
}


def generate_mould(typology: str, geom_kw: dict, mix_kw: dict, proc_kw: dict, *,
                   kind: str = "silicone", skin_t: float = 6.0,
                   wall: float = 6.0, draft_deg: float = 2.0,
                   pitch: float | None = None) -> dict:
    """Generate a mould for the current design. Thin: `mould_cast` does the work.

    `kind="silicone"` -> two-pour interlocking tooling (jacket A, jacket B, core).
    `kind="rigid"`    -> a split negative the mix is poured straight into.

    Everything is mesh CSG, so this returns in seconds rather than the minutes the
    voxel path needed, and there is no pitch to coarsen: the only grid left is the
    one the aeration solve runs on, and that is the design's own scoring pitch.
    """
    from .. import mould_cast as mc

    geom = make_geom(typology, **geom_kw)
    mix = Mix(**{k: v for k, v in mix_kw.items() if k in Mix.__dataclass_fields__})
    proc = Process(**{k: v for k, v in proc_kw.items()
                      if k in Process.__dataclass_fields__})
    spec = mc.CastSpec(goal=kind, silicone_t=skin_t, wall=wall,
                       draft_deg=draft_deg, d_max=mix.d_max,
                       jam_mult=JAM_RATIO_LIT)
    res = mc.build_mould(geom, spec, pitch=pitch or 0.0,
                         cure_days=proc.cure_days or 28.0, rh_pct=proc.rh_pct)

    win = res["window"]
    res["summary"] = {
        "kind": kind, "typology": typology, "pitch_mm": res["pitch"],
        "parting_axis": res["parting"]["axis"],
        "parting_coord_mm": res["parting"]["plane"],
        "parting_reason": res["parting"]["reason"],
        "undercut": res["parting"]["undercut"],
        "window_d_mm": win["d_mm"], "window_spacing_mm": win["spacing_mm"],
        "open_area_frac": win["open_area_frac"],
        "cover_surrogate": win["cover_surrogate"],
        "meets_coverage": win["meets_target"],
        "coverage_limited_by": win["limited_by"],
        "L_eff_mm": res["L_eff_mm"], "L_dry_mm": res["L_dry_mm"],
        "plastic_cm3": res["plastic_cm3"],
        "silicone_mass_g": res["silicone_mass_g"],
        "silicone_volume_mm3": res["silicone_volume_mm3"],
        "n_pillars": res["n_pillars"],
        "procedure": res["procedure"],
        "checks": res["checks"],
        "parts": list(res["parts"]),
        "report": res["report"],
    }
    return res


#: Directory each role is written to inside the zip. The folder IS the instruction:
#: a flat archive of eight STLs gives a printer no way to tell that two of them are a
#: former for casting a third, and printing the skin is a silent failure — the rigid
#: copy fits the jacket perfectly and releases nothing.
ROLE_DIR = {"print": "1_print_these", "cast_silicone": "2_cast_these_in_silicone"}

#: Smallest connected component worth writing as its own STL. Below this a "piece" is
#: debris a boolean shed, not a part: a 2 mm speck is unprintable, unfindable in a
#: slicer, and dilutes the file list the fabrication manifest depends on.
MIN_PRINTABLE_MM3 = 200.0


def mould_stl_bundle(res: dict, *, prefix: str, roles=("print",)) -> tuple:
    """Zip every part as an STL, with a fabrication manifest.

    Much shorter than it used to be, because the parts arrive as meshes. The voxel
    path had to marching-cubes each occupancy grid, split it into connected
    components, and guard against shipping debris as printable parts; mesh CSG
    either yields a valid solid or raises, and `checks` has already reported the
    body count.
    """
    import io
    import zipfile

    manifest, buf = [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, mesh in res["parts"].items():
            role = PART_ROLE.get(name, "print")
            if role not in roles:
                continue
            fn = f"{ROLE_DIR.get(role, role)}/{prefix}_{name}.stl"
            z.writestr(fn, mesh.export(file_type="stl"))
            r = res["report"][name]
            manifest.append({"file": fn, "role": role, "part": name,
                             "volume_cm3": r["volume_cm3"],
                             "watertight": r["watertight"],
                             "bodies": r["bodies"], "bbox_mm": r["bbox_mm"]})
        z.writestr("MANIFEST.txt", _bundle_readme(res, manifest, prefix))
    return buf.getvalue(), manifest


def bundled_mesh(zip_bytes: bytes, name: str):
    """Load one part back out of the bundle, as a Trimesh.

    The preview reads the ARCHIVE rather than keeping the meshes alongside it, for
    two reasons. It is what the user actually downloads — a preview built from a
    separately-held mesh could drift from the file in the zip, and on this path the
    whole point is that the part list and the archive agree. And `mould_record`
    deliberately returns no voxel grids, so re-meshing on demand is not an option
    that costs nothing.
    """
    import io
    import zipfile
    import trimesh
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with z.open(name) as fh:
            return trimesh.load(io.BytesIO(fh.read()), file_type="stl")


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
                 kind: str = "silicone", skin_t: float = 6.0,
                 wall: float = 6.0, draft_deg: float = 2.0,
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

    Returns: summary (json-able), the zip bytes, the manifest, and the parts table.
    """
    if progress:
        progress(f"solving the {kind} mould…")
    res = generate_mould(typology, geom_kw, mix_kw, proc_kw, kind=kind,
                         skin_t=skin_t, wall=wall, draft_deg=draft_deg)
    s = res["summary"]

    if progress:
        progress("writing the printable set…")
    prefix = f"{kind}_{typology}"
    blob, manifest = mould_stl_bundle(res, prefix=prefix)

    parts = [{"part": p, "role": PART_ROLE.get(p, "print")} for p in s["parts"]]

    # One flag over the driver's own checks, because a failing row in a table above
    # a primary-styled download button is not a warning anybody reads. The archive is
    # still offered — the parts are real geometry and a caster may want them anyway —
    # but the reason travels with it.
    blockers = [c["check"] for c in s["checks"] if not c["pass"]]
    return {
        "kind": kind, "typology": typology,
        "summary": jsonable(s),
        "zip": blob, "manifest": manifest, "parts": parts,
        "manufacturable": not blockers, "blockers": blockers,
        "zip_name": f"mould_{prefix}.zip",
        "n_files": len(manifest),
        "n_print": sum(1 for m in manifest if m["role"] == "print"),
        "n_cast": sum(1 for m in manifest if m["role"] == "cast_silicone"),
    }


def _bundle_readme(res: dict, manifest: list, prefix: str) -> str:
    """The fabrication order, in the archive. The parts do not explain themselves."""
    s = res["summary"]
    L = [f"{prefix} — generated by biocast mould_cast", ""]
    if s["kind"] == "silicone":
        L += ["SILICONE TOOLING — TWO POURS, then cast the mix in the rubber.",
              "Everything here is PRINTED. The mould itself is what comes out of the",
              "pour; there is no STL for it because you do not print rubber.", ""]
    else:
        L += ["RIGID SPLIT MOULD. Print both halves and cast the mix straight in.", ""]
    L += [f"  {i}. {step}" for i, step in enumerate(s["procedure"], start=1)]
    L += ["",
          f"  parting        axis {s['parting_axis']} at "
          f"{s['parting_coord_mm']:.1f} mm — {s['parting_reason']}"]
    if s["kind"] == "silicone":
        L += [f"  silicone       {s['silicone_mass_g']:.0f} g "
              f"({s['silicone_volume_mm3']/1000:.0f} cm3)",
              f"  breather holes {s['window_d_mm']:.0f} mm at "
              f"{s['window_spacing_mm']:.0f} mm pitch, formed by "
              f"{s['n_pillars']} pillars", "",
              "THE SKIN IS A SEALED FACE, NOT A BREATHABLE ONE. An enclosing skin",
              "cements NOTHING — the windows are the only thing that aerates the",
              "cast, so keep the jacket's holes over the skin's on assembly."]
    else:
        L += ["", "CURE THE HALVES OPEN-FACED. A mould face is no-flux for oxygen;",
              "assembling them early turns the parting face from an oxygen source",
              "into a sealed interface and reproduces the solid-cast failure."]
    L += ["", f"  coverage       {s['cover_surrogate']:.3f} against 0.85 "
          f"({'meets' if s['meets_coverage'] else 'MISSES'} it) — "
          f"{s['coverage_limited_by']}"]
    L += ["", "CHECKS"]
    L += [f"  [{'ok  ' if c['pass'] else 'FAIL'}] {c['check']}"
          + (f" — {c['detail']}" if c.get("detail") else "") for c in s["checks"]]
    L += ["", "FILES",
          *[f"  {m['file']:44s} {m['volume_cm3']:9.1f} cm3  "
            f"{'x'.join(f'{v:.0f}' for v in m['bbox_mm'])} mm" for m in manifest]]
    L += ["", "Not a structural sign-off. Nothing here has been cast; verification is",
          "geometric and transport-based. See docs/mould_notes.md."]
    return "\n".join(L)


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
