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

import os
import sys
from contextlib import contextmanager
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
from ..physics import strength as stg
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


# --------------------------------------------------------------------------- workers
#: Resident memory one scoring worker needs, in MB. Measured, not guessed: a single
#: `evaluate` peaks about 190 MB above baseline on the block and tile grids (the
#: occupancy, depth, distance and SDF arrays coexist), and a forked child adds little
#: on top because the interpreter and the numpy/scipy/trimesh pages are shared
#: copy-on-write with the pool's parent. Rounded up for the pages a child does dirty.
#:
#: This is a real constraint rather than a formality. The search holds no grids
#: between candidates, but N workers hold N grids AT THE SAME MOMENT, so the memory
#: is what actually bounds the worker count on the small containers this app deploys
#: to — not the core count.
WORKER_MEM_MB = 260

#: Never exceed this many workers however many cores are found. Past roughly this
#: point the pool is bounded by memory bandwidth on the same voxel grids rather than
#: by cores, and the scheduling and pickling overhead starts to show.
WORKER_CAP = 16


def _cgroup_cpu_quota() -> float | None:
    """CPUs this container is actually allowed, or None if unconstrained.

    Both of this project's deploy targets are containers — the Dockerfile serves
    Hugging Face Spaces and Cloud Run — and inside one, `os.cpu_count()` reports the
    HOST's cores. A 2-vCPU Space reports 16, and a pool built on that number is worse
    than staying serial: the workers time-slice onto two cores while their voxel
    grids all sit in RAM at once.
    """
    try:                                                    # cgroup v2
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return float(quota) / float(period)
    except Exception:
        pass
    try:                                                    # cgroup v1
        root = Path("/sys/fs/cgroup/cpu")
        q = float((root / "cpu.cfs_quota_us").read_text())
        p = float((root / "cpu.cfs_period_us").read_text())
        if q > 0 and p > 0:
            return q / p
    except Exception:
        pass
    return None


def _available_mb() -> float | None:
    """Memory we may actually use, from the cgroup limit or /proc/meminfo."""
    best = None
    for p in ("/sys/fs/cgroup/memory.max",                  # v2
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # v1
        try:
            raw = Path(p).read_text().strip()
            if raw != "max":
                # v1 writes a sentinel near 2**63 to mean "no limit"
                v = float(raw) / 1e6
                if v < 1e9:
                    best = v if best is None else min(best, v)
        except Exception:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                v = float(line.split()[1]) / 1024.0
                best = v if best is None else min(best, v)
                break
    except Exception:
        pass
    return best


def cpu_budget(requested: int | None = None) -> int:
    """How many scoring workers to run. Detected, never assumed.

    Order of authority, narrowest first:

      BIOCAST_WORKERS   explicit override, for when the operator knows better
      cgroup quota      what the container is actually allowed (see above)
      CPU affinity      what this process is permitted to run on; `taskset` and most
                        schedulers restrict this without touching `os.cpu_count()`
      os.cpu_count()    last resort

    Then clamped by `WORKER_MEM_MB` against available memory, and by `WORKER_CAP`.
    Returning 1 is a normal answer, not a failure — the caller runs in-process.
    """
    env = os.environ.get("BIOCAST_WORKERS", "").strip()
    if requested is None and env:
        try:
            requested = int(env)
        except ValueError:
            requested = None
    if requested is not None and int(requested) > 0:
        return max(1, min(int(requested), WORKER_CAP))

    if hasattr(os, "process_cpu_count"):            # 3.13+, affinity-aware
        n = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):          # Linux
        n = len(os.sched_getaffinity(0)) or 1
    else:
        n = os.cpu_count() or 1

    quota = _cgroup_cpu_quota()
    if quota is not None:
        n = min(n, max(1, int(quota)))

    mem = _available_mb()
    if mem is not None:
        # leave the parent its own headroom before dividing the rest into workers
        n = min(n, max(1, int((mem - WORKER_MEM_MB) // WORKER_MEM_MB)))

    return int(max(1, min(n, WORKER_CAP)))


def describe_workers(n: int) -> str:
    """One line on where the worker count came from, for the interface to show."""
    if os.environ.get("BIOCAST_WORKERS", "").strip():
        src = "set by BIOCAST_WORKERS"
    elif _cgroup_cpu_quota() is not None:
        src = "from this container's CPU quota"
    else:
        src = "from the CPUs this process may run on"
    mem = _available_mb()
    note = f", {mem/1024:.1f} GB available" if mem else ""
    return f"{n} worker{'s' if n != 1 else ''} — {src}{note}"


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


@lru_cache(maxsize=1)
def load_mechanics(mec_path: str = "") -> LitParams | None:
    """The mechanics literature set itself, for the capacity model.

    `load_physics` reduces the same file to (low, mode, high) triangles, which is
    the wrong shape for UCS: on construction waste there is ONE study, so a
    triangular prior would invent a mode the measurement does not have.
    `physics.strength` reads the rows directly instead — hence a second accessor
    over the same file rather than more fields on `PhysicsInputs`.

    Returns None rather than raising when the file is absent: capacity is a
    reported extra, and the module's own defaults are the same numbers. That is
    the opposite of `load_physics`, which raises, because there the package
    defaults silently coincide with several retrieved values and a wrong path
    would be invisible.
    """
    mp = _find_param_file(mec_path, "mechanics_params.json")
    if mp is None:
        return None
    try:
        return LitParams(str(mp))
    except Exception:
        return None


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


#: (notch depth, root radius) for the Inglis stress-concentration factor. Defined
#: in `physics.strength` and re-exported here under the name callers already use:
#: the scorer and the capacity model must not be able to disagree about what the
#: notch is, and they would if each kept its own copy.
notch_of = stg.notch_of


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

    # Capacity rides alongside the score on the SAME geometry, notch and seed
    # style, and deliberately does not enter it: the four subscores answer
    # "will this solidify", capacity answers "what can it carry once it has".
    # It never marks a design infeasible — see `physics.strength`.
    cap = stg.load_capacity(design, diag, phys=load_mechanics(), n_mc=n_mc,
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
        "capacity_kN": cap["capacity_kN"],
        "capacity_lo_kN": cap["capacity_lo_kN"],
        "capacity_hi_kN": cap["capacity_hi_kN"],
        "ucs_nom_MPa": cap["ucs_nom_MPa"],
        "critical_section_mm2": cap["critical_section_mm2"],
        "critical_section_voxel_mm2": cap["critical_section_voxel_mm2"],
        "critical_section_rel_diff": cap["critical_section_rel_diff"],
        "critical_section_agrees": cap["critical_section_agrees"],
        "critical_section_plane": cap["critical_section_plane"],
        "kt_used": cap["kt_used"],
        "c90_benchmark_MPa": cap["c90_benchmark_MPa"],
        "c90_ratio": cap["c90_ratio"],
        "strength_provenance": cap["strength_provenance"],
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


def rank_strength(r: dict) -> tuple:
    """Sort key for the strongest VIABLE shape, best last.

    Feasibility leads here for exactly the reason it leads in `rank_design`, and
    the measured case in `search_shapes` is the same one: an 18 mm-walled vessel
    that breaks two rules cannot be cast, so whatever capacity it computes is the
    capacity of an object that does not exist. Capacity is a weaker basis for
    overriding feasibility than the score is, not a stronger one — it is a
    literature UCS envelope on a single study times an ASSUMED organism derating,
    so it must not be able to buy a design past a broken hard rule.

    The 5th percentile leads the capacity comparison for the same reason
    `rank_design` ranks on `score_lo`: the interval spans the ASSUMED 0.3-0.7
    derating on top of a log-uniform envelope, so the median rewards whichever
    design happens to have the widest one. `score_lo` breaks the remaining ties,
    which keeps two designs of equal capacity ordered by which is likelier to
    cement at all.
    """
    return (-int(r.get("n_fail", 0)), float(r.get("capacity_lo_kN", 0.0)),
            float(r.get("score_lo", 0.0)))


#: The two things a search can be asked to optimise. "viability" is the default
#: and reproduces the previous behaviour exactly.
OBJECTIVES = {"viability": rank_design, "strength": rank_strength}


def _row_from(r: dict) -> dict:
    """The scalar summary of one scored design — the only part a search row needs.

    Kept apart from `evaluate` because it is also the pool's return payload, and what
    `evaluate` returns is mostly unshippable: `_mesh` is a Trimesh and `_diag` carries
    the full voxel stack, so returning the raw dict would pickle 190 MB of grids back
    down a pipe per candidate to compute a dozen floats from them. Shared by the
    parallel and serial paths so the two cannot drift.
    """
    return {"score": r["score"], "score_lo": r["score_lo"], "score_hi": r["score_hi"],
            "feasible": r["feasible"], "n_fail": r["n_fail"],
            "limiting": r["dominant_failure_mode"],
            "section_mm": r["min_section_measured_mm"],
            "volume_cm3": r["volume_mm3"] / 1000.0,
            "cemented_fraction": r["cemented_fraction"],
            "capacity_kN": r["capacity_kN"], "capacity_lo_kN": r["capacity_lo_kN"],
            "failed": ", ".join(r["failed_rules"])}


def _score_one(args) -> dict | None:
    """Score one design and return only its summary. The pool's unit of work.

    Module-level and taking plain data on purpose: a forkserver or spawn child
    unpickles this by qualified name, so it cannot be a closure inside
    `search_shapes`, and the `derive` callable is applied in the PARENT rather than
    shipped — `derive_geom` lives in `biocast.gui.app`, and importing that module in a
    worker would execute `st.set_page_config` at import time.

    Returns None for a geometry that fails to build, exactly as the serial path did;
    an unbuildable candidate is a normal outcome of sampling a box, not an error.
    """
    typology, geom_kw, mix_kw, proc_kw, jam_ratio, n_mc = args
    try:
        r = evaluate(typology, geom_kw, mix_kw, proc_kw,
                     jam_ratio=jam_ratio, n_mc=n_mc)
    except Exception:
        return None
    return _row_from(r)


def _worker_init() -> None:
    """Keep each worker single-threaded.

    The pool already uses every core, so a worker whose numpy also tries to is
    oversubscribing by the worker count. These are read lazily by the threading
    layers, and the heavy calls here (`distance_transform_edt`, marching cubes,
    elementwise SDF work) are single-threaded regardless — this is belt and braces
    for the ones that are not.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


@contextmanager
def _no_main_reexec():
    """Stop a forkserver/spawn worker from re-running the app script.

    `multiprocessing.spawn` ships the parent's `__main__` path to each new worker,
    which executes it with `runpy.run_path(..., run_name="__mp_main__")` so that
    anything defined there exists for unpickling. Under Streamlit that path is the
    studio: Streamlit installs a synthetic `__main__` whose `__file__` is the running
    script, and `streamlit_app.py` calls `main()` at module level with no
    `if __name__ == "__main__"` guard to stop it — so every worker would boot its own
    copy of the whole app, which is both the CPU this change exists to save and a
    stream of context-less Streamlit warnings.

    Nothing sent to a worker needs `__main__`. `_score_one` and its arguments resolve
    inside `biocast.gui.engine`, which the forkserver preloads; `derive` is applied in
    the parent for this very reason. With no `__file__` and no `__spec__` on
    `__main__`, `get_preparation_data` sends neither key and the child skips the
    fixup entirely.

    Held across the submits, not just construction: `ProcessPoolExecutor` starts
    workers lazily, so the first `submit` is what triggers the copy.
    """
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    had_file = hasattr(main, "__file__")
    file_, spec = getattr(main, "__file__", None), getattr(main, "__spec__", None)
    try:
        if had_file:
            del main.__file__
        main.__spec__ = None
        yield
    finally:
        if had_file:
            main.__file__ = file_
        main.__spec__ = spec


def _make_pool(n_workers: int):
    """A process pool, or None to stay in-process.

    FORKSERVER, not the platform default. Streamlit runs the script in a worker
    thread, and plain `fork` from a multithreaded process inherits whatever locks
    were held at the instant of the call — the classic way to get a child that
    deadlocks inside malloc before it reaches any of our code. A forkserver forks
    from a clean single-threaded process instead.

    Preloading this module into that server is what keeps forkserver affordable:
    without it every child imports numpy, scipy and trimesh for itself and pays
    ~200 MB, whereas children forked from a server that already imported them share
    those pages copy-on-write. It also moves the import cost off the first candidate.
    """
    if n_workers <= 1:
        return None
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("forkserver" if "forkserver" in methods else "spawn")
        if hasattr(ctx, "set_forkserver_preload"):
            ctx.set_forkserver_preload(["biocast.gui.engine"])
        return ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                   initializer=_worker_init)
    except Exception:
        # A sandbox that forbids subprocesses is a reason to run serially, not to
        # fail the search. The caller's results are identical either way.
        return None


def search_shapes(typology: str, space: dict, choices: dict, derive,
                  mix_kw: dict, proc_kw: dict, *, jam_ratio: float = JAM_RATIO_LIT,
                  n_random: int = 24, n_refine: int = 2, seed: int = 0,
                  n_mc: int = 150, start: dict | None = None,
                  workers: int | None = None, objective: str = "viability",
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

    `objective` chooses what "best" means. "viability" is the default and the
    behaviour above, ranking on `score_lo`. "strength" swaps the tiebreak for
    `capacity_lo_kN` via `rank_strength` — but keeps `-n_fail` in front, so the
    strongest VIABLE shape is what comes back rather than the strongest shape.
    The paragraph above is the reason: the 18 mm-walled vessel that outranked the
    27 mm one on score alone cannot be cast at either, and a capacity computed
    for it is the capacity of an object that does not exist. Capacity is the
    weaker basis for that override, not the stronger one — its interval carries
    an ASSUMED organism derating on top of a single-study envelope. Everything
    else about the search is identical, including the RNG stream, so the two
    objectives explore the same candidates and differ only in which they keep.

    EVERY CANDIDATE IS SAMPLED ON THE SLIDER'S OWN GRID. The search used to draw
    continuous values while the "use this design" button rounded them to each
    slider's `step` on the way back — so the design handed over was not the design
    that was scored, and the Explore tab could report a feasible winner that the
    Design tab immediately rejected. Measured over 60 shell draws at d_max = 4 mm,
    that rounding alone changed the verdict on 5 of them, in both directions. It is
    worse than the step sizes suggest because the section is measured off a 2 mm
    voxel grid: a 0.15 mm nudge to `wall` moves it a whole voxel, and at
    d_max = 4 mm the jamming limit is 24.0 mm, which is exactly where the measured
    section tends to land. `_snap` makes the handover lossless instead of nearly so.

    THE SAMPLING STAGE RUNS ON EVERY CORE `cpu_budget` FINDS; the refinement does not,
    and cannot. A compass search accepts the first improving move and continues from
    the point it just moved to, so the later probes of a sweep depend on how the
    earlier ones turned out — evaluating them together would score a different set of
    designs and could return a different winner. Since the sampling stage is both the
    larger share of the budget and, by the paragraph above, the stage that actually
    finds the design, that is where the cores go. `workers=1` is not a fallback path:
    it runs the identical code in-process.

    RESULTS DO NOT DEPEND ON THE WORKER COUNT. Every candidate is drawn from the RNG
    and deduplicated BEFORE any of them is dispatched, and the rows are collected back
    in draw order rather than completion order — so the same seed gives the same
    table, in the same order, on 2 cores or 32.

    Returns every candidate scored, best first.
    """
    try:
        rank = OBJECTIVES[objective]
    except KeyError:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}, "
                         f"got {objective!r}") from None
    rng = np.random.default_rng(int(seed))
    names = list(space)
    rows, seen = [], set()

    def _snap(k: str, v: float) -> float:
        """Put a value on the grid the slider for `k` can actually hold."""
        s = space[k]
        st = float(s.get("step") or 0.0)
        if st <= 0:
            return float(np.clip(v, s["lo"], s["hi"]))
        return float(np.clip(round(float(v) / st) * st, s["lo"], s["hi"]))

    all_names = names + list(choices)
    budget = int(n_refine) * len(all_names) * 4     # ceiling, not a target
    total = int(n_random) + budget + (1 if start else 0)
    done = 0
    pool = None

    def _tick():
        nonlocal done
        done += 1
        if progress:
            progress(min(done / max(total, 1), 1.0), done, total)

    def _key(g):
        return tuple(round(float(g[k]), 4) if isinstance(g[k], (int, float)) else g[k]
                     for k in sorted(g))

    def _score_batch(cands: list[dict]) -> list[dict]:
        """Score a whole batch, in draw order, across the pool if there is one.

        `derive` is applied HERE rather than in the worker: it is pure dict work that
        costs nothing, and it lives in the Streamlit module, which a worker must not
        import. See `_score_one`.
        """
        fresh, dups = [], 0
        for g in cands:
            k = _key(g)
            if k in seen:
                dups += 1
                continue
            seen.add(k)
            fresh.append(g)
        for _ in range(dups):                   # a duplicate still consumed a draw
            _tick()
        if not fresh:
            return []

        work = [(typology, derive(typology, g), mix_kw, proc_kw, jam_ratio, n_mc)
                for g in fresh]
        subs: list[dict | None] = [None] * len(fresh)
        landed = [False] * len(fresh)

        def _serially(idx):
            for i in idx:
                subs[i] = _score_one(work[i])
                landed[i] = True
                _tick()

        if pool is None or len(fresh) == 1:
            _serially(range(len(work)))
        else:
            from concurrent.futures import as_completed
            from concurrent.futures.process import BrokenProcessPool
            futs = {pool.submit(_score_one, w): i for i, w in enumerate(work)}
            try:
                # ticked as they land, so the bar tracks real progress; STORED by
                # index, so `rows` stays in draw order however completions interleave.
                for f in as_completed(futs):
                    i = futs[f]
                    subs[i], landed[i] = f.result(), True
                    _tick()
            except (BrokenProcessPool, OSError):
                # A worker died mid-flight — in practice the OOM killer, on a box
                # where BIOCAST_WORKERS was set above what `cpu_budget` would have
                # allowed. Finish what is left in-process: a slow search is a better
                # outcome than a lost one, and `_score_one` is pure, so re-running
                # the unfinished candidates is safe.
                _serially([i for i, ok in enumerate(landed) if not ok])

        got = []
        for g, sub in zip(fresh, subs):
            if sub is None:                     # geometry that would not build
                continue
            row = {**g, **sub}
            rows.append(row)
            got.append(row)
        return got

    def _score(g):
        got = _score_batch([g])
        return got[0] if got else None

    # Drawn in full BEFORE anything is scored, so the candidate set comes from the
    # seed alone and not from how the pool happens to interleave.
    batch = []
    # `start` only if it covers every searched parameter: a partial start would be
    # scored with dataclass defaults filling the gaps, and could then be picked as
    # `best` — whose missing keys the refinement's `cur` would immediately fail on.
    if start and all(k in start for k in all_names):
        batch.append({k: (_snap(k, start[k]) if k in space else start[k])
                      for k in all_names})
    for _ in range(int(n_random)):
        g = {k: _snap(k, rng.uniform(s["lo"], s["hi"])) for k, s in space.items()}
        for k, opts in choices.items():
            g[k] = opts[int(rng.integers(len(opts)))]
        batch.append(g)

    with _no_main_reexec():
        pool = _make_pool(min(cpu_budget(workers), len(batch)))
        try:
            _score_batch(batch)
        finally:
            if pool is not None:
                # Shut down HERE, not at the end. The refinement below is sequential,
                # so the workers have nothing left to do, and N idle workers holding
                # N voxel grids is the largest thing this function ever keeps alive.
                pool.shutdown(wait=True)
                pool = None

    if not rows:
        return []
    best = max(rows, key=rank)
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
                r = _score({**cur, k: opt})     # `_score_batch` does the ticking
                used += 1
                if r and rank(r) > rank(best):
                    best, cur = r, {kk: r[kk] for kk in all_names}
                    improved = True
        for k in names:
            s = space[k]
            # at least one slider notch: below that the snapped trial is the point
            # we are already standing on, and `_score` dedupes it away as a no-op.
            step = max(frac * (s["hi"] - s["lo"]), float(s.get("step") or 0.0))
            for sign in (+1, -1):
                if used >= budget:
                    break
                trial = dict(cur)
                trial[k] = _snap(k, cur[k] + sign * step)
                r = _score(trial)
                used += 1
                if r and rank(r) > rank(best):
                    best, cur = r, {kk: r[kk] for kk in all_names}
                    improved = True
                    break                     # move on; this direction paid off
        if not improved:
            frac /= 2.0                       # only shrink when a full sweep fails

    rows.sort(key=rank, reverse=True)
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
