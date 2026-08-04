"""Retrodiction test: does the model reproduce the paper's own outcomes?

The paper reports two facts we can test against, and it reports them
qualitatively, so the test is a RANKING test, not a calibration:

  FAIL    early prototypes — solid forms, "incompatibility with ... optimal
          surface ratio for heterotrophic bacterial growth", cracking and
          incomplete mineralisation from uneven drying and oxygenation (Fig. 5)
  SUCCESS the two halves of an oval egg-like structure, hollow, cast in a split
          mould (Figs. 6-7)

A model that assigns the solid form a lower score than the hollow split shell,
and attributes the solid form's failure to aeration/drying rather than to
something else, has reproduced the experiment. A model that ranks them the other
way round is wrong regardless of how well-sourced its parameters are.
"""
from __future__ import annotations

import json
import numpy as np

from biocast.params import ShellParams, Design, Mix, Process, LitParams
from biocast.grammars import shell as sh, sdf
from biocast.physics import fields as fl
from biocast import score as sc, constraints as cn

PITCH = 2.0


def solid_ovoid_mesh(p: ShellParams, pitch: float = PITCH):
    """The paper's early prototype: same outer form, NO cavity, NO aperture."""
    X, Y, Z, origin, shape = sh._grid(p, pitch)
    zn = np.clip(Z / p.c, -1.0, 1.0)
    taper = np.clip(1.0 - p.ovoid * (zn + 1.0) / 2.0, 0.15, None)
    with np.errstate(over="ignore", invalid="ignore"):
        f = (np.abs(X / (p.a * taper)) ** p.n + np.abs(Y / (p.b * taper)) ** p.n
             + np.abs(Z / p.c) ** p.n) - 1.0
    occ = np.nan_to_num(f, nan=1.0) <= 0.0
    d = sdf.signed_distance_from_binary(occ, pitch)
    return sdf.mesh_field(d, origin, pitch), d, origin, pitch


def evaluate(mesh, design, *, split: bool, phys, label: str,
             notch_depth=0.0, n_mc=400, field=None):
    """Diagnostics + score for one case.

    `split` marks the split-mould workflow: during curing the two halves are cast
    open-faced, so the parting plane is atmosphere. That is the geometric content
    of the paper's Fig. 6, and it is modelled by adding the parting plane to the
    oxygen source set.
    """
    diag = fl.geometric_diagnostics(
        mesh, PITCH,
        parting_axis=2 if split else None,
        parting_frac=0.5 if split else None,
        include_parting=split, field=field)
    res = sc.score_design(design, diag, phys=phys, n_mc=n_mc,
                          notch_depth_mm=notch_depth)
    verdicts = cn.check(design, cn.Thresholds(), diag=res | {
        "sa_to_vol": diag["sa_to_vol"],
        "max_wall_thickness": diag["max_wall_thickness"]})
    summ = cn.summarise(verdicts)
    return {
        "label": label,
        "sa_to_vol": diag["sa_to_vol"],
        "max_wall_thickness": diag["max_wall_thickness"],
        "volume_mm3": diag["volume_mesh_mm3"],
        "cemented_fraction": res["cemented_fraction"],
        "score": res["score"],
        "score_lo": res["score_lo"],
        "score_hi": res["score_hi"],
        "sub_aeration": res["sub_aeration"],
        "sub_drying": res["sub_drying"],
        "sub_castability": res["sub_castability"],
        "sub_structural": res["sub_structural"],
        "dominant_failure_mode": res["dominant_failure_mode"],
        "failure_mode_text": res["failure_mode_text"],
        "penetration_depth_nom_mm": res["penetration_depth_nom_mm"],
        "feasible": summ["feasible"],
        "n_fail": summ["n_fail"],
        "failed_rules": summ["failed_rules"],
    }


def main():
    kin = LitParams("micp_kinetics_params.json")
    mec = LitParams("mechanics_params.json")
    phys = sc.PhysicsInputs.from_lit(kin, mec)

    mix = Mix(d_max=4.0)
    proc_split = Process(split_mould=True)
    proc_mono = Process(split_mould=False)

    base = ShellParams()
    rows = []

    # CASE A — the paper's failed early prototype: solid ovoid, monolithic mould
    solid_p = ShellParams(wall=base.c, aperture_r=0.0)   # wall >= semi-axis => no cavity
    m_solid, d_solid, o_solid, p_solid = solid_ovoid_mesh(base)
    fld_solid = (d_solid, o_solid, p_solid)
    rows.append(evaluate(m_solid, Design(geom=solid_p, mix=mix, proc=proc_mono),
                         split=False, phys=phys,
                         field=fld_solid,
                         label="A: solid ovoid, monolithic mould (paper Fig. 5 — FAILED)"))

    # CASE B — same solid form but cast in halves: isolates the split-mould effect
    rows.append(evaluate(m_solid, Design(geom=solid_p, mix=mix, proc=proc_split),
                         split=True, phys=phys,
                         field=fld_solid,
                         label="B: solid ovoid, split mould (isolates parting-plane effect)"))

    # CASE C — hollow shell, monolithic: isolates the hollowing effect
    m_shell, d_sh, o_sh, p_sh = sh.build(base, pitch=PITCH, return_field=True)
    fld_shell = (d_sh, o_sh, p_sh)
    rows.append(evaluate(m_shell, Design(geom=base, mix=mix, proc=proc_mono),
                         split=False, phys=phys,
                         field=fld_shell,
                         label="C: hollow shell, monolithic mould (isolates hollowing)"))

    # CASE D — the paper's successful design: hollow shell, split mould
    rows.append(evaluate(m_shell, Design(geom=base, mix=mix, proc=proc_split),
                         split=True, phys=phys,
                         field=fld_shell,
                         label="D: hollow shell, split mould (paper Figs. 6-7 — SUCCEEDED)"))

    # CASE E — a deliberately sharp-cornered variant to exercise the notch term
    sharp = ShellParams(fillet_r=0.5)
    m_sharp, d_sp, o_sp, p_sp = sh.build(sharp, pitch=PITCH, return_field=True)
    rows.append(evaluate(m_sharp, Design(geom=sharp, mix=mix, proc=proc_split),
                         split=True, phys=phys, notch_depth=8.0,
                         field=(d_sp, o_sp, p_sp),
                         label="E: hollow shell, sharp rim r=0.5 mm (notch-sensitivity probe)"))

    json.dump(rows, open("validation_paper.json", "w"), indent=1)

    hdr = f"{'case':62s} {'S/V':>7s} {'t_max':>6s} {'cem':>5s} {'score':>6s} {'[5-95%]':>13s} {'mode':>12s} {'feas':>5s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:62]:62s} {r['sa_to_vol']:7.4f} {r['max_wall_thickness']:6.1f} "
              f"{r['cemented_fraction']:5.2f} {r['score']:6.3f} "
              f"[{r['score_lo']:.3f},{r['score_hi']:.3f}] {r['dominant_failure_mode'][:12]:>12s} "
              f"{str(r['feasible']):>5s}")

    a, d = rows[0], rows[3]
    print("\nRETRODICTION:")
    print(f"  paper FAILED case A score = {a['score']:.3f}  (mode: {a['dominant_failure_mode']})")
    print(f"  paper SUCCESS case D score = {d['score']:.3f}  (mode: {d['dominant_failure_mode']})")
    ok_rank = d["score"] > a["score"]
    ok_mode = a["dominant_failure_mode"] in ("aeration", "drying")
    print(f"  ranking correct: {ok_rank};  failure attributed to aeration/drying: {ok_mode}")
    return rows


if __name__ == "__main__":
    main()
