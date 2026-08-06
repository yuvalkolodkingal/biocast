"""Regenerate every mould STL, both goals, for all three typologies.

The STLs are deliberately NOT committed — a mesh is a derived artefact of the
design, so it is rebuilt rather than stored. This script is also the integration
test: every part goes through the same checks the Mould tab reports, and a failure
is printed rather than swallowed.

    PYTHONPATH=. python examples/regenerate_moulds.py --out stl/moulds

Runtime is seconds per mould. The voxel generators this replaced took two to six
MINUTES each, because they booleaned 6-24 M voxel grids and ran an oxygen field
solve per candidate window pitch; the only grid left here is the one the aeration
surrogate runs on, at the design's own scoring pitch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from biocast import mould_cast as mc
from biocast.params import BlockParams, ShellParams, TileParams

#: The baseline designs the verification tables refer to. The shell wall and the
#: block face-shell/web are NOT the grammar defaults: they are the values the
#: sieve-vs-cure study settled on for a 28-day cure at d_max = 4 mm.
GEOMS = {
    "shell": ShellParams(wall=26.0),
    "block": BlockParams(face_shell=37.0, web=30.0),
    "tile": TileParams(),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="stl/moulds", help="output directory")
    ap.add_argument("--goal", choices=["silicone", "rigid", "both"], default="both")
    ap.add_argument("--typology", choices=[*GEOMS, "all"], default="all")
    ap.add_argument("--d-max", type=float, default=4.0,
                    help="largest aggregate fragment (mm); sets the window bore")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(GEOMS) if a.typology == "all" else [a.typology]
    goals = ["silicone", "rigid"] if a.goal == "both" else [a.goal]

    problems = []
    for name in names:
        for goal in goals:
            t = time.time()
            spec = mc.CastSpec(goal=goal, d_max=a.d_max)
            res = mc.build_mould(GEOMS[name], spec, cure_days=28.0, rh_pct=90.0)
            s, win = res["parting"], res["window"]
            print(f"[{goal:8s} {name:5s}] {time.time() - t:.1f} s | "
                  f"parting axis {s['axis']} at {s['plane']:.1f} mm "
                  f"(undercut {s['undercut']:.3f})")
            print(f"    windows {win['d_mm']:.0f} mm at {win['spacing_mm']:.0f} mm, "
                  f"cover {win['cover_surrogate']:.3f} "
                  f"{'meets' if win['meets_target'] else 'MISSES'} 0.85 | "
                  f"plastic {res['plastic_cm3']:.0f} cm3"
                  + (f" | silicone {res['silicone_mass_g']:.0f} g "
                     f"({res['n_pillars']} pillars)" if goal == "silicone" else ""))

            for part, mesh in res["parts"].items():
                f = out / f"{goal}_{name}_{part}.stl"
                mesh.export(f)
                r = res["report"][part]
                print(f"      {f.name:34s} {f.stat().st_size / 1e6:5.1f} MB  "
                      f"{r['volume_cm3']:8.1f} cm3  bodies {r['bodies']}  "
                      f"watertight {r['watertight']}")

            for c in res["checks"]:
                if not c["pass"]:
                    problems.append(f"{goal} {name}: {c['check']} — {c['detail']}")

    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S) — reported, not suppressed:")
        for p in problems:
            print("  -", p)
        print("\nA coverage miss is a property of the CAST DESIGN and its cure, not "
              "of the mould: the limit is drying depth, and more open area cannot "
              "fix it. See docs/mould_notes.md.")
    else:
        print("all parts passed every check")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
