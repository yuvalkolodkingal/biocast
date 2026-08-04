"""Regenerate every design and mould STL referenced in the docs.

The large meshes are not committed (a filleted block at 2.5 mm voxel pitch is ~17 MB
of ASCII STL, and they are derived data). This script rebuilds them from the
parameters recorded in `data/pareto_front.csv` and `docs/mould_notes.md` in about a
minute.

    PYTHONPATH=. python examples/regenerate_meshes.py --out stl

Only the small shell parts are committed, as a smoke test that the pipeline works
without running anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from biocast.params import BlockParams, ShellParams, TileParams
from biocast.grammars import block as bl
from biocast.grammars import shell as sh
from biocast.grammars import tile as tl
from biocast import mould

# Best design per typology from the 6912-cell sweep (data/pareto_front.csv).
# d_max = 2 mm, 21 d at 85 % RH unless the name says otherwise.
BEST = {
    "shell_best_dmax2mm": ShellParams(
        a=58.2, b=75.0, c=71.7, n=2.1, ovoid=0.40,
        wall=19.3, aperture_r=13.5, fillet_r=6.5),
    "shell_best_dmax4mm": ShellParams(
        a=78.0, b=81.0, c=115.0, n=3.3, ovoid=0.0,
        wall=30.6, aperture_r=27.0, fillet_r=10.0),
    "block_best_dmax4mm": BlockParams(
        face_shell=32.0, web=36.0, n_cores=3, fillet_r=6.0, core_taper=0.8),
    "tile_best_dmax4mm": TileParams(
        t=37.0, pattern="radial", groove_depth=4.8, groove_width=33.0,
        groove_pitch=42.0, fillet_r=10.0, joint=6.0, thick_tile=True),
}

# Mould designs (docs/mould_notes.md) sit on the 28-day-cure branch, where the
# feasible window is open at d_max = 4 mm.
MOULD_VESSEL = ShellParams(a=55.0, b=55.0, c=78.0, n=2.4, ovoid=0.28,
                           wall=26.0, aperture_r=16.0, fillet_r=8.0)
MOULD_BLOCK = BlockParams(face_shell=38.0, web=32.0, n_cores=2,
                          fillet_r=8.0, core_taper=2.0)

PITCH = {"shell": 1.5, "block": 2.5, "tile": 1.2}


def build(p, pitch=None):
    mod = {"shell": sh, "block": bl, "tile": tl}[p.typology]
    return mod.build(p, pitch=pitch or PITCH[p.typology])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stl", type=Path)
    ap.add_argument("--skip-moulds", action="store_true")
    args = ap.parse_args()
    (args.out / "designs").mkdir(parents=True, exist_ok=True)
    (args.out / "moulds").mkdir(parents=True, exist_ok=True)

    for name, p in BEST.items():
        mesh = build(p)
        assert mesh.is_watertight, f"{name} not watertight — that is a bug, not tolerance"
        mesh.export(args.out / "designs" / f"{name}.stl")
        print(f"{name:26s} vol {mesh.volume/1000:8.1f} cm3  watertight")
        if p.typology == "shell":
            lo, up = sh.split_halves(mesh)
            lo.export(args.out / "designs" / f"{name}_lower.stl")
            up.export(args.out / "designs" / f"{name}_upper.stl")

    if args.skip_moulds:
        return

    # The mould builders return voxel OCCUPANCY GRIDS for each part, not meshes
    # (only `obj_mesh` is a Trimesh). Converting with `occ_to_mesh` is required —
    # iterating for objects that happen to have `.export` silently writes the cast
    # object and none of the mould parts.
    for tag, (fn, p, wanted) in {
        "shell": (mould.build_shell_mould, MOULD_VESSEL,
                  ["lower", "upper", "core_lo", "core_up"]),
        "block": (mould.build_block_mould, MOULD_BLOCK, ["lower", "upper"]),
    }.items():
        parts = fn(p)
        origin, pitch = parts["origin"], parts["pitch"]
        for part in wanted:
            m = mould.occ_to_mesh(parts[part], origin, pitch)
            m.export(args.out / "moulds" / f"mould_{tag}_{part}.stl")
            print(f"mould_{tag}_{part:9s} vol {m.volume/1000:8.1f} cm3  "
                  f"watertight={m.is_watertight}")
        bal = parts.get("balance", {})
        if bal:
            print(f"  volume balance: {bal}")


if __name__ == "__main__":
    main()
