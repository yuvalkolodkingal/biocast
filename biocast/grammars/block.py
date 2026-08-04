"""Hollow-core masonry unit grammar (CMU-type).

Proportions from the team's notes: a 20x20x40 cm module, 40-50 % void, ~3.2 cm face
shells, ~2.5 cm webs. The notes give only that nominal module; the 390x190x190 mm
defaults used here are the corresponding ACTUAL dimensions from standard masonry
practice (a nominal size includes the ~10 mm mortar joint), which the notes do not
state.

For bio-cementation the cores are not only a material saving: they are the
oxygen supply to the interior of every web, which is why a block with the
standard void fraction is a far better MICP candidate than a solid brick of the
same footprint. Every core corner is filleted — the team's notes are explicit
that a corner in brittle bio-cement is where the crack starts.
"""
from __future__ import annotations

import numpy as np
import trimesh

from ..params import BlockParams
from . import sdf


def core_dims(p: BlockParams) -> tuple[float, float]:
    """(core_width_along_L, core_depth_along_W) for the given parameter set."""
    core_w = (p.L - 2 * p.face_shell - (p.n_cores - 1) * p.web) / max(p.n_cores, 1)
    core_d = p.W - 2 * p.face_shell
    return core_w, core_d


def core_centers(p: BlockParams) -> list[float]:
    """x-coordinates of the core centres (block centred on the origin)."""
    core_w, _ = core_dims(p)
    xs = []
    x = -p.L / 2 + p.face_shell
    for _ in range(p.n_cores):
        xs.append(x + core_w / 2)
        x += core_w + p.web
    return xs


def void_fraction(p: BlockParams) -> float:
    core_w, core_d = core_dims(p)
    if core_w <= 0 or core_d <= 0:
        return 0.0
    return (p.n_cores * core_w * core_d * p.H) / (p.L * p.W * p.H)


def build(p: BlockParams, pitch: float = 2.5, fillet: bool = True,
          return_field: bool = False):
    """Return the watertight block mesh (mm), origin at the block centre."""
    core_w, core_d = core_dims(p)
    if core_w <= 0 or core_d <= 0:
        raise ValueError(f"invalid block: core {core_w:.1f}x{core_d:.1f} mm is non-positive")

    pad = p.fillet_r + 4 * pitch
    hi = np.array([p.L / 2 + pad, p.W / 2 + pad, p.H / 2 + pad])
    X, Y, Z, origin, shape = sdf.make_grid(-hi, hi, pitch)

    r = p.fillet_r if fillet else 0.0

    # outer body, filleted on all twelve edges by rounded intersection of slabs
    dx = np.abs(X) - (p.L / 2 - r)
    dy = np.abs(Y) - (p.W / 2 - r)
    dz = np.abs(Z) - (p.H / 2 - r)
    if r > 0:
        # exact rounded box: distance to the inner box minus r
        ax, ay, az = np.maximum(dx, 0), np.maximum(dy, 0), np.maximum(dz, 0)
        body = (np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
                + np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0) - r)
    else:
        body = sdf.sd_box(X, Y, Z, p.L / 2, p.W / 2, p.H / 2)

    # cores: vertical prisms through the full height, tapered for release
    taper = np.tan(np.deg2rad(max(p.core_taper, 0.0)))
    for xc in core_centers(p):
        # taper widens the core toward the top (draft for demoulding)
        grow = taper * (Z + p.H / 2)
        hw = core_w / 2 + grow
        hd = core_d / 2 + grow
        cdx = np.abs(X - xc) - np.maximum(hw - r, 1e-6)
        cdy = np.abs(Y) - np.maximum(hd - r, 1e-6)
        ax, ay = np.maximum(cdx, 0), np.maximum(cdy, 0)
        core2d = np.sqrt(ax ** 2 + ay ** 2) + np.minimum(np.maximum(cdx, cdy), 0.0) - r
        core = np.maximum(core2d, np.abs(Z) - (p.H / 2 + pitch * 3))
        body = sdf.op_round_subtract(body, core, r)

    # optional decorative face grooves running vertically on both long faces
    if p.groove_count > 0 and p.groove_depth > 0 and p.groove_width > 0:
        pitch_x = p.L / (p.groove_count + 1)
        for i in range(1, p.groove_count + 1):
            xg = -p.L / 2 + i * pitch_x
            gd = np.abs(X - xg) - p.groove_width / 2
            for sgn in (+1, -1):
                gy = sgn * (Y - sgn * p.W / 2) + p.groove_depth  # depth into the face
                groove = np.maximum(gd, -gy)
                groove = np.maximum(groove, np.abs(Z) - p.H / 2)
                body = sdf.op_round_subtract(body, groove, min(r, p.groove_width / 2))

    mesh = sdf.mesh_field(body, origin, pitch)
    if mesh.body_count > 1:
        mesh = max(mesh.split(only_watertight=False), key=lambda m: m.volume)
    mesh.fill_holes()
    trimesh.repair.fix_normals(mesh)
    if return_field:
        return mesh, body, origin, pitch
    return mesh
