"""Relief paving/cladding tile grammar (Panot-type).

Team notes: 200x200 mm, 40 mm thick, relief 2-3 mm deep (<10 % of thickness),
channels ~10 mm wide, for drainage and slip resistance rather than stiffening.

Four relief patterns are provided. `groove_width` is the parameter the granular
jamming rule bites on hardest: the notes assume 2-3 x d_max is enough, but the
retrieved jamming literature puts the critical aperture ratio near 5-6 x d_max
for angular grains, so a 10 mm channel with 4 mm waste is flagged. The generator
does not silently widen it — it reports the conflict and lets the designer choose.
"""
from __future__ import annotations

import numpy as np
import trimesh

from ..params import TileParams
from . import sdf


def _pattern_field(X: np.ndarray, Y: np.ndarray, p: TileParams) -> np.ndarray:
    """2D signed field of the groove network; <0 inside a groove."""
    w = p.groove_width / 2.0
    pitch = max(p.groove_pitch, 1e-6)

    if p.pattern == "grid":
        fx = np.abs(((X + pitch / 2) % pitch) - pitch / 2) - w
        fy = np.abs(((Y + pitch / 2) % pitch) - pitch / 2) - w
        return np.minimum(fx, fy)

    if p.pattern == "diagonal":
        u = (X + Y) / np.sqrt(2.0)
        v = (X - Y) / np.sqrt(2.0)
        fu = np.abs(((u + pitch / 2) % pitch) - pitch / 2) - w
        fv = np.abs(((v + pitch / 2) % pitch) - pitch / 2) - w
        return np.minimum(fu, fv)

    if p.pattern == "radial":
        r = np.sqrt(X ** 2 + Y ** 2)
        fr = np.abs(((r + pitch / 2) % pitch) - pitch / 2) - w
        return fr

    if p.pattern == "flower":
        # the classic Panot four-petal motif: a rosette of arcs per cell
        cx = ((X + pitch / 2) % pitch) - pitch / 2
        cy = ((Y + pitch / 2) % pitch) - pitch / 2
        th = np.arctan2(cy, cx)
        r = np.sqrt(cx ** 2 + cy ** 2)
        petal_r = pitch * 0.32 * (1.0 + 0.35 * np.cos(4 * th))
        return np.abs(r - petal_r) - w

    raise ValueError(f"unknown pattern {p.pattern!r}")


def build(p: TileParams, pitch: float = 1.0, fillet: bool = True,
          return_field: bool = False):
    """Return the watertight tile mesh (mm), origin at the tile centre.

    Grid pitch defaults finer than the other grammars because the relief is only
    a few millimetres deep and would otherwise be lost to discretisation.
    """
    r = p.fillet_r if fillet else 0.0
    pad = max(r, p.groove_depth) + 4 * pitch
    hi = np.array([p.L / 2 + pad, p.W / 2 + pad, p.t / 2 + pad])
    X, Y, Z, origin, shape = sdf.make_grid(-hi, hi, pitch)

    # tile body with filleted perimeter edges
    r_edge = min(r, p.t / 2 - pitch, min(p.L, p.W) / 4)
    r_edge = max(r_edge, 0.0)
    dx = np.abs(X) - (p.L / 2 - r_edge)
    dy = np.abs(Y) - (p.W / 2 - r_edge)
    dz = np.abs(Z) - (p.t / 2 - r_edge)
    if r_edge > 0:
        ax, ay, az = np.maximum(dx, 0), np.maximum(dy, 0), np.maximum(dz, 0)
        body = (np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
                + np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0) - r_edge)
    else:
        body = sdf.sd_box(X, Y, Z, p.L / 2, p.W / 2, p.t / 2)

    # relief grooves cut into the top face only
    if p.groove_depth > 0 and p.groove_width > 0:
        f2d = _pattern_field(X, Y, p)
        top = p.t / 2
        # groove solid: the pattern prism from (top - depth) upward
        gz = sdf.sd_slab_z(Z, top - p.groove_depth, top + 2 * pitch + p.groove_depth)
        groove = np.maximum(f2d, gz)
        r_root = min(r, p.groove_width / 2.0, p.groove_depth)
        body = sdf.op_round_subtract(body, groove, r_root if fillet else 0.0)

    mesh = sdf.mesh_field(body, origin, pitch)
    if mesh.body_count > 1:
        mesh = max(mesh.split(only_watertight=False), key=lambda m: m.volume)
    mesh.fill_holes()
    trimesh.repair.fix_normals(mesh)
    if return_field:
        return mesh, body, origin, pitch
    return mesh


def tile_array(p: TileParams, nx: int = 2, ny: int = 2, pitch: float = 1.2):
    """Lay out an nx-by-ny composition with the designed joint gap.

    The joint is the feed/oxygen path between units in a wall or pavement, which
    is why it is a hard constraint rather than a detailing preference.
    """
    base = build(p, pitch=pitch)
    step_x = p.L + p.joint
    step_y = p.W + p.joint
    parts = []
    for i in range(nx):
        for j in range(ny):
            m = base.copy()
            m.apply_translation([(i - (nx - 1) / 2) * step_x,
                                 (j - (ny - 1) / 2) * step_y, 0.0])
            parts.append(m)
    return trimesh.util.concatenate(parts)
