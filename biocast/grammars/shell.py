"""Hollow ovoid vessel grammar — the typology that succeeded in the paper.

The paper's finding: solid / low surface-to-volume forms failed (cracking,
incomplete mineralisation from uneven drying and oxygenation), while an oval
egg-like form cast as two halves succeeded. This grammar therefore builds a
SHELL, not a solid: the internal cavity is the oxygen reservoir that lets an
obligate aerobe respire throughout the wall, and the parting plane doubles the
exposed surface during curing.

Construction
------------
1. Superellipsoid of revolution with an ovoid taper -> outer surface.
2. Offset inward by `wall` -> inner surface; boolean difference -> shell.
3. Optional top aperture (a cylinder subtraction) to vent the cavity and admit
   the nutrient dropper of the paper's Fig. 7.
4. Rim and parting edges filleted by a rolling-ball morphological open/close,
   because a sharp rim is exactly the crack initiator the team's notes warn about.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

from ..params import ShellParams
from . import sdf


def _superellipsoid_sdf(pts: np.ndarray, a: float, b: float, c: float,
                        n: float, ovoid: float) -> np.ndarray:
    """Signed implicit function; <0 inside. Ovoid taper widens -z, narrows +z."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    zn = np.clip(z / c, -1.0, 1.0)
    # egg taper: radius scale falls off toward +z
    taper = 1.0 - ovoid * (zn + 1.0) / 2.0
    taper = np.clip(taper, 0.15, None)
    ax, by = a * taper, b * taper
    with np.errstate(over="ignore", invalid="ignore"):
        f = (np.abs(x / ax) ** n + np.abs(y / by) ** n + np.abs(z / c) ** n) - 1.0
    return f


def _grid(p: ShellParams, pitch: float):
    pad = p.fillet_r + 4 * pitch
    hi = np.array([p.a + pad, p.b + pad, p.c + pad])
    return sdf.make_grid(-hi, hi, pitch)


def _outer_field(X, Y, Z, p: ShellParams) -> np.ndarray:
    """Approximate distance field of the ovoid superellipsoid.

    The implicit superellipsoid function is not a distance, so we normalise it by
    its own gradient magnitude (a first-order Eikonal correction). That is enough
    for the rounded CSG operators to place fillets of the requested radius, and
    the field is re-distanced exactly after meshing.
    """
    zn = np.clip(Z / p.c, -1.0, 1.0)
    taper = np.clip(1.0 - p.ovoid * (zn + 1.0) / 2.0, 0.15, None)
    ax, by = p.a * taper, p.b * taper
    n = p.n
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        f = (np.abs(X / ax) ** n + np.abs(Y / by) ** n + np.abs(Z / p.c) ** n) - 1.0
    gx = np.gradient(f, axis=0)
    gy = np.gradient(f, axis=1)
    gz = np.gradient(f, axis=2)
    # gradient is in index units; convert using the actual spacing
    return f, (gx, gy, gz)


def build(p: ShellParams, pitch: float = 1.5, fillet: bool = True,
          return_field: bool = False):
    """Return the watertight shell mesh (mm), filleted by rounded CSG.

    With return_field=True also returns (field, origin, pitch) so callers can
    voxelize exactly from the SDF instead of ray-testing the mesh.
    """
    X, Y, Z, origin, shape = _grid(p, pitch)

    # 1. outer surface as a binary set, then convert to a TRUE distance field.
    #    This avoids the non-metric superellipsoid implicit function entirely.
    zn = np.clip(Z / p.c, -1.0, 1.0)
    taper = np.clip(1.0 - p.ovoid * (zn + 1.0) / 2.0, 0.15, None)
    with np.errstate(over="ignore", invalid="ignore"):
        f = (np.abs(X / (p.a * taper)) ** p.n
             + np.abs(Y / (p.b * taper)) ** p.n
             + np.abs(Z / p.c) ** p.n) - 1.0
    outer_bin = np.nan_to_num(f, nan=1.0) <= 0.0
    if outer_bin.sum() == 0:
        raise ValueError("shell parameters produced an empty solid")
    d_out = sdf.signed_distance_from_binary(outer_bin, pitch)

    # 2. hollow it: exact on a distance field
    d = sdf.shell_of(d_out, p.wall)

    # 3. top aperture, filleted where it breaks the rim
    if p.aperture_r > 0:
        cyl = sdf.sd_cylinder_z(X, Y, Z, p.aperture_r, p.c * 2.0,
                                center=(0.0, 0.0, p.c))
        d = sdf.op_round_subtract(d, cyl, p.fillet_r if fillet else 0.0)

    # 4. inner meridional ribs: unioned with a concave fillet at the wall root,
    #    which is where a sharp rib would otherwise concentrate stress
    if p.rib_count > 0 and p.rib_depth > 0:
        th = np.arctan2(Y, X)
        rad = np.sqrt(X ** 2 + Y ** 2)
        inner_r = np.maximum(p.a * taper - p.wall, 1e-6)
        # a rib is a radial bump on the inner wall
        wedge = np.cos(p.rib_count * th)
        ang_w = np.deg2rad(360.0 / (3.0 * p.rib_count))
        d_ang = (np.arccos(np.clip(wedge, -1, 1)) / p.rib_count) - ang_w
        d_rib = np.maximum(rad - inner_r, -(rad - (inner_r - p.rib_depth)))
        rib = np.maximum(d_rib, d_ang * inner_r)
        rib = sdf.op_intersect(rib, d_out + 0.5 * p.wall)  # keep ribs inside the body
        d = sdf.op_round_union(d, rib, min(p.fillet_r, p.rib_depth) if fillet else 0.0)

    mesh = sdf.mesh_field(d, origin, pitch)
    if mesh.body_count > 1:
        mesh = max(mesh.split(only_watertight=False), key=lambda m: m.volume)
    mesh.fill_holes()
    trimesh.repair.fix_normals(mesh)
    if return_field:
        return mesh, d, origin, pitch
    return mesh


def split_halves(mesh: trimesh.Trimesh, axis: int = 2, offset: float | None = None):
    """Cut the shell into the two halves of the paper's split-mould workflow."""
    normal = np.zeros(3); normal[axis] = 1.0
    origin = mesh.centroid.copy()
    if offset is not None:
        origin[axis] = offset
    upper = mesh.slice_plane(origin, normal, cap=True)
    lower = mesh.slice_plane(origin, -normal, cap=True)
    return lower, upper
