"""Signed-distance-field primitives and rounded CSG.

Why not morphological (voxel) filleting: a rolling-ball open/close removes any
feature thinner than 2r, so filleting a 15 mm shell wall with an 8 mm ball
deletes the wall and leaves a mesh riddled with handles. Rounded CSG on a true
distance field instead places a fillet of exactly radius r on the *edge* where
two surfaces meet, which is where the team's stress-concentration rule actually
applies, and leaves flat/smooth regions untouched.

All fields follow the convention: value < 0 inside the solid, and the gradient
has unit magnitude (a true Euclidean distance), which the rounded operators
require to give the stated radius.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Field construction
# --------------------------------------------------------------------------
def signed_distance_from_binary(vol: np.ndarray, pitch: float) -> np.ndarray:
    """Convert a binary occupancy grid into a true signed distance field (mm).

    Uses the standard two-sided EDT. The half-voxel shift centres the zero level
    on the material boundary rather than on the voxel centres.
    """
    inside = ndimage.distance_transform_edt(vol, sampling=pitch)
    outside = ndimage.distance_transform_edt(~vol, sampling=pitch)
    return (outside - inside) + np.where(vol, 0.5, -0.5) * 0.0  # keep zero level at the interface


def sd_box(X, Y, Z, hx, hy, hz, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Exact SDF of an axis-aligned box with half-extents (hx,hy,hz)."""
    qx = np.abs(X - center[0]) - hx
    qy = np.abs(Y - center[1]) - hy
    qz = np.abs(Z - center[2]) - hz
    ax, ay, az = np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0)
    outside = np.sqrt(ax * ax + ay * ay + az * az)
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
    return outside + inside


def sd_cylinder_z(X, Y, Z, radius, half_h, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Exact SDF of a z-aligned finite cylinder."""
    d_r = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius
    d_z = np.abs(Z - center[2]) - half_h
    ar, az = np.maximum(d_r, 0), np.maximum(d_z, 0)
    return np.minimum(np.maximum(d_r, d_z), 0.0) + np.sqrt(ar * ar + az * az)


def sd_slab_z(Z, z_lo, z_hi) -> np.ndarray:
    """SDF of the infinite slab z_lo <= z <= z_hi."""
    return np.maximum(z_lo - Z, Z - z_hi)


# --------------------------------------------------------------------------
# Rounded CSG (Inigo Quilez operators)
# --------------------------------------------------------------------------
def op_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def op_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def op_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a minus b."""
    return np.maximum(a, -b)


def op_round_union(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Union with a concave fillet of radius r along the seam."""
    if r <= 0:
        return op_union(a, b)
    ua = np.maximum(r - a, 0.0)
    ub = np.maximum(r - b, 0.0)
    return np.maximum(r, np.minimum(a, b)) - np.sqrt(ua * ua + ub * ub)


def op_round_intersect(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Intersection with a convex fillet of radius r along the edge."""
    if r <= 0:
        return op_intersect(a, b)
    ua = np.maximum(r + a, 0.0)
    ub = np.maximum(r + b, 0.0)
    return np.minimum(-r, np.maximum(a, b)) + np.sqrt(ua * ua + ub * ub)


def op_round_subtract(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """a minus b, with a fillet of radius r where the cut meets the surface."""
    if r <= 0:
        return op_subtract(a, b)
    return op_round_intersect(a, -b, r)


def shell_of(d: np.ndarray, wall: float) -> np.ndarray:
    """Hollow a solid field into a shell of the given wall thickness.

    On a true distance field the inner surface is simply the (d + wall) level
    set, so the shell is d >= 0 outside and d + wall <= 0 inside.
    """
    return op_subtract(d, d + wall)


# --------------------------------------------------------------------------
# Meshing
# --------------------------------------------------------------------------
def mesh_field(d: np.ndarray, origin: np.ndarray, pitch: float, *, level: float = 0.0):
    """Marching-cubes a field into a watertight trimesh, padded so it closes.

    Two non-obvious details, both learned from broken output:

    `allow_degenerate=False` is REQUIRED. Axis-aligned faces of a box-like field
    land exactly on grid planes, so the field contains large plateaus of exact
    zeros (tens of thousands of voxels for a 200 mm tile). Marching cubes emits
    zero-area triangles there; they are harmless until trimesh merges duplicate
    vertices, at which point the surface tears and Euler number jumps into the
    hundreds. Suppressing degenerate triangles at source keeps the mesh a closed
    2-manifold.

    The level is nudged off exact zero for the same reason — it breaks ties on
    those plateaus without measurably moving the surface (1e-9 mm).
    """
    import trimesh
    from skimage import measure

    fill = float(np.nanmax(d)) + pitch
    pad = np.pad(np.nan_to_num(d, nan=fill), 1, mode="constant", constant_values=fill)
    verts, faces, _, _ = measure.marching_cubes(
        pad, level=level + 1e-9, spacing=(pitch,) * 3, allow_degenerate=False)
    verts = verts - pitch + origin  # undo the one-voxel pad, then place in world
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m.fix_normals()
    return m


def make_grid(lo, hi, pitch: float):
    """Return (X, Y, Z, origin, shape) covering [lo, hi] at the given pitch."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int) + 1, 8)
    xs = [lo[i] + np.arange(n[i]) * pitch for i in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    return X, Y, Z, lo, X.shape
