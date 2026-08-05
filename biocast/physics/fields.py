"""Geometric field diagnostics on a cast body.

Everything here works on a voxel sampling of the mesh interior, which lets one
code path serve all three typologies (no analytic special-casing).

Fields computed
---------------
depth      : distance from each interior voxel to the nearest exposed surface (mm)
thickness  : local wall thickness = 2 x the distance to the nearest surface for
             a voxel on the medial surface; reported as the distribution of 2*depth
sa_to_vol  : exposed surface area / solid volume (1/mm)

The `exposed` surface is not simply the mesh boundary: in a split-mould cast the
parting face is open to air during curing (paper Fig. 6), and an internal cavity
communicating with the outside is also an oxygen source. `exposure_mask` encodes
that, and the depth field is a multi-source distance transform from all exposed
faces at once.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def voxelize(mesh, pitch: float):
    """Return (occ, origin, pitch): boolean occupancy grid of the solid.

    Uses a winding-number/ray test (`mesh.contains`) on voxel centres rather than
    `VoxelGrid.fill()`. This is load-bearing for hollow typologies: `fill()` uses
    a flood/morphological fill that does not distinguish "inside the material"
    from "inside a sealed internal cavity", so it floods the shell's cavity and
    reports a 14 mm-walled vessel as a solid lump — a 27 % volume overestimate
    that silently destroys the aeration subscore (every design then looks solid,
    and the cemented fraction saturates at 1.0 for all of them).

    Prefer `occupancy_from_field` when the caller still has the generating SDF:
    `trimesh.contains` runs a ray/winding test per point and its memory use is
    not proportional to the chunk you hand it, so on object-scale grids it gets
    the kernel OOM-killed even when the grid itself is small. Every grammar in
    this package builds from a field, so that path is both cheaper and exact.
    """
    lo = mesh.bounds[0] - pitch
    hi = mesh.bounds[1] + pitch
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int) + 1, 4)
    axes = [lo[i] + np.arange(n[i]) * pitch for i in range(3)]
    occ = np.zeros(tuple(n), dtype=bool)

    # chunk over the slowest axis, small enough that the ray engine stays bounded
    chunk = max(1, int(2e5 // max(n[1] * n[2], 1)))
    Y, Z = np.meshgrid(axes[1], axes[2], indexing="ij")
    yz = np.column_stack([Y.ravel(), Z.ravel()])
    for i0 in range(0, n[0], chunk):
        i1 = min(i0 + chunk, n[0])
        xs = axes[0][i0:i1]
        pts = np.empty((len(xs) * yz.shape[0], 3))
        pts[:, 0] = np.repeat(xs, yz.shape[0])
        pts[:, 1] = np.tile(yz[:, 0], len(xs))
        pts[:, 2] = np.tile(yz[:, 1], len(xs))
        occ[i0:i1] = mesh.contains(pts).reshape(len(xs), n[1], n[2])

    return occ, lo, pitch


def occupancy_from_field(d: np.ndarray, origin: np.ndarray, pitch: float):
    """Occupancy grid straight from a signed-distance field (d <= 0 is solid).

    This is the preferred entry point: exact by construction, no ray casting, and
    it preserves internal cavities that a flood fill would close.
    """
    return (np.asarray(d) <= 0.0), np.asarray(origin, float), float(pitch)


def exposure_mask(occ: np.ndarray, *, parting_axis: int | None = None,
                  parting_index: int | None = None,
                  open_top: bool = False) -> np.ndarray:
    """Boolean grid marking air voxels that act as oxygen sources.

    All air voxels connected to the outside of the bounding box are sources;
    fully sealed internal pockets are not. If a parting plane is given, the
    solid faces adjacent to it are treated as exposed too.
    """
    air = ~occ
    # air connected to the grid boundary = the outside atmosphere (and any cavity
    # that communicates with it, e.g. through an aperture)
    pad = np.pad(air, 1, constant_values=True)
    lbl, _ = ndimage.label(pad)
    outside_label = lbl[0, 0, 0]
    connected = (lbl == outside_label)[1:-1, 1:-1, 1:-1]
    src = air & connected

    if parting_axis is not None and parting_index is not None:
        # a thin air slab at the parting plane: during split-mould curing the two
        # halves are cast open-faced, so this plane is atmosphere
        sl = [slice(None)] * 3
        sl[parting_axis] = slice(max(parting_index, 0), parting_index + 1)
        slab = np.zeros_like(occ)
        slab[tuple(sl)] = True
        src = src | slab
    return src


def exposure_mask_in_mould(occ: np.ndarray, mould_occ: np.ndarray, *,
                           parting_axis: int | None = None,
                           parting_index: int | None = None,
                           open_parting_face: bool = False) -> dict:
    """Oxygen sources for a body sitting IN a mould, where mould faces are no-flux.

    `exposure_mask` answers "which air can reach the atmosphere" for a bare body,
    and every air voxel connected to the grid boundary qualifies. That is the right
    question for a demoulded body and the wrong one for a body still in its mould:
    the air path from the cast surface to the atmosphere runs through the mould wall,
    which is solid. Using the bare mask on a moulded body silently grants a fully
    enclosed cavity the same atmosphere access as an open-faced cast — and since the
    aeration subscore saturates when everything looks exposed, every mould then
    scores as if it were the paper's successful open-faced Fig. 6 case, including the
    fully enclosed geometry that reproduces its Fig. 5 failure.

    This matters more for an elastomeric mould than it looks. Silicone is highly
    oxygen-permeable in absolute terms (350-800 Barrer for PDMS), which invites the
    conclusion that a silicone skin breathes. It does not, because the comparison
    that decides cementation is against DRAINED PORES, not against water:

        drained-pore equivalent permeability  D_eff/(R T) ~ 2.3e-10 mol m/(m2 s Pa)
        silicone at 520 Barrer                            ~ 1.7e-13
        => a 6 mm skin carries ~300x the resistance of the 26 mm wall behind it

    Against water-filled pores the same skin is transparent (~1e-3 of their
    resistance), which is exactly why the intuition misleads: PDMS beats water and
    loses to air by two and a half orders of magnitude. The vapour side compounds it
    — a 6 mm silicone skin passes on the order of 1 % of the free evaporation rate,
    and since oxygen only travels far through pores evaporation has already drained,
    throttling drying throttles aeration too.

    So both rigid and elastomeric mould faces are treated as no-flux here, and only
    genuinely OPEN area — breather windows, the open parting face of a split mould,
    an aperture that is not covered by mould — acts as atmosphere. `mould_occ` must
    be on the same grid as `occ`; `mould_auto` returns its parts on the object's
    grid for this reason.

    Returns the mask plus the open-area bookkeeping, because "what fraction of the
    cast surface is actually open" is the number that decides whether a mould can
    cement at all, and it should be reported rather than left implicit.
    """
    air = ~occ & ~mould_occ                      # void that is neither cast nor mould
    pad = np.pad(air, 1, constant_values=True)
    lbl, _ = ndimage.label(pad)
    connected = (lbl == lbl[0, 0, 0])[1:-1, 1:-1, 1:-1]
    src = air & connected

    if open_parting_face and parting_axis is not None and parting_index is not None:
        # A split mould cured open-faced: the parting plane is atmosphere even where
        # mould material flanks it. This is the ONE case where mould geometry does not
        # block, and it is the mechanism behind the paper's successful cast — so it is
        # opt-in rather than assumed, because assembling the halves early converts
        # that face back into a sealed interface.
        sl = [slice(None)] * 3
        sl[parting_axis] = slice(max(parting_index, 0), parting_index + 1)
        slab = np.zeros_like(occ)
        slab[tuple(sl)] = True
        src = src | (slab & ~mould_occ)

    def _contact(mask):
        n = 0
        for ax in range(3):
            for shift in (1, -1):
                nb = np.roll(mask, shift, axis=ax)
                idx = [slice(None)] * 3
                idx[ax] = 0 if shift == 1 else -1
                nb[tuple(idx)] = False
                n += int((occ & nb).sum())
        return n

    open_faces = _contact(src)
    sealed_faces = _contact(mould_occ)
    total = open_faces + sealed_faces
    return {"src": src, "open_faces": open_faces, "sealed_faces": sealed_faces,
            "open_area_frac": float(open_faces / max(total, 1)),
            "note": ("mould faces treated as no-flux; only open area is atmosphere"
                     if total else "no cast/mould contact found — check grids align")}


def depth_field(occ: np.ndarray, src: np.ndarray, pitch: float) -> np.ndarray:
    """Distance (mm) from every solid voxel to the nearest oxygen source voxel."""
    # EDT of the complement of the source set gives distance-to-source everywhere
    dist = ndimage.distance_transform_edt(~src, sampling=(pitch, pitch, pitch))
    out = np.full(occ.shape, np.nan)
    out[occ] = dist[occ]
    return out


def geometric_diagnostics(mesh, pitch: float, *, parting_axis: int | None = None,
                          parting_frac: float | None = None,
                          include_parting: bool = True,
                          field: tuple | None = None) -> dict:
    """Compute the geometry-only diagnostics used by the constraint checker.

    `field` is an optional (d, origin, pitch) tuple from a grammar's
    `build(..., return_field=True)`. Pass it whenever available: occupancy then
    comes from the SDF exactly, with internal cavities preserved and no ray
    casting. `mesh` is still used for its area/volume reference values.
    """
    if field is not None:
        d_fld, origin, pitch = field
        occ, origin, pitch = occupancy_from_field(d_fld, origin, pitch)
    else:
        occ, origin, pitch = voxelize(mesh, pitch)
    if occ.sum() == 0:
        raise ValueError("voxelisation produced an empty solid; reduce pitch")

    p_idx = None
    if include_parting and parting_axis is not None:
        frac = 0.5 if parting_frac is None else parting_frac
        p_idx = int(round(frac * (occ.shape[parting_axis] - 1)))

    src = exposure_mask(occ, parting_axis=parting_axis, parting_index=p_idx)
    depth = depth_field(occ, src, pitch)
    d = depth[occ]

    vol_vox = occ.sum() * pitch ** 3
    # exposed area: count solid-face/source-face contacts
    faces = 0
    for ax in range(3):
        for shift in (1, -1):
            nb = np.roll(src, shift, axis=ax)
            # zero out the wrapped plane
            idx = [slice(None)] * 3
            idx[ax] = 0 if shift == 1 else -1
            nb[tuple(idx)] = False
            faces += (occ & nb).sum()
    area_vox = faces * pitch ** 2

    return {
        "voxel_pitch": pitch,
        "n_solid_voxels": int(occ.sum()),
        "volume_voxel_mm3": float(vol_vox),
        "volume_mesh_mm3": float(mesh.volume),
        "area_mesh_mm2": float(mesh.area),
        "exposed_area_mm2": float(area_vox),
        "sa_to_vol": float(area_vox / vol_vox) if vol_vox > 0 else 0.0,
        "sa_to_vol_mesh": float(mesh.area / mesh.volume) if mesh.volume > 0 else 0.0,
        "depth_mean": float(np.nanmean(d)),
        "depth_p50": float(np.nanpercentile(d, 50)),
        "depth_p95": float(np.nanpercentile(d, 95)),
        "depth_max": float(np.nanmax(d)),
        "max_wall_thickness": float(2 * np.nanmax(d)),
        "mean_wall_thickness": float(2 * np.nanmean(d)),
        "_grid": {"occ": occ, "src": src, "depth": depth, "origin": origin, "pitch": pitch},
    }
