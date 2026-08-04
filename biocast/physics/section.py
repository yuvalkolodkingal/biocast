"""Geometry-resolved minimum section thickness from the occupancy grid.

Why this is needed
------------------
`score._infer_min_feature` takes the narrowest mould passage from the NOMINAL
parameters: for a shell that is `wall` (or the aperture width if smaller). That is
correct only while the aperture bore stays inside the cavity. It does not: the
shell grammar subtracts `sd_cylinder_z(..., half_h=2*c, center=(0,0,c))`, which
spans z from -c to 3*c and therefore bores the FULL height of the body. Whenever
aperture_r exceeds the cavity radius (a*taper - wall) the bore eats into the wall
and the remaining annulus is much thinner than `wall`.

Measured case: a=40.3, wall=35.9, aperture_r=23.7. An equatorial slice through the
solid shows two 10.0 mm-wide segments, and the medial-ridge measure below returns
min_section_p5_mm = 8.0 mm (absolute min 5.66, median 12.0) against a 35.9 mm
nominal wall — a 4.5x overstatement, which turns a 2.0 x d_max passage at
d_max = 4 mm (inside Vani's always-clog band) into an apparent 9 x d_max and hands
it castability = 1.0. Feeding the measured value in instead is what stops the
sweep from recommending a mould that cannot be filled.

Measure
-------
Local thickness on the medial ridge: 2 x the distance-to-air field sampled at its
own local maxima. The p5 of that distribution is used as the design's minimum
section (robust to single-voxel ridge noise at discretisation scale); the absolute
minimum is reported alongside so the spread is visible.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def min_section(occ: np.ndarray, pitch: float) -> dict:
    """Minimum local section thickness (mm) of a voxelised solid."""
    if occ.sum() == 0:
        return dict(min_section_p5_mm=0.0, min_section_min_mm=0.0,
                    median_section_mm=0.0, n_ridge_voxels=0)
    dist = ndimage.distance_transform_edt(occ, sampling=(pitch,) * 3)
    ridge = (dist >= ndimage.maximum_filter(dist, size=3)) & (dist > pitch)
    if not ridge.any():                      # section at or below one voxel
        t = 2.0 * float(dist.max())
        return dict(min_section_p5_mm=t, min_section_min_mm=t,
                    median_section_mm=t, n_ridge_voxels=0)
    th = 2.0 * dist[ridge]
    return dict(min_section_p5_mm=float(np.percentile(th, 5)),
                min_section_min_mm=float(th.min()),
                median_section_mm=float(np.median(th)),
                n_ridge_voxels=int(ridge.sum()))
