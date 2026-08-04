"""Oxygen transport and the biocementation penetration field.

Why this module decides everything
----------------------------------
The source paper's central experimental result is that solid forms failed
("incompatibility with ... optimal surface ratio for heterotrophic bacterial
growth") while a hollow ovoid cast in halves succeeded. The retrieved transport
parameters explain why, quantitatively, and the gap is enormous:

    dissolved-O2 path (saturated pores) : L ~ 0.3 mm   (0.13 - 1.58)
    gas-phase path (air-filled pores)   : L ~ 57 mm    (12 - 513)

A ~190x difference. Bio-cementation of anything thicker than a millimetre is
therefore only possible through the GAS phase, which means the design problem is
not "get bacteria in" but "keep an air-filled pore network connected to the
atmosphere everywhere in the wall". That is a geometry problem, and it is the one
this module scores.

Model
-----
Steady zero-order reaction-diffusion in the solid body:

    D_eff * lap(C) = R      wherever C > 0
    C = C_0                 on exposed (atmosphere-facing) surfaces
    C >= 0                  everywhere  (no respiration where O2 is exhausted)

Zero-order is the correct kinetic limit here because pore O2 (0.24 mol/m3
dissolved, 8.4 mol/m3 gas) greatly exceeds the bacterial half-saturation
constant (single-digit micromolar), so respiration runs at its maximum rate until
oxygen is simply gone. The C >= 0 constraint makes this a linear complementarity
(obstacle) problem rather than a plain Poisson solve; it is handled with an
active-set iteration, each step of which is a Poisson solve by conjugate
gradients on the currently-active voxels.

The 1D slab solution of the same equations, a = sqrt(2*D_eff*C_0/R), is the
`analytic_penetration_depth` below and is what the literature rows report; the
field solve generalises it to real geometry, where a corner fed from two sides
cements deeper than a slab and an internal boss cements less.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg


MM_PER_M = 1000.0


def analytic_penetration_depth(D_eff_m2s: float, C0_mol_m3: float,
                               R_mol_m3_s: float) -> float:
    """Zero-order slab penetration depth in mm: a = sqrt(2*D*C0/R)."""
    if R_mol_m3_s <= 0:
        return float("inf")
    a_m = np.sqrt(2.0 * D_eff_m2s * C0_mol_m3 / R_mol_m3_s)
    return float(a_m * MM_PER_M)


def millington_quirk(phi: float, sw: float, *, gas: bool = True,
                     a: float = 10.0 / 3.0, b: float = 2.0) -> float:
    """Millington-Quirk relative diffusivity factor.

    gas=True  -> uses air-filled porosity eps = phi*(1-Sw)
    gas=False -> uses water-filled porosity theta = phi*Sw
    Returns the multiplier on the free-fluid diffusivity.
    """
    phi = float(np.clip(phi, 1e-6, 1.0))
    sw = float(np.clip(sw, 0.0, 1.0))
    eps = phi * (1.0 - sw) if gas else phi * sw
    if eps <= 0:
        return 0.0
    return float(eps ** a / phi ** b)


def effective_diffusivity(D_free_m2s: float, phi: float, sw: float, *,
                          gas: bool = True) -> float:
    return D_free_m2s * millington_quirk(phi, sw, gas=gas)


# --------------------------------------------------------------------------
# Field solve
# --------------------------------------------------------------------------
def solve_oxygen(occ: np.ndarray, src: np.ndarray, pitch_mm: float, *,
                 D_eff_m2s: float, C0_mol_m3: float, R_mol_m3_s: float,
                 max_active_iter: int = 12, cg_tol: float = 1e-8,
                 cg_maxiter: int = 600) -> dict:
    """Solve the obstacle-constrained steady reaction-diffusion field.

    Parameters
    ----------
    occ : bool grid, True where the cast body is solid
    src : bool grid, True on air voxels that act as atmosphere (Dirichlet C=C0)
    pitch_mm : voxel edge length in mm

    Returns a dict with the concentration field (mol/m3), the oxygenated mask,
    and the fraction of the body that receives oxygen.
    """
    h_m = pitch_mm / MM_PER_M
    idx = np.full(occ.shape, -1, dtype=np.int64)
    solid = np.flatnonzero(occ.ravel())
    idx.ravel()[solid] = np.arange(solid.size)
    n = solid.size
    if n == 0:
        raise ValueError("empty body")

    # Neighbour bookkeeping: for each solid voxel, count solid+source neighbours
    shifts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    rows, cols, vals = [], [], []
    # Discretisation. The PDE is D*lap(C) = R (respiration is a SINK, so the
    # Laplacian is positive inside the body). Multiplying by -h^2 puts it in the
    # positive-definite form used below:
    #     6*C_i - sum_neighbours(C) = -R*h^2/D  + (C0 for each atmosphere face)
    # Getting this sign wrong makes respiration a source and the interior
    # concentration exceed the boundary value, which is how the error was caught.
    rhs = np.full(n, -R_mol_m3_s * h_m ** 2 / D_eff_m2s, dtype=float)

    diag = np.zeros(n)
    for s in shifts:
        nb_occ = np.roll(occ, s, axis=(0, 1, 2))
        nb_src = np.roll(src, s, axis=(0, 1, 2))
        # kill wrapped planes
        for ax, sh in enumerate(s):
            if sh != 0:
                sl = [slice(None)] * 3
                sl[ax] = 0 if sh > 0 else -1
                nb_occ[tuple(sl)] = False
                nb_src[tuple(sl)] = False
        nb_idx = np.roll(idx, s, axis=(0, 1, 2))

        # solid-solid coupling
        m_ss = occ & nb_occ
        if m_ss.any():
            i = idx[m_ss]
            j = nb_idx[m_ss]
            rows.append(i); cols.append(j); vals.append(-np.ones(i.size))
            diag[i] += 1.0
        # solid-atmosphere: Dirichlet contribution to the rhs
        m_sd = occ & nb_src
        if m_sd.any():
            i = idx[m_sd]
            diag[i] += 1.0
            np.add.at(rhs, i, C0_mol_m3)
        # solid-void (sealed internal air, no O2): zero-flux, contributes nothing

    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    A = sparse.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()

    # Obstacle problem: enforce C >= 0 by a primal-dual active-set iteration.
    #
    # A naive "drop every voxel that went negative, permanently" loop fails badly
    # when most of the body is anoxic: the first unconstrained solve is negative
    # nearly everywhere, the whole domain is deactivated at once, and the answer
    # collapses to C == 0 with no oxygenated shell at all. The fix is to let
    # voxels RE-ENTER the free set — a voxel is constrained only while the
    # residual there says respiration still outstrips supply — and to iterate to
    # a fixed point of the active set rather than shrinking it monotonically.
    # Seeding matters enormously for cost. The active set GROWS at only one voxel
    # layer per iteration (a Poisson solve only sees one voxel past the current
    # front), so starting from "everything free" needs as many iterations as the
    # front has voxels — 100+ for a thick body, and it silently stops short if the
    # iteration cap hits first. Starting from the analytic slab depth puts the
    # front within one voxel of the answer immediately, and the iteration then
    # only has to apply local geometric corrections (a corner fed from two faces
    # cements deeper than a slab; an internal boss cements less).
    L_seed = analytic_penetration_depth(D_eff_m2s, C0_mol_m3, R_mol_m3_s)
    if np.isfinite(L_seed):
        dist_to_src = ndimage.distance_transform_edt(~src, sampling=(pitch_mm,) * 3)
        free = (dist_to_src.ravel()[solid] <= L_seed)
        if not free.any():           # penetration below one voxel: keep the skin alive
            free = dist_to_src.ravel()[solid] <= pitch_mm
    else:
        free = np.ones(n, dtype=bool)
    C = np.zeros(n)
    prev_free = None
    for _ in range(max_active_iter):
        if not free.any():
            break
        Af = A[free][:, free].tocsr()
        bf = rhs[free]
        sol, _info = cg(Af, bf, x0=C[free], rtol=cg_tol, maxiter=cg_maxiter)
        Cn = np.zeros(n)
        Cn[free] = np.maximum(sol, 0.0)
        C = Cn

        # active set = voxels pinned at zero whose residual keeps them there
        resid = A @ C - rhs
        new_free = (C > 0) | (resid < 0)
        # a voxel adjacent to the atmosphere is always free (it is fed directly)
        if new_free.sum() == 0:
            break
        if prev_free is not None and np.array_equal(new_free, prev_free):
            free = new_free
            break
        prev_free, free = new_free, new_free

    C = np.maximum(C, 0.0)

    field = np.full(occ.shape, np.nan)
    field.ravel()[solid] = C
    oxy = np.zeros(occ.shape, dtype=bool)
    oxy.ravel()[solid] = C > 0

    # Resolution guard. The front position is first-order accurate in the voxel
    # pitch, so a penetration depth spanning only a few voxels is not resolved.
    # This matters in practice: the dissolved-O2 depth is ~0.3 mm, which is
    # sub-voxel for any object-scale grid, so a saturated-pore run will report a
    # cemented fraction of ~0 with a warning rather than a spuriously precise
    # number. That is the physically correct conclusion (a wet-pore cast does not
    # bio-cement beyond its skin), but the caller must know it is grid-limited.
    n_vox_in_depth = L_seed / pitch_mm if np.isfinite(L_seed) else np.inf
    resolved = n_vox_in_depth >= 4.0

    return {
        "C": field,
        "oxygenated": oxy,
        "oxygenated_fraction": float(oxy.sum() / occ.sum()),
        "depth_voxels": float(n_vox_in_depth),
        "depth_resolved": bool(resolved),
        "resolution_warning": (
            None if resolved else
            f"penetration depth {L_seed:.3f} mm spans only {n_vox_in_depth:.1f} voxels at "
            f"pitch {pitch_mm} mm; the cemented fraction is grid-limited and should be read "
            "as 'skin only', not as a converged number"),
        "C_mean": float(np.nanmean(field)),
        "C0": C0_mol_m3,
        "R": R_mol_m3_s,
        "D_eff": D_eff_m2s,
        "analytic_depth_mm": analytic_penetration_depth(D_eff_m2s, C0_mol_m3, R_mol_m3_s),
    }


def cemented_fraction_from_depth(depth_mm: np.ndarray, occ: np.ndarray,
                                 L_mm: float) -> float:
    """Fraction of the body within one penetration depth of an exposed surface.

    This is the cheap surrogate for the field solve: it is what the literature
    depth rows directly support, and it is used for the Monte Carlo sweep where a
    full PDE solve per sample would be prohibitive.
    """
    d = depth_mm[occ]
    if d.size == 0:
        return 0.0
    return float(np.mean(d <= L_mm))
