"""Drying front, pore saturation, and the aeration-drying coupling.

This module supplies the link that makes the paper's result fall out of the
physics rather than being asserted.

The problem with treating saturation as a free constant
------------------------------------------------------
Oxygen reaches bacteria ~190x further through air-filled pores (~57 mm) than
through water-filled ones (~0.3 mm). If pore saturation Sw is a global sampled
constant, then every geometry gets the same penetration depth and the aeration
subscore cannot distinguish a 16 mm shell wall from a 100 mm solid lump — which
is exactly the degenerate behaviour observed before this module existed
(cemented fraction 1.00 for every design, including the ones the paper reports
as failures).

But Sw is not independent of geometry. A cast body starts saturated with
inoculation and cementation liquid. Air can only enter where evaporation has
already removed that liquid, and evaporation proceeds inward from exposed
surfaces at a rate bounded by the stage-1/stage-2 transition (0.5-2.5 mm/day,
midpoint 1.5). So over a curing period there is a finite AIR-ENTRY DEPTH:

    L_dry = (E * t_cure) / (phi * dS)

Inside that depth the pores are drained enough for the gas-phase oxygen path;
beyond it they remain liquid-filled and only the 0.3 mm dissolved path applies,
which is indistinguishable from anoxic at architectural scale.

The effective oxygen penetration is therefore the SMALLER of the air-entry depth
and the gas-phase reaction-diffusion depth:

    L_eff = min(L_dry, L_gas)

and this is what makes thickness matter. A 16 mm shell wall exposed on both faces
is fully within L_dry. A 100 mm solid ovoid has a core that never drains, never
sees air, and never cements — the paper's "incomplete mineralisation" (Fig. 5).

The same L_dry drives the cracking term with opposite sign: a body thin enough to
dry uniformly does not develop the differential-shrinkage restraint that cracks
it, whereas one whose surface dries while its core stays wet does. This is why
the paper lists cracking and incomplete mineralisation together as symptoms of
the same geometric defect.
"""
from __future__ import annotations

import numpy as np


def air_entry_depth(E_mm_per_day: float, t_cure_days: float, porosity: float,
                    delta_saturation: float = 0.5,
                    D_vapour_mm2_per_day: float | None = None,
                    rh_pct: float = 90.0) -> float:
    """Depth (mm) to which evaporation has drained pores enough to admit air.

    Two regimes, and using the wrong one is a large error.

    STAGE 1 (capillary-supported): the drying front stays hydraulically connected
    to the surface, evaporation runs at the constant rate E, and the drained depth
    grows LINEARLY, L = E*t/(phi*dS).

    STAGE 2 (vapour-diffusion-limited): the liquid connection breaks, the
    vaporisation plane retreats into the body, and further drying must carry
    vapour out through the already-dry layer. The front then advances as the
    SQUARE ROOT of time, L = sqrt(2*D_v*t/(phi*dS)), because the diffusive path
    lengthens as the front recedes.

    WHICH BRANCH ACTUALLY GOVERNS: stage 1, in every regime this model is used in.
    At cure times of days-to-weeks the stage-2 expression evaluates to hundreds or
    thousands of millimetres (e.g. E=1.5, t=14 d, RH=90 % gives stage2 = 761 mm
    against stage1 = 10.5 mm), so the `min` selects stage 1 throughout and the
    vapour branch is effectively dormant. It is retained as a guard for long cures
    and as documentation of the physical ceiling, NOT because it is the operative
    constraint — an earlier version of this docstring wrongly credited it with
    bounding the depth.

    What actually bounds the drained depth is the HUMIDITY DISCOUNT on stage 1.
    Curing at high RH (the paper's humidity-controlled split-mould setup, Fig. 6)
    deliberately slows drying: a nearly-saturated atmosphere removes water slowly
    no matter how thin the section. Without the (1 - RH) factor, stage 1 alone
    gives 1.5 mm/day x 14 d / (0.4 x 0.5) = 105 mm, which would say a 96 mm solid
    lump dries out completely and should cement as well as a 16 mm shell — the
    opposite of the paper's result. With the factor at RH = 90 % the depth becomes
    10.5 mm, which is what correctly leaves a bulky cast with a permanently
    saturated, permanently anoxic core.

    The practical consequence is that RH is the strongest process lever in the
    model: it enters linearly through (1 - RH) and therefore moves the feasible
    wall-thickness window more than either the evaporation rate or the cure
    duration over their plausible ranges.
    """
    denom = max(porosity * delta_saturation, 1e-6)
    rh_factor = max(1.0 - rh_pct / 100.0, 0.02)
    t = max(t_cure_days, 0.0)

    stage1 = max(E_mm_per_day, 0.0) * rh_factor * t / denom

    if D_vapour_mm2_per_day is None:
        # Effective vapour diffusivity through the drained layer. Water vapour in
        # air is ~2.4e-5 m2/s = 2.07e6 mm2/day; a Millington-Quirk style
        # tortuosity factor for a partly-drained pack is O(1e-2), and the (1-RH)
        # gradient scales the flux.
        D_vapour_mm2_per_day = 2.07e6 * 0.02 * rh_factor
    stage2 = np.sqrt(2.0 * D_vapour_mm2_per_day * t / denom)

    return float(min(stage1, stage2))


def effective_penetration(L_gas_mm: float, L_dry_mm: float,
                          L_liquid_mm: float = 0.3) -> dict:
    """Combine the transport-limited and drainage-limited depths.

    Returns the governing depth and which mechanism limits it, so the estimator
    can attribute a failure to the right cause instead of just reporting a number.
    """
    L_eff = float(min(L_gas_mm, L_dry_mm))
    if L_dry_mm < L_gas_mm:
        limiter = "drainage"      # pores still liquid-filled: geometry too bulky
    else:
        limiter = "diffusion"     # air present but O2 consumed before it gets deep
    return {
        "L_eff_mm": max(L_eff, L_liquid_mm),
        "L_gas_mm": float(L_gas_mm),
        "L_dry_mm": float(L_dry_mm),
        "L_liquid_mm": float(L_liquid_mm),
        "limiter": limiter,
    }


def saturation_field(depth_mm: np.ndarray, L_dry_mm: float, *,
                     sw_dry: float = 0.35, sw_wet: float = 0.95) -> np.ndarray:
    """Local pore saturation as a function of distance from an exposed surface.

    Linear ramp from `sw_dry` at the surface to `sw_wet` at the air-entry depth,
    constant `sw_wet` beyond. Used for reporting and for the field-resolved solve;
    the scalar `effective_penetration` above is what the Monte Carlo consumes.
    """
    if L_dry_mm <= 0:
        return np.full_like(depth_mm, sw_wet, dtype=float)
    frac = np.clip(depth_mm / L_dry_mm, 0.0, 1.0)
    return sw_dry + (sw_wet - sw_dry) * frac


def drying_uniformity(max_thickness_mm: float, L_dry_mm: float) -> float:
    """Ratio of half-thickness to air-entry depth; >1 means a core stays wet.

    This is the dimensionless group the cracking subscore keys on: at values
    below ~1 the section dries as a whole, above ~1 the surface shrinks against a
    saturated core, which is the restrained-shrinkage cracking condition.
    """
    if L_dry_mm <= 0:
        return float("inf")
    return float((max_thickness_mm / 2.0) / L_dry_mm)
