# Grasshopper implementation specification

**Target:** rebuild the `biocast` generator and constraint checker as a Rhino/Grasshopper
definition, so the rules run inside the modelling environment and reject unbuildable
geometry *before* the mould is printed — the workflow the project notes asked for
("such a system will save a huge amount of trial and error, because it will reject
geometries liable to create breakage points before the mould printing stage").

This document is a build specification, not a tutorial. It states what each cluster
must compute, which rules are hard gates, and — importantly — which parts of the
Python model have **no faithful Grasshopper equivalent** and must be either
approximated or left in Python.

---

## 0. Scope and honest division of labour

| Capability | Grasshopper | Notes |
|---|---|---|
| Parametric geometry, all three typologies | **Yes** | Native surface/solid modelling is better than the Python SDF approach here |
| Fillet-everywhere enforcement | **Yes** | `Fillet Edge` / `Blend`; radius driven by a slider bound to `d_max` |
| Dimensional constraint rules (fillet, groove, joint, void, standards) | **Yes** | Pure arithmetic on parameters; `Cull`/`Dispatch` + colour feedback |
| Jamming / bridging gate | **Yes** | Arithmetic gate, `w >= 6 d_max` |
| Wall-thickness field (local thickness everywhere) | **Partly** | See §5 — needs a sampling workaround; no native medial-axis |
| Surface-area-to-volume ratio | **Yes** | `Volume` + `Area` components |
| Oxygen reaction–diffusion field solve | **No** | Needs a PDE solve on a voxel grid. Use the closed-form depth (§6) in GH; keep the field solve in Python |
| Monte-Carlo uncertainty propagation | **Impractical** | Hundreds of parameter draws per design. Approximate with a three-point low/nominal/high evaluation (§7) |
| Split-mould negative generation | **Yes** | `Solid Difference`, `Cap`, boolean-safe if tolerances are set (§8) |

The pragmatic split: **Grasshopper owns geometry and every algebraic rule; Python owns
the field solve and the uncertainty.** A `score_lite` computed entirely in Grasshopper
(§7) tracks the full model closely enough for interactive design steering, and the
Python package is the authority for a final number.

---

## 1. Input parameters

Group these as a single `Parameters` cluster with labelled sliders. Values in **bold**
are the ones every rule keys on.

### Material / mix
| Name | Type | Default | Range | Meaning |
|---|---|---|---|---|
| **`d_max`** | slider, mm | 4.0 | 1–8 | Largest aggregate fragment. Drives fillet radius, groove width, minimum section |
| `d50` | slider, mm | 1.0 | 0.2–4 | Median particle size |
| `porosity` | slider | 0.40 | 0.30–0.50 | Packed-bed void fraction |
| `caco3_target` | slider, % | 8.0 | 1–20 | Target carbonate content (gate at 3 %, §4) |

### Process / curing
| Name | Type | Default | Range | Meaning |
|---|---|---|---|---|
| **`cure_days`** | slider | 14 | 3–60 | Cure duration. Sets drained depth |
| **`rh_pct`** | slider, % | 90 | 50–99 | Curing relative humidity. High RH deliberately slows drying |
| `E_evap` | slider, mm/day | 1.5 | 0.5–2.5 | Evaporation rate (stage-1/2 transition band) |
| `split_mould` | boolean | true | — | Cast in halves → the parting plane is an oxygen source |

### Geometry — Shell (ovoid vessel)
`a`, `b` (semi-axes, mm, 40–90) · `c` (long semi-axis, 60–140) · `n` (superellipsoid
exponent, 2.0–4.0) · `ovoid` (egg taper, 0–0.45) · **`wall`** (8–40) ·
`aperture_r` (0–30) · **`fillet_r`** (4–16) · `rib_count` (0–8) · `rib_depth` (0–10)

### Geometry — Block (CMU-type)
`L` 390 · `W` 190 · `H` 190 · `n_cores` 2–3 · **`face_shell`** 25–50 ·
**`web`** 19–40 · **`fillet_r`** 6–16 · `core_taper` 0–4° · optional
`groove_depth` / `groove_width` / `groove_count`

### Geometry — Tile (Panot-type)
`L` 200 · `W` 200 · **`t`** 25–60 · `pattern` (grid | diagonal | flower | radial) ·
**`groove_depth`** 2–8 · **`groove_width`** 8–40 · `groove_pitch` 30–80 ·
**`fillet_r`** 4–14 · **`joint`** 3–12 · `thick_tile` boolean

---

## 2. Component graph

```
[Parameters cluster]
      |
      +---> [Rule Gate cluster] ------> pass/fail booleans + messages
      |            |                    (evaluate BEFORE geometry: it is free)
      |            v
      |      [Cull Pattern] -- drop failing candidates
      |
      +---> [Typology switch: Stream Filter]
                 |
        +--------+--------+--------------+
        v                 v              v
   [Shell cluster]  [Block cluster]  [Tile cluster]
        |                 |              |
        +--------+--------+--------------+
                 v
        [Fillet All Edges]  <- radius = fillet_r, hard-bound to d_max floor
                 v
        [Diagnostics cluster]  -> Volume, Area, S/V, thickness samples (§5)
                 v
        [Score Lite cluster]   -> subscores + total (§7)
                 v
        +--------+--------+
        v                 v
   [Preview: colour   [Mould cluster] (§8)
    by score]              v
                      [Export STL]
```

Evaluate the rule gate **before** building geometry. Every rule in §3–§4 is arithmetic
on the parameters, so a failing candidate costs nothing to reject, and in a large
sweep this is the difference between an interactive definition and an unusable one.

---

## 3. Hard rules (reject — these are gates, not warnings)

Implement each as a boolean expression. Feed all of them into a single `And` gate whose
output drives `Cull Pattern`, and surface the individual failures as text so the designer
sees *which* rule bit.

```python
# fillet — project notes: a corner in brittle bio-cement is where the crack starts
R1:  fillet_r >= 1.5 * d_max                      # hard floor
R1b: fillet_r >= 2.0 * d_max                      # preferred (warn only)

# groove depth — keep a strong core, do not let the unit snap through the grooved plane
R2:  groove_depth <= t / 3                        # block
R2t: groove_depth <= t / 4                        # thick tile

# groove width — the team's rule
R3:  groove_width >= 2.0 * d_max                  # notes' minimum
R3b: groove_width >= 3.0 * d_max                  # notes' "safe"

# groove width — the JAMMING rule from the literature. THIS OVERRIDES R3.
R4:  groove_width >= 6.0 * d_max                  # accept  (Zuriguel Rc=4.94 spheres, 6.0 angular)
R4b: groove_width >= 8.0 * d_max                  # safe    (Vani dense-suspension divergence 8.1)
R4c: groove_width >= 3.0 * d_max                  # ABSOLUTE FLOOR — below this it always clogs

# minimum castable section
R5:  min_section >= 5.0 * d_max                   # ACI 318 26.4.2.1(a)(5) inverted (wall)
R5t: tile_t      >= 3.0 * d_max                   # ACI slab-depth clause

# masonry standards
R6:  face_shell >= 32.0                           # ASTM C90, 8 in and wider
R7:  web        >= 19.0                           # ASTM C90 current minimum
R7b: web        >= 25.0                           # the team's own conservative rule (warn)

# joint — feed solution and oxygen must reach the tile edges in a composition
R8:  joint >= 3.0

# aerobic geometry (needs diagnostics, §5)
R9:  sa_to_vol >= 0.030                           # 1/mm
R10: max_wall_thickness <= 2 * L_eff              # §6
```

**R3 vs R4 is a real conflict and the definition must not hide it.** The project notes
specify `w >= 2–3 d_max`; the granular-jamming literature puts the critical
aperture-to-particle ratio at 4.94 (spheres) to 6.0 (angular grains), with dense
suspensions diverging at 8.1 and *certain* clogging below 3. At `d_max = 4 mm` the notes
give 8–12 mm while the literature demands 24 mm. Show **both** thresholds on the canvas,
gate on R4, and label R3 as "team rule (superseded — see jamming criterion)". A ~10 mm
Panot channel at 4 mm aggregate sits essentially at the always-clog boundary.

---

## 4. Derived quantities and secondary gates

```python
# block core dimensions
core_w = (L - 2*face_shell - (n_cores - 1)*web) / n_cores
core_d = W - 2*face_shell
void_fraction = n_cores * core_w * core_d * H / (L * W * H)     # target 0.40–0.50

R11: core_w > 2 * fillet_r        and  core_d > 2 * fillet_r
R12: min(core_w, core_d) >= 6 * d_max          # cores must not jam either
R13: core_taper >= 1.0                         # draft, or the green body will not release

# tile
R14: groove_pitch > groove_width               # or the relief merges
R15: groove_depth <= 0.10 * t                  # Panot practice; regulated ceiling is 5 mm

# shell
R16: wall < min(a, b, c)                       # or there is no cavity
R17: 2 * aperture_r >= 6 * d_max               # the aperture is a feed passage: jamming applies
R18: fillet_r <= wall                          # the fillet must fit in the wall

# carbonate gate
R19: caco3_target >= 3.0        # below ~3 % a specimen cannot stand unconfined
```

---

## 5. Diagnostics cluster — and the thickness problem

`Volume` and `Area` are native, so `sa_to_vol = Area / Volume` is trivial. **Local wall
thickness is the hard part: Grasshopper has no medial-axis or signed-distance component.**

Recommended workaround, in order of preference:

1. **Ray-based sampling (native).** Populate the solid's surface with points
   (`Populate Geometry`, ~500–2000), shoot an inward ray along the reversed surface
   normal at each (`Mesh Ray` / `Brep | Line`), and take the distance to the first
   opposite intersection. The distribution of those distances approximates the local
   thickness field; its maximum is what R10 needs. This is what the Python code computes
   exactly as `2 x` the distance transform, and the ray estimate is adequate for gating.
2. **Section-based.** For the shell and block, thickness is analytic from the parameters
   (`wall`, `face_shell`, `web`) — just compute it and skip sampling. Use sampling only
   for free-form variants.
3. **Dendro / Cocoon plugin** if the team is willing to install one: both expose a real
   voxel/SDF pipeline, which makes the thickness field and the offset-based hollowing
   direct rather than approximate. If a plugin is acceptable, prefer this.

Report `max_wall_thickness`, `mean_wall_thickness`, `sa_to_vol`, `volume`, `area`.

---

## 6. Oxygen and drying — the closed forms to implement

The full PDE solve does not belong in Grasshopper, but the **1-D closed forms it
generalises do**, and they carry almost all of the design signal. Implement as an
`Expression` component chain:

```python
# --- gas-phase reaction-diffusion depth ---------------------------------
# Millington-Quirk relative diffusivity on the AIR-filled porosity
eps    = porosity * (1 - sw)                      # sw = liquid saturation, ~0.5 nominal
f_MQ   = eps**(10.0/3.0) / porosity**2
D_eff  = f_MQ * D_O2_gas                          # D_O2_gas = 2.209e-5 m^2/s
L_gas  = sqrt(2 * D_eff * C_O2_gas / R_O2) * 1000 # mm;  C_O2_gas = 8.42 mol/m^3
                                                  #      R_O2    = 1.116e-3 mol/(m^3 s)

# --- drained (air-entry) depth: TWO regimes, take the smaller ------------
rh_factor = max(1 - rh_pct/100, 0.02)
denom     = porosity * dS                         # dS ~ 0.5 saturation drop
stage1    = E_evap * rh_factor * cure_days / denom               # capillary-fed, linear in t
D_vap     = 2.07e6 * 0.02 * rh_factor                            # mm^2/day
stage2    = sqrt(2 * D_vap * cure_days / denom)                  # vapour-limited, sqrt(t)
L_dry     = min(stage1, stage2)

# --- what actually governs ----------------------------------------------
L_eff   = max(min(L_gas, L_dry), 0.3)   # 0.3 mm = the dissolved-O2 floor
limiter = "drainage" if L_dry < L_gas else "diffusion"
```

Units: work in metres inside the `L_gas` expression, then convert to millimetres —
mixing them is the easiest way to get a plausible-looking wrong answer.

**Which branch governs, and what actually bounds the depth.** In every regime this model
is used in, `stage1` is the smaller term and therefore governs: at cure times of
days-to-weeks `stage2` evaluates to hundreds of millimetres (E = 1.5, t = 14 d, RH = 90 %
gives 761 mm against 10.5 mm). Implement the `min` anyway as a guard for long cures, but
do not expect the vapour branch to engage.

The term that does the work is the **`rh_factor` discount on stage 1**. Without it,
stage 1 gives `1.5 mm/day x 14 d / (0.4 x 0.5) = 105 mm`, which would say a 96 mm solid
lump dries out completely and should cement as well as a 16 mm shell — the *opposite* of
the experimental result. With it at RH = 90 % the depth is 10.5 mm, which correctly leaves
a bulky cast with a permanently saturated, anoxic core.

Consequence for the definition: **RH is the strongest process lever you expose.** It
enters linearly through `(1 - RH)`, so over plausible ranges it moves the feasible
wall-thickness window more than either evaporation rate or cure duration — dropping
RH 90 % → 70 % triples the drained depth (10.5 → 31.5 mm), against 2.0× for doubling the
cure to 28 days and 1.7× for the fastest evaporation rate. Put the RH slider where the
designer will find it.

Constants to expose as a locked `Constants` cluster (values and provenance in
`micp_kinetics_params.json`): `D_O2_gas = 2.209e-5 m^2/s`, `C_O2_gas = 8.42 mol/m^3`,
`R_O2 = 1.116e-3 mol/(m^3 s)` (range 1.27e-4 – 3.34e-3), `D_O2_water = 2.0e-9 m^2/s`,
`C_O2_sat = 0.24 mol/m^3`.

---

## 7. `score_lite` — the Grasshopper-side estimator

Four subscores in [0,1], **multiplied** (series requirements: a body that cannot be
filled does not get a second chance to be well oxygenated — a mean would let a good
aeration score hide a fatal casting defect).

```python
# --- aeration: what fraction of the body is within reach of oxygen ---
# Exact version integrates the distance field. Lite version uses the thickness ratio.
cem_frac  = min(2 * L_eff / max_wall_thickness, 1.0)
S_aer     = 1 / (1 + exp(-12 * (cem_frac - 0.85) / 0.15 * 0.5))

# --- drying uniformity: half-thickness vs how far drying actually reaches ---
R_dry     = (max_wall_thickness / 2) / L_dry
S_dry     = 1.0 if R_dry <= 1 else R_dry ** -1.5

# --- castability: narrowest passage vs the jamming threshold ---
ratio     = min_feature / d_max
S_cast    = 1.0 if ratio >= 8.0 else (0.02 if ratio <= 3.0 else 0.02 + 0.98*((ratio-3)/5)**1.2)

# --- structural: Inglis notch factor, converted by the Weibull modulus ---
Kt        = 1 + 2*sqrt(notch_depth / fillet_r)      # notch_depth = groove_depth, or 0
S_str     = 1.0 if Kt <= 2.0 else (Kt/2.0) ** (-m_weibull/4.0)    # m_weibull ~ 14

score     = S_aer * S_dry * S_cast * S_str
```

`min_feature` = the narrowest passage the wet mix must flow through: `min(wall, 2*aperture_r)`
for the shell; `min(face_shell, web, groove_width, core_w, core_d)` for the block;
`min(t, groove_width)` for the tile.

**Uncertainty without Monte Carlo.** Evaluate `score` three times — with
`R_O2` and `E_evap` at their low, nominal and high literature values — and report the
spread as an interval. It is coarser than the Python sampling but it preserves the
essential honesty: the intervals are wide, and two designs whose intervals overlap
should be treated as tied rather than ranked.

Colour the preview by `score` (`Gradient` component) and label each candidate with its
dominant failure mode = the name of the smallest subscore.

---

## 8. Mould cluster

1. `Bounding Box` the object, inflate by the chosen mould wall thickness → mould block.
2. Find the parting plane: the widest cross-section. Sweep a plane along the candidate
   axis, take `Area` of each section, pick the maximum. **Verify rather than assume** —
   for a tapered ovoid the widest section is *not* at mid-height.
3. `Solid Difference` (mould block − object), then split by the parting plane and `Cap`.
4. Draft: taper mould walls ≥ 1–2°. A green bio-cement body will not survive a
   zero-draft pull.
5. Registration keys: ≥ 3 conical dowel/socket pairs, **asymmetrically placed** so the
   halves mate one way only.
6. Vents and feed passages: **≥ 6 × d_max (24 mm at d_max = 4 mm)**. The jamming
   criterion applies to the mould's own apertures exactly as it does to the object's
   features — a 5 mm vent on this mould will block.
7. Verify: both halves closed (`Is Solid`), and
   `vol(lower) + vol(upper) + vol(object) ≈ vol(mould block)`. That volume balance
   catches boolean failures that look fine in the viewport.

Set Rhino's absolute tolerance to ~0.01 mm before booleans; the default is often too
coarse for filleted intersections at this scale and produces naked edges.

---

## 9. Validation the definition must reproduce

Before trusting the definition, confirm it retrodicts the experiment:

| Case | Geometry | Expected |
|---|---|---|
| A | Solid ovoid, monolithic | **Low** score, dominant mode **aeration** — the paper's Fig. 5 failure |
| D | Hollow shell (wall ≈ 14–16 mm), split mould | **Higher** score, aeration satisfied — the paper's Figs. 6–7 success |
| E | Hollow shell, `fillet_r = 0.5 mm`, 8 mm groove | Collapses on **structural** (Kt ≈ 9) |

The Python reference run gives cemented fraction 0.45 for A versus 1.00 for D. If the
Grasshopper `score_lite` ranks A above D, the `L_dry` expression is wrong — check that
`min(stage1, stage2)` is being taken and that the RH factor is applied.

---

## 10. Known divergences from the Python model

State these on the canvas so nobody over-trusts the definition:

1. **No field solve.** `cem_frac` from a thickness ratio ignores that a corner fed from
   two faces cements deeper than a slab, and that an internal boss cements less. It is
   optimistic for concave geometry.
2. **Three-point instead of Monte Carlo.** Underestimates the tail width; the reported
   interval is narrower than the true one.
3. **Ray-sampled thickness** is an approximation of the exact distance-transform
   thickness; increase the sample count until `max_wall_thickness` stabilises.
4. **The oxygen constants are literature values for sand and S. pasteurii-style systems,
   with *B. subtilis* respiration.** They carry an organism mismatch flagged in the
   parameter files, and `phi_f_frac` (biofilm volume fraction) is an assumption spanning
   a full decade — it multiplies oxygen demand linearly and is the weakest link in the
   chain.
5. **The success score is a ranking instrument, not a calibrated probability.** No
   pass/fail dataset was available to fit it. Feeding in even a modest table of the lab's
   own outcomes would turn it into a calibrated model.
