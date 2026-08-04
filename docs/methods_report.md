# Growing shapes for bio-concrete: a geometry generator with a success estimator

**Methods, parameter provenance, validation, and limits**

For: *Still Life — Engineering Symbiotic Relationships through Bacterial Biomineralization*
(A. Ioshpe, I. Kolodkin-Gal, Scojen Institute for Synthetic Biology, Reichman University)

---

## 1. What this does, and what question it answers

The project casts objects from construction-waste aggregate bio-cemented by *Bacillus
subtilis*. The pictorial reports one hard geometric lesson: early prototypes **failed**
— cracking and incomplete mineralisation, attributed to uneven drying and oxygenation,
and to "incompatibility with … optimal surface ratio for heterotrophic bacterial growth"
— while a **hollow, oval, egg-like form cast as two halves succeeded**, with no visible
scar at the join.

The team's design notes add quantitative casting rules: fillet rather than chamfer, with
`r ≥ 1.5–2 × d_max`; groove depth `≤ t/3` (block) or `≤ t/4` (thick tile); groove width
`≥ 2–3 × d_max` to avoid aggregate bridging; CMU proportions (200×200×400 mm, 40–50 %
void, ~32 mm face shells, ~25 mm webs); Panot proportions (200×200×40 mm, 2–3 mm relief,
~10 mm channels). The notes ask for these to be encoded "as hard parameters" so that a
model "will reject geometries liable to create breakage points before the mould printing
stage."

This work delivers that, and answers a question the pictorial poses but cannot settle
from two data points: **which geometries will bio-cement completely, why the failures
failed, and what to change.** The deliverable is a Python package
(`biocast`) with three shape grammars, a machine-checkable constraint rule set, a
physically-grounded success estimator with propagated uncertainty, mould generation, and
a Grasshopper specification for rebuilding it in Rhino.

The estimator is a **ranking instrument, not a calibrated probability** — see §9.

---

## 2. The governing physics, and why it is a geometry problem

*B. subtilis* is an obligate aerobe: without O₂ as terminal electron acceptor it does not
respire, and without respiration there is no microbially induced carbonate. So the
binding question is how deep oxygen reaches into a cast body.

### 2.1 Two transport paths, 190× apart

Oxygen can reach bacteria dissolved in pore water or as gas in air-filled pores. Solving
the zero-order reaction–diffusion slab (§2.2) with retrieved parameters:

| path | effective diffusivity | penetration depth |
|---|---|---|
| dissolved O₂, water-filled pores | 5.05 × 10⁻¹⁰ m² s⁻¹ | **0.3 mm** (0.13–1.58) |
| gas-phase O₂, air-filled pores | 6.46 × 10⁻⁷ m² s⁻¹ | **57 mm** (12–513) |

A factor of ~190. Bio-cementation of anything thicker than a millimetre is therefore
**only possible through the gas phase**, and the design problem is not "get bacteria in"
but *keep an air-filled pore network connected to the atmosphere throughout the wall.*
That is geometry, and it is what the estimator scores.

### 2.2 Reaction–diffusion model

Steady state inside the solid body:

$$D_{\text{eff}}\,\nabla^2 C = R \quad\text{where } C>0, \qquad C = C_0 \text{ on exposed surfaces},\qquad C \ge 0$$

Zero-order kinetics is the correct limit: pore O₂ (0.24 mol m⁻³ dissolved, 8.42 mol m⁻³
gas) greatly exceeds any bacterial half-saturation constant (single-digit µM), so
respiration runs at maximum rate until oxygen is gone. The `C ≥ 0` constraint makes this
a linear complementarity (obstacle) problem, solved by an active-set iteration with a
conjugate-gradient Poisson solve per step (`biocast.physics.oxygen.solve_oxygen`).

The 1-D slab solution of the same equations,

$$a = \sqrt{\frac{2 D_{\text{eff}} C_0}{R}}$$

is what the literature depth rows report and what the Monte Carlo consumes; the field
solve generalises it to real geometry, where a corner fed from two faces cements deeper
than a slab and an internal boss cements less.

Effective diffusivity follows Millington–Quirk on the **air-filled** porosity
$\varepsilon = \phi(1-S_w)$:

$$D_{\text{eff}} = \frac{\varepsilon^{10/3}}{\phi^{2}}\,D_{\text{O}_2,\text{gas}}$$

### 2.3 The coupling that makes thickness matter

Saturation $S_w$ cannot be treated as a free constant. A cast body starts saturated with
inoculation and cementation liquid; air only enters where evaporation has removed it. Over
a cure of duration $t$ there is a finite **air-entry depth**:

$$L_{\text{dry}} = \min\!\left(\underbrace{\frac{E\,(1-\mathrm{RH})\,t}{\phi\,\Delta S}}_{\text{stage 1, capillary-fed}},\ \underbrace{\sqrt{\frac{2 D_v t}{\phi \Delta S}}}_{\text{stage 2, vapour-limited}}\right)$$

and the effective penetration is

$$L_{\text{eff}} = \min\!\left(L_{\text{gas}},\, L_{\text{dry}}\right)$$

**Which branch governs, and why it matters.** In every regime tested, stage 1 is the
smaller term: at days-to-weeks cures stage 2 evaluates to 381–2730 mm and never binds. The
term doing the work is the **humidity discount** $(1-\mathrm{RH})$. Without it, stage 1
gives 1.5 mm/day × 14 d / (0.4 × 0.5) = **105 mm**, which would say a 96 mm solid lump
dries out completely and should cement as well as a 16 mm shell — the *opposite* of the
experimental result. With it at 90 % RH the depth is **10.5 mm**, which correctly leaves a
bulky cast with a permanently saturated, permanently anoxic core.

Consequence: **RH is the strongest process lever in the model.** Dropping 90 % → 70 %
triples the drained depth (10.5 → 31.5 mm), against 2.0× for doubling the cure to 28 days
and 1.7× for the fastest evaporation rate.

This single coupling reproduces both of the paper's Fig. 5 symptoms from one cause:
a section too thick to drain has an anoxic core (incomplete mineralisation) *and* a
surface that shrinks against a saturated interior (cracking).

---

## 3. Shape grammars

Three parametric families, all built as signed-distance fields and meshed by marching
cubes (`biocast.grammars`).

| grammar | typology | source of proportions |
|---|---|---|
| `shell` | hollow ovoid vessel, split into halves | the paper's successful Figs. 6–7 design |
| `block` | hollow-core masonry unit | team notes + ASTM C90 |
| `tile` | relief paving/cladding tile | team notes + Barcelona Panot / Orden VIV/561/2010 |

**Fillets, never chamfers, by construction.** All edges are rounded with exact
signed-distance CSG operators (`op_round_union`, `op_round_intersect`,
`op_round_subtract`), which place a fillet of precisely radius *r* on the edge where two
surfaces meet. A chamfer cannot be expressed in the grammar, which is the team's rule
enforced structurally rather than by checking after the fact.

An earlier implementation filleted morphologically (a rolling-ball open/close on the
voxel grid). That is wrong and was discarded: an 8 mm ball cannot roll inside a 15 mm
shell wall, so the operation deleted the wall and produced a mesh with ~97 spurious
handles (Euler number −192 instead of 4). The distance-field operators do not have this
failure because they act on the edge, not the volume.

**Mesh quality.** All meshes are watertight with the topology their geometry implies:
shell Euler 0 (genus-1: hollow with an aperture), block Euler −2 (two through-cores),
tile Euler 2 (closed). Across the 6723-cell sweep, **6723/6723 meshes were watertight**.
Volumes agree with independent voxel counts to 0.8 % and with rebuilt export meshes to
0.46 %.

---

## 4. Constraint rule set

Each rule is a predicate returning a verdict tagged by origin — **TEAM** (the project's
own notes), **LIT** (retrieved literature), **STD** (a standard), **GEOM**
(self-consistency). `severity="fail"` rejects the design before mould printing;
`"warn"` means castable but suboptimal.

| rule | threshold | origin |
|---|---|---|
| `fillet_radius_min` | `r ≥ 1.5 × d_max` | TEAM |
| `fillet_radius_preferred` | `r ≥ 2.0 × d_max` | TEAM (warn) |
| `groove_depth_max` | `h ≤ t/3` block, `t/4` thick tile | TEAM |
| `groove_width_min` / `_safe` | `w ≥ 2 × d_max` / `3 × d_max` | TEAM |
| `groove_jamming`, `aperture_not_jamming`, `measured_section_not_jamming` | passage `≥ jam_ratio × d_max` | LIT |
| `min_section_thickness` | `≥ 3 × d_max` (nominal), `5 × d_max` from ACI 318 26.4.2.1(a)(5) | LIT |
| `face_shell_min` / `web_min` | 32 mm / 25 mm | STD / TEAM |
| `void_fraction_band` | 40–50 % | STD (warn) |
| `joint_min` | `≥ 3 mm` for feed and O₂ passage | TEAM |
| `relief_fraction` | `≤ 10 %` of tile thickness | STD (warn) |
| `surface_to_volume_min` | `≥ 0.030 mm⁻¹` | LIT |
| `penetration_coverage` | `≥ 85 %` of body cemented | LIT |
| `draft_for_release` | `≥ 1°` | GEOM (warn) |

### 4.1 Where the team's rules were confirmed, and where they were not

| team rule | verdict | evidence |
|---|---|---|
| Fillet, never chamfer | **Confirmed, and it is the right instinct** | Inglis $K_t = 1+2\sqrt{h/r}$ diverges as the root sharpens; with Weibull *m* = 11.5–16.8 for porous cement composites, failure probability scales ≈ $K_t^m$ |
| Face shell ≈ 32 mm | **Confirmed exactly** | ASTM C90 Table 1: 1¼ in = 31.75 mm for 8 in and wider units |
| Panot relief 2–3 mm | **Confirmed, conservative** | Orden VIV/561/2010 caps directional grooves at 5 mm |
| Void 40–50 % | **Realistic, but not a C90 limit** | C90 constrains face shell, web, and normalised web area — not void ratio |
| Web ≈ 25 mm | **Superseded, conservative** | ASTM C90 since 2011-b permits 19.1 mm for all widths; 25 mm matches the pre-2011 6 in requirement. Kept as a build rule, but do not cite C90 for it |
| **Groove width ≥ 2–3 × d_max** | **Too permissive by 2–3×** | See below |

**The bridging rule is the one substantive disagreement.** Granular flow through an
aperture jams below a critical aperture-to-particle ratio, and the measured values are
much higher than the notes assume:

- $R_c = 4.94 \pm 0.03$ for spherical beads, rising to **6.0 for angular (rice-shaped)
  grains**, with material, density, elasticity and roughness having no measurable effect —
  it is pure geometry (Zuriguel et al. 2005, `10.1103/PhysRevE.71.051303`)
- for particles suspended in liquid, a constriction **always** clogs below `W/d = 3`, and
  for a dense suspension near maximum packing the divergence sits at `W_c/d = 8.1`
  (Vani, Escudier & Sauret 2022, `10.1039/D2SM00962E`)

Recommended: **accept at `w ≥ 6 d_max`, safe at `w ≥ 8 d_max`, certain failure below
`3 d_max`.** At the project's `d_max = 4 mm` that is 24 mm accept / 32 mm safe, against
the notes' 8–12 mm. **The notes' lower bound sits at the always-clogs boundary, and the
~10 mm Panot channel with 4 mm waste is in the same regime.**

In the sweep, 1233 cells fail the measured-section jamming rule, and **576 of those have a
section that clears the team's own 3 × d_max rule** — they would have been printed and
cast starved. That number is the concrete cost of the rule being too permissive.

**One internal inconsistency, stated plainly.** The shipped `Thresholds.jam_ratio`
defaults to **4.0**, not the literature's 4.94–6.0, because `Thresholds.from_lit` looks
for parameter keys that the mechanics file does not use (it stores the value under the
symbol `R_c = D_outlet/d_particle`). The engine therefore gates *less* strictly than this
report recommends. Counts of designs whose measured section falls below each candidate
threshold: **657** below 3×, **1233** below 4× (the shipped gate), **1812** below 4.94×,
**2706** below 6× (recommended), 4233 below 8×. To adopt the recommendation, set
`Thresholds(jam_ratio=6.0)` explicitly. This is left visible rather than silently
patched because it changes which designs are rejected and the choice belongs to the team.

---

## 5. Success estimator

Four subscores in [0,1], **multiplied**:

$$S = S_{\text{aer}} \times S_{\text{dry}} \times S_{\text{cast}} \times S_{\text{str}}$$

Multiplication, not averaging, because these are series requirements: a body that cannot
be filled does not get a second chance to be well oxygenated, and a mean would let a good
aeration score hide a fatal casting defect.

| subscore | form | rationale |
|---|---|---|
| $S_{\text{aer}}$ | logistic in cemented volume fraction, centred on 0.85 | the paper defines success as **complete** solidification, so partial coverage is penalised, not scored pro rata |
| $S_{\text{dry}}$ | $R^{-1.5}$ for $R>1$, where $R = (t_{\max}/2)/L_{\text{dry}}$ | at $R\le1$ the section drains and shrinks together; above 1 the surface shrinks against a wet core — the restrained-shrinkage condition, which cracks at ~75 % of splitting tensile strength |
| $S_{\text{cast}}$ | ramp between `3 × d_max` (certain clog) and `jam_safe × d_max` | granular jamming, §4.1 |
| $S_{\text{str}}$ | $(K_t/2)^{-m/4}$, $K_t = 1+2\sqrt{h/r}$ | Inglis notch factor converted to failure risk through the Weibull modulus |

**Uncertainty.** Every literature parameter carries a `[low, high]` range and is resampled
(triangular) per Monte Carlo draw; the reported score is the median with a 5–95 %
interval. Ranking by the median while showing the interval is the honest use of this
model — **the intervals are wide, and designs whose intervals overlap should be treated
as tied.** The dominant contributor is the biofilm volume fraction (0.01–0.10, a 10× span
that multiplies oxygen demand linearly), which is an assumption with no retrieved source.

**Failure attribution.** The smallest subscore names the failure mode, mapped to the
paper's own observations: `aeration` → incomplete mineralisation (anoxic core, Fig. 5);
`drying` → cracking from uneven drying (Fig. 5); `castability` → aggregate bridging /
starved mould feature; `structural` → crack initiation at a stress riser.

---

## 6. Validation: retrodicting the paper

Five cases, holding everything constant except the variable under test. The paper reports
outcomes qualitatively, so this is a **ranking** test, not a calibration.

| case | geometry | S/V (mm⁻¹) | t_max | cemented frac | score [5–95 %] | mode | feasible |
|---|---|---|---|---|---|---|---|
| **A** | solid ovoid, one-piece mould — *paper Fig. 5, FAILED* | 0.075 | 96 mm | **0.45** | 0.000 [0.000, 0.000] | **aeration** | no |
| B | solid ovoid, split mould | 0.127 | 76 mm | 0.57 | 0.000 [0.000, 0.003] | aeration | no |
| C | hollow shell, one-piece mould | 0.219 | 16 mm | 1.00 | 0.088 [0.000, 0.108] | castability | yes |
| **D** | hollow shell, split mould — *paper Figs. 6–7, SUCCEEDED* | 0.263 | 16 mm | **1.00** | 0.088 [0.000, 0.108] | castability | yes |
| E | hollow shell, sharp rim (r = 0.5 mm) | 0.260 | 16 mm | 1.00 | 0.001 [0.000, 0.001] | **structural** | no |

**The retrodiction succeeds on both counts.** The form the paper reports as failing is
rejected, with the failure attributed to aeration — the paper's own stated cause — and
only 45 % of its volume ever reaches oxygen. The form the paper reports as succeeding is
accepted with complete coverage. Case E confirms the notch term bites: removing the fillet
alone collapses an otherwise-sound design.

Cases B and C separate the two design moves. **Hollowing is the decisive one** (C vs A:
cemented fraction 0.45 → 1.00); the split mould alone recovers only 0.45 → 0.57, because
a parting plane adds exposed area but does not reduce the distance from the core to the
nearest surface. This is a testable prediction the pictorial does not distinguish: *a
solid form cast in halves should still fail.*

**Solver verification.** `solve_oxygen` reproduces the analytic zero-order slab to 0.1 %
in the unexhausted regime, and its front position converges with voxel pitch (24.9 % error
at 0.05 mm pitch on a 0.47 mm depth, falling as pitch decreases). A resolution guard flags
any run where the penetration depth spans fewer than 4 voxels, so the sub-voxel
dissolved-O₂ case reports "skin only" rather than a spuriously precise number.

---

## 7. Design space

**6912 design cells** = 768 geometries (256 Sobol points per typology) × 9 process cells
(`d_max ∈ {2,3,4} mm` × cure ∈ {14 d/90 % RH, 28 d/90 % RH, 21 d/85 % RH}). 6723 cells
were meshed, diagnosed, and scored with 250 Monte Carlo draws each. **3223 cells (46.6 %)
are feasible** — shell 1312, tile 1160, block 751.

### 7.1 Why designs are rejected

| limiting subscore | cells | | rule fired | cells |
|---|---|---|---|---|
| aeration (anoxic core) | 2654 | | `penetration_coverage` | 1755 |
| castability (bridging) | 2385 | | `measured_section_not_jamming` | 1233 |
| structural | 1381 | | `web_min` | 657 |
| drying | 303 | | `face_shell_min` | 639 |

The two dominant causes are **the paper's own failure mode** and **granular jamming**.
Structure almost never rejects a design (1381 structurally limited, only 254 failing a
structural rule) — which is the team's fillet rule working: once `r ≥ 1.5 × d_max` is
enforced, the notch problem is controlled and transport becomes binding.

### 7.2 The sieve-versus-cure trade

Two limits act on the same section thickness from opposite directions: the jamming floor
(`≥ 6 × d_max`) and the drying ceiling (`≤ 2 × L_dry`). Computed air-entry depths:
**10.5 mm** at 14 d/90 % RH, **21.0 mm** at 28 d/90 % RH, **23.6 mm** at 21 d/85 % RH.

| d_max | jamming floor | 14 d/90 % RH (ceiling 21.0) | 28 d/90 % RH (42.0) | 21 d/85 % RH (47.2) |
|---|---|---|---|---|
| 2 mm | 12.0 mm | 12.0 – 21.0 mm | 12.0 – 42.0 mm | 12.0 – 47.2 mm |
| 3 mm | 18.0 mm | 18.0 – 21.0 mm (marginal) | 18.0 – 42.0 mm | 18.0 – 47.2 mm |
| 4 mm | 24.0 mm | **closed** (24.0 > 21.0) | 24.0 – 42.0 mm | 24.0 – 47.2 mm |

Designs scoring above 0.9, from the swept data:

| d_max | 14 d/90 % RH | 28 d/90 % RH | 21 d/85 % RH |
|---|---|---|---|
| 2 mm | 21 | 164 | 228 |
| 3 mm | **0** | 65 | 94 |
| 4 mm | **0** | 12 | 17 |

**At the project's stated 4 mm aggregate with a 14-day 90 % RH cure, the window is
closed**: 62 cells still pass the rule set, but the best score anywhere is 0.379 and not
one design clears 0.9. Both escapes work, and they are not equivalent:

- **Change the cure** (keep 4 mm waste, go to 21 d/85 % RH): feasible cells 62 → **379**
  (6.1×), best score 0.379 → 0.998, designs above 0.9: 0 → **17**.
- **Sieve to 2 mm** (keep the 14-day cure): feasible cells 62 → **152** (2.5×), designs
  above 0.9: 0 → **21**.

**Recommendation: cure at ~85 % RH for ~21 days and do not sieve.** It needs no sieving
line, keeps the waste stream as-is, and yields six times more feasible geometry. Note that
lowering RH does more per day than extending time (23.6 mm in 21 days at 85 % RH beats
21.0 mm in 28 days at 90 % RH) — the humidity gradient, not elapsed time, drains the
pores. A minimum-intervention option also exists: at 4 mm the floor and ceiling meet
exactly at a 16-day cure (both 24.0 mm), so **17 days is the first schedule that actually
opens the window**; and on the sieving side the wedge is closed at exactly
`d_max = 3.5 mm` (floor and ceiling both 21.0 mm), so sieving must go **below** 3.5 mm —
3.4 mm opens it — for the existing 14-day schedule to work.

### 7.3 Recommended designs

Best overall (all at `d_max = 2 mm`, 21 d/85 % RH):

- **Shell** `shell_209`, **0.998 [0.998, 0.998]** — a = 58.2, b = 75.0, c = 71.7 mm,
  n = 2.1, ovoid = 0.40, wall 19.3, aperture_r 13.5, fillet 6.5 mm. Material 599.6 cm³,
  cavity 292.9 cm³, section 9.7 × d_max, cemented fraction 1.00, K_t 1.79. The only design
  in the sweep whose interval is degenerate at the ceiling — it stays above 0.99 at every
  corner of the literature ranges.
- **Block** `block_181`, 0.998 [0.182, 0.998] — face shell 33.9, web 29.2 mm, 2 cores,
  fillet 7.9 mm, taper 3.7°.
- **Tile** `tile_034`, 0.236 [0.189, 0.291] — flower pattern, t = 28.8 mm.

At the project's stated `d_max = 4 mm` (21 d/85 % RH): **shell** `shell_171`, 0.998
[0.189, 0.998], wall 30.6 mm — the wall must grow to clear the 24 mm jamming floor, which
costs **4.7× the material** of the 2 mm-waste shell for the same score. That ratio is the
real price of not sieving, and it is still better than a closed window.

### 7.4 Tiles are capped at 0.236, and this is a real result

Across all 2304 tile cells, $K_t \ge 3.0$ (range 3.00–3.78), because the builder clamps
the groove root radius to `min(fillet_r, w/2, groove_depth)` — so `r/h ≤ 1` and Inglis
gives $K_t \ge 3$ **by construction**. With *m* ≈ 14 the structural subscore cannot exceed
0.237. A groove cut into a brittle cast body *is* a notch. The fix is not a parameter
change but a change of relief type: a **shallow dished relief whose root radius exceeds
its depth** (Panot practice — 2–3 mm deep, ~10 mm wide, r ≈ 5 mm gives r/h ≈ 1.7,
$K_t ≈ 2.5$) rather than a groove. The swept range (`groove_depth` 2–8 mm) never reaches
that regime, which is itself a finding: **the tile grammar should be extended with a
dished-relief variant.**

---

## 8. Moulds

Split-mould negatives were generated for the vessel (6 parts: 2 halves + 2 loose cores)
and the block (2 halves), both on the 28-day-cure branch. Full detail in
`mould_notes.md`; the results that bear on design:

- **The widest section is not the equator.** The ovoid taper puts it at z = −21 mm
  (r = 50.04 mm vs 48.82 mm at the equator). Parting at the equator would leave a 1.2 mm
  re-entrant lip — a spalled edge on a green body, and invisible in a render.
- **The jamming criterion cuts both ways.** Aggregate feed and vent passages must
  *exceed* `6 × d_max` = 24 mm; **liquid-only drains must stay *below* `3 × d_max` = 12 mm**
  so they pass spent solution while retaining aggregate. Sizing a drain at 24 mm makes a
  hole the mix falls out of.
- **The vessel cannot use an integral core boss.** Its cavity is ~45 mm across at the
  parting plane but the aperture bore is only Ø32, so a drafted boss would be pinned at
  Ø32 and cast a 32 mm wall where 26 mm was designed. Two loose cores are required.
- **Verification:** all 6 parts watertight, winding-consistent, genus matching the counted
  through-holes; volume balance closes with **0.0 mm³ unattributed**; kinematic release
  sweeps clear for every part.
- **Open decision flagged, not resolved:** the block's as-cast web is 28 mm over the
  prismatic band (clears the 25 mm C90 floor) but dips to 16 mm in the last 8 mm at each
  end. That dip is the block's own 8 mm edge fillet — the fillet the team's rule requires —
  not draft relief; the nominal body measures the same there. Since C90 measures web
  thickness at its thinnest point, this unit would not pass if measured through the
  filleted end zone. Either reduce `fillet_r` to ~6 mm (still ≥ 1.5 × d_max) or accept it
  as a non-structural demonstrator cast. The choice is the team's.
- **Print:** PETG (PLA hydrolyses in a warm alkaline 28-day cure), 20 mm walls, parting
  face down, **PTFE liner mandatory** — FDM layer lines are a mechanical key and a green
  body will tear its surface off without one. Cure the halves **separately, open face up**:
  assembling early converts the parting face from an oxygen source into a sealed interface
  and reproduces the Fig. 5 failure.

---

## 9. Limits — read before using any number here

1. **The score is a ranking instrument, not a calibrated probability.** No pass/fail
   dataset was available to fit it: the pictorial reports success qualitatively (Fig. 2)
   with no numeric strength, CaCO₃ %, or penetration values. Use it to compare designs,
   not to predict a yield.
2. **The intervals are wide, sometimes uninformative.** `block_181` spans [0.182, 0.998];
   the block in the typology comparison spans 0.000–0.476 around a median of 0.002 — a
   factor of 200. Any ranking inside an overlapping band is noise.
3. **The weakest input is `phi_f_frac`** (biofilm volume fraction, 0.01–0.10, `ASSUMED`,
   no retrieved source). It multiplies oxygen demand linearly and alone spans a decade.
4. **Organism mismatch throughout the strength literature.** Nearly all MICP strength data
   is for *Sporosarcina pasteurii* (ureolytic, fast). One retrieved study on the actual
   substrate class — MICP-treated construction and demolition waste — gives UCS 724 kPa
   maximum, 490 ± 149 kPa typical, four to fifteen times weaker than clean sand, and that
   is still *S. pasteurii*; a *B. subtilis* penalty sits on top. One encouraging
   counterpoint: *B. subtilis* with calcium formate **exceeded** the ureolytic control on
   both surface resistance and carbonate content in a dune-stabilisation study.
5. **There is no usable UCS-versus-CaCO₃ relationship.** Pooled power-law, exponential and
   linear fits give R² of 0.008, −0.045 and 0.004 — two of them *worse than predicting the
   mean*. This is the finding, not a fitting failure: 0.12–0.16 MPa versus 1.65–1.82 MPa
   occur at the **same** sub-1.4 % carbonate content, differing only in whether an additive
   localised the precipitate at grain contacts. **Placement, not quantity, sets strength.**
   A hard gate at 3 % CaCO₃ (below which nothing stands unconfined) is used instead of a
   curve.
6. **$K_t$ is a relative ranking.** The source notes Inglis overestimates $K_t$ by >30 %
   against FE for rough-surface valleys, and a bio-cemented waste aggregate is neither
   homogeneous nor isotropic at aggregate scale.
7. **Eleven kinetics rows rest on unopened sources.** Of 43 rows: 20 MEASURED, 11
   SECONDARY, 9 DERIVED, 3 ASSUMED; by retrieval level, 28 FULL_TEXT, 9 ABSTRACT, 5
   METADATA_ONLY, 1 NO_SOURCE. The Millington–Quirk exponents and the Boudreau tortuosity
   relation rest on CrossRef metadata alone. `D_O2_water` was independently corroborated
   against a fully-retrieved source (agreement 0.1 % at 25 °C).
8. **No mechanical test.** Every strength statement is a geometric stress-concentration
   argument plus a literature Weibull modulus. Nothing here substitutes for casting a
   specimen and breaking it.
9. **Cure levels are coarse** (three cells). Given how strong the RH effect is, a dedicated
   RH sweep at fixed duration is worth running before committing a schedule.

### What would most improve this model

In priority order:

1. **A pass/fail table of your own casts** — geometry, `d_max`, cure schedule, and whether
   it solidified completely. Even 15–20 rows would convert the score from a ranking into a
   calibrated probability and let the subscore weights be fitted rather than assumed.
2. **One measurement of biofilm volume fraction** in an inoculated pack, which collapses
   the widest uncertainty in the chain.
3. **Sectioning one failed cast** and measuring the cemented depth directly — that single
   number tests the central `L_eff` prediction.
4. **A UCS test on your own waste** with *B. subtilis*, to replace the *S. pasteurii*
   proxy.

---

## 10. Parameter provenance

All parameters live in two machine-readable files with per-row provenance:
`micp_kinetics_params.json` (43 rows) and `mechanics_params.json` (42 rows, plus 11
UCS-vs-CaCO₃ data points and 21 $K_t$ points). Every row carries value, low, high, units,
`evidence_class`, organism, substrate, DOI, and notes showing derivation arithmetic;
kinetics rows additionally carry `retrieval_level`, and a validator asserts that no row
can be `MEASURED` without `FULL_TEXT`. **34 unique DOIs.**

Key sources by role:

| role | value | source |
|---|---|---|
| O₂ diffusivity in air | 2.209 × 10⁻⁵ m² s⁻¹ | NIST Technical Note 2279, `10.6028/nist.tn.2279` |
| O₂ diffusivity in water | 2.0 × 10⁻⁹ m² s⁻¹ | `10.1021/jp952903y` (not retrieved; corroborated by Stewart 2003, retrieved) |
| *B. subtilis* respiration | 7.3 mmol O₂ (g DCW)⁻¹ h⁻¹ | retrieved full text |
| jamming criterion | R_c = 4.94 (spheres), 6.0 (angular) | Zuriguel et al. 2005, `10.1103/PhysRevE.71.051303` |
| wet clogging | always clogs below W/d = 3; W_c/d = 8.1 dense | Vani, Escudier & Sauret 2022, `10.1039/D2SM00962E` |
| notch factor | $K_t = 1+2\sqrt{h/r}$ | classical Inglis (1913) equivalent-ellipse result. Reached via Gebrehiwot et al. 2023 `10.23998/rm.124815`, where it is Eq. 5 — that paper's full text was NOT retrieved, so treat the citation as the route to a textbook formula, not as a verified source |
| Weibull modulus | m = 11.5–16.8 | Jiang et al. 2024 |
| UCS on demolition waste | 490 ± 149 kPa, max 724 kPa | Fouladi et al. 2024, `10.1007/s11440-024-02396-8` |
| no pooled UCS–CaCO₃ fit | R² ≤ 0.01 | Fu, Saracho & Haigh 2023, `10.1016/j.bgtech.2023.100002` |
| CaCO₃ floor | ~3 % to stand unconfined | Fu, Saracho & Haigh 2023 |
| evaporation transition | 0.5–2.5 mm day⁻¹ | Shokri & Or 2011 |
| shrinkage crack criterion | σ = 0.75 × splitting tensile | JCI, after van Breugel & Lokhorst 2001 |
| tensile/compressive ratio | 0.19–0.25 | Nafisi et al. 2020, `10.1139/cgj-2019-0230` |
| *B. subtilis* CaCO₃ yield | 2.8 % (formate), 1.3 % (acetate) | Hemayati et al. |
| face shell / web minima | 32 mm / 19 mm | ASTM C90 Table 1 (via NCMA/CMHA) |
| Panot relief ceiling | 5 mm grooves, 4 mm studs | Orden VIV/561/2010; Barcelona Plec Tècnic |
| aggregate vs form width | `d_max ≤ W/5`, slab `≥ 3 d_max` | ACI 318 26.4.2.1(a)(5) |

Project sources: the pictorial manuscript and the design notes of 19/07/2026, consolidated
in `design_brief_sources.md`.

---

## 11. Reproducing this

```bash
tar xzf biocast_pkg.tar.gz
# put micp_kinetics_params.json and mechanics_params.json alongside
PYTHONPATH=. python validate_paper.py        # reproduces the §6 table
```

```python
from biocast.params import ShellParams, Design, Mix, Process, LitParams
from biocast.grammars import shell
from biocast.physics import fields
from biocast import score, constraints

p = ShellParams(a=58.2, b=75.0, c=71.7, n=2.1, ovoid=0.40,
                wall=19.3, aperture_r=13.5, fillet_r=6.5)
mesh, fld, origin, pitch = shell.build(p, pitch=1.5, return_field=True)
diag = fields.geometric_diagnostics(mesh, pitch, parting_axis=2,
                                    parting_frac=0.5, field=(fld, origin, pitch))
d = Design(geom=p, mix=Mix(d_max=2.0), proc=Process(cure_days=21, rh_pct=85))
phys = score.PhysicsInputs.from_lit(LitParams("micp_kinetics_params.json"),
                                    LitParams("mechanics_params.json"))
print(score.score_design(d, diag, phys=phys))
print(constraints.summarise(constraints.check(d, constraints.Thresholds(jam_ratio=6.0), diag=diag)))
```

**Two pitfalls that will bite anyone extending this.** Always pass
`field=(fld, origin, pitch)` into `geometric_diagnostics` — trimesh's `.fill()` floods
internal cavities (27 % volume overestimate, which makes every hollow design look solid
and saturates the aeration subscore at 1.0 for everything), and `mesh.contains()` on an
object-scale grid exhausts memory. And any new marching-cubes code must pass
`allow_degenerate=False`: flat faces land on grid planes, produce tens of thousands of
exact zeros, and the mesh tears when vertices merge.
