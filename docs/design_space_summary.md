# Design-space sweep — MICP success versus castability

768 geometries (256 Sobol points per typology) × 9 process cells
(d_max ∈ {2, 3, 4} mm × cure ∈ {14 d/90 % RH, 28 d/90 % RH, 21 d/85 % RH}) = **6912 design
cells**. 747 geometries survived the parameter-level constraint filter and were voxelised,
diagnosed and scored with 250 Monte Carlo draws each; **6723 cells carry a full geometry +
score**, and every mesh built in the sweep was watertight (6723/6723).

| | cells | share |
|---|---|---|
| sampled | 6912 | 100 % |
| geometry built + scored | 6723 | 97.3 % |
| rejected on parameters alone (never meshed) | 189 | 2.7 % |
| **feasible** (no hard-rule failure) | **3223** | **46.6 %** |

Feasible by typology: shell 1312, tile 1160, block 751.

---

## 1. Why designs were rejected

**Limiting subscore across all 6723 scored cells** (the subscore that came out lowest; solid
count = cells that a hard rule also rejected):

| limiting subscore | cells | of which infeasible |
|---|---|---|
| aeration (anoxic core) | 2654 | 1840 |
| castability (aggregate bridging) | 2385 | 1312 |
| structural (crack at a stress riser) | 1381 | 254 |
| drying (uneven-shrinkage cracking) | 303 | 94 |

**Hard rules that fired** (a cell can fail several):

| rule | cells | origin |
|---|---|---|
| `penetration_coverage` — under 85 % of the body reaches the cementation threshold | 1755 | LIT |
| `measured_section_not_jamming` — measured narrowest section below 4 × d_max | 1233 | LIT |
| `web_min` — web under the 25 mm ASTM C90-type floor | 657 | STD |
| `measured_section_over_dmax` — measured section below 3 × d_max | 657 | LIT |
| `face_shell_min` — face shell under 32 mm | 639 | STD |
| `aperture_not_jamming` | 450 | LIT |
| `groove_jamming` | 285 | LIT |
| `fillet_radius_min` — fillet under 1.5 × d_max | 183 | TEAM |
| `min_section_thickness` (nominal) | 120 | LIT |
| `groove_pitch_gt_width` | 99 | GEOM |
| `groove_depth_max` — deeper than t/4 | 63 | TEAM |

The two dominant rejection causes are the paper's own failure mode (an anoxic core that never
completely solidifies) and granular jamming in the mould. Structure is almost never what
rejects a design — 1381 cells are structurally limited but only 254 fail a structural rule.
That is the quantitative form of the team's fillet rule working as intended: once r ≥ 1.5 ×
d_max is enforced, the notch problem is controlled and transport takes over as the binding
constraint.

### A correction the sweep forced

`score._infer_min_feature` takes the narrowest mould passage from the *nominal* parameters. For
the shell that is `wall`, and it is wrong whenever the aperture bore eats into the wall it is
supposed to be measuring: the grammar subtracts a cylinder spanning z ∈ [−c, 3c], so
`aperture_r` bores the **full height** of the body, not just the cap. Measured case
(a = 40.3, wall = 35.9, aperture_r = 23.7, d_max = 4 mm): an equatorial slice through the solid
shows two 10.0 mm-wide segments, and the medial-ridge measure the sweep actually uses gives a 5th
-percentile section of **8.0 mm against a 35.9 mm nominal wall** — a 4.5× overstatement that
turned a 2.0 × d_max passage (inside the always-clog band) into an apparent 9 × d_max and handed
it castability = 1.00 and an overall score of 0.998.

The sweep therefore measures the narrowest section from the voxel field directly (2 × the
distance-to-air field on its own medial ridge, 5th percentile) and feeds *that* into castability,
plus two geometry-resolved rules (`measured_section_over_dmax`,
`measured_section_not_jamming`) that the parameter-level rules cannot see. The flagged design
drops from 0.998 to **0.020** and is now correctly rejected. This moved 257 cells from feasible
to infeasible (3480 → 3223) and is the difference between recommending a mould that can be
filled and one that cannot.

---

## 2. Sieve versus cure, quantified from the swept data

The conflict is between two limits on the same section thickness:

- **jamming floor**: every passage ≥ ~6 × d_max (Zuriguel 2005 R_c = 4.94 spheres, 6.0 angular)
- **drying ceiling**: a section only dries uniformly to ~2 × L_dry, where L_dry is the air-entry
  depth reached by the end of the cure

Air-entry depths computed by the model: **L_dry = 10.5 mm** at 14 d/90 % RH, **21.0 mm** at
28 d/90 % RH, **23.6 mm** at 21 d/85 % RH. The window between floor and ceiling closes exactly
where the brief predicted:

| d_max | jamming floor 6 × d_max | 14 d/90 % RH (ceiling 21.0) | 28 d/90 % RH (ceiling 42.0) | 21 d/85 % RH (ceiling 47.2) |
|---|---|---|---|---|
| 2 mm | 12.0 mm | 12.0 – 21.0 mm | 12.0 – 42.0 mm | 12.0 – 47.2 mm |
| 3 mm | 18.0 mm | 18.0 – 21.0 mm (marginal) | 18.0 – 42.0 mm | 18.0 – 47.2 mm |
| 4 mm | 24.0 mm | **closed** (24.0 > 21.0) | 24.0 – 42.0 mm | 24.0 – 47.2 mm |

Designs scoring above 0.9, counted across all three typologies:

| d_max | 14 d/90 % RH | 28 d/90 % RH | 21 d/85 % RH |
|---|---|---|---|
| 2 mm | 21 | 135 | 164 |
| 3 mm | **0** | 54 | 72 |
| 4 mm | **0** | 12 | 15 |

Mean score over feasible cells:

| d_max | 14 d/90 % RH | 28 d/90 % RH | 21 d/85 % RH |
|---|---|---|---|
| 2 mm | 0.531 | 0.517 | 0.548 |
| 3 mm | 0.352 | 0.436 | 0.472 |
| 4 mm | 0.216 | 0.355 | 0.391 |

**The brief's conflict is confirmed and sharpened.** At d_max = 4 mm with a 14-day 90 % RH cure,
62 cells still pass the rule set but the best score anywhere is **0.379** and *not one design*
clears 0.9 — the window is closed and what survives is marginal. Both escapes work, and they are
not equivalent:

- **Extend/dry the cure** (d_max stays 4 mm, cure → 21 d/85 % RH): feasible cells 62 → **379**
  (6.1×), best score 0.379 → **0.998**, designs above 0.9: 0 → **15**.
- **Sieve to 2 mm** (cure stays 14 d/90 % RH): feasible cells 62 → **152** (2.5×), best score
  → 0.998, designs above 0.9: 0 → **21**.

Changing the cure is the stronger lever *and* the cheaper one: it needs no sieving line, keeps
the waste stream as-is, and yields six times more feasible geometry than shortening the cure and
sieving. Lowering RH from 90 % to 85 % does more per day than extending 14 → 28 days at 90 % RH
(L_dry 23.6 mm in 21 days versus 21.0 mm in 28 days) — the humidity gradient, not elapsed time,
is what drains the pores. **Recommendation: cure at 85 % RH for ~21 days and do not sieve.**

A note on the team's own rule: the notes give 2–3 × d_max as the groove/passage minimum. Every
retrieved jamming source puts the critical ratio at 4.94–6.0 (and Vani 2022 reports certain
clogging below 3), so the team's rule sits at or below the always-clog boundary. Of the 6912
cells, 1233 fail the measured-section jamming check; **576 of those have a measured section that
clears the team's own 3 × d_max rule** and would therefore have been printed and starved. That
576 is the concrete cost of the 2–3× rule being 2–3× too permissive.

---

## 3. Pareto fronts

Objectives: maximise score, minimise material volume, and for the shell also maximise enclosed
cavity volume (the paper's Fig. 7 living-colony space). Fronts taken over feasible, meshed cells
only.

| typology | feasible cells | non-dominated | best score |
|---|---|---|---|
| shell | 1312 | 105 | 0.998 |
| block | 751 | 1 | 0.998 |
| tile | 1160 | 4 | 0.236 |

The shell front is broad (105 members) because the third objective — cavity volume — genuinely
trades against material volume, so many geometries are incomparable. The block front collapses
to a single point: with face shell and web pinned near their standard minima by ASTM-type rules,
score saturates and only volume separates designs, leaving one winner.

**Tiles are capped at 0.236 and no tile can do better.** Across all 2304 tile cells, K_t ≥ 3.0
everywhere (range 3.00–3.78), because the tile builder clamps the groove root radius to
`min(fillet_r, w/2, groove_depth)` — the root radius can never exceed the groove depth, so
r/h ≤ 1 and Inglis gives K_t ≥ 3 by construction. With a Weibull modulus m ≈ 14 the structural
subscore cannot exceed 0.237. This is a real result, not a scoring artefact: a groove cut into a
brittle cast body is a notch, and the only way to raise the tile's ceiling is to change the
relief geometry — a shallow dished relief whose root radius exceeds its depth (Panot practice:
2–3 mm deep, ~10 mm wide, r ≈ 5 mm gives r/h ≈ 1.7, K_t ≈ 2.5) rather than a groove. The swept
span (groove_depth 2–8 mm) never reaches that regime.

---

## 4. Recommended designs

### Best overall (all land at d_max = 2 mm, 21 d/85 % RH)

**Shell — `shell_209`, score 0.998 [0.998, 0.998]**
a = 58.2, b = 75.0, c = 71.7 mm, n = 2.1, ovoid = 0.40, wall = 19.3, aperture_r = 13.5,
fillet_r = 6.5 mm. Material 599.6 cm³, enclosed cavity 292.9 cm³, measured section 19.3 mm
(9.7 × d_max), max local thickness 20.4 mm, S/V 0.219 mm⁻¹, cemented fraction 1.00, K_t 1.79.
The only design in the whole sweep whose 5–95 % interval is degenerate at the ceiling — it stays
above 0.99 across every corner of the literature ranges. Split into halves of 311.0 and
288.6 cm³.

**Block — `block_181`, score 0.998 [0.182, 0.998]**
face_shell 33.9, web 29.2 mm, 2 cores, fillet_r 7.9 mm, core_taper 3.7°. Material 5889.9 cm³,
measured section 20.0 mm, max thickness 42.4 mm, S/V 0.088 mm⁻¹, cemented fraction 1.00,
K_t 1.79. Sits just above both ASTM-type floors, which is what the standard proportions were
selected for; the wide lower bound reflects that a 42 mm section is only cementable at the
favourable end of the transport ranges.

**Tile — `tile_034`, score 0.236 [0.189, 0.291]**
flower pattern, t = 28.8, groove_depth 5.8, groove_width 39.8, groove_pitch 46.4, fillet_r 9.1,
joint 3.8 mm. Material 899.7 cm³, measured section 24.0 mm, S/V 0.112 mm⁻¹, cemented fraction
1.00, K_t 3.00. Aeration, drying and castability are all at 1.00; the score is entirely the
structural ceiling described above.

### Best at the project's stated d_max = 4 mm (also 21 d/85 % RH)

**Shell — `shell_171`, score 0.998 [0.189, 0.998]**
a = 77.5, b = 80.6, c = 114.6 mm, n = 3.3, ovoid = 0.00, wall = 30.6, aperture_r = 26.9,
fillet_r = 10.4 mm. Material 2799.8 cm³, cavity 1476.4 cm³, measured section 30.5 mm
(7.6 × d_max), max thickness 32.0 mm, cemented fraction 1.00. The wall has to grow to 30.6 mm to
clear the 24 mm jamming floor, which is why the object is 4.7× the material of the 2 mm-waste
shell for the same score — that ratio is the real cost of not sieving, and it is still the
better trade than a closed window.

**Block — `block_090`, score 0.904 [0.013, 0.998]**
face_shell 32.5, web 36.1 mm, 3 cores, fillet_r 6.1 mm, core_taper 0.8° (below the 1° draft
preference — a warn, not a fail). Material 7724.8 cm³, measured section 30.0 mm, max thickness
45.0 mm.

**Tile — `tile_089`, score 0.232 [0.016, 0.290]**
radial pattern, t = 37.0, groove_depth 4.8, groove_width 33.4, groove_pitch 42.1, fillet_r 9.9,
joint 5.5 mm. Material 1289.4 cm³, measured section 33.4 mm.

---

## 5. Honest limits

- **The intervals are wide and should govern how these are used.** Only `shell_209` has a
  5-percentile bound at the ceiling; `block_181` spans [0.182, 0.998] and `block_090`
  [0.013, 0.998]. Designs whose intervals overlap are tied. The single largest contributor is
  the biofilm volume fraction (0.01–0.10, a 10× span that multiplies oxygen demand linearly).
- **The score ceiling is 0.9975, not 1.0**, set by the aeration logistic at cemented fraction
  1.00. Many designs sit exactly there, so the median alone does not separate them — export
  ranking used the 5th-percentile bound as the tie-break.
- **K_t is a relative ranking, not an absolute stress multiplier.** The source notes that Inglis
  overestimated K_t by more than 30 % against FE for rough-surface valleys, and that a
  bio-cemented waste aggregate is neither homogeneous nor isotropic at aggregate scale.
- **Voxel discretisation** quantises measured sections to the grid: shell/block/tile pitches
  2.0/2.5/1.2 mm in the sweep, 1.4/1.8/1.0 mm for the exports. Rebuilt export volumes agree with
  the swept values to within 0.46 %.
- **Cure-schedule levels are coarse** (three cells). The RH effect is strong enough that a
  dedicated RH sweep at fixed duration would be worth running before committing a schedule.
- **No mechanical test.** Every strength statement is a geometric stress-concentration argument
  combined with a literature Weibull modulus; nothing here substitutes for casting and breaking
  a specimen.

## Files

- `design_space.csv` — 6912 rows × 85 columns: every parameter, both parameter-level and
  geometry-resolved verdicts, all four subscores, score with 5–95 % interval, dominant failure
  mode, cemented fraction, penetration depths, K_t, S/V, thicknesses, volumes.
  `feasible` and `failed_rules` are the summary columns; `evaluated_stage` distinguishes
  `geometry+score` from `parameters_only`.
- `pareto_front.csv` — 110 non-dominated rows with the trade-off columns and front size.
- `stl_manifest.csv` — 20 exported meshes (all watertight) with Euler number, volume, and the
  rebuild-versus-sweep volume delta.
- `out/stl/*.stl` — the recommended designs; shells also as `__lower_half` / `__upper_half`.
