# ISSB 2026 poster abstract

5th conference of the Israeli Society for Synthetic Biology
Tel Aviv University, Bar Shira Auditorium, 3 September 2026

---

## Title

**biocast: a parametric geometry generator with a physically grounded success estimator for bio-cemented construction waste**

## Authors

[to be added]

---

## Abstract (249 words)

biocast is a Python package that generates cast geometries for *Bacillus subtilis* biomineralisation on ground construction waste and estimates whether each one will solidify completely. Three parametric shape grammars, a hollow vessel, a masonry block and a relief tile, are built as signed-distance fields and meshed by marching cubes. Rounded CSG operators place a fillet of exact radius on every edge, and a chamfer cannot be expressed in the grammar at all, so the team's fillet-not-chamfer rule is enforced structurally rather than checked after the fact. Each candidate then runs against roughly twenty machine-checkable predicates, every one tagged by evidence origin: project practice, literature, published standard, or geometric self-consistency.

Scoring uses four subscores that multiply rather than average, because they are series requirements: a body that cannot be filled gets no credit for good oxygenation. The aeration term comes from an obstacle-constrained reaction-diffusion solve for oxygen on the occupancy grid, coupled to an evaporation front, so wall thickness and cure humidity enter through transport rather than through a heuristic. Every literature parameter carries a range and is resampled per Monte Carlo draw, so the output is a median with a 5 % to 95 % interval, a hard-rule verdict, and a named failure mode.

Validation is by retrodiction. The solid form the source project reports as failing is rejected at 45 % cemented fraction on aeration; the hollow split-mould shell it reports as succeeding is accepted at 100 %. Across a 6912-cell sweep all 6723 meshes were watertight and volumes matched independent voxel counts to 0.8 %. The package also emits split-mould negatives with a closed volume balance and ships an interactive design studio.

---

## Trimmed to 200 words, if the form is tighter

biocast is a Python package that generates cast geometries for *Bacillus subtilis* biomineralisation on ground construction waste and estimates whether each will solidify completely. Three parametric shape grammars, a hollow vessel, a masonry block and a relief tile, are built as signed-distance fields, so rounded CSG operators place exact fillets and a chamfer cannot be expressed at all. Each candidate runs against roughly twenty machine-checkable rules tagged by evidence origin.

Scoring uses four subscores that multiply rather than average, because they are series requirements. The aeration term comes from an obstacle-constrained reaction-diffusion solve for oxygen on the occupancy grid, coupled to an evaporation front, so wall thickness and cure humidity enter through transport rather than a heuristic. Literature parameters are resampled per Monte Carlo draw, so the output is a median with a 5 % to 95 % interval and a named failure mode.

Validation is by retrodiction: the solid form the source project reports as failing is rejected at 45 % cemented fraction on aeration, and the hollow split-mould shell it reports as succeeding is accepted at 100 %. Across a 6912-cell sweep all 6723 meshes were watertight, with volumes matching independent voxel counts to 0.8 %.

---

## Where each number comes from

All traceable in `docs/methods_report.md`:

- three grammars, exact fillets, chamfer inexpressible: §3
- rule set and evidence tags: §4
- multiplied subscores, Monte Carlo intervals, failure attribution: §5
- 45 % / 100 % retrodiction: §6
- 6912 cells swept, 6723 meshed, watertight, 0.8 % volume agreement: §3 and §7
- mould volume balance closing at 0.0 mm³ unattributed: §8

If a reviewer presses on what the score means, §9 is the honest answer: it ranks designs, it is not a calibrated probability, and no pass/fail dataset existed to fit it. The abstract claims retrodiction and mesh verification, not predictive accuracy.

---

## Re-checked against the automatic mould work

`validate_paper.py` was re-run after the automatic mould generators and the
sealed-face boundary condition landed (`docs/mould_auto_notes.md`). **Every number
the abstract quotes is unchanged**: case A (solid, monolithic) 0.45 cemented
fraction, rejected on aeration; case D (hollow, split) 1.00, accepted; ranking
correct; failure attributed to aeration/drying. The retrodiction claim stands as
written.

That is not a coincidence — the validator already modelled the mould the way the
new `exposure_mask_in_mould` does, adding the parting plane as atmosphere only for
the split case. The new work generalises that treatment to any mould geometry
rather than correcting it.

Two things a reviewer might now ask, both worth having an answer ready for.

**"Does the mould itself block oxygen?"** Yes, and it is now modelled: a mould face
— rigid or silicone — is a no-flux boundary, so only genuinely open area acts as
atmosphere. Curing the halves *open-faced* is what makes the split mould work;
assembling them early converts the parting face back into a sealed interface. On
the rigid tile mould the same body reads 1.000 demoulded, 0.515 fully enclosed,
1.000 with the parting face open.

**"Would a flexible mould help?"** Not by breathing. Silicone is highly
oxygen-permeable in absolute terms but carries ~294x the diffusive resistance of
the drained pore network behind it, and a 6 mm skin passes under a thousandth of
free evaporation. It earns its place by releasing without cavity draft — worth
5.0 mm of web on the vessel and block — not by transport. An enclosed silicone
skin cements **0.000**; a windowed one reaches 0.861 (vessel) to 0.885 (tile).

Neither point changes the abstract's claims, so no edit to the 249- or 200-word
text is needed. They are recorded here because both are natural poster questions
and the answers are now computed rather than speculative.
