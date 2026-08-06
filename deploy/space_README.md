---
title: Bio-concrete Design Studio
emoji: 🧱
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
fullWidth: true
short_description: Scored cast geometries for B. subtilis bio-cementation
---

# Bio-concrete design studio

Generates cast geometries for *Bacillus subtilis* MICP (microbially induced calcium
carbonate precipitation) on ground construction waste, checks them against a
machine-readable rule set, and estimates how likely each is to solidify completely —
with propagated uncertainty and an attributed failure mode.

Built for the *Still Life* project (A. Ioshpe, I. Kolodkin-Gal, Scojen Institute for
Synthetic Biology, Reichman University).

**Four tabs**

- **Design** — move sliders, see the meshed body in 3D with a sectioning plane, the
  four subscores, every constraint verdict, and download the STL.
- **Mould** — generate a mould for the shape you have dialled in, preview any part
  in 3D, and download the printable set as a zip with a fabrication manifest. Two
  goals: a **rigid split negative** to pour the mix straight into, or **two-pour
  silicone tooling** — print jacket A, jacket B and the master, cast the rubber in
  them, then cast the mix in the rubber. Nothing is hand-tuned per shape: the parting
  axis is the one that undercuts least, the tooling is the part's own silhouette
  grown by a wall, and the breather pitch is stepped down until the body meets the
  aeration criterion. It returns in seconds.
- **Process window** — the castability floor against the drying ceiling, and what cure
  or sieve opens the window when they cross.
- **Explore** — finds the shape most likely to cement: samples the design space to
  locate a promising basin, then runs a compass search inside it, ranking on the 5th
  percentile of the score rather than the median so a wide interval cannot flatter a
  design. One click pushes the winner into the Design sliders, and the Mould tab then
  builds the tooling for it.

**A mould face does not breathe.** Both paths treat rigid and silicone mould faces as
no-flux, so only genuinely open area acts as atmosphere. Silicone is highly
oxygen-permeable in absolute terms and still carries ~294x the diffusive resistance of
the drained pore network behind a 6 mm skin — it beats water and loses to air, which is
why the permeability intuition misleads. A fully enclosed silicone skin cements
**nothing**; the generator therefore sizes a breather lattice against the oxygen solve.
Silicone earns its place by releasing without cavity draft, worth about 5 mm of section
returned to the designer, not by transport.

Mould generation on this Space runs at a coarsened voxel pitch, stated in the tab, so
the solve fits the container's memory. Coverage is nearly pitch-independent, but
regenerate locally at the grammar pitch for final print files.

**Read the score honestly.** It ranks designs; it is *not* a calibrated probability, as
no pass/fail dataset was available to fit it. The interval matters more than the median,
and two designs whose intervals overlap should be treated as tied.

The mechanism is oxygen transport: *B. subtilis* is an obligate aerobe, and oxygen
reaches ~57 mm through drained pores but only ~0.3 mm through water-filled ones. That
factor of 190 is why a solid lump keeps an anoxic core while a thin shell cements
throughout.

---

This Space is a mirror of the `biocast` repository, deployed from `streamlit_app.py`.
This file is the Space card only — it is not the repository README.
