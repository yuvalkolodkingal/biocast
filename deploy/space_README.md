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

**Three tabs**

- **Design** — move sliders, see the meshed body in 3D with a sectioning plane, the
  four subscores, every constraint verdict, and download the STL.
- **Process window** — the castability floor against the drying ceiling, and what cure
  or sieve opens the window when they cross.
- **Explore** — randomised search inside the current mix and cure settings, ranked.

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
