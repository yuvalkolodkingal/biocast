# Bio-concrete design studio — how to run it

An interactive front end on the `biocast` engine: move sliders, watch the score and
every constraint verdict update, download the STL when a design passes.

## Run

```bash
tar xzf biocast_pkg.tar.gz                 # if starting from the archive
pip install streamlit                      # plus the engine's own deps
PYTHONPATH=. streamlit run biocast/gui/app.py
```

It opens at `http://localhost:8501`. Keep `micp_kinetics_params.json` and
`mechanics_params.json` in the project root — the app loads the literature ranges
from them and falls back to package defaults if they are missing.

## The three tabs

**Design.** Pick a typology, move the geometry sliders. Every change re-meshes,
re-diagnoses and re-scores in about a second. You get:

- a verdict banner (castable / rejected) with the score and its 5–95 % interval
- the four subscores, with the limiting one flagged — they multiply, so the
  weakest sets the total
- a **3D shape preview** of the meshed body: drag to orbit, shift-drag to pan,
  scroll to zoom. **Section** cuts it on an X, Y or Z plane and fills the cut face
  in red, which is how to read a wall thickness, a block web or whether a cavity
  actually closed; **Flip** takes the other half and **Facets** shades each
  triangle flat, showing the voxel steps the geometry was marched at. The camera
  and the section survive a slider move, so you can watch a wall thicken in
  section. Untick *Show the 3D shape preview* if the ~0.5 MB per update is
  awkward on a remote connection
- geometry and transport tables, including the **measured** narrowest section
- the full constraint table, each rule tagged by where it came from: *your notes*,
  *literature*, *standard*, or *geometry*
- optional oxygen field solve, drawing the mid-plane section with the anoxic core
  in dark red
- STL download (plus the lower half for split-mould shells) and a JSON record

**Process window.** The castability floor (`≥ 6 × d_max`) against the drying
ceiling (`≤ 2 × L_dry`), with your mix marked. When they cross there is no feasible
section thickness, and the tab tells you the two ways out: the shortest cure that
opens the window, or the sieve target that does. The heat map at the bottom gives
the minimum cure for every combination of aggregate size and humidity.

**Explore.** Finds the shape most likely to cement, inside the current mix and cure
settings. Random sampling, then a compass search — each parameter up and down, the
step halving only when a whole sweep fails to improve. The design currently on the
sliders is scored first, so the search can never hand back something worse than what
you have.

Sampling is where the designs come from. **The refinement has not been shown to find
better ones**: across all three typologies it matched the best sampled design every
time and never beat it, at both a 25 % and a 10 % first step. It confirms that a design
is a local optimum for single-parameter moves, which is worth something, and it is the
only stage that can improve on a hand-tuned starting point — but spend budget on
samples first.

**Ranking is feasibility-first**, and this is not a detail. A design that breaks a
hard rule cannot be cast, so no score buys its way past one that can; ties are settled
by the 5th percentile of the score, then the median. Ranking on score alone inverts
the ordering in practice — measured on the vessel at `d_max = 4 mm`, an 18 mm wall
breaking *two* rules scores 0.207 against a 27 mm wall breaking *one* and scoring
0.000 at the 5th percentile. On score alone the search parks on the 18 mm design and
rejects every refinement.

Measured, vessel at `d_max = 4 mm`, 21 d / 85 % RH, 24 samples and 2 refinement
levels — 84 designs in 88 s on two cores:

| | broken rules | score | feasible |
|---|---|---|---|
| slider defaults | 1 | 0.378 | no |
| best found | **0** | **0.657** | **yes** |

**Use this design** writes the winner into the Design tab's sliders; the Mould tab
then generates the pattern, the former and the jacket for it. Everything is
downloadable as CSV.

## Two things the GUI does differently from the raw package

**It scores the measured section, not the nominal one.** `score._infer_min_feature`
reads the narrowest passage off the parameters, which overstates it by up to 4.5×
when the aperture bore eats into the shell wall. The app measures it from the
occupancy grid instead, so castability reflects the passage that actually exists.
Both numbers are shown; when they diverge by more than 30 % you get a warning.

**It defaults to the literature jamming threshold.** The package ships
`jam_ratio = 4.0`; the granular-flow literature says 4.94 (spheres) to 6.0 (angular
grains). The app defaults to **6.0** and exposes the slider under *Rule set*,
because that choice changes which designs are rejected and it should be visible
rather than inherited. Your notes' 2–3 × d_max sits at the always-clogs boundary.

## Reading the score honestly

The number ranks designs; it is **not a calibrated probability** — there was no
pass/fail dataset to fit it against. The interval matters more than the median: it
is driven mainly by biofilm volume fraction (0.01–0.10, assumed, spans a decade),
and **two designs whose intervals overlap should be treated as tied.** The app says
so when the interval exceeds 0.4.

## Worked example — the three cases in the overview figure

| design | cure | score | verdict |
|---|---|---|---|
| solid, 76 mm section | 14 d, 90 % RH | 0.000 | rejected — `penetration_coverage`, 61 % coverage against the 85 % floor, 39 % of the body anoxic |
| hollow, 14 mm wall | 14 d, 90 % RH | 0.082 | rejected — `measured_section_not_jamming`, 20 mm section against the 24 mm floor |
| hollow, 26 mm wall | 21 d, 85 % RH | 0.655 | **accepted**, fully cemented, castability limiting at 0.71 |

The first is the paper's Fig. 5 failure reproduced from the physics. The second
shows why the successful geometry still needs a process change at 4 mm aggregate.
The third is what to build.

## Extending it

`biocast/gui/engine.py` is the only place the GUI touches physics — one
`evaluate()` call returns everything the interface displays. To add a typology,
write the grammar in `biocast/grammars/`, add it to `PITCH` and the dispatch dicts
in `engine.py`, then add its slider block to `geom_controls` in `app.py`. No
scoring code changes, and the preview needs no changes at all — it draws whatever
mesh `evaluate()` returns.

## A note on the preview mesh

`biocast/gui/viewer.py` is a self-contained WebGL canvas, so the GUI still needs
nothing beyond Streamlit and fetches nothing from a CDN. Two consequences worth
knowing:

- **What you see is decimated, what you download is not.** Marching cubes gives
  45k–113k triangles; above 50k the preview is vertex-clustered for display, and
  the caption says so with both counts. `Download STL` always writes the
  full-resolution mesh.
- **The terracing is real.** The shell surface comes from a distance transform of
  a *binary* voxelisation (`grammars/shell.py`), so it carries ~1 voxel of
  stair-stepping, which *Facets* mode shows plainly. It is in the exported STL
  too, and at a 2 mm pitch it is far below the tolerances the scores turn on —
  but it is the mesh, not a rendering artefact.
