# Mould generation — design record

`biocast/mould_cast.py`. Two goals from one geometry core:

- **silicone** — two-pour interlocking tooling: print `jacket_a`, `jacket_b` and
  `core`, cast the rubber in them, cast the mix in the rubber.
- **rigid** — a split negative the mix is poured straight into.

Runtime is **2–7 s per mould**. The voxel generators this replaced took 2–6 minutes.

---

## 1. What it produces

Measured at `d_max = 4 mm`, 28 d / 90 % RH, on the baseline designs
(`ShellParams(wall=26)`, `BlockParams(face_shell=37, web=30)`, `TileParams()`):

| | shell | block | tile |
|---|---|---|---|
| parting axis / plane | 2 @ −16.6 mm | 2 @ −75.2 mm | 2 @ 0.0 mm |
| undercut score | 0.364 | 0.076 | 0.002 |
| window Ø / pitch | 10 / 28 mm | 10 / 24 mm | 10 / 34 mm |
| cover surrogate | 0.906 | 0.897 | 0.921 |
| pillars formed | 9 | 80 | 36 |
| plastic, silicone goal | 1413 cm³ | 15842 cm³ | 2715 cm³ |
| silicone | 1170 g | 4140 g | 589 g |
| plastic, rigid goal | 1410 cm³ | 10384 cm³ | 767 cm³ |
| every part 1 body, watertight | ✓ | ✓ | ✓ |
| coverage meets 0.85 | ✓ | ✓ | ✓ |

`PYTHONPATH=. python examples/regenerate_moulds.py` rebuilds all six sets in about
half a minute and re-runs every check.

## 2. Why this replaced the voxel generators

`mould_auto.py` and `mould_silicone.py` booleaned occupancy grids — 6–24 M voxels,
a few dozen full-grid arrays, minutes per solve. They were precise, and they kept
producing parts that verified beautifully and could not be built. The vessel's
former failed its own release sweep to the end, through four separate fixes:
transverse window pillars that sheared through the cured rubber, a parting membrane
trapped under the upper skin, a tongue ring that `d_out` also placed *inside* the
bore, and a shellwall-thick plug sitting in the vessel's own aperture.

Every one of those is a question about **getting rigid tooling out of rigid
tooling**. The geometry here does not ask it.

## 3. The two ideas that do the work

Both come from `auto-mold-generator`, rebuilt here on trimesh + manifold booleans.

**A box for the plastic, a hugging silhouette for the rubber.** The jacket and the
rigid block are axis-aligned boxes, as in
[`automated_3d_mold_generator`](https://github.com/Lion4re/automated_3d_mold_generator);
the silicone chamber is the part's own shadow offset by the skin thickness.

That split is deliberate and it is what fixed the appearance. A box face is two
triangles, where an offset silhouette has to be triangulated and comes out as a fan of
long slivers from a single vertex — **628 cap triangles on the vessel against 12**.
Those slivers were the "stretched out" look. Buying the clean version in RUBBER would
be expensive; buying it in filament costs 1.02x on the tile, 1.18x on the vessel and
1.66x on the block, and the silicone bill does not move at all.

**A fused core, so nothing can get stuck.** The part stands in the middle of the
tooling on a bottom flange and the silicone fills the gap around it. The elastomer
is what demoulds, and it demoulds by stretching — the one thing it is good at.

The flange is cut from the part's **bottom section**, not its shadow. On a part with
a chamfered base the shadow fills the chamfer in, and the rubber would never take
that shape.

## 4. What was kept from the voxel path, and why

The geometry is not the point of this project. §10–11 of
[`mould_auto_notes.md`](mould_auto_notes.md) still hold and are load-bearing:

> A silicone face is a **no-flux boundary**. A 6 mm skin carries ~294× the oxygen
> resistance of the drained pore network behind it. A fully enclosed skin cements
> **0.000** — anoxic by construction, not merely poor.

So `size_windows` still steps the lattice pitch down until the body meets the
coverage criterion, and the tooling still grows **pillars** across the gap so the
cast skin demoulds already perforated. A mould that cannot breathe is not simpler,
it is wrong.

Coverage uses the **drained-depth surrogate**, not the reaction–diffusion solve. It
is what the ladder can afford per candidate pitch, and it only ever *chooses* a
pitch — it is never reported as the design's cemented fraction. The two are not
interchangeable and the return says which it is.

## 5. Decisions that are measured rather than assumed

**The parting axis is searched.** All three axes are scored and the one that
undercuts least wins — the vessel picks axis 2 at an undercut of 0.364 against 0.66
and 0.65 for x and y. Undercut is the lateral protrusion of each slice past its
inboard neighbour, accumulated away from the plane and normalised by the largest
section, so it is scale-free and comparable between axes.

**The plane goes at the centre of the max-area plateau, not the argmax.** On a
uniform prism every slice has the same area, so an argmax lands wherever
floating-point noise peaks — usually an end, which splits the part into a lid and
everything else. The tile parts at exactly 0.0 mm because of this.

**Draft subdivides before it warps.** Warping only moves existing vertices, so a
long flat face with corner vertices only would scale uniformly instead of tapering,
and the draft would be reported but not present — the same failure §2 of the old
record documents for the voxel path.

**Keys are projected onto the real outline.** `sqrt(area/pi)` is the right radius
only for a disc; on the block's 390 × 190 footprint it lands outside the ring at the
ends and inside it on the flanks, and a key placed there floats free of the jacket.
Measured as 2–3 disconnected bodies per half before the fix. Three keys at
0/140/250° mate one way only, so a half cannot be clamped on 120° out.

**The jacket has a lid as well as a floor.** Without one the upper half is an open
ring: its pillars have nothing to hang from and come off as loose rod — measured, 37
separate bodies in one jacket half. The pour goes through a spout in the lid.

**Splitting is a boolean, never `slice_plane`.** Its cap is a fan triangulation from
a single vertex, so a 120 mm parting face came out as a star of enormous slivers —
visible on every split part, and degenerate triangles a slicer then has to chew
through. Intersecting with an oversized half-space box re-triangulates the cut face
properly, which is how `automated_3d_mold_generator` does it too.

**Subdivision in `apply_draft` is capped.** It exists so a long flat face gains
vertices to taper around, and on the tile it fired four times and turned 56 k
triangles into 224 k — the 11 MB STLs. Capped at 4x the input or 120 k faces.

**Keys are frusta, not cones.** A sharp apex prints badly, locates nothing once the
tip rounds over, and reads as a spike in every render.

**Pillars run along the draw only.** In the *skin* a bore is a cut and any direction
works; the tooling has to **form** it with a solid pillar, and a pillar across the
pull shears through the cured rubber when the mould opens. This costs open area and
is the only version that can be built.

## 6. The checks, and why there are only four

The voxel path ran volume balances, release sweeps, Euler counts and topology
partitions because its booleans could silently misplace material on a grid. Mesh CSG
either yields a valid solid or raises. What is left is what the geometry cannot
guarantee:

| check | why it can fail |
|---|---|
| every part is a single printable body | a key or pillar detached from the jacket |
| every part is watertight | a boolean produced a non-manifold result |
| windows meet the aeration criterion | the limit is drying depth, not open area |
| window bore clears the clog band | a bore under 2 × d_max self-dams |
| **the fill port opens into rubber** | a port on the plan centroid misses an annular gap |
| **the cast skin is one piece** | the window pillars severed the rubber |
| the halves do not overlap | a mis-signed boolean gave both the same material |

The last three were added after the port check caught a real failure. **The fill port
is placed by measurement, not by construction**: candidate positions come from the
pole of inaccessibility of the gap's own cross-section at several heights, and each is
scored by the volume of rubber its actual bore opens into. Ranking on realised
intersection is the only thing that works, and both failure modes were measured — a
port on the plan centroid opens into the vessel's aperture bore and touches **0.0 cm³**
of rubber, and a pole of inaccessibility taken on the wrong section still missed on the
block, on a 53 517 mm² section that looked perfectly healthy. Both now open ~4.6–4.9 cm³.

The check has a discrimination control: forcing the plan-centroid placement makes it
fail on both the vessel and the block, so a pass is evidence rather than decoration.

## 7. What this record does not cover

- **Nothing here has been cast.** Verification is geometric and transport-based.
- **Silicone volume is the cost driver and this geometry uses more of it.** A
  vertical prism around a curved body holds more rubber than a conformal skin: the
  vessel is 906 g here against 316 g for the old conformal offset. That is the price
  of tooling that opens, and it is a real price — see §13 of the old record.
- **Marching-cubes stair-steps.** The grammars mesh a voxel field, so every part
  inherits a step at every voxel on the cavity wall. The silhouette is de-staircased
  but the cavity surface is not; fixing it means smoothing the design mesh before CSG
  or meshing the field finer, and both change the cast geometry slightly.
- **The silicone mechanicals are supplier values** for named grades, not
  measurements on the rubber this team would buy, and none were measured after
  28 days in an alkaline urea/CaCl₂ solution.
