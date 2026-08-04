# SOURCE EXTRACT 1 — "Still Life" pictorial (Ioshpe & Kolodkin-Gal, Reichman University)
Bacillus subtilis MICP on granular composites / ground construction waste.

Key statements (verbatim-derived):
- Success rate defined as COMPLETE SOLIDIFICATION of the aggregate; significantly higher in
  bacteria-treated vs untreated aggregates (Fig 2). Construction waste = highly viable substrate.
- Pore analysis by automatic image analysis: bacteria reduce porosity and fill gaps (Fig 3).
- Initial attempts FAILED due to "incompatibility with microbial fermentation and optimal surface
  ratio for heterotrophic bacterial growth" (B. subtilis cannot grow without O2 as terminal
  electron acceptor). => AEROBIC constraint is the primary geometric driver.
- A design that considers surface properties -- the two halves of an OVAL EGG-LIKE structure
  (Fig 6) -- proved SUCCESSFUL. Absence of a "scar" between halves => self-healing across the
  parting interface.
- Fig 5: early prototypes show cracking and incomplete mineralization due to UNEVEN DRYING and
  OXYGENATION challenges.
- Fig 6: split-mold technique casts in halves to ensure oxygen exposure + unified drying, with
  humidity control.
- Fig 7: final artifact has a HOLLOW internal structure to house the living colony; maintained
  with a nutrient dropper.
- NO numeric values are reported in the paper: no OD600, no cementation solution molarity, no
  curing time/temperature/RH, no cycle count, no UCS, no CaCO3 %, no penetration depth, no
  sample dimensions. All quantitative parameters must come from external literature.
- Cited works in the reference list include: Dhami/Reddy/Mukherjee 2013 Front Microbiol
  (biomineralization review, 10.3389/fmicb.2013.00314); Jroundi et al. 2017 Nat Commun (stone
  consolidation by self-inoculation); Bicer et al. (B. subtilis ATCC 6633 marble biohealing,
  ureolytic biocalcification); Dade-Robertson et al. Microb Biotechnol (growing buildings with
  bacterial biofilms); Haystead, J., Gilmour, K., Sherry, A., Dade-Robertson, M., and Zhang, M.
  (2024). Effect of (in)organic additives on microbially induced calcium carbonate precipitation.
  J Appl Microbiol 135. 10.1093/jambio/lxad309; Kolodkin-Gal lab papers on calcium
  signaling/deposition in B. subtilis multicellularity (Trends Microbiol 31:1225-1237; iScience 2022).

# SOURCE EXTRACT 2 — project design notes (translated from Hebrew, WhatsApp, 19/07/2026)
These are the team's own hard casting rules, to be encoded as constraints.

FILLET vs CHAMFER:
- Fillet (smoothing radius) is unambiguously preferred. In brittle materials like bio-cement or
  concrete, ANY angle -- even the obtuse angle of a chamfer -- acts as an engineering weak point
  creating STRESS CONCENTRATION. Under load the crack initiates exactly at that corner. A fillet
  distributes load smoothly over a continuous curve.
- HOW MUCH: rule of thumb, radius must be at least 1.5x to 2x the largest aggregate size.
  If construction-waste fragments reach 4 mm, fillet radius should be ~6 to 8 mm.
  => r >= 1.5 * d_max  (target 2.0 * d_max)

GROOVE / FLUTE THICKNESS AND DEPTH:
- Grooves cannot be too thin or too deep, else they weaken the material or simply fail to cast.
- MAX DEPTH: to keep a strong core and avoid an object that snaps in two, grooves may go at most
  1/3 of the total unit thickness (for a BLOCK), or 1/4 for a THICK TILE.
  => h_groove <= t/3 (block); h_groove <= t/4 (thick tile)
- MIN WIDTH: a groove must be at least 2x to 3x the largest aggregate in the mix. If the groove in
  the mould is too narrow, small stones JAM at the groove opening, block passage of sand and
  bacteria (phenomenon called BRIDGING), and the groove comes out defective and starved of material.
  => w_groove >= 2 * d_max (min), 3 * d_max (safe)

INDUSTRY PROPORTION FORMULAS:
- Standard hollow block (CMU): classic dimensions 20x20x40 cm. Void volume (cores) is ~40% to 50%
  of the block. Outer face-shell thickness ~3.2 cm; internal web thickness (partitions between
  cores) ~2.5 cm. This proportion gives dramatic material and weight savings without losing
  load-bearing capacity.
- Barcelona Panot tile: standard 20x20 cm, thickness 4 cm. The recessed relief depth producing the
  pattern is ONLY 2 to 3 mm (less than 10% of tile thickness); recessed channel width is usually
  around 1 cm. These grooves are for water drainage and slip prevention on sidewalks and have NO
  structural stiffening role.

PARAMETRIC MODEL INTENT (team's own words):
- Build an interactive parametric layout tool / algorithm (web app or Rhino Grasshopper) that
  encodes the rules of thumb as HARD PARAMETERS so it produces tiles satisfying both structural
  requirements and bacterial needs, and REJECTS geometries that would create breakage points
  before the mould-printing stage.
- Explicit rules named: minimum fillet radius r >= 1.5*d_max; maximum groove depth
  h_groove <= (1/3)*t_tile; JOINT TOLERANCES -- minimum gap between tiles in a composition to
  allow passage of feed solution and oxygen.
- Motivation: "such a system will save a huge amount of trial and error, because it will reject
  geometries liable to create breakage points before the mould printing stage."
