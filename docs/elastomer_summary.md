# Elastomer and silicone mould parameters for B. subtilis bio-cementation

Polydimethylsiloxane is one of the most oxygen-permeable polymers in common use, 600 Barrer against
about 10 for polyethylene terephthalate, so the expectation going into this retrieval was that a
silicone mould would breathe and that the silicone path would relax the aeration constraint the rest of
this project is organised around. It does the opposite. The comparison that decides the question is not
silicone against another polymer but silicone against the thing it replaces at the boundary, and what a
mould face replaces is open air. Reading the oxygen permeability of a 6 mm silicone skin,
2.01e-13 mol m m-2 s-1 Pa-1, against the gas-phase permeability of the 26 mm drained aggregate wall
behind it, 2.56e-10 mol m m-2 s-1 Pa-1, the skin carries **294 times the diffusive resistance of the
wall it covers** (220 to 505 across the 350-800 Barrer envelope). Against *saturated* pores the same
skin adds under one percent, a ratio of 0.0064 — and that second number is exactly why the intuition
misleads. PDMS genuinely beats water by a wide margin, and it loses to air by two and a half orders of
magnitude. *Bacillus subtilis* needs drained pores. A silicone mould face is therefore a no-flux
boundary, and only genuinely open area is atmosphere.

Water vapour tells the same story from the other side. Silicone is the most vapour-permeable common
elastomer, 23,000 Barrer for water against 600 for oxygen in the same film and the same published
table, and yet at 6 mm and at the cure's own driving force — pore air saturated, chamber at 90 % RH,
30 C — the skin passes 0.85 g m-2 day-1. The drying model's open face at the same relative humidity
passes 150 g m-2 day-1. The skin transmits 0.57 % of that, and the drying penetration depth at 28 days
falls from 21.0 mm to 0.12 mm, a section ceiling of 0.24 mm where the open-face ceiling is 42 mm. No
geometry has a 0.24 mm section. What silicone buys is undercut tolerance and zero-draft release, and it
should be specified for those reasons and not sold as breathable.

The full table is in [elastomer_params.json](../data/elastomer_params.json): 35 rows, 15 MEASURED,
14 DERIVED with the arithmetic written out in each `notes` field, and 6 ASSUMED. Five DOIs were
retrieved. Nine rows carry an explicit condition-mismatch flag, and the reason that count is high is
worth stating at the top: **not one retrieved source measured a property at 30 C, 28 days, pH ~9, in
contact with a calcium/urea cementation solution.** Every row is a transfer from somewhere else, and
each one names where from.

## What was retrieved

### Transport across the mould face

[Blume et al. 1991](<https://doi.org/10.1016/0376-7388(91)80008-t>) is the load-bearing source, and it
is load-bearing for a specific reason: its Table 1 reports oxygen at 600 Barrer and water at
23,000 Barrer *on the same film in the same table*, so the two can be compared without an inter-study
correction. The film is two-component RTV 615, addition-cured. The mismatches are that the table is at
40 C rather than 30 C and that RTV 615 is an unfilled optical-grade silicone rather than a filled mould
rubber. Both push the same way: gas permeability rises with temperature and filler lowers it, so 600
Barrer overstates the mould-rubber value at 30 C and the no-flux conclusion is conservative. The
oxygen-to-nitrogen ratio of 600/280 = 2.14 is the canonical PDMS selectivity, which is the check that
the film was defect-free rather than pinholed.

The Barrer-to-SI conversion is a derived row of its own rather than a constant to be trusted, because
it is easy to get wrong by a factor of ten and the whole comparison rests on it: 1 Barrer =
1e-10 cm3(STP) cm cm-2 s-1 cmHg-1, and converting each factor gives
1e-10/22414 x 1e-2 x 1e4 / 1333.22 = 3.3464e-16 mol m m-2 s-1 Pa-1. The project's working constant
3.348e-16 agrees to 0.05 %; the residual is whether 22414 or 22400 cm3 mol-1 is used for the STP molar
volume, and nothing turns on it.

### Hardness to modulus

Two relations were requested and both are recorded, along with the finding that the choice does not
matter where this project works and does matter above it. The ASTM-style exponential form
`E = exp(0.0235 SA - 0.6403)` MPa and [Gent 1958](https://doi.org/10.5254/1.3542351)
`E = 0.0981(56 + 7.62336 SA) / (0.137505(254 - 2.54 SA))` give 1.067 and 1.142 MPa at Shore 30A, a 7 %
spread that is a tie inside the scatter either relation admits. At Shore 50A they give 1.707 and
2.456 MPa, a 44 % divergence, and at 60A it is 67 %. Mould skins are specified at 15A to 30A, so quote
about 1.1 MPa at Shore 30A and do not agonise; if a harder rubber above 40A is ever specified, state
which relation produced the modulus and treat it as a factor-of-1.5 quantity.

There is a provenance trap in the exponential form that the file records explicitly. The Dow technical
bulletin that states it writes `log10(E) = 0.0235 S - 0.6403`, base ten, while the widely propagated
engineering form uses `exp()`, base e. These are different functions. At Shore 30A they give 1.161 and
1.067 MPa and the difference is invisible; at Shore 60A they give 5.884 and 2.159 MPa and the
difference is a factor of 2.7. The base-e form is what this project uses. Gent's own paper is the
better-founded of the two — its abstract argues the theoretical relation is "more appropriate than the
empirical one for small indentations" and that its constants "are not subject to experimental
uncertainty" — but Gent validates above roughly 40 hardness units, so at Shore 15-30 it too is being
used below its stated window. [Qi, Joyce & Boyce 2003](https://doi.org/10.5254/1.3547752) supplies the
third opinion by simulating the indentation itself, and concludes the durometer "still can be used as a
reasonable approximation of the initial neo-Hookean modulus unless the limiting extensibility is known
to be small" — which is to say the mapping degrades precisely for materials that cannot stretch far,
and mould silicones can.

### Alkaline compatibility over a 28-day cure

[Masson et al. 2022](https://doi.org/10.1016/j.engfailanal.2022.106305) is the only retrieved study that
aged a silicone in alkali on the right timescale, and it is simultaneously the most relevant and the
most over-severe source in the file. A silicone membrane was immersed at pH 13.5 — chosen as
representative of fresh concrete — at 40, 50, 60 and 70 C for up to 30 days. The rubber **stiffens
rather than softening**: Shore A rose from an initial 76, rubbery-plateau storage modulus crossed the
paper's 4 MPa failure criterion at about 10 days at 60 C and beyond 40 days at 40 C, relative crosslink
density roughly tripled, and FTIR showed loss of methyl content from the Si-CH3 bands at 1260 and
798 cm-1. Cracks were photographed at 23 days / 50 C. Dakin-Arrhenius on the stiffening kinetics gave
60 kJ mol-1 and a predicted service life just over a year, which the authors themselves flag as "a very
short time in comparison with established performance for silicones in construction" where sealants
"will readily perform well over 20 years."

The same paper supplies the mechanism for why alkali matters at all: it reports 96 kJ mol-1 for
water-driven hydrolysis of straight-chain PDMS, and 21 kJ mol-1 in acidic or alkaline water where a
nucleophile can polarise the Si-O-Si segment. The siloxane backbone is not inert to base; alkali drops
the barrier by a factor of 4.6.

Transferring that to the biocast case requires two extrapolations and the file writes both out. The
temperature step from 40 C to 30 C at 60 kJ mol-1 slows the stiffening by 2.1x. The pH step from 13.5
to about 9 — the initial culture pH recorded in `micp_kinetics_params.json` for the non-ureolytic
route — is 3.2e4 in hydroxide activity, and on a nucleophile-catalysed reaction that is by far the
larger factor. One cure should leave a 30A skin serviceable. **What is not supported is any claim about
a mould library**, because the exposure repeats and the failure mode is cumulative and irreversible.

### The polyesters, where the existing record needed correcting

The project record says PLA hydrolyses under warm wet alkaline conditions and PETG holds. Both halves
are supportable in *mechanism* and neither is supported by a measurement at biocast conditions, and the
more interesting correction is that the reason PETG is the better choice is not that it hydrolyses more
slowly but that it hydrolyses in a different *mode*.

[Schneider et al. 2020](https://doi.org/10.3390/polym12081711) treated printed PLA in 0.1 to 5.0 mol/L
NaOH and found that below 0.2 mol/L "no height changes are observed even at extended periods of time,"
that above 3.0 mol/L the scaffolds dissolve completely within 48 h, and — importantly — that across the
whole range "the treatment has no effect on the E-modulus of the scaffolds." At 48-hour timescales the
attack is surface erosion and the bulk survives. But 0.2 mol/L NaOH is pH 13.3 and biocast runs four
pH units milder, so this study does not show that a PLA mould fails in one cure. What condemns PLA is
[Laycock et al. 2017](https://doi.org/10.1016/j.progpolymsci.2017.02.004): PLA degrades by **bulk**
erosion, so molecular weight and toughness fall before any mass loss or visible change appears. The
tabulated half-life of a polyester bond in neutral water at 25 C is 3.3 years, so 28 days at pH 9 will
not dissolve a PLA mould — it may quietly embrittle it, and the damage is invisible on inspection and
accumulates across reuses.

[Kawahara et al. 2016](https://doi.org/10.2115/fiberst.2016-0005) gives PET the opposite character. Etching PET
fibres in 1.0 to 2.8 M NaOH at 40 to 60 C, the activation energies cluster at 62.4 to 69.5 kJ mol-1 and
the attack is topochemical: the fibre thins from the outside in with a rectilinear diameter-versus-time
relation. Surface-limited damage stays where it can be seen and does not embrittle the section. The
grade mismatch is real and worth naming — the study is oriented semicrystalline PET homopolymer fibre,
while PETG is the glycol-modified amorphous copolymer used for printing, and amorphous regions
hydrolyse faster, so PETG should be somewhat less resistant than these fibres. No PETG-specific
alkaline dataset was retrieved.

So the ranking silicone > PETG > PLA holds, but it is driven by erosion mode — stiffening, then surface,
then bulk — rather than by rate, and it is labelled ASSUMED because no retrieved source compares the
three.

### Mould-rubber mechanicals

These come from named supplier technical data sheets and are labelled `manufacturer_tds`, not MEASURED
in the peer-reviewed sense, and the grade is named in every row. At matched Shore 30A the tin-cure
(condensation) rubber Mold Max 30 is the *stronger* material — 3.98 MPa tensile and 21.9 kN m-1 Die B
tear against 2.90 MPa and 15.4 kN m-1 for platinum-cure Mold Star 30, so +37 % and +42 %. Tin-cure is
not the weaker chemistry mechanically. Its disadvantages are dimensional and temporal: 0.2 % linear
cure shrinkage against under 0.1 % for platinum, because condensation cure evolves a small molecule and
addition cure does not, plus the shorter library life that the supplier's own literature notes for
accelerated tin systems. On a 400 mm object 0.2 % is 0.8 mm, which is the same order as the 2-3 mm
Panot relief tolerance and the measured 2.49 mm/face block relief, so shrinkage is dimensionally
significant at architectural scale. **Platinum cure is the correct choice, on dimensional grounds
rather than on strength.**

Tear, not tensile, is the property that ends a mould's life: a mould fails at a nick propagating from a
thin web or a sharp internal corner during demoulding. That is the same notch-sensitivity argument this
project already applies to the cast object, and the design consequence is to fillet the silicone's
internal corners as well as the object's.

### Demould mechanics

The undercut strain relation is derived rather than retrieved, and the derivation is in the `notes`
because the result is load-bearing. A skin spanning a chord of length `s` that must clear a re-entrant
feature of depth `u` stretches along two legs of run `s/2` and rise `u`, giving
`eps = sqrt(1 + (2u/s)^2) - 1`. It depends on the aspect ratio alone and carries no absolute length
scale, which is why a small deep undercut and a large deep one demand identical strain. Two
idealisations make it a *lower* bound: strain is taken as uniform along the chord where a real skin
localises it at the lip, and the two-straight-leg path is the shortest clearing path where a real skin
bends around a radius.

At a safety factor of 4 on elongation at break — four separate degradations justify it, since the
tabulated figure is a single monotonic pull on virgin dry rubber at 23 C, and the service condition is
repeated pulls at uncontrolled rate on a nicked surface after alkaline exposure that the aging data show
*reduces* elongation — the allowable strain on a Mold Star 30 skin is 84.8 % and the maximum undercut
aspect ratio is `u/s = 0.78`. For tin-cure Mold Max 30 it is 0.72. **Undercuts are not the binding limit
for an elastomeric skin**: `u/s = 0.5`, a deep undercut by any rigid-mould standard, needs only 41.4 %
strain, half the allowable. Softer grades go further — an Ecoflex-class rubber at 900 % elongation
permits `u/s` up to 1.55, and the 250-300 % elongation often assumed for "silicone" turns out to be a
mould-grade property rather than a silicone property.

A rigid backing jacket is mandatory and the two components do different jobs, which should not be
conflated. The skin tolerates undercuts; the jacket holds the skin to shape against mix pressure. At
1.1 MPa the skin is roughly four orders of magnitude less stiff than a plywood or GRP jacket, so an
unbacked skin bulges and the cast section thickens uncontrollably — and section thickness is the very
quantity the drying ceiling constrains. Disassembly order is jacket off the skin first, then peel the
skin off the cast; peeling with the jacket still on forces the skin to stretch against a rigid
constraint, which is how skins tear at their thin sections.

## What had to be assumed

Six rows are ASSUMED and three of them are consequential. The 6 mm skin thickness is an engineering
choice with no source; every transport number in the file scales off it, linearly for oxygen resistance
and inversely for vapour transmission, so a redesigned skin invalidates those rows arithmetically
rather than conceptually. The safety factor of 4 on elongation is judgement, argued from four
independent reasons the tabulated elongation overstates service capacity, and it sets `u/s_max`
directly. The material ranking is an ordinal judgement assembled from three sourced rows.

## The three weakest rows

1. `WVTR_tds` — the figure of 2000 g m-2 day-1 at 50 um has **no retrieved primary source and no stated
   test temperature or humidity gradient**, which is precisely the defect this file exists to prevent.
   It is retained only because it cross-checks the Blume et al.-derived route to 21 %, and it is flagged in the
   file as not for use in a model. The row that should be used is `WVTR_cure`.
2. `sil_28d_verdict` — the expected condition of a skin after one cure rests on an Arrhenius
   extrapolation across 10 K and 4.5 pH units from the only available dataset, with no measurement at
   the target conditions and no basis whatever for cumulative damage across a mould library.
3. `E(SA)_exp` — the base-e versus base-ten provenance ambiguity means the same published constants
   yield 2.159 or 5.884 MPa at Shore 60A. Usable below about 40A, where it agrees with Gent; not above.

A fourth caveat is not a row but a scope limit. The 21 % agreement between the two independent WVTR
routes is reassuring about the *arithmetic* and says nothing about whether Fickian
thickness-limited transport is the right model for a filled mould rubber with a real surface, which is
the assumption both routes share.

## The one measurement that would most improve this set

Shore A and Die B tear strength on the actual mould rubber after 28 days' immersion in the actual
cementation solution at 30 C, repeated after 1, 5 and 10 simulated cure cycles. It needs a durometer, a
tear die and a beaker. It would replace the weakest row in the file with a measurement at the real
conditions, and it addresses the one thing the entire alkaline-stability section cannot currently speak
to, which is reuse.

## What this file does not cover

Jacket sizing is already handled by `mix_pressure` and `wall_deflection` in `mould.py` and is not
re-derived here. Release agents and platinum-cure surface inhibition are not addressed. Nothing was
retrieved on whether contact with silicone affects the cementation solution or the *B. subtilis*
culture itself; that is a wet-lab question and out of scope for a geometry and transport parameter set.
