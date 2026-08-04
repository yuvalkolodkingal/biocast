# MICP transport and kinetics parameters for B. subtilis bio-cementation

Oxygen reaches a bio-cemented casting through its air-filled pores, not through its pore water, and
that single fact sets the geometry. Dissolved-O2 diffusivity in water is 2.0e-9 m2 s-1
([Han & Bartels 1996](https://doi.org/10.1021/jp952903y)) against 2.2e-5 m2 s-1 for O2 in the gas
phase ([NIST Technical Note 2279](https://doi.org/10.6028/nist.tn.2279)), four orders of magnitude, and pore air
at 30 C carries 8.42 mol m-3 of O2 against 0.240 mol m-3 in air-saturated water
([Bok et al. 2023](https://doi.org/10.3389/fnuen.2023.1158109)). Feeding the colony by dissolved
oxygen alone supports an aerobic layer 0.13–1.6 mm thick; feeding it through connected air-filled
pores supports 12–513 mm. The design brief's observation that solid early prototypes failed for lack
of oxygen while the split-mould hollow-core geometry succeeded is the physical signature of that
gap, and the constraint checker should encode it as a limit on wall thickness at a given pore
saturation rather than as a generic "keep it thin" rule.

The full table is in [micp_kinetics_params.json](micp_kinetics_params.json): 43 rows, 20 MEASURED,
11 SECONDARY, 9 DERIVED with the arithmetic written out in each `notes` field, and 3 ASSUMED. Ten rows
are for *B. subtilis* itself; fifteen carry an explicit organism-mismatch flag. Every row also carries a
`retrieval_level` recording how much of its source was actually opened, and no row is labelled MEASURED
unless that source's full text was read: see *Provenance limits* below.

## The two numbers that matter most

### Effective oxygen diffusivity

Three tortuosity models were retrieved rather than one, and they disagree by a factor of 2.1 at the
same porosity. That spread is the honest uncertainty and should be sampled, not averaged. At
phi = 0.40 (the measured column porosity in
[Ebigbo et al. 2012](https://doi.org/10.1029/2011wr011714)) and full saturation, Millington-Quirk
gives D_eff/D_0 = 0.40^(10/3)/0.40^2 = 0.295, the Archie form with cementation exponent 1.5
([Hamamoto et al. 2010](https://doi.org/10.1029/2009wr008424)) gives 0.253, and the Boudreau
sediment relation tau^2 = 1 - ln(phi^2) ([Boudreau 1996](https://doi.org/10.1016/0016-7037%2896%2900158-5))
gives 0.141. Multiplying through by D_O2_water(25 C) yields 2.8e-10 to 5.9e-10 m2 s-1.

The exponent itself is contested. Millington-Quirk's 10/3
([Millington & Quirk 1961](https://doi.org/10.1039/tf9615701200)) remains the default, but
[Ghanbarian & Hunt 2014](https://doi.org/10.1002/2013wr014790) tested 71 experiments and 632 data
points and found gas diffusion follows a power law in air-filled porosity with exponent 2.0 in 66 of
them: Millington-Quirk over-penalises partial saturation. The JSON carries `low = 2.0` on that row
so the Monte Carlo can straddle both.

In the gas phase the same correction applied to air-filled porosity eps = phi(1-Sw) gives
2.0e-6 m2 s-1 at Sw = 0.3 falling to 3.0e-8 m2 s-1 at Sw = 0.8. Even the driest-pore case at
Sw = 0.8 still beats the saturated dissolved-phase path by roughly a hundredfold, which is why
saturation, and therefore the drying schedule, is the governing state variable.

### Volumetric oxygen consumption rate

This is built from a *B. subtilis*-specific respiration measurement rather than transferred from
another organism. [Hu et al. 2017](https://doi.org/10.1186/s12934-017-0764-z) report specific O2
uptake of 4.56 mmol gDCW-1 h-1 at dissolved oxygen below 10% air saturation and
10.02 mmol gDCW-1 h-1 at 35% saturation, both at 37 C on glucose with respiratory quotient near
unity. Multiplying by a dry biofilm density of 10–12 kg m-3 — fitted at 10 g L-1 for MICP sand
columns by Ebigbo et al. and measured at 12 mg cm-3 for a laboratory biofilm by
[Stewart 2003](https://doi.org/10.1128/jb.185.5.1485-1491.2003) — gives 1.27e-2 to 3.34e-2
mol m-3 s-1 *within* the biofilm. Scaling by the biofilm's share of pore space gives the bulk sink
term of 1.3e-4 to 3.3e-3 mol m-3 s-1 that the reaction-diffusion model consumes.

Two independent checks bracket that result. Stewart's own worked example, using
mu = 0.80 h-1 for *P. aeruginosa*, yields 9.7e-2 mol m-3 s-1, 2.9x above our high bound, as
expected, since Hu et al. measured mu = 0.035–0.092 h-1 for *B. subtilis* in the same experiment
that produced the uptake rates. Coming from an entirely different direction,
[Çelik & Çalık 2004](https://doi.org/10.1021/bp0342351) measured volumetric O2 uptake of
0.001–0.003 mol m-3 s-1 in a *Bacillus* bioreactor, a window our point estimate and high bound sit
inside. The agreement is encouraging but not a validation of the weakest input: a stirred bioreactor
is far denser in biomass than a sand pack, so the match partly reflects the assumed biofilm volume
fraction compensating.

## Penetration depths: what the literature actually observes

Measured cementation depths are consistently decimetres or less, and the mechanism is
near-injection clogging rather than reagent exhaustion.
[Cheng & Cord-Ruwisch 2014](https://doi.org/10.1080/01490451.2013.836579) treated 2 m columns by
surface percolation and found that in fine sand (<0.3 mm) repeated treatments clogged the injection
end and limited cementation depth to under 1 m, while three-dimensional fine-sand trials cemented
80% of the material to 2–2.5 MPa only to a depth of 20 cm. In coarse sand (>0.5 mm) the clogging did
not occur and strength of 850–2067 kPa was achieved along the entire 2 m. Grain size is a
first-order lever, and the brief's crushed waste with d_max up to 4 mm sits favourably in the coarse
regime. At the shallow end, single sprayed applications produce a crust of a few millimetres to
about 25 mm ([Lai et al. 2023](https://doi.org/10.3390/ma16186211);
[Fu et al. 2023](https://doi.org/10.1016/j.bgtech.2023.100002)), and even with active pumping the
uniformly-reinforced dimension stays at 0.09–0.15 m radius
([Wang et al. 2026](https://doi.org/10.1371/journal.pone.0349797)).

Two findings from this literature invert the intuition that more bacteria and faster reaction are
better. Lai et al. measured *deeper* effective reinforcement at *lower* cell density, because the
bioflocculation lag stretches from 45–55 min at 1e9 cells mL-1 to 540 min at 6.7e7 cells mL-1,
letting solution infiltrate before it gels. And
[Konstantinou et al. 2021](https://doi.org/10.1038/s41598-021-85712-6) found that high urease
activity left CaCO3 "high at the top of the specimens and low or nil at the bottom" across a
specimen only 150 mm tall, concluding that activity below 10 mmol L-1 h-1 is what delivers uniform
cementation. The entire recent uniformity literature, from low-pH injection to low-temperature dual
inhibition ([Huang et al. 2025](https://doi.org/10.3390/ma18112514)), is engineering deliberate
slowness. **Our organism being a slow carbonate producer is an asset for uniformity, not only a
handicap for strength.**

## Organism mismatch, and one case where B. subtilis wins

Most quantitative MICP work uses *Sporosarcina pasteurii*, and its values must not be transferred
silently. The mismatch cuts in a direction worth naming: Fu et al. record that S. pasteurii urease
can stay active under anoxia because the enzyme was synthesised earlier during aerobic culture, so a
ureolytic system is partly buffered against oxygen loss in a way ours is not. In the non-ureolytic
route the aerobic oxidation *is* the carbonate-generating step: formate or acetate is oxidised to
CO2, which hydrates to bicarbonate and then carbonate, so no oxygen means no carbonate, full stop.
[Li et al. 2017](https://doi.org/10.1080/01490451.2017.1303553) measured UCS differing by a factor of
100 between aerated and air-restricted MICP with S. pasteurii; for our pathway that transfer is
conservative.

One premise in the design brief needs correcting. *B. subtilis* is not literally an obligate aerobe:
[Nakano & Zuber 1998](https://doi.org/10.1146/annurev.micro.52.1.165) review its anaerobic growth on
nitrate or nitrite as terminal electron acceptor, and by fermentation, via the ResD/ResE–FNR
regulatory pathway. The aerobic geometric constraint nonetheless stands, on stoichiometric rather
than viability grounds.

Against that, the one head-to-head comparison retrieved has *B. subtilis* outperforming the ureolytic
control. [Hemayati et al. 2023](https://doi.org/10.1038/s41598-023-33070-w) optimised non-ureolytic
MICP with *B. subtilis* on calcium formate and calcium acetate and compared it with an S. pasteurii
control at matched application rate: B. subtilis with calcium formate gave 190 kPa surface resistance
and 2.8% CaCO3 against the control's 132 kPa and 2.41%, with no ammonia by-product. Their optimum was
50 g L-1 calcium source (0.384 M as formate, 0.316 M as acetate, the same window the ureolytic
literature converged on), OD 1 with no benefit beyond, calcium-to-bacteria volume ratio 1, initial
pH 9, and 9 days curing at 30 C. Formate reached 87% of its final yield within 3 days where acetate
reached only 45%. The caveats are that this is a thin sprayed dune crust rather than a cast monolith,
its resistances are two orders below multi-cycle ureolytic column protocols, and the polymorph was
vaterite rather than calcite, whose metastability the paper does not address.

## Protocol numbers and yield

Cycle counts in the retrieved protocols span 4 to 48 and are a function of bacterial retention rather
than a material constant: [Lambert & Randall 2019](https://doi.org/10.1016/j.watres.2019.05.069)
needed 48 cycles over 4 days for 2.7 MPa in a full-size brick mould, Konstantinou et al. used 15
injections at 24 h intervals, Huang et al. reached 2.5 MPa in four cycles with dual inhibition, and
pre-trapping bacteria on the sand with an aminosilane cut seven injections to three
([Ugur et al. 2024](https://doi.org/10.1021/acsami.3c13971)). Longer intervals consistently improve
chemical efficiency, crystal quality and uniformity; the 24 h figure is the conservative choice.

No paper reports CaCO3 yield per cycle directly, so it is derived two ways in the JSON. From Lambert
& Randall's brick geometry (1.718 L mould, 650 mL pore volume implying porosity 0.378, 2.830 kg sand
at an assumed 2650 kg m-3 grain density), influent calcium of 0.09–0.13 M at their reported ~98%
calcium efficiency gives 5.74–8.29 g CaCO3 per cycle, or 0.20–0.29% by mass, integrating to
9.7–14.1% over 48 cycles. From Konstantinou et al.'s specimen at ~50% chemical efficiency the figure
is 0.35% per cycle. Against Fu et al.'s threshold that a specimen needs about 3% CaCO3 to stand alone
unconfined (and under 2% suffices when confined), 0.2–0.7% per cycle puts the self-supporting point at
roughly 4–15 cycles.

## Weakest links, in order

1. **Biofilm volume fraction of pore space (ASSUMED, 1–10%).** No retrieved source measures it for an inoculated sand pack. It multiplies the volumetric O2 sink linearly and by itself spans a decade, dominating the uncertainty on the most important derived number in the set, ahead of the measured respiration rate.
2. **Curing relative humidity (ASSUMED, 90–100%, no DOI).** The largest outright gap. No measured optimum exists for biocemented granular composites, and the parameter is not free to maximise: high RH keeps cells hydrated but keeps saturation high, which is exactly what collapses gas-phase oxygen supply. RH and oxygenation must be co-optimised, not set independently.
3. **Critical drying rate for crack initiation.** Does not appear to exist in the literature for a biocemented granular composite. The closest anchor is the transport physics: [Shokri & Or 2011](https://doi.org/10.1029/2010wr010284) find the onset of vapour-diffusion-limited stage-2 evaporation falls in a narrow band of 0.5–2.5 mm d-1, beyond which the vaporisation plane jumps below the surface and drying becomes depth-nonuniform. Holding imposed drying at or below that band is an inference about the brief's "uneven drying" failure, not a measured crack criterion.
4. **Archie saturation exponent (ASSUMED, 2.0).** Controls how fast oxygen supply collapses with rising saturation; the functional form is sourced but the exponent is conventional.
5. **Penetration depth for our organism.** Every measured depth is ureolytic. The clogging-limited depths from S. pasteurii probably do not bind for slow non-ureolytic B. subtilis, since the oxygen-limited depth should bind first, but this is reasoning, not measurement.
6. **Porosity of the actual substrate.** Ground construction waste packing porosity is unmeasured; retrieved values span 0.378 to 0.69. It enters twice, through the tortuosity denominator and through the aggregate mass that fixes CaCO3 percentage.

## Provenance limits

Eleven of the 43 rows cite a paper whose full text could not be retrieved: the publisher returned no
open-access copy via Unpaywall, Semantic Scholar, PMC or CrossRef text-mining links. Those rows are
labelled SECONDARY rather than MEASURED. No numeric value changed when they were relabelled; the
values are unaffected and, where possible, corroborated from a source that was fully retrieved. The
distinction matters because a reader deciding how hard to lean on a given number needs to know whether
it was read from the paper or from its abstract.

Nine of the eleven rest on a publisher or PubMed abstract that states the quoted figure verbatim:
D_O2_water, m_Archie, L_cem_fine, L_cem_coarse, R_grout, E_crit, anaerobic_capability and
k_O2_penalty. Several have independent support from papers that were opened. The Han & Bartels
interpolation formula for D_O2_water is corroborated by Stewart 2003, whose Table 1 gives
20.0e-6 cm2 s-1 at 25 C against the formula's 1.998e-5 (0.1% agreement) and quotes 2.68e-5 at 37 C
against 2.624e-5 (2%). The Cheng & Cord-Ruwisch depth limits and the Li et al. oxygen penalty are
both independently summarised in [Fu et al. 2023](https://doi.org/10.1016/j.bgtech.2023.100002), which
was fully retrieved.

The remaining two are weaker still, resting on CrossRef metadata alone with no abstract available: the
Millington-Quirk exponents and the Boudreau tortuosity relation. Both are standard, textbook-level
forms reproduced from domain knowledge rather than from any text retrieved here, and the
Millington-Quirk form is at least described secondhand in two sources that were retrieved. The
Boudreau relation is the single weakest-provenance row in the file. Its numerical influence is
bounded and known: it is the most restrictive of the three tortuosity models and therefore sets the
low end of D_O2_eff_sat. Anyone for whom that lower bound is decision-relevant should obtain the paper
before relying on it.

One transcription caveat is worth recording. The inoculum densities in Ebigbo et al.'s Table 1 were
read from PDF text extraction that rendered decimal points as colons, so the entries appear as
"14:0 x 10^9" and similar. These are seven-field rows whose leading digit is the column index, not
part of the mantissa: stripping it recovers column numbers 1, 2, 3, 4 in sequence and yields
1.3e7 to 4.0e9 cfu mL-1. Two checks confirm the alignment. The recovered indices run exactly 1 to 4 in
order, which no alternative parse reproduces, and column 1 is the only row with "no" in the
rinse-influent field while also carrying the most calcium pulses, matching the paper's account of
clogging at the injection end of column 1 and the modified rinse-first strategy in columns 2 to 4. The
JSON records the rejected alternative parse alongside the accepted one. That range is contextual only:
the N_cells_inoc row's own value and bounds come from Lai et al. 2023, whose full text was retrieved.

## Why oxygen and not carbon or calcium

For the aerobic non-ureolytic pathway, oxygen is the limiting reactant by a wide margin. Acetate
diffuses at 1.21e-9 m2 s-1 against dissolved O2 at 2.0e-9 (Stewart 2003, Table 1), only 60% slower,
but it is supplied at roughly 0.32 M against oxygen's 0.24 mM, a molar supply ratio near 1300:1.
Neither the carbon source nor the calcium can plausibly become limiting before oxygen does, which
justifies building the success estimator around an oxygen transport model and treating nutrient
delivery as a secondary constraint.
