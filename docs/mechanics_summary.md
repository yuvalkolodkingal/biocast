# Strength, failure mechanics and standards for B. subtilis MICP cast objects

The two numbers the constraint checker leans on hardest pull in opposite directions from the team's
notes. The stress-concentration rule is sound: the preference for fillets over chamfers is exactly
right, and the Inglis equivalent-ellipse solution ([Gebrehiwot et al. 2023](https://doi.org/10.23998/rm.124815))
gives it a closed form, `Kt = 1 + 2*sqrt(h/r)`, that a checker can evaluate directly. The groove-width
rule is not sound: the granular bridging literature puts the jamming threshold at six to eight particle
diameters of aperture, two to three times the team's stated `w >= 2-3 d_max`, whose lower bound sits
precisely at the ratio below which a constricted suspension is reported to clog every time. Everything
else in this document is calibration around those two findings, plus one structural caveat that colours
all of it: every strength number available is for the wrong organism.

## Material strength

### Organism mismatch

No paper reporting unconfined compressive strength against CaCO3 content for *Bacillus subtilis*-cemented
granular material was found. The MICP strength literature is built almost entirely on *Sporosarcina
pasteurii*, chosen precisely because of its "pronounced urease enzyme production aptitude", and the values below inherit that.
What can be said about *B. subtilis* is mechanistic rather than mechanical. Hemayati and colleagues ran non-ureolytic heterotrophic precipitation with *B. subtilis* on
calcium formate and calcium acetate and found it produced more carbonate than *B. amyloliquefaciens*,
with formate outperforming acetate and the formate/*B. subtilis* pairing delivering 87% of its total
carbonate within three days ([Hemayati et al. 2023](https://doi.org/10.1038/s41598-023-33070-w)). The same
paper writes the pathway out explicitly as an oxidative one — formate and acetate are consumed with O2 via
formate dehydrogenase to yield CO2. That is the mechanistic backing for the source paper's aerobic
constraint and for treating surface-area-to-volume as a first-order design variable rather than an
aesthetic one. It is not backing for a strength value.

The recommendation for the estimator is therefore to sample UCS from the *S. pasteurii* distribution and
then apply an explicit, clearly-labelled derating factor (a uniform 0.3 to 0.7 is a defensible guess for a
slower, non-ureolytic pathway) and to mark that factor ASSUMED in every output. A silent transfer would
turn a 4x-uncertain guess into a false prediction. The indirect *B. subtilis* numbers that do exist are
additive-to-cement percentages: [Zhang et al. 2023](https://doi.org/10.1007/s12665-023-10899-y) tabulate
compressive strength gains of +25.8%, +19.26% and +15% for *B. subtilis* in cement mortar from three
separate studies, which quantify a supplement to Portland cement and cannot be converted into a standalone
bio-cement strength.

### UCS versus CaCO3 content

Eleven data points were extracted across seven studies. Fitting them pooled produces power-law,
exponential and linear forms whose R-squared values are 0.008, -0.045 and 0.004, two of them negative
in linear space, meaning every functional form performs worse than predicting the mean. That result is
not a tuning problem to be optimised away. The comprehensive review by
[Fu, Saracho & Haigh 2023](https://doi.org/10.1016/j.bgtech.2023.100002) reaches it from a far larger
dataset, reporting that published correlations are variously linear, polynomial and exponential and that
"the massive data discrepancy in Fig. 2a shows how difficult and unreliable it is to describe the relation with a single trendline". Their pooled envelope spans CaCO3 from below 1% to over 35% and UCS from below 50 kPa to over 18 MPa.

The cleanest demonstration of why comes from a single pair of specimens.
[Almajed et al. 2019](https://doi.org/10.1038/s41598-018-38361-1) measured 0.12-0.16 MPa and 1.65-1.82 MPa
at the *same* carbonate content below 1.4%, a twelve-fold strength spread, differing only in whether
non-fat milk powder had localised the precipitate at interparticle contacts. Carbonate content does not
determine strength; carbonate *placement* does. The corollary appears in
[Terzis & Laloui 2018](https://doi.org/10.1038/s41598-018-19895-w), where medium-grained sand reached
3-12 MPa over calcite contents of 5-10% while fine sand plateaued near 2.5 MPa across the same band —
because in coarse voids the crystals precipitate away from grain contacts where they would do structural work.

What the checker should use instead of a pooled fit is a substrate-conditioned draw with a hard gate.
The gate is that a specimen needs "a minimum Ccc of circa 3%" to stand up without
confinement at all ([Fu et al. 2023](https://doi.org/10.1016/j.bgtech.2023.100002)); below that, "complete
solidification" in the source paper's own sense has not happened. Above it, the substrate class sets the
range, and for this project that range is uncomfortable. The one study on the actual substrate, washed
recycled sand from demolition waste, reached a maximum of 724 kPa at optimum conditions of 0.5 mol/L
cementation media, 30 °C and twelve treatment cycles, with 490 ± 149 kPa at 1 mol/L
([Fouladi et al. 2024](https://doi.org/10.1007/s11440-024-02396-8)). That is roughly four to fifteen times
below the clean-sand values, and it was achieved with *S. pasteurii*, so the *B. subtilis* penalty stacks
on top. For within-study interpolation over clean medium sand, the Terzis endpoints solve exactly to
`UCS = 0.12 * C^2.00`, quadratic in carbonate content, valid only over 5-10%, from two reported values so
no R-squared exists, and the steepest and hence most optimistic relation retrieved.

Set against ordinary construction materials, the picture is sobering rather than fatal. A cast MICP
bio-brick reached 2.7 MPa after 48 treatment cycles over four days, which the authors note falls slightly
below their national minimum for non-facing brick and well below conventional face brick at 9-12.5 MPa
([Lambert & Randall 2019](https://doi.org/10.1016/j.watres.2019.05.069)). ASTM C90 requires 13.8 MPa net-area
strength for a loadbearing masonry unit. [Beatty, Williams & Srubar 2022](https://doi.org/10.1146/annurev-matsci-081720-105303)
put biomineralized building materials "on the order of 5 MPa" against Portland-cement
concrete "on the order of 50 MPa", with MICP-stabilized soils reaching up to 14 MPa
in the best cases. A bio-cemented waste-aggregate object is a non-structural or lightly-loaded element;
the geometry generator should not be permitted to produce something whose service loads assume otherwise.

### Tensile capacity and brittleness

For a brittle cast object the compressive number is rarely what kills it. Direct-tension tests on
heavily-cemented MICP sands gave 210-710 kPa across carbonate contents of 3.8-14.4%, and the
tensile-to-compressive ratio "ranged from 0.19 to 0.25 depending upon the particle size"
([Nafisi et al. 2020](https://doi.org/10.1139/cgj-2019-0230)) — a brittleness ratio of 4.0 to 5.3 by
reciprocal. That ratio looks generous next to concrete's 0.07-0.11, but only because MICP's compressive
strength is low, not because the material is tough. The strain data make the point properly: tensile
failure arrived at 0.02-0.04% strain against 0.21-0.36% in compression, so "the ratio of
the tensile strain at failure to that of unconfined compressive strength is about 0.1". Tension
reaches its limit an order of magnitude earlier in strain than compression does. Surface-energy
measurements in the same paper (work of cohesion 82.8 mJ/m² for CaCO3, work of adhesion 84.2 mJ/m² at the
CaCO3-silica interface) indicate that "cohesive failure within calcium carbonate bonds is
more likely to occur than adhesive failure at particle contacts". Cracks run through the cement, not along the grain, so bond volume at contacts sets tensile capacity.

Because no flexural data exists for MICP material, weak lime mortar serves as the proxy for a
compressive-to-flexural ratio. Dividing compressive by flexural strength mix-by-mix and age-by-age through
Table 3 of [Costigan et al. 2015](https://doi.org/10.1016/j.jobe.2015.10.001) gives nineteen pairs spanning
2.80 to 8.90 with a median of 4.59 — for instance CL90 at one year, 1.4/0.5 = 2.80; NHL5 at one year,
13.3/2.9 = 4.59; NHL3.5 at six months, 8.9/1.0 = 8.90. Flexural strength overestimates true tensile
strength, so the real compressive-to-tensile ratio sits above these figures. The proxy brackets the MICP
measurement rather than replacing it.

## Failure mechanics

### Stress concentration and notch sensitivity

Two families of stress-concentration data were retrieved, and they are not interchangeable — the
non-dimensional group means something different in each, which is the single most common way a Kt curve
gets misapplied.

For grooves, flutes and the panot relief, a cut of depth `h` with root radius `r` into a section, the
governing form is the Inglis equivalent-ellipse solution, `Kt = 1 + 2*sqrt(h/r)`
([Gebrehiwot et al. 2023](https://doi.org/10.23998/rm.124815), Eq. 5). It reduces to the classical `Kt = 3`
for a semicircular notch and diverges as the root sharpens: `r/h = 0.5` gives 3.83, `r/h = 0.2` gives 5.47,
`r/h = 0.05` gives 9.94, `r/h = 0.01` gives 21.0. The divergence is the formal content of the team's claim
that any angle, even a chamfer's obtuse one, is a weak point, and it is worse than a design choice: a square-cut root has its radius set by mould resolution and particle size rather than by the
designer, so Kt becomes a manufacturing outcome. The same paper is candid that Inglis is
"applicable for shallow profiles that can be approximated to an elliptical geometry"
and that when benchmarked against finite-element analysis for rough-surface valleys, the Inglis and Neuber
forms overestimated Kt by more than 30%. For a brittle cast object that is the correct direction to err.

For a filleted shoulder or change of section, chart values read off Peterson (Pilkey 1997) for axial
tension with the step height equal to the fillet radius sit in a narrow band of 1.79-2.02 across `r/d` from
0.005 to 0.25, with independent finite-element values for the flat-plate equivalent spanning 1.29-2.56
over the same geometry set ([Prajapati & Patel 2023](https://doi.org/10.17576/jkukm-2023-35%281%29-14)).
Note the normaliser: `r/d` against the smaller shaft diameter, not the notch depth. This family is valid
for a small radius on a large section and is the wrong tool for a deep groove in a thin tile; use it as a
sanity floor for section changes and use Inglis for everything cut into a surface.

Kt alone understates the danger, though, because it is an elastic stress ratio and what matters is failure
probability. The flaw-based argument supplies the conversion. Weibull size-effect fits on foamed concrete
give moduli of 11.5-16.8 with R-squared of 0.96-0.999, decreasing as density decreases because
"foamed concrete with a high porosity is brittle", and falling below the 14-34 reported
for normal concrete for want of stiff aggregates to stabilise crack propagation
([Jiang et al. 2024](https://doi.org/10.1016/j.matdes.2024.112841)). Under weakest-link statistics with a
finite modulus `m`, raising local stress by `Kt` raises failure probability by roughly `Kt^m`: at `m = 12` a modest `Kt` of 1.5 costs a factor of about 130 in survival probability. [Bažant 2019](https://doi.org/10.1098/rspa.2018.0617)
adds the structural-size dimension, showing that quasibrittle strength distributions transit from Weibullian
to Gaussian as size grows because the weakest-link chain has a finite number of links, one per representative
volume element — so a small object made of a coarse composite has few links, scatters widely, and is governed
by whichever single flaw is worst. A groove root is a manufactured flaw of known location. This is why the
checker should treat Kt as a risk multiplier through `Kt^m`, not as a stress correction.

### Drying shrinkage and cracking

The shrinkage magnitudes relevant here come from an aggregate-free printable mortar, which is the closest
available analogue to a fines-rich bio-cemented cast: total 28-day shrinkage of -5269 µm/m with standard
deviation below 327, falling to -4797 and -3976 µm/m with 2% and 4% shrinkage-reducing admixture
([Federowicz et al. 2020](https://doi.org/10.3390/ma13112590)). For context, the same paper's literature
summary puts autogenous shrinkage at 330-850 µm/m for w/c 0.2-0.4, notes that ASTM C157 total shrinkage of
high-performance concrete rarely exceeds 800 µm/m, and gives roughly 1200 µm/m for aggregate-free mortars
long-term. The absence of coarse aggregate to restrain the paste is what drives the high figures.

The intervention that works is not chemical. Foil isolation reduced maximum deformation to -1031 µm/m,
"only 20% of the deformations of the base mix", against the 23% reduction the best
admixture dose achieved — a five-fold benefit from suppressing drying versus a fifth from additives. That
result directly corroborates the source paper's split-mould-with-humidity-control strategy and its Fig. 5
observation of cracking from uneven drying. It also flags a standards problem: ASTM C90 caps linear drying
shrinkage of a masonry unit at 0.065%, i.e. 650 µm/m. The uncured mortar figure is eight times that limit, and even sealed-cured it exceeds it by 1.6 times. A fines-rich bio-cemented cast will not meet the masonry-unit
shrinkage specification without both humidity control and real coarse aggregate.

For the cracking criterion itself, the Japan Concrete Institute state-of-the-art report supplies a usable
threshold, citing temperature-stress-testing-machine work by van Breugel and Lokhorst: the tensile stress
at crack initiation was "approximately 75% of the splitting tensile strength", and the report notes this
held across both cement types and four mixture proportions irrespective of the degree of hydration
([Mihashi & Leite 2004](https://doi.org/10.3151/jact.2.141)). The same
report gives uniaxial tensile strength as about 88% of splitting tensile strength and, for its safety
formulation, assumes coefficients of variation of 10% on shrinkage stress and 8% on splitting tensile
strength — which is exactly the spread the Monte Carlo needs. The checker rule is therefore: crack if
restrained stress reaches 0.75 f_ct,spl, with those CoVs propagated.

Differential drying across the wall thickness is where the evidence runs out numerically. The JCI report's
analysis chain is explicitly sectional (unrestrained shrinkage strain distributions inside the member
section, then restrained shrinkage and stress distributions, then crack analysis) and it notes that faster
thermal loading can produce a gradient through the cross-section that must be considered. But no citable
numeric thickness or gradient threshold for a MICP or bio-cemented composite was found. That row is marked
ASSUMED in the parameter table with its engineering basis stated: self-restraint stress scales with the
surface-to-core moisture difference, which grows with both section thickness and drying rate, so the
defensible rule is to cap thickness and slow the drying rather than to trust a threshold. The source paper's
own Fig. 5 and Fig. 6 are the qualitative evidence that this mechanism governs the failures actually observed.

## Standards and geometry rules

### Masonry and tile standards

| Quantity | Team's note | Standard | Verdict |
|---|---|---|---|
| CMU face-shell thickness | ~3.2 cm | 1-1/4 in (32 mm) for 8 in nominal and wider; 1 in (25 mm) at 6 in; 3/4 in (19 mm) at 3-4 in — ASTM C90 Table 1 | **Agree**, exact match to the 8-in-and-wider row |
| CMU web thickness | ~2.5 cm | 3/4 in (19.1 mm) for all unit widths since ASTM C90-11b | **Disagree**, the team's figure matches the superseded pre-2011 requirement; theirs is conservative, but do not cite C90 for it |
| Web area | not stated | Minimum normalized web area 6.5 in²/ft² (45,140 mm²/m²), replacing equivalent web thickness | **Not in notes**, and this is the better constraint form: an area budget lets the generator trade web count against thickness |
| Block module | 20×20×40 cm | ASTM nominal is 8×8×16 in (203×203×406 mm); 190×190×390 mm is the metric block's actual jointed size | **Both exist**, 20×20×40 is the metric nominal, not the ASTM one |
| Void fraction | 40-50% | Not a tabulated C90 limit; C90 controls geometry via thickness minima and web area. Only related tabulated figure is the solid-unit definition, net area ≥75% of gross | **Not found as a standard**, mark ASSUMED and derive achievable void from the thickness rules |
| Panot plan size and thickness | 20×20 cm, 4 cm | Barcelona's Plec Tècnic makes the 4 cm panot the default footway paving on a 15 cm concrete base, with 8 cm only at vehicle crossings; manufacturers list 20×20×4 as the most common piece | **Agree** |
| Panot relief depth | 2-3 mm, <10% of thickness | Orden VIV/561/2010 Art. 45 caps indicator-paving grooves at 5 mm depth and warning studs at 4 mm height | **Consistent and conservative**, 2-3 mm sits inside the 5 mm regulated ceiling, and 5/40 = 12.5% bounds the team's <10% claim |
| Panot channel width | ~1 cm | **Not found**, the Plec specifies bedding, base and a 3 mm joint but not relief geometry; VIV/561/2010 regulates depth and stud height, deferring width to UNE 127029; EN 1339 explicitly excludes tactility | **Not found**, retained as ASSUMED; 10 mm is plausible and satisfies the bridging criterion for d_max up to 3.3 mm |
| Relief tolerance | not stated | Molded ribs and scores within ±1/16 in (1.6 mm) of both specified dimension and specified placement, CMHA CMU-TEC-001 | **Not in notes**, and it bites: a 2-3 mm designed relief carries ±50-80% tolerance, so anything shallower than ~3 mm is inside manufacturing noise |

Sources for the ASTM C90 table values are the reproductions in STRUCTURE magazine's *Changing Masonry
Standards* and *Just the FAQs* columns (Jason Thompson, NCMA) and CMHA CMU-TEC-001; the Barcelona figures
come from the Ajuntament's *Plec Tècnic de Pavimentació* (2017) and Orden VIV/561/2010 (BOE-A-2010-4057).

### Granular bridging criterion

This is the number the constraint checker depends on most, and it contradicts the team's rule by a factor
of two to three.

The primary measurement is a dry-silo experiment across roughly fifty orifice sizes. Fitting mean avalanche
size to `<s> = A/(Rc - R)^gamma` gave `gamma = 6.9 ± 0.2`, `A = 9900 ± 100` and a critical dimensionless
radius `Rc = 4.94 ± 0.03` for spherical beads, where `R = phi/r` is the orifice radius over the bead radius —
which reads directly as the aperture-to-particle-diameter ratio ([Zuriguel et al. 2005](https://doi.org/10.1103/PhysRevE.71.051303)).
Above `Rc`, flow never stops; below it, arching arrests it. Critically, bead material, density, elasticity
and surface roughness had no measurable effect: the phenomenon is pure geometry, which is what makes it
transferable to crushed waste. Shape, however, does matter and pushes the threshold up: the same paper
reports `Rc = 5.05` for pasta grains and `6.0` for rice, concluding that the nearer to spherical the shape, the lower `Rc` becomes. Crushed construction waste is angular, so the upper end applies. Zuriguel's own review
corroborates the sphere figure independently, noting "a divergence of the avalanche size was
reported for an outlet diameter about 5 times the bead diameter" while recording that the existence of
a true critical size is settled in 3D but contested in 2D ([Zuriguel 2014](https://doi.org/10.4279/pip.060014)).

The wet analogue is closer still to a cementation slurry, and it is less forgiving. Studying non-Brownian
suspensions through printed millifluidic constrictions, [Vani, Escudier & Sauret 2022](https://doi.org/10.1039/D2SM00962E)
report two thresholds: an absolute floor where "a constricted system will eventually clog for
small enough width to particle diameter ratio W/d < 3, even for small" solid fraction; and, for a dense
suspension near maximum packing, a fitted divergence at `Wc/d = 8.1` with `gamma = 7.9`, noting that in that
dense regime "the clogging of the constriction seems to follow the behavior observed in silos
for dry and immersed granular materials". A stiff cast mix *is* a dense suspension, so 8 is the relevant
divergence rather than 3. The 2025 Annual Review confirms the ratio as the governing parameter, calling
"the neck-to-particle size ratio D/d" the control parameter for clog formation, with a higher ratio letting
more particles escape before a clog forms and so lowering clogging probability, and gives
the minimal bridge as `n_min = (D/d)^2 + 1` particles ([Marin & Souzy 2025](https://doi.org/10.1146/annurev-fluid-030124-112742)).

Combining the conservative branch of each (4.94 spherical floor, 6.0 angular correction, 8.1 dense-suspension
divergence, 3.0 always-clogs floor) gives the checker thresholds: **accept at `w >= 6 d_max`, safe at
`w >= 8 d_max`, certain failure below `3 d_max`**. At `d_max = 4 mm` that is 24 mm to accept and 32 mm to be
safe, against certain failure below 12 mm. The team's rule of `2-3 d_max` yields 8-12 mm: their lower bound
sits *at* the always-clogs boundary and their upper bound just above it. Their ~1 cm panot channel is in the
same regime. This needs resolving before mould geometry is committed.

Two things soften the conclusion slightly. ACI 318-19 §26.4.2.1(a)(5) limits nominal maximum aggregate to the
least of one-fifth the narrowest form dimension, one-third the slab depth, and three-quarters the minimum bar
clear spacing — inverting to `t >= 5 d_max` for a wall and `>= 3 d_max` for a slab, and to `s >= (4/3) d_max`
for clear spacing. That spacing multiple of 1.33 is far *less* conservative than the bridging number, but it
describes vibrated concrete with a fluid mortar phase passing reinforcement, which is the wrong physics for a
stiff non-vibrated waste mix; the code's own escape clause, that the limits may be waived where workability
and consolidation permit placement without honeycombs, is precisely the assumption that fails here. And
gradation is not a one-way penalty: [Mahawish, Bouazza & Gates 2018](https://doi.org/10.1007/s11440-017-0604-7)
found a gap-graded distribution improved bio-cemented coarse aggregate to a maximum UCS near 575 kPa at 75%
coarse / 25% fine, and that the peak strength was not associated with maximum carbonate precipitation, because
the fines supply bridging contacts. Paired with the Terzis fine-sand plateau, this means there is an optimum
`d_max` rather than a monotonic one: coarse waste needs fines to build contacts, but too fine and the
carbonate spreads across too many contacts to bond any of them well.

### Basis of the fillet rule

The team's `r >= 1.5 d_max`, target `2.0 d_max`, has no direct literature source and is marked ASSUMED. It is
nonetheless defensible on two retrieved grounds, and they are different grounds from the groove rule — which
is the likely origin of the confusion, since both were apparently set from the same "multiples of `d_max`"
intuition. On stress, the Inglis form makes radius-relative-to-feature-depth the controlling group: at
`r = 2 d_max = 8 mm` on a 2-3 mm panot relief, `r/h > 2.5` and `Kt < 2.3`, whereas a square root at the
aggregate scale (`r ~ 0.5 mm`, `h = 3 mm`) gives `Kt ~ 5.9`. On castability, the radius must exceed the
particle size or aggregate cannot fill the fillet at all. The fillet multiple is a fill-and-stress rule and
is roughly right; the groove-width multiple is a bridging rule and is too small by two to three times.

## Open gaps

Three values could not be sourced and are marked ASSUMED with their basis stated rather than dressed up.
There is no numeric transfer factor from *S. pasteurii* to *B. subtilis* strength, so the derating range is a
labelled guess. There is no citable thickness or gradient threshold for differential drying in a bio-cemented
composite, so the rule is directional. And there is no official specification for panot channel width — the
municipal document controls bedding and joints, the accessibility order controls depth and stud height and
defers the rest to UNE 127029, and EN 1339 excludes tactility from its scope altogether. The most valuable
single experiment the team could run against this table is a UCS-versus-carbonate series on their own waste
aggregate with their own organism, which would collapse the widest uncertainty in the whole set.
