"""Score a single design and print the verdict — the smallest useful example.

    PYTHONPATH=. python examples/score_one_design.py
"""
from biocast.gui import engine as E

# Best shell from the 6912-cell sweep (data/pareto_front.csv), d_max = 2 mm.
r = E.evaluate(
    "shell",
    dict(a=58.2, b=75.0, c=71.7, n=2.1, ovoid=0.40,
         wall=19.3, aperture_r=13.5, fillet_r=6.5),
    dict(d_max=2.0),
    dict(cure_days=21, rh_pct=85, split_mould=True),
    n_mc=400,
)

print(f"score           {r['score']:.3f}  (5-95%: {r['score_lo']:.3f} - {r['score_hi']:.3f})")
print(f"feasible        {r['feasible']}")
print(f"limiting term   {r['dominant_failure_mode']} - {r['failure_mode_text']}")
print(f"cemented frac   {r['cemented_fraction']:.3f}")
print(f"reachable depth {r['penetration_depth_nom_mm']:.1f} mm "
      f"(gas {r['L_gas_nom_mm']:.0f}, drained {r['L_dry_nom_mm']:.1f}) "
      f"limited by {r['penetration_limiter']}")
print(f"section         {r['min_section_measured_mm']:.1f} mm measured "
      f"({r['section_over_dmax']:.1f} x d_max), nominal {r['min_feature_nominal_mm']:.1f} mm")
print()
for key in ("aeration", "drying", "castability", "structural"):
    print(f"  {key:12s} {r['sub_' + key]:.3f}")
if r["failed_rules"]:
    print("\nfailed rules:")
    for v in r["verdicts"]:
        if not v["passed"] and v["severity"] == "fail":
            print(f"  {v['rule']}: {v['message']}")
