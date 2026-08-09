"""Tests for the load-capacity estimate.

Run:  PYTHONPATH=. python -m pytest tests/ -q

Every assertion here is a RANKING or a GATE, never a calibration. There is no
measured capacity for a B. subtilis cast on construction waste to check against —
that absence is the whole reason `physics.strength` reports an interval and a
provenance string instead of a number. So the tests check the things the model is
actually allowed to claim: that a fillet beats a sharp root, that more section
beats less, that the carbonate gate is a hard zero, and that the answer stays far
below the masonry benchmark it is reported against.

The last test is the important one for the rest of the package: capacity is
ADDITIVE OUTPUT. `validate_paper.py` must still print its table byte for byte,
because that table is the model's only retrodiction of a real experiment and a
strength feature has no business moving it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from biocast.params import BlockParams, Design, Mix, Process, ShellParams, TileParams
from biocast.physics import fields as fl
from biocast.physics import strength as stg
from biocast.gui import engine as E

ROOT = Path(__file__).resolve().parents[1]

#: The rim of a shell is a change of section, not a groove, so the grammar gives
#: it no notch depth to measure a root radius against — see `strength.notch_of`.
#: A rim step is a property of the load case, and this is the same 8 mm one
#: `validate_paper.py` case E uses to exercise the notch term on a vessel.
RIM_NOTCH_MM = 8.0


def _diag(geom, pitch=None):
    """Mesh and voxel diagnostics for one geometry, as `engine.evaluate` does."""
    mesh, fld, origin, p = E.build_mesh(geom, pitch)
    return fl.geometric_diagnostics(mesh, p, parting_axis=2, parting_frac=0.5,
                                    field=(fld, origin, p))


def _capacity(geom, diag, **kw):
    design = Design(geom=geom, mix=Mix(), proc=Process(), name="t")
    return stg.load_capacity(design, diag, phys=E.load_mechanics(), **kw)


# ---------------------------------------------------------------- notch effect
def test_filleted_rim_beats_sharp_rim():
    """Same vessel, two rim radii: the fillet must carry more.

    This is the fillet-not-chamfer rule in the team's notes, in capacity terms.
    It arrives entirely through Kt — the two shells have the SAME critical
    section, because the neck is set by the bore and the wall, not by the rim —
    so the test isolates the notch term rather than confounding it with geometry.
    """
    sharp = ShellParams(fillet_r=0.5)
    round_ = ShellParams(fillet_r=8.0)

    c_sharp = _capacity(sharp, _diag(sharp), notch_depth_mm=RIM_NOTCH_MM)
    c_round = _capacity(round_, _diag(round_), notch_depth_mm=RIM_NOTCH_MM)

    assert c_round["kt_used"] < c_sharp["kt_used"]
    assert c_round["capacity_kN"] > c_sharp["capacity_kN"]
    assert c_round["capacity_lo_kN"] > c_sharp["capacity_lo_kN"]
    # and the win is the notch alone, not a different amount of material
    assert c_round["critical_section_mm2"] == pytest.approx(
        c_sharp["critical_section_mm2"], rel=1e-9)


def test_sharp_root_is_not_a_divide_error():
    """A zero root radius gives Kt = inf and capacity 0, not a crash or a NaN.

    That is the honest reading of a square-cut root in a brittle cast, and it is
    the same convention `score.s_structural` uses.
    """
    geom = TileParams()
    cap = _capacity(geom, _diag(geom), notch_depth_mm=3.0, root_radius_mm=0.0)
    assert not np.isfinite(cap["kt_used"])
    assert cap["capacity_kN"] == 0.0
    assert cap["capacity_hi_kN"] == 0.0


# ---------------------------------------------------------------- section rules
@pytest.mark.parametrize("thin,thick", [(24.0, 40.0), (32.0, 34.0)])
def test_thicker_block_section_is_not_smaller(thin, thick):
    """More face shell is more net area, and so more capacity, all else equal.

    Monotone by construction: a thicker face shell shrinks the cores in both
    directions. The test exists because the analytic net area has to MERGE
    overlapping cores at high draft, and an off-by-one there breaks monotonicity
    before it breaks anything visible.
    """
    a = BlockParams(face_shell=thin)
    b = BlockParams(face_shell=thick)
    A_thin = stg.critical_section_mm2(a, _diag(a))
    A_thick = stg.critical_section_mm2(b, _diag(b))
    assert A_thick >= A_thin

    c_thin = _capacity(a, _diag(a))
    c_thick = _capacity(b, _diag(b))
    assert c_thick["capacity_kN"] >= c_thin["capacity_kN"]


def test_block_cores_merge_at_high_draft():
    """Heavy draft merges adjacent cores; the section must not double-count them.

    Summing n * core_w * core_d instead of merging the x-extents understates the
    section by ~20 % on a three-core unit at 5 deg, which is exactly the size of
    error the occupancy grid is there to catch.
    """
    geom = BlockParams(n_cores=3, core_taper=5.0)
    check = stg.section_check(geom, _diag(geom))
    assert check["grid_agrees"], check


@pytest.mark.parametrize("typ,kw", [
    ("shell", {}), ("shell", dict(wall=40.0, aperture_r=10.0)),
    ("block", {}), ("block", dict(face_shell=20.0, web=15.0)),
    ("tile", {}), ("tile", dict(pattern="flower")),
    ("tile", dict(pattern="radial", groove_depth=5.0)),
])
def test_analytic_section_matches_the_occupancy_grid(typ, kw):
    """The analytic section is within tolerance of the voxels on the same plane.

    Known exceptions, both documented in `strength.section_check` and both
    REPORTED rather than asserted away: a shell whose aperture fillet is a large
    fraction of its wall, where the grammar's blend eats a neck the analytic
    annulus does not model, and a closed shell whose cavity shuts near a pole,
    where the grid is simply coarse. Neither is in this list.
    """
    geom = E.make_geom(typ, **kw)
    check = stg.section_check(geom, _diag(geom))
    assert check["rel_diff"] <= stg.SECTION_CHECK_TOL, check


# ---------------------------------------------------------------- UCS sampling
def test_carbonate_gate_is_a_hard_zero():
    """Below ~3 % CaCO3 there is no capacity at all, whatever the geometry.

    Fu et al. 2023: a specimen needs "a minimum Ccc of circa 3%" to stand without
    confinement. A body that cannot stand up is not weak, it is absent — so this
    is a discontinuity on purpose and not a steep penalty.
    """
    rng = np.random.default_rng(0)
    assert sample_zero(rng, 2.0) == 0.0
    assert sample_zero(rng, 2.999) == 0.0
    assert sample_zero(rng, 3.0) > 0.0

    geom = BlockParams()
    diag = _diag(geom)
    design = Design(geom=geom, mix=Mix(caco3_achieved_pct=2.0), proc=Process())
    cap = stg.load_capacity(design, diag, phys=E.load_mechanics(), n_mc=64)
    assert cap["capacity_kN"] == 0.0
    assert cap["capacity_hi_kN"] == 0.0
    assert cap["ucs_nom_MPa"] == 0.0
    # the geometry is still measured — only the material is missing
    assert cap["critical_section_mm2"] > 0.0


def sample_zero(rng, pct):
    return stg.sample_ucs(rng, "rca", pct, mec=E.load_mechanics())


def test_rca_stays_an_order_of_magnitude_under_the_c90_benchmark():
    """RCA capacity must land >= 10x below ASTM C90's 13.8 MPa net-area minimum.

    The source table says MICP on waste is 5-20x under C90 BEFORE the organism
    derating; with the ASSUMED 0.3-0.7 on top it cannot come close. If this test
    ever fails, the sampler has started quoting clean-sand numbers for waste, and
    the interface would be telling someone a bio-cemented unit is a structural
    CMU substitute.
    """
    rng = np.random.default_rng(7)
    mec = E.load_mechanics()
    draws = np.array([stg.sample_ucs(rng, "rca", 8.0, mec=mec) for _ in range(4000)])

    assert draws.max() <= stg.C90_BENCHMARK_MPa / 10.0
    assert np.median(draws) <= stg.C90_BENCHMARK_MPa / 10.0

    geom = BlockParams()
    cap = _capacity(geom, _diag(geom), n_mc=400)
    assert cap["c90_ratio"] >= 10.0
    assert cap["ucs_nom_MPa"] * 10.0 <= stg.C90_BENCHMARK_MPa


def test_waste_is_weaker_than_clean_sand_and_both_are_derated():
    """The substrate class has to change which measurement is quoted."""
    mec = E.load_mechanics()
    rca = np.median([stg.sample_ucs(np.random.default_rng(i), "rca", 8.0, mec=mec)
                     for i in range(400)])
    sand = np.median([stg.sample_ucs(np.random.default_rng(i), "clean_sand", 8.0,
                                     mec=mec) for i in range(400)])
    assert rca < sand
    # the derating is applied, so nothing reaches the undegraded envelope's top
    lo, hi = stg._ucs_bounds("rca", mec)
    assert rca < hi * stg.DERATING_B_SUBTILIS[1]


def test_unknown_substrate_class_raises():
    """A typo must not silently fall through to the stronger envelope."""
    with pytest.raises(ValueError, match="substrate_class"):
        stg.sample_ucs(np.random.default_rng(0), "recycled", 8.0)
    with pytest.raises(ValueError, match="substrate_class"):
        Mix(substrate_class="recycled")


def test_capacity_is_reproducible_and_carries_its_provenance():
    geom = ShellParams()
    diag = _diag(geom)
    a = _capacity(geom, diag, n_mc=200, seed=3)
    b = _capacity(geom, diag, n_mc=200, seed=3)
    assert a["capacity_kN"] == b["capacity_kN"]
    assert a["capacity_lo_kN"] <= a["capacity_kN"] <= a["capacity_hi_kN"]
    for phrase in ("ASSUMED", "B. subtilis", "Not a structural sign-off"):
        assert phrase in a["strength_provenance"]


# ---------------------------------------------------------------- engine wiring
def test_evaluate_reports_capacity_without_gating_feasibility():
    """Capacity reaches the result dict, and never changes the verdict.

    A design with the carbonate gate unmet has zero capacity and must still be
    judged castable on the geometry alone — capacity is a reported metric, not a
    fifth subscore and not a hard rule.
    """
    common = dict(cure_days=21.0, rh_pct=85.0, split_mould=True)
    ok = E.evaluate("block", {}, dict(d_max=2.0), common, n_mc=64)
    gated = E.evaluate("block", {}, dict(d_max=2.0, caco3_achieved_pct=1.0),
                       common, n_mc=64)

    for key in ("capacity_kN", "capacity_lo_kN", "capacity_hi_kN", "ucs_nom_MPa",
                "critical_section_mm2", "kt_used", "strength_provenance"):
        assert key in ok

    assert gated["capacity_kN"] == 0.0
    assert gated["feasible"] == ok["feasible"]
    assert gated["n_fail"] == ok["n_fail"]
    assert gated["score"] == pytest.approx(ok["score"])

    row = E._row_from(ok)
    assert row["capacity_kN"] == ok["capacity_kN"]
    assert row["capacity_lo_kN"] == ok["capacity_lo_kN"]


def test_strength_objective_ranks_on_capacity_behind_feasibility():
    """`rank_strength` puts broken rules first and capacity second."""
    a = dict(n_fail=0, capacity_lo_kN=1.0, score_lo=0.9)
    b = dict(n_fail=0, capacity_lo_kN=5.0, score_lo=0.1)
    broken = dict(n_fail=1, capacity_lo_kN=99.0, score_lo=0.99)

    assert E.rank_strength(b) > E.rank_strength(a)          # capacity breaks the tie
    assert E.rank_strength(a) > E.rank_strength(broken)     # feasibility leads
    # the default objective is unchanged and still ranks on score_lo
    assert E.OBJECTIVES["viability"] is E.rank_design
    assert E.rank_design(a) > E.rank_design(b)


def test_engine_notch_of_is_the_strength_module_s():
    """One definition of the notch, so the scorer and the capacity cannot differ."""
    assert E.notch_of is stg.notch_of
    assert E.notch_of(ShellParams(fillet_r=6.0)) == (0.0, 6.0)
    assert E.notch_of(TileParams(groove_depth=3.0, groove_width=10.0,
                                 fillet_r=8.0)) == (3.0, 3.0)


# ---------------------------------------------------------------- regression
def test_validate_paper_table_is_byte_identical():
    """The paper retrodiction must not move. Strength is additive output only.

    Run in a scratch directory because `validate_paper.py` writes
    `validation_paper.json` beside itself; the literature JSONs it names are
    looked up by bare filename and are not found from either location, so the
    run is identical wherever it happens.
    """
    expected = (Path(__file__).parent / "data" / "validate_paper_table.txt").read_bytes()
    out = subprocess.run([sys.executable, str(ROOT / "validate_paper.py")],
                         cwd=Path(__file__).parent / "data" / "_run",
                         env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
                         capture_output=True, check=True)
    assert out.stdout == expected


@pytest.fixture(autouse=True, scope="module")
def _scratch_run_dir():
    d = Path(__file__).parent / "data" / "_run"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    for f in d.glob("*"):
        f.unlink()
    d.rmdir()
