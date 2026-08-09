"""Parameter schema for bio-cemented (MICP) cast geometry.

All lengths are in millimetres unless a field name ends in `_m` (metres) or the
docstring says otherwise. Physics parameters are SI internally; the conversion
happens at the boundary in `biocast.physics`.

Provenance classes
------------------
MEASURED : reported directly in a retrieved publication
DERIVED  : arithmetic on reported values (derivation recorded in the source JSON)
ASSUMED  : engineering default with no direct source
TEAM     : the project team's own stated rule (design notes, 19/07/2026)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Literal

Typology = Literal["shell", "block", "tile"]


# --------------------------------------------------------------------------
# Mix / substrate
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Mix:
    """The granular substrate being bio-cemented.

    d_max is the single most influential parameter in the whole rule set: it
    sets the fillet radius floor, the groove-width floor (via jamming) and the
    minimum castable section thickness.

    Two fields exist only for the load-capacity estimate and neither is a
    measurement of this project's material:

    `caco3_achieved_pct` is ASSUMED. It is what the treatment schedule is
    believed to have deposited, and `physics.strength` reads it for ONE purpose:
    the hard gate at ~3 % below which a specimen is not self-supporting at all
    (Fu et al. 2023). It is deliberately not used as a strength predictor —
    pooled UCS-vs-CaCO3 fits over the retrieved data give R^2 <= 0.01, and the
    same carbonate content has produced strengths 12x apart. `caco3_target` is
    the schedule's target, kept as a record of intent; no model reads it.

    `substrate_class` SELECTS which measured UCS envelope applies — "rca" for
    crushed construction and demolition waste (the project's actual substrate,
    and the weakest: one study, 0.34-0.72 MPa) or "clean_sand" for the pooled
    clean-sand band. It changes which measurement is quoted, not how it is used.
    """

    d_max: float = 4.0            # mm, largest aggregate fragment (team notes: waste reaches 4 mm)
    d50: float = 1.0              # mm, median particle size
    porosity: float = 0.38        # -, packed bed void fraction
    saturation: float = 0.65      # -, fraction of pore volume filled with liquid during curing
    caco3_target: float = 8.0     # % by mass, the treatment schedule's target (recorded, unread)
    caco3_achieved_pct: float = 8.0   # % by mass, ASSUMED — see below
    substrate_class: str = "rca"  # "rca" (crushed C&D waste) | "clean_sand"

    def __post_init__(self):
        if self.d_max <= 0:
            raise ValueError("d_max must be positive")
        if not 0 < self.porosity < 1:
            raise ValueError("porosity must be in (0,1)")
        if not 0 <= self.saturation <= 1:
            raise ValueError("saturation must be in [0,1]")
        if self.substrate_class not in ("rca", "clean_sand"):
            raise ValueError(f"substrate_class must be 'rca' or 'clean_sand', "
                             f"got {self.substrate_class!r}")


# --------------------------------------------------------------------------
# Process / curing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Process:
    """Inoculation and curing schedule."""

    cycles: int = 4               # number of feed/mineralisation cycles
    cure_days: float = 14.0       # total curing duration
    temp_C: float = 30.0          # curing temperature
    rh_pct: float = 90.0          # curing relative humidity
    split_mould: bool = True      # cast in halves (paper Fig. 6) -> parting face is O2-exposed
    forced_aeration: bool = False # active air supply into an internal cavity


# --------------------------------------------------------------------------
# Geometry parameter sets, one per grammar
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ShellParams:
    """Hollow ovoid vessel, cast as two halves — the paper's successful typology.

    Superellipsoid profile: |x/a|^n + |y/b|^n + |z/c|^n = 1 with an ovoid taper
    applied along +z so the form is egg-like rather than symmetric.
    """

    typology: Typology = "shell"
    a: float = 55.0               # mm, semi-axis x
    b: float = 55.0               # mm, semi-axis y
    c: float = 78.0               # mm, semi-axis z (long axis)
    n: float = 2.4                # superellipsoid exponent (2 = ellipsoid, >2 = boxier)
    ovoid: float = 0.28           # 0 = symmetric ellipsoid, >0 = egg taper
    wall: float = 14.0            # mm, shell wall thickness
    aperture_r: float = 16.0      # mm, radius of the top opening (0 = fully closed)
    fillet_r: float = 8.0         # mm, fillet radius at the aperture rim / parting edges
    rib_count: int = 0            # meridional stiffening ribs on the inner wall
    rib_depth: float = 0.0        # mm, rib protrusion into the cavity


@dataclass(frozen=True)
class BlockParams:
    """Hollow-core masonry unit (CMU-type).

    Team notes give the classic 200x200x400 mm module with ~40-50 % void,
    ~32 mm face shells and ~25 mm webs.
    """

    typology: Typology = "block"
    L: float = 390.0              # mm, length  (nominal 400 module)
    W: float = 190.0              # mm, width   (nominal 200 module)
    H: float = 190.0              # mm, height  (nominal 200 module)
    n_cores: int = 2              # number of hollow cores
    face_shell: float = 32.0      # mm, outer wall thickness
    web: float = 25.0             # mm, internal partition thickness
    fillet_r: float = 8.0         # mm, fillet radius on core corners and outer edges
    core_taper: float = 2.0       # deg, draft on core walls for mould release
    groove_depth: float = 0.0     # mm, decorative face groove depth
    groove_width: float = 0.0     # mm, decorative face groove width
    groove_count: int = 0


@dataclass(frozen=True)
class TileParams:
    """Relief paving/cladding tile (Panot-type).

    Team notes: 200x200 mm, 40 mm thick, relief only 2-3 mm deep (<10 % of
    thickness) with channels ~10 mm wide; the relief is for drainage and slip
    resistance, NOT stiffening.
    """

    typology: Typology = "tile"
    L: float = 200.0              # mm
    W: float = 200.0              # mm
    t: float = 40.0               # mm, tile thickness
    pattern: str = "grid"         # grid | diagonal | flower | radial
    groove_depth: float = 3.0     # mm, recessed relief depth
    groove_width: float = 10.0    # mm, recessed channel width
    groove_pitch: float = 50.0    # mm, centre-to-centre channel spacing
    fillet_r: float = 8.0         # mm, fillet radius at groove roots and perimeter
    joint: float = 6.0            # mm, designed joint gap to the neighbouring tile
    thick_tile: bool = True       # True -> the stricter t/4 groove-depth limit applies


GeomParams = ShellParams | BlockParams | TileParams


@dataclass(frozen=True)
class Design:
    """A complete design: geometry + mix + process."""

    geom: GeomParams
    mix: Mix = field(default_factory=Mix)
    proc: Process = field(default_factory=Process)
    name: str = "design"

    @property
    def typology(self) -> Typology:
        return self.geom.typology

    def to_dict(self) -> dict:
        """Plain data for export. `asdict` carries the new Mix fields for free;
        the provenance block is here because they must not be read as
        measurements once this dict has left the process."""
        return {
            "name": self.name,
            "typology": self.typology,
            "geom": asdict(self.geom),
            "mix": asdict(self.mix),
            "proc": asdict(self.proc),
            "provenance": {
                "mix.caco3_achieved_pct":
                    "ASSUMED — no measurement of this project's material",
                "mix.substrate_class":
                    "SELECTS which measured UCS envelope applies; 'rca' rests on "
                    "a single study (Fouladi et al. 2024)",
            },
        }

    def variant(self, **kw) -> "Design":
        """Return a copy with geometry fields overridden."""
        return replace(self, geom=replace(self.geom, **kw))


# --------------------------------------------------------------------------
# Literature-backed constants, loaded from the phase-0 JSON artifacts
# --------------------------------------------------------------------------
class LitParams:
    """Thin accessor over the retrieved parameter JSON files.

    Each entry carries value/low/high, so `get` returns the nominal and
    `bounds` returns the Monte-Carlo range used for uncertainty propagation.
    """

    def __init__(self, *paths: str | Path):
        self.entries: dict[str, dict] = {}
        self.sources: dict[str, dict] = {}
        self.extra: dict[str, object] = {}
        for p in paths:
            if p is None:
                continue
            p = Path(p)
            if not p.exists():
                continue
            blob = json.loads(p.read_text())
            for e in blob.get("parameters", []):
                key = e.get("symbol") or e.get("name")
                if key:
                    self.entries[key] = e
                    if e.get("name"):
                        self.entries.setdefault(e["name"], e)
            for k, v in blob.items():
                if k not in ("parameters", "schema"):
                    self.extra[k] = v

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str, default: float | None = None) -> float:
        e = self.entries.get(key)
        if e is None:
            if default is None:
                raise KeyError(f"parameter {key!r} not in retrieved literature set")
            return default
        v = e.get("value")
        return float(v) if v is not None else float(default)

    def bounds(self, key: str, default: tuple | None = None) -> tuple[float, float]:
        e = self.entries.get(key)
        if e is None:
            if default is None:
                raise KeyError(key)
            return default
        v = e.get("value")
        lo = e.get("low", v)
        hi = e.get("high", v)
        lo = float(lo if lo is not None else v)
        hi = float(hi if hi is not None else v)
        return (min(lo, hi), max(lo, hi))

    def provenance(self, key: str) -> dict:
        e = self.entries.get(key, {})
        return {
            "evidence_class": e.get("evidence_class"),
            "organism": e.get("organism"),
            "substrate": e.get("substrate"),
            "source_doi": e.get("source_doi"),
            "source_title": e.get("source_title"),
            "notes": e.get("notes"),
        }
