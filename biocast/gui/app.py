"""Bio-concrete design studio — interactive shape generator with live scoring.

Run:  PYTHONPATH=. streamlit run biocast/gui/app.py

Four tabs:
  Design    move sliders, see the mesh in 3D, the four subscores, and every
            constraint verdict update live; download STL.
  Mould     generate, verify and mesh a rigid or silicone mould for that shape in
            one action, and download every part as a zip with a fabrication
            manifest. The silicone path is two castings: it also emits the master
            pattern and the printed former the rubber itself is poured in.
  Process   the feasible window as a map — where the castability floor and drying
            ceiling cross, and what cure or sieve opens it.
  Explore   randomised search inside the current process settings, ranked.

Everything numeric comes from `biocast.gui.engine`, which calls the validated
package. The GUI holds no physics of its own.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from biocast.gui import engine as E
from biocast.gui import viewer as V

st.set_page_config(page_title="Bio-concrete design studio", layout="wide",
                   initial_sidebar_state="expanded")

SUB_LABEL = {
    "aeration": "Aeration",
    "drying": "Drying uniformity",
    "castability": "Castability",
    "structural": "Structural",
}
SUB_HELP = {
    "aeration": "Fraction of the body oxygen can reach. B. subtilis is an obligate "
                "aerobe, so an anoxic core never cements.",
    "drying": "Half-thickness against the depth evaporation actually drains. Above "
              "1.0 the surface shrinks against a wet core and cracks.",
    "castability": "Narrowest passage against the aggregate size. Granular flow jams "
                   "below ~6 x d_max, so the mould starves.",
    "structural": "Inglis notch factor converted through the Weibull modulus. A sharp "
                  "root is where a brittle cast cracks.",
}
ORIGIN_LABEL = {"TEAM": "your notes", "LIT": "literature", "STD": "standard",
                "GEOM": "geometry"}


# ----------------------------------------------------------------------------- sidebar
def sidebar_process():
    st.sidebar.header("Mix and process")
    st.sidebar.caption("These apply to every typology and drive the feasible window.")

    d_max = st.sidebar.slider(
        "Max aggregate size d_max (mm)", 1.0, 8.0, 4.0, 0.5,
        help="Largest fragment in the ground waste. Sets the castability floor at "
             "6 x d_max and the fillet floor at 1.5 x d_max.")
    porosity = st.sidebar.slider("Packed porosity", 0.30, 0.50, 0.38, 0.01)

    st.sidebar.markdown("**Curing**")
    cure_days = st.sidebar.slider(
        "Cure duration (days)", 3, 60, 21, 1,
        help="Longer cure drains deeper, which raises the ceiling on section thickness.")
    rh_pct = st.sidebar.slider(
        "Relative humidity (%)", 50, 99, 85, 1,
        help="The strongest process lever in the model: it enters linearly through "
             "(1 - RH). Lower RH drains faster.")
    split_mould = st.sidebar.checkbox(
        "Split mould (cast in halves)", True,
        help="The parting face is open to atmosphere during curing, so it acts as an "
             "oxygen source. This is the paper's Fig. 6 technique.")

    st.sidebar.markdown("---")
    with st.sidebar.expander("Rule set", expanded=False):
        jam_ratio = st.slider(
            "Jamming threshold (x d_max)", 3.0, 8.0, E.JAM_RATIO_LIT, 0.1,
            help="Measured critical aperture-to-particle ratio: 4.94 spheres, 6.0 "
                 "angular grains, 8.1 dense suspension; below 3.0 it always clogs. "
                 "Your notes say 2-3, which sits at the always-clog boundary.")
        n_mc = st.select_slider("Monte Carlo draws", [100, 200, 300, 500, 800], 300,
                                help="More draws tighten the interval estimate, not "
                                     "the interval itself.")
        st.caption(
            "The package default is 4.0; this app defaults to 6.0 on the granular-flow "
            "literature. The choice changes which designs are rejected.")

    return (dict(d_max=d_max, porosity=porosity),
            dict(cure_days=float(cure_days), rh_pct=float(rh_pct),
                 split_mould=split_mould),
            jam_ratio, n_mc)


#: THE SEARCH SPACE, IN ONE PLACE. `geom_controls` builds its sliders from this and
#: `search_shapes` samples and refines inside it, so the shape you can dial in by hand
#: and the shape the search can reach are the same set.
#:
#: They were not. The explorer drew `wall` from 8-42 mm against a slider offering
#: 6-45, `a` and `b` from 40-90 against 30-100, and the tile's `groove_pitch` from
#: 30-80 against 25-90 — so the search could not reach designs the sliders offered,
#: and a design the search found could sit outside what the sliders could reproduce.
#: Two hand-maintained copies of the same bounds is one copy too many.
#:
#: `floor` is a multiple of d_max that raises the DEFAULT (not the bound), so the
#: interface opens on a design that already respects the rule instead of on one the
#: verdict table immediately rejects.
GEOM_SPACE = {
    "shell": {
        "a": dict(label="Semi-axis a (mm)", lo=30.0, hi=100.0, val=55.0, step=1.0),
        "b": dict(label="Semi-axis b (mm)", lo=30.0, hi=100.0, val=55.0, step=1.0),
        "c": dict(label="Semi-axis c, long (mm)", lo=40.0, hi=150.0, val=78.0, step=1.0),
        "n": dict(label="Superellipsoid exponent n", lo=2.0, hi=4.0, val=2.4, step=0.1,
                  help="2 = ellipsoid, higher = boxier"),
        "ovoid": dict(label="Egg taper", lo=0.0, hi=0.45, val=0.28, step=0.01),
        "wall": dict(label="Wall thickness (mm)", lo=6.0, hi=45.0, val=22.0, step=0.5,
                     help="Castability floor here is 6 x d_max"),
        "aperture_r": dict(label="Aperture radius (mm)", lo=0.0, hi=35.0, val=16.0,
                           step=0.5,
                           help="A feed passage, so the bore diameter must also clear "
                                "the jamming limit"),
        "fillet_r": dict(label="Fillet radius (mm)", lo=2.0, hi=20.0, val=8.0, step=0.5,
                         floor=1.5, help="Your rule: >= 1.5 x d_max"),
    },
    "block": {
        "face_shell": dict(label="Face shell (mm)", lo=20.0, hi=55.0, val=34.0,
                           step=1.0, help="ASTM C90 minimum is 32 mm for 8 in units"),
        "web": dict(label="Web (mm)", lo=15.0, hi=45.0, val=30.0, step=1.0,
                    help="C90 permits 19 mm; your notes say 25 mm"),
        "core_taper": dict(label="Core draft (deg)", lo=0.0, hi=5.0, val=2.0, step=0.1,
                           help="Needed for release of a fragile green body"),
        "fillet_r": dict(label="Fillet radius (mm)", lo=2.0, hi=20.0, val=8.0, step=0.5,
                         floor=1.5, help="Your rule: >= 1.5 x d_max"),
        "groove_depth": dict(label="Face groove depth (mm)", lo=0.0, hi=20.0, val=0.0,
                             step=0.5),
        "groove_width": dict(label="Face groove width (mm)", lo=0.0, hi=50.0, val=0.0,
                             step=1.0),
    },
    "tile": {
        "t": dict(label="Thickness (mm)", lo=20.0, hi=60.0, val=40.0, step=1.0),
        "groove_depth": dict(label="Relief depth (mm)", lo=1.0, hi=10.0, val=3.0,
                             step=0.5,
                             help="Panot practice is 2-3 mm; the regulated ceiling "
                                  "is 5 mm"),
        "groove_width": dict(label="Channel width (mm)", lo=5.0, hi=45.0, val=24.0,
                             step=1.0, help="Jamming floor is 6 x d_max"),
        "groove_pitch": dict(label="Channel pitch (mm)", lo=25.0, hi=90.0, val=50.0,
                             step=1.0),
        "fillet_r": dict(label="Fillet radius (mm)", lo=2.0, hi=18.0, val=8.0, step=0.5,
                         floor=1.5),
        "joint": dict(label="Joint gap (mm)", lo=2.0, hi=15.0, val=6.0, step=0.5),
    },
}

#: Discrete parameters, which the optimiser enumerates rather than perturbs.
GEOM_CHOICES = {"shell": {}, "block": {"n_cores": [2, 3]},
                "tile": {"pattern": ["grid", "diagonal", "flower", "radial"]}}

#: Which column each control lands in, so the layout survives the table-driven build.
GEOM_COLS = {
    "shell": (("a", "b", "c"), ("n", "ovoid", "wall"), ("aperture_r", "fillet_r")),
    "block": (("face_shell", "web"), ("n_cores", "core_taper"),
              ("fillet_r", "groove_depth", "groove_width")),
    "tile": (("t", "pattern"), ("groove_depth", "groove_width", "groove_pitch"),
             ("fillet_r", "joint")),
}

GEOM_HEADING = {
    "shell": "**Hollow ovoid vessel** — the typology that succeeded in the paper",
    "block": "**Hollow-core masonry unit** — CMU module, 390 x 190 x 190 mm",
    "tile": "**Relief paving tile** — Panot type, 200 x 200 mm",
}


def geom_default(typology: str, name: str, d_max: float) -> float:
    s = GEOM_SPACE[typology][name]
    return float(min(s["hi"], max(s["val"], s.get("floor", 0.0) * d_max)))


def geom_key(typology: str, name: str) -> str:
    """Widget key, so a design found by the search can be written into the sliders.

    Streamlit only lets a widget's value be set through `session_state` under its own
    key, and only before the widget is built for that run — which is why the search's
    "use this design" writes the keys and then reruns rather than returning a dict.
    """
    return f"geom_{typology}_{name}"


def derive_geom(typology: str, g: dict) -> dict:
    """The parameters the grammars want that are implied rather than dialled."""
    g = dict(g)
    if typology == "block":
        g["groove_count"] = 2 if g.get("groove_depth", 0.0) > 0 else 0
    if typology == "tile":
        g["thick_tile"] = g.get("t", 40.0) >= 35.0
    return g


def geom_controls(typology: str, d_max: float) -> dict:
    """Typology-specific sliders, built from `GEOM_SPACE`.

    A design handed over by the search is applied HERE, at the top, rather than
    written by the button that chose it. Streamlit refuses to modify a widget's key
    once that widget has been instantiated in the current run, and `tab_design` runs
    before `tab_explore` in every run — so the button's own
    `st.session_state[geom_key(...)] = v` raised `StreamlitAPIException` on the first
    slider it touched. Staging the values under a plain key and applying them on the
    next run, before any widget exists, is the way round it.
    """
    pend = st.session_state.pop("geom_pending", None)
    if pend and pend["typology"] == typology:
        for k, v in pend["values"].items():
            st.session_state[geom_key(typology, k)] = v
    st.markdown(GEOM_HEADING[typology])
    space, choices = GEOM_SPACE[typology], GEOM_CHOICES[typology]
    fillet_floor = 1.5 * d_max
    out = {}
    for col, names in zip(st.columns(3), GEOM_COLS[typology]):
        with col:
            for name in names:
                key = geom_key(typology, name)
                if name in choices:
                    opts = choices[name]
                    out[name] = st.selectbox(name.replace("_", " ").capitalize(),
                                             opts, key=key)
                    continue
                s = space[name]
                hint = s.get("help", "")
                if s.get("floor"):
                    hint += f" = {s['floor'] * d_max:.1f} mm at d_max = {d_max:g} mm"
                elif "6 x d_max" in hint:
                    hint += f" = {6 * d_max:.0f} mm"
                if key not in st.session_state:
                    st.session_state[key] = geom_default(typology, name, d_max)
                out[name] = st.slider(s["label"], s["lo"], s["hi"], step=s["step"],
                                      key=key, help=hint or None)
    return derive_geom(typology, out)


# ----------------------------------------------------------------------------- views
def score_header(r: dict):
    """Verdict banner + the score with its interval, which is the honest headline."""
    lo, hi = r["score_lo"], r["score_hi"]
    if r["feasible"]:
        st.success(f"**Castable** — predicted success {r['score']:.3f} "
                   f"(5-95 %: {lo:.3f} - {hi:.3f})")
    else:
        st.error(f"**Rejected** — {r['n_fail']} rule"
                 f"{'s' if r['n_fail'] != 1 else ''} failed. "
                 f"Predicted success {r['score']:.3f} (5-95 %: {lo:.3f} - {hi:.3f})")

    if hi - lo > 0.4:
        st.caption("The interval is wide: the literature ranges alone span most of "
                   "[0,1] here, so treat this as a ranking, not a probability.")

    cols = st.columns(4)
    for col, key in zip(cols, ["aeration", "drying", "castability", "structural"]):
        v = r[f"sub_{key}"]
        limiting = (r["dominant_failure_mode"] == key)
        col.metric(SUB_LABEL[key] + (" ← limiting" if limiting else ""),
                   f"{v:.3f}", help=SUB_HELP[key])
    st.caption(f"Subscores multiply, so the weakest sets the total. "
               f"Limiting term: **{SUB_LABEL[r['dominant_failure_mode']]}** — "
               f"{r['failure_mode_text']}")


def preview_panel(r: dict):
    """The mesh itself, before the tables that describe it."""
    st.subheader("Shape preview")
    mesh = r["_mesh"]
    if len(mesh.faces) == 0:
        st.warning("The mesh came back empty — nothing to draw. Check the geometry "
                   "sliders; a wall thicker than the semi-axis leaves no solid.")
        return
    p = V.stl_viewer(mesh)
    note = ("Drag to orbit, shift-drag to pan, scroll to zoom. **Section** slices the "
            "body on a plane — the red faces are the cut, which is how to read wall "
            "thickness, the cores, and whether a cavity actually closed.")
    if p["nFacesFull"] > p["nFaces"]:
        note += (f" The preview is simplified to {p['nFaces']:,} of "
                 f"{p['nFacesFull']:,} triangles for display; the STL download below "
                 "is the full-resolution mesh.")
    st.caption(note)
    if not r["watertight"]:
        st.warning("This mesh is not watertight, so the preview may show holes and the "
                   "volume-based numbers are unreliable.")


def geometry_panel(r: dict):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Geometry**")
        st.dataframe(pd.DataFrame([
            ("Volume", f"{r['volume_mm3']/1000:.1f} cm³"),
            ("Surface / volume", f"{r['sa_to_vol']:.4f} mm⁻¹"),
            ("Max section", f"{r['max_wall_thickness_mm']:.1f} mm"),
            ("Measured narrowest section",
             f"{r['min_section_measured_mm']:.1f} mm  "
             f"({r['section_over_dmax']:.1f} × d_max)"),
            ("Nominal narrowest passage", f"{r['min_feature_nominal_mm']:.1f} mm"),
            ("Watertight", "yes" if r["watertight"] else "NO — mesh error"),
        ], columns=["", "value"]), hide_index=True, width="stretch")
        if r["min_section_measured_mm"] < 0.7 * r["min_feature_nominal_mm"]:
            st.warning(
                f"The measured section ({r['min_section_measured_mm']:.1f} mm) is well "
                f"below the nominal wall ({r['min_feature_nominal_mm']:.1f} mm) — the "
                "aperture bore is eating into it. Castability is scored on the measured "
                "value, which is the passage the mix must actually flow through.")
    with c2:
        st.markdown("**Transport**")
        st.dataframe(pd.DataFrame([
            ("Cemented fraction", f"{r['cemented_fraction']:.2f}"),
            ("Reachable depth L_eff", f"{r['penetration_depth_nom_mm']:.1f} mm"),
            ("  gas-phase limit", f"{r['L_gas_nom_mm']:.0f} mm"),
            ("  drained depth", f"{r['L_dry_nom_mm']:.1f} mm"),
            ("Limited by", r["penetration_limiter"]),
            ("Drying ratio", f"{r['drying_ratio']:.2f}"),
            ("Notch factor Kt", f"{r['kt']:.2f}"),
        ], columns=["", "value"]), hide_index=True, width="stretch")


def verdict_table(r: dict):
    rows = []
    for v in r["verdicts"]:
        rows.append({
            "": "FAIL" if (not v["passed"] and v["severity"] == "fail")
                else ("warn" if not v["passed"] else "ok"),
            "rule": v["rule"],
            "source": ORIGIN_LABEL.get(v["origin"], v["origin"]),
            "value": None if v["value"] is None else round(float(v["value"]), 2),
            "limit": None if v["limit"] is None else round(float(v["limit"]), 2),
            "detail": v["message"],
        })
    df = pd.DataFrame(rows)
    order = {"FAIL": 0, "warn": 1, "ok": 2}
    df = df.sort_values("", key=lambda s: s.map(order)).reset_index(drop=True)
    st.dataframe(df, hide_index=True, width="stretch", height=340)
    st.caption("Rules tagged *your notes* come from the design notes; *literature* "
               "from retrieved papers; *standard* from ASTM C90 / Panot / ACI 318.")


def section_figure(r: dict):
    """Mid-plane oxygen field — where the design does and does not cement.

    Built with `Figure()` rather than `plt.subplots()` on purpose. `st.pyplot(fig)`
    defaults to `clear_figure=False` whenever a figure is passed, so a pyplot-created
    figure stays in matplotlib's global `Gcf` registry forever — and every tab body
    re-runs on every widget interaction, so a long session accumulates hundreds of
    them. That eats exactly the RAM headroom `mould_pitch_for` is calibrated against,
    which would surface as an out-of-memory kill during a silicone solve rather than
    as the figure leak it is. A bare `Figure` is never registered.
    """
    ox = r.get("_oxygen")
    if ox is None:
        return None
    C, occ, pitch = ox["C"], ox["occ"], ox["pitch"]
    j = occ.shape[1] // 2
    sl_occ = occ[:, j, :].T
    sl_C = np.where(sl_occ, C[:, j, :].T, np.nan)

    fig = Figure(figsize=(5.2, 4.4))
    ax = fig.subplots()
    ax.imshow(np.where(sl_occ, 0.0, np.nan), origin="lower", cmap="Greys",
              vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(sl_C, origin="lower", cmap="viridis", vmin=0.0, vmax=8.42,
                   interpolation="nearest")
    anox = sl_occ & ~(sl_C > 0)
    if anox.any():
        ax.contourf(anox.astype(float), levels=[0.5, 1.5], colors=["#7b241c"], alpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Mid-plane pore O$_2$  ·  {ox['anoxic_fraction']*100:.0f} % anoxic",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="mol m$^{-3}$", fraction=0.046)
    fig.tight_layout()
    if ox.get("resolution_warning"):
        st.caption(ox["resolution_warning"])
    return fig


def window_figure(mix, proc, jam_ratio):
    """The feasible window: castability floor against drying ceiling.

    `Figure()` rather than `plt.subplots()` — see `section_figure`. This one runs
    unguarded on every single script run, so it is the larger of the two leaks.
    """
    dmx = np.linspace(1.0, 8.0, 200)
    floor = jam_ratio * dmx
    w_now = E.feasible_window(mix["d_max"], proc["cure_days"], proc["rh_pct"], jam_ratio)
    ceiling = w_now["ceiling_mm"]

    fig = Figure(figsize=(6.6, 4.3))
    ax = fig.subplots()
    ax.fill_between(dmx, floor, ceiling, where=floor <= ceiling, alpha=0.25,
                    color="#2e7d32", label="feasible section band")
    ax.plot(dmx, floor, color="#2e7d32", lw=2,
            label=f"castability floor, {jam_ratio:g} × d_max")
    ax.axhline(ceiling, color="#b8860b", lw=2,
               label=f"drying ceiling, 2 × L_dry = {ceiling:.1f} mm")
    ax.plot(dmx, 3.0 * dmx, ls=":", color="#c0392b", lw=1.4,
            label="always clogs below 3 × d_max")
    ax.axvline(mix["d_max"], color="0.35", lw=1, ls="--")
    ax.plot([mix["d_max"]], [ceiling], "o", color="0.2", ms=6)
    ax.annotate(f"your mix\nd_max = {mix['d_max']:g} mm",
                (mix["d_max"], ceiling), textcoords="offset points",
                xytext=(8, -28), fontsize=9, color="0.2")
    ax.set_xlabel("Max aggregate size d_max (mm)")
    ax.set_ylabel("Section thickness (mm)")
    ax.set_ylim(0, max(60, ceiling * 1.4))
    ax.set_xlim(1, 8)
    ttl = ("Window is OPEN" if w_now["open"] else "Window is CLOSED")
    ax.set_title(f"{ttl} at d_max = {mix['d_max']:g} mm, "
                 f"{proc['cure_days']:.0f} d at {proc['rh_pct']:.0f} % RH", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig, w_now


# ----------------------------------------------------------------------------- tabs
def _evaluate_memo(typology, geom_kw, mix, proc, jam_ratio, n_mc, with_field):
    """The Design-tab solve, memoised on its full input set for the session.

    `st.tabs` is client-side only: every tab body executes on every script run, and
    every widget interaction anywhere on the page is a script run. So each click in
    the Mould tab — including the STL download — was silently re-running a 0.6-0.9 s
    geometry-and-score solve for the Design tab as a side effect.

    Held in `session_state` rather than `st.cache_data`, which pickles the return
    value: this one carries a Trimesh and the diagnostic voxel grids, and
    round-tripping those through pickle costs more than the solve it would save.
    """
    sig = json.dumps([typology, geom_kw, mix, proc, jam_ratio, n_mc, with_field],
                     sort_keys=True, default=str)
    hit = st.session_state.get("_design_memo")
    if hit is not None and hit[0] == sig:
        return hit[1]
    r = E.evaluate(typology, geom_kw, mix, proc, jam_ratio=jam_ratio,
                   n_mc=n_mc, with_field=with_field)
    st.session_state["_design_memo"] = (sig, r)
    return r


def tab_design(typology, mix, proc, jam_ratio, n_mc):
    geom_kw = geom_controls(typology, mix["d_max"])
    # The Mould tab generates a mould for whatever the Design tab currently shows,
    # so the live geometry has to be reachable from there. Keyed by typology so
    # switching typology cannot hand the mould generator another shape's parameters.
    st.session_state["geom_kw"] = geom_kw
    st.session_state["geom_kw_typology"] = typology
    c1, c2 = st.columns(2)
    show_prev = c1.checkbox(
        "Show the 3D shape preview", value=True,
        help="Draws the meshed body in the browser. Turn it off if you are moving "
             "sliders on a slow connection — it sends about 0.5 MB per update.")
    show_field = c2.checkbox(
        "Solve the oxygen field (slower, shows the anoxic core)", value=False,
        help="Runs the reaction-diffusion solve and draws a mid-plane section.")

    with st.spinner("Building geometry and scoring…"):
        r = _evaluate_memo(typology, geom_kw, mix, proc, jam_ratio, n_mc, show_field)

    score_header(r)
    st.markdown("---")
    if show_prev:
        preview_panel(r)
        st.markdown("---")
    geometry_panel(r)

    if show_field:
        fig = section_figure(r)
        if fig is not None:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.pyplot(fig)
            with c2:
                ox = r["_oxygen"]
                st.markdown("**What the field shows**")
                st.write(
                    f"Evaporation has drained {ox['L_dry_mm']:.1f} mm from every "
                    f"air-exposed face. Inside that shell oxygen is available; deeper "
                    f"the pores are still water-filled, where oxygen reaches only "
                    f"~0.3 mm. {ox['anoxic_fraction']*100:.0f} % of this body is "
                    "anoxic and will not cement.")
                if ox["anoxic_fraction"] > 0.15:
                    st.info("To shrink the anoxic core: reduce the section, add a "
                            "cavity, lower the RH, or extend the cure.")

    st.markdown("---")
    st.subheader("Constraint verdicts")
    verdict_table(r)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    mesh = r["_mesh"]
    tag = f"{typology}_dmax{mix['d_max']:g}_{proc['cure_days']:.0f}d{proc['rh_pct']:.0f}rh_score{r['score']:.3f}"
    # `data` is a callable, so the mesh is only serialised when someone actually
    # clicks. Every tab body re-runs on every widget interaction, so the eager form
    # exported a 2-6 MB STL into the media manager on each slider move. Safe here
    # because the button re-renders on every run — a deferred payload is dropped the
    # moment its element stops being referenced.
    if len(mesh.faces) == 0:
        c1.warning("No solid to export — the geometry sliders leave nothing to mesh.")
    else:
        c1.download_button("Download STL", lambda: mesh.export(file_type="stl"),
                           file_name=f"{tag}.stl", mime="model/stl",
                           width="stretch", on_click="ignore")
    rec = {k: v for k, v in r.items() if not k.startswith("_")}
    rec.update({"geom": geom_kw, "mix": mix, "proc": proc})
    c2.download_button("Download report (JSON)",
                       lambda: json.dumps(rec, indent=1, default=str),
                       file_name=f"{tag}.json", mime="application/json",
                       width="stretch", on_click="ignore")
    if typology == "shell" and proc["split_mould"] and len(mesh.faces):
        from biocast.grammars import shell as sh
        try:
            lo, up = sh.split_halves(mesh)
            c3.download_button("Download lower half",
                               lambda: lo.export(file_type="stl"),
                               file_name=f"{tag}_lower.stl", mime="model/stl",
                               width="stretch", on_click="ignore")
        except Exception as exc:                       # pragma: no cover
            c3.caption(f"half export unavailable: {exc}")
    return r


def tab_process(mix, proc, jam_ratio):
    st.markdown(
        "Two limits act on the same section thickness from opposite directions. "
        "**Castability** needs it thick enough that aggregate does not bridge; "
        "**drying** needs it thin enough that evaporation drains the whole section. "
        "When the floor rises above the ceiling, no thickness works and the fix is "
        "process, not geometry.")
    fig, w = window_figure(mix, proc, jam_ratio)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(fig)
    with c2:
        st.metric("Drained depth L_dry", f"{w['L_dry_mm']:.1f} mm")
        st.metric("Feasible band width",
                  f"{w['width_mm']:.1f} mm" if w["open"] else "closed",
                  delta=None)
        if w["open"]:
            st.success(f"Sections from **{w['floor_mm']:.0f} to {w['ceiling_mm']:.0f} mm** "
                       "clear both limits.")
        else:
            st.error(f"No feasible thickness: castability wants ≥ {w['floor_mm']:.0f} mm "
                     f"but drying allows ≤ {w['ceiling_mm']:.0f} mm.")
            t_open = E.min_cure_to_open(mix["d_max"], proc["rh_pct"], jam_ratio)
            sieve = E.max_dmax_to_open(proc["cure_days"], proc["rh_pct"], jam_ratio)
            st.markdown("**Two ways out**")
            if t_open:
                st.write(f"- Cure for **{t_open:.0f} days** at {proc['rh_pct']:.0f} % RH "
                         f"(from {proc['cure_days']:.0f})")
            st.write(f"- Or sieve the waste below **{sieve:.1f} mm** "
                     f"(from {mix['d_max']:g})")
            st.caption("Lowering RH is usually cheapest: it enters linearly through "
                       "(1 - RH), so it moves the ceiling faster than added days.")

    st.markdown("---")
    st.subheader("Cure schedule map")
    st.caption("Shortest cure that opens the window, by aggregate size and humidity. "
               "Blank = no cure up to 120 days opens it.")
    dms = [1, 2, 3, 4, 5, 6, 7, 8]
    rhs = [60, 70, 75, 80, 85, 90, 95]
    grid = []
    for dm in dms:
        row = {"d_max (mm)": dm}
        for rh in rhs:
            t = E.min_cure_to_open(float(dm), float(rh), jam_ratio)
            row[f"{rh} %"] = None if t is None else int(t)
        grid.append(row)
    gdf = pd.DataFrame(grid).set_index("d_max (mm)")
    st.dataframe(gdf.style.background_gradient(cmap="RdYlGn_r", axis=None)
                 .format(na_rep="—", precision=0), width="stretch")


def tab_explore(typology, mix, proc, jam_ratio):
    st.subheader("Find the shape most likely to cement")
    st.markdown(
        "Samples the design space to find a promising basin, then walks downhill "
        "inside it one parameter at a time. Every candidate is meshed, diagnosed and "
        "scored by the same engine as the Design tab — the search adds no shortcut "
        "model. When it finishes you can push the winner straight into the sliders, "
        "and the **Mould** tab will then build the silicone mould for it.")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
    n = c1.select_slider("Random samples", [12, 24, 48, 96], 48,
                         help="This stage is where the designs actually come from. "
                              "Spend budget here first.")
    n_ref = c2.select_slider("Refinement depth", [0, 1, 2, 3], 1,
                             help="Compass search: each parameter up and down, the "
                                  "step halving only when a whole sweep fails. It "
                                  "confirms a local optimum rather than finding "
                                  "better designs — measured across all three "
                                  "typologies it matched the best sampled design "
                                  "every time and never beat it. Raise it only when "
                                  "polishing a design you already like.")
    seed = c3.number_input("Seed", 0, 9999, 0, 1)
    n_par = len(GEOM_SPACE[typology]) + len(GEOM_CHOICES[typology])
    total = int(n) + int(n_ref) * n_par * 4 + 1
    c4.metric("Designs to score", f"≤ {total}",
              help=f"1 current + {n} random + at most {n_ref} x {n_par} x 4 refining. "
                   "The refinement stops as soon as it stops improving, so the real "
                   "count is usually lower.")
    st.caption(f"Up to about {total * 1.3:.0f} s at roughly 1.3 s per design on two "
               f"cores. The design currently on the sliders is scored first, so the "
               f"search can never hand back something worse than what you have.")

    if st.button("Search", type="primary"):
        prog = st.progress(0.0, text="scoring…")
        cur = (st.session_state.get("geom_kw", {})
               if st.session_state.get("geom_kw_typology") == typology else None)
        with st.spinner("Searching…"):
            rows = E.search_shapes(
                typology, GEOM_SPACE[typology], GEOM_CHOICES[typology], derive_geom,
                mix, proc, jam_ratio=jam_ratio, n_random=int(n), n_refine=int(n_ref),
                seed=int(seed), start=cur,
                progress=lambda f, d, t: prog.progress(f, text=f"scored {d}/{t}"))
        prog.empty()
        # Same reason the Mould tab holds its result: any widget interaction re-runs
        # the script with this button False, and a search whose results vanish the
        # moment you touch anything cannot be acted on.
        st.session_state["search"] = {
            "typology": typology, "rows": rows,
            "settings": {"mix": dict(mix), "proc": dict(proc), "jam": jam_ratio},
        }

    got = st.session_state.get("search")
    if not got or got["typology"] != typology:
        st.info("Press **Search**. Nothing is computed until you do.")
        return
    rows = got["rows"]
    if not rows:
        st.warning("No candidate built successfully. Try a different typology or mix.")
        return
    if got["settings"] != {"mix": dict(mix), "proc": dict(proc), "jam": jam_ratio}:
        st.warning("The mix or cure settings have changed since this search ran. "
                   "Scores below are for the settings it ran under.")

    best = rows[0]
    nf = sum(1 for r in rows if r["feasible"])
    st.markdown("#### Best design found")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Broken rules", int(best["n_fail"]),
              help="Ranked on this FIRST, then on score. A design that breaks a hard "
                   "rule cannot be cast at all, so no score should buy its way past "
                   "one that can — and on score alone the ordering really does invert: "
                   "an 18 mm-walled vessel breaking two rules outscores a 27 mm one "
                   "breaking only the aperture rule.")
    m2.metric("Score", f"{best['score']:.3f}",
              help=f"5–95 % interval {best['score_lo']:.3f} – {best['score_hi']:.3f}. "
                   f"Ties on broken rules are settled by the 5th percentile first, "
                   f"because the intervals are wide and mostly driven by an assumed "
                   f"biofilm volume fraction — ranking on the median alone promotes "
                   f"whichever design has the widest interval.")
    m3.metric("Limited by", SUB_LABEL.get(best["limiting"], best["limiting"]))
    m4.metric("Cemented fraction", f"{best['cemented_fraction']:.3f}")
    st.write(f"**{nf} of {len(rows)} feasible.** "
             + ("" if best["feasible"] else
                f"**The best design still fails:** {best['failed']}. "))

    keys = list(GEOM_SPACE[typology]) + list(GEOM_CHOICES[typology])
    st.dataframe(pd.DataFrame([{"parameter": k, "value": (
        round(best[k], 2) if isinstance(best[k], float) else best[k])} for k in keys]),
        hide_index=True, width="stretch")

    b1, b2 = st.columns([1, 2])
    if b1.button("⬅ Use this design", type="primary", width="stretch"):
        vals = {}
        for k in keys:
            v = best[k]
            if k in GEOM_SPACE[typology]:
                s = GEOM_SPACE[typology][k]
                # snap to the slider's own step, or Streamlit rejects the value
                v = float(np.clip(round(float(v) / s["step"]) * s["step"],
                                  s["lo"], s["hi"]))
            vals[k] = v
        # staged, not written — `geom_controls` applies it on the next run, before the
        # widgets exist. See its docstring.
        st.session_state["geom_pending"] = {"typology": typology, "values": vals}
        st.session_state.pop("mould_record", None)   # it described the old shape
        st.rerun()
    b2.caption("Writes these values into the **Design** tab's sliders. The **Mould** "
               "tab then generates the pattern, the former and the jacket for it.")

    st.markdown("#### Every candidate scored")
    df = pd.DataFrame(rows)
    show = keys + ["n_fail", "score", "score_lo", "score_hi", "feasible", "limiting",
                   "section_mm", "volume_cm3", "failed"]
    st.dataframe(df[show].round(3), hide_index=True, width="stretch", height=340)
    st.download_button("Download results (CSV)", df.to_csv(index=False),
                       file_name=f"search_{typology}.csv", mime="text/csv",
                       on_click="ignore")
    fe = df[df.feasible]
    if len(fe) and (fe.score_hi - fe.score_lo).median() > 0.4:
        st.caption("Median interval among feasible designs exceeds 0.4 — designs whose "
                   "intervals overlap should be treated as tied, not ranked.")


# ----------------------------------------------------------------------------- main
#: What a generated mould was generated FROM. Held beside the result so a stale
#: result announces itself instead of being read as the current one — the sliders
#: above it keep moving after the solve, and a mould that no longer matches them is
#: the single most misleading thing this tab could show.
def _mould_settings(typology, kind, geom_kw, mix, proc, skin_t, deflect) -> dict:
    return {"typology": typology, "kind": kind, "geom": dict(geom_kw),
            "mix": dict(mix), "proc": dict(proc),
            "skin_t": float(skin_t), "deflect": float(deflect)}


#: The two-stage chain, stated as a procedure rather than left to be inferred from a
#: list of part names. Someone opening a zip of eight STLs cannot tell that three of
#: them exist only to manufacture a fourth.
PROCEDURE_MD = """\
**The silicone path is two castings, not one.** You print a former, cast the rubber
mould in it, and cast the bio-concrete in the rubber.

1. **Print** `pattern` — a solid positive of the body you designed. This is a
   sacrificial master, not a part of the mould.
2. **Print** `pour_shell_lower` + `pour_shell_upper`, seat the pattern inside on the
   window pillars, clamp shut on the tongue-and-groove rim, and pour
   **{silicone_mass_g:.0f} g** of silicone in through the spout. The pillars hold the
   pattern at the {skin_t_mm:.0f} mm skin offset *and* form the breather windows, so
   the skin demoulds already perforated. — *{pour_procedure}*
3. **Demould the skin.** That cured rubber is the mould the bio-concrete is cast in.
4. **Print** `jacket_lower` + `jacket_upper` and clamp the skin inside them — silicone
   at ~1.1 MPa cannot hold a section against mix pressure. Keep the jacket's windows
   over the skin's: a window with jacket behind it is not a window.
5. **Cast the mix**, cure open-faced, then take the jacket off the skin *first* and
   peel the skin off the cast. The reverse tears the skin.
"""

#: Per-part one-liners. Keyed on the part name so a part that gains a role but no
#: explanation shows an empty note rather than a wrong one.
PART_NOTE = {
    "pattern": "the sacrificial master — print it, pour silicone around it, discard",
    "pour_shell_lower": "the former you pour silicone into (lower half)",
    "pour_shell_upper": "the former you pour silicone into (upper half)",
    "parting_plate": "the wall between the two skin halves; lift it out over the "
                     "pattern before opening the lower half",
    "jacket_lower": "carries mix pressure; draft lives here, not on the cast",
    "jacket_upper": "carries mix pressure; draft lives here, not on the cast",
    "skin": "THE MOULD — cast this in rubber, do not print it",
    "skin_lower": "THE MOULD, lower half — cast this in rubber, do not print it",
    "skin_upper": "THE MOULD, upper half — cast this in rubber, do not print it",
    "skin_core_lining": "rubber lining of the hollow core; squeezed out through the "
                        "cast's aperture, not peeled",
    "core": "rigid backing behind the core lining",
    "lower": "mould half; cure it open-faced",
    "upper": "mould half; cure it open-faced",
    "core_lo": "loose core, withdrawn from the lower half",
    "core_up": "loose core, withdrawn from the upper half",
}


def tab_mould(typology, mix, proc, jam_ratio):
    """Generate a printable mould for the current design and show its verification.

    Two mould types, and the choice is a transport decision before it is a
    fabrication one — which is why the aeration comparison is shown before the
    part list rather than buried under it.
    """
    st.subheader("Mould generation")
    st.caption(
        "Everything is derived from the design's own field: parting plane from the "
        "measured re-entrant volume, mould wall from the plate deflection target, "
        "flange from what it must house, cores from the kinematic release condition. "
        "Nothing is hand-tuned per typology.")

    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        kind = st.radio(
            "Mould type", ["rigid", "silicone"], horizontal=True,
            format_func=lambda s: {"rigid": "Rigid (printed FDM)",
                                   "silicone": "Silicone skin + jacket"}[s],
            help="A silicone skin releases by stretching, so the cast object needs "
                 "no draft relief. It does NOT breathe — see the aeration panel.")
    with c2:
        deflect = st.slider("Wall deflection target (mm)", 0.02, 0.30, 0.10, 0.01,
                            help="A mould that bulges casts an out-of-tolerance "
                                 "section, and section is what the drying and "
                                 "oxygen models are most sensitive to.")
    with c3:
        skin_t = st.slider("Silicone skin (mm)", 3.0, 12.0, 6.0, 0.5,
                           disabled=(kind != "silicone"),
                           help="Thicker is more durable but a worse vapour "
                                "barrier, and silicone is the cost driver.")

    st.caption(
        "A mould solve is heavier than a design score. Measured on two cores (what a "
        "basic hosted container gives you): **rigid ~15 s, silicone ~2-3 min**, "
        "because the silicone path runs an oxygen field solve per boundary condition "
        "and again per candidate window pitch. It is faster on a workstation. Meshing "
        "every part adds ~10-30 s and runs in the same click, so the STL download is "
        "ready as soon as the numbers are.")

    # Fall back to the grammar defaults rather than another typology's parameters:
    # geom_controls only runs when the Design tab renders, and passing a shell's
    # `wall` to a tile would either raise or silently build the wrong body.
    geom_kw = (st.session_state.get("geom_kw", {})
               if st.session_state.get("geom_kw_typology") == typology else {})

    # WHY THE RESULT IS STORED RATHER THAN RENDERED INLINE.
    #
    # Streamlit re-runs the entire script on every widget interaction, and a button
    # reads True only on the run that follows its own click. The previous version
    # guarded this whole tab with `if not st.button("Generate"): return`, so the FIRST
    # thing a user did after generating — clicking "Prepare printable STLs", or the
    # download button itself — re-ran the script with the generate button False and
    # returned before reaching any of it. The tab emptied and no STL could ever be
    # obtained. Everything below therefore renders from `st.session_state`, and the
    # button only writes to it.
    want = _mould_settings(typology, kind, geom_kw, mix, proc, skin_t, deflect)
    if st.button(f"Generate {kind} mould", type="primary"):
        if not geom_kw:
            st.info("Using the grammar's default parameters — open the **Design** tab "
                    "first to mould the shape you have dialled in.")
        box = st.empty()
        with st.spinner("Generating, verifying and meshing every part…"):
            rec = E.mould_record(typology, geom_kw, mix, proc, kind=kind,
                                 skin_t=skin_t, deflect_target_mm=deflect,
                                 progress=lambda m: box.caption(m))
        box.empty()
        rec["settings"] = want
        st.session_state["mould_record"] = rec

    rec = st.session_state.get("mould_record")
    if rec is None:
        st.info("Nothing is computed until you press **Generate** — a mould solve is "
                "too heavy to run on every slider move.")
        return
    if rec["settings"] != want:
        st.warning(
            f"Showing the **{rec['settings']['kind']} {rec['settings']['typology']}** "
            f"mould generated earlier. The settings above have changed since — press "
            f"**Generate {kind} mould** to rebuild for the current ones. The download "
            f"below is still the older mould.")
    kind, typology = rec["kind"], rec["typology"]
    s = rec["summary"]
    if s["pitch_coarsened"]:
        st.info(
            f"Generated at **{s['pitch_mm']:.2f} mm voxel pitch** — {s['pitch_reason']}. "
            "The silicone path holds several dozen full-grid arrays plus the oxygen "
            "solve, and the grammar pitch exceeds the memory a hosted container has. "
            "Coverage is nearly pitch-independent here (the tile reads 0.885 at 2.0 mm "
            "against 0.879 at 3.0 mm), but read window diameter and skin thickness as "
            "resolved to about this pitch. For final STLs regenerate locally at the "
            "grammar pitch with `examples/regenerate_moulds.py`.")

    # ---- aeration first: it decides whether the mould can cement at all -----
    st.markdown("#### Will it cement in the mould?")
    if kind == "silicone":
        st.warning(
            f"**A silicone face is a no-flux boundary, not a breathable one.** The "
            f"{s['skin_t_mm']:.0f} mm skin carries "
            f"**{s['R_skin_over_R_drained_wall']:.0f}x** the oxygen resistance of the "
            f"drained pore network behind it, and passes "
            f"{s['wvtr_frac_of_free_evap']*100:.2f} % of the free evaporation rate — "
            f"which collapses the drained depth to "
            f"{s['L_dry_behind_skin_mm']:.2f} mm. Against *saturated* pores the same "
            f"skin is transparent (ratio {s['R_skin_over_R_saturated_wall']:.4f}), "
            f"which is why the permeability intuition misleads: PDMS beats water and "
            f"loses to air. Silicone buys undercut tolerance and zero-draft release.")
        a = pd.DataFrame([
            {"boundary condition": "fully enclosed skin", "open area": 0.0,
             "cemented fraction": s["cemented_frac_enclosed"]},
            {"boundary condition": f"windowed ({s['window_spacing_mm']:.0f} mm pitch)",
             "open area": s["open_area_frac"],
             "cemented fraction": s["cemented_frac_windowed"]},
            {"boundary condition": "rigid split mould, parting face open",
             "open area": None,
             "cemented fraction": s["cemented_frac_rigid_baseline"]},
        ])
        st.dataframe(a, hide_index=True, width="stretch",
                     column_config={
                         "open area": st.column_config.NumberColumn(format="%.3f"),
                         "cemented fraction": st.column_config.NumberColumn(
                             format="%.3f")})
        if s["meets_coverage"]:
            st.success(
                f"Windowed skin reaches {s['cemented_frac_windowed']:.3f} at "
                f"{s['open_area_frac']*100:.1f} % open area, clearing the 0.85 "
                f"criterion. Read it as *at* the criterion, not comfortably above: "
                f"the oxygen demand term alone spans a factor of ~26.")
        else:
            st.error(
                f"Windowed skin reaches only {s['cemented_frac_windowed']:.3f}, below "
                f"the 0.85 criterion, and the limit is **{s['coverage_limited_by']}**. "
                f"More open area will not fix a drying limit — change the cure "
                f"(lower RH is the strongest lever) or cast open-faced.")
    else:
        a = rec["aeration"]
        st.dataframe(pd.DataFrame([
            {"boundary condition": "demoulded body (reference)",
             "cemented fraction": a["demoulded"]["cemented_fraction"]},
            {"boundary condition": "closed rigid mould (all faces sealed)",
             "cemented fraction": a["enclosed"]["cemented_fraction"]},
            {"boundary condition": "split mould, parting face open",
             "cemented fraction": a["open_faces_only"]["cemented_fraction"]},
        ]), hide_index=True, width="stretch",
            column_config={"cemented fraction":
                           st.column_config.NumberColumn(format="%.3f")})
        st.caption(
            f"A rigid mould face is no-flux too. Curing the halves **open-faced** is "
            f"what makes the difference — assembling them early converts the parting "
            f"face from an oxygen source into a sealed interface and reproduces the "
            f"source paper's solid-cast failure. Drained depth L_dry = "
            f"{a['L_dry_mm']:.1f} mm at "
            f"{rec['settings']['proc']['cure_days']:.0f} d / "
            f"{rec['settings']['proc']['rh_pct']:.0f} % RH — the cure this mould was "
            f"generated for. These are drained-depth values, not the field solve.")

    # ---- decisions and verification ----------------------------------------
    st.markdown("#### What the generator decided")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Parting plane", f"{s['parting_coord_mm']:.1f} mm",
              help=s.get("parting_reason", ""))
    d2.metric("Mould wall", f"{s['mould_wall_mm']:.0f} mm" if kind == "rigid"
              else f"{s['jacket_wall_mm']:.0f} mm (jacket)")
    if kind == "rigid":
        d3.metric("Draft", f"{s['draft_requested_deg']:.2f}°",
                  help=f"relief {s['relief_per_face_mm']:.2f} mm per face")
        d4.metric("Cores", s["core_strategy"].replace("_", " "),
                  help=s["core_reason"])
    else:
        d3.metric("Silicone", f"{s['silicone_mass_g']:.0f} g",
                  help=f"{s['silicone_volume_mm3']/1000:.0f} cm3 at the spec density")
        d4.metric("Web returned", f"{s['web_returned_mm']:.1f} mm",
                  help="section the zero-draft skin gives back to the designer; "
                       "useful only if it stays inside the drying ceiling")

    checks = [("volume balance closes", s["balance_exact"],
               f"unattributed {s['unattributed_mm3']:.1f} mm3"),
              ("aperture classes pass", s["apertures_pass"],
               "aggregate >= 6 x d_max, drains <= 3 x d_max")]
    if kind == "silicone":
        checks += [("release order (jacket first)", s["release_order_ok"],
                    "and the control does interfere: "
                    f"{s['release_control_interferes']}"),
                   ("undercut strain within allowable", s["undercuts_ok"],
                    f"worst {s['worst_undercut_strain_pct']:.1f} % vs "
                    f"{s['allowable_strain_pct']:.1f} % allowable"),
                   ("jacket stiff enough perforated", s["jacket_adequate"], ""),
                   ("former: spout and vent reach every cavity",
                    s["pour_shell_reaches_cavity"],
                    f"measured intersection per cavity body, and no material taken "
                    f"from the far half — spout {s['spout_d_mm']:.0f} mm, vent "
                    f"{s['vent_d_mm']:.0f} mm"),
                   ("former: casts the skin that was SCORED",
                    s["cavity_matches_skin"]["ok"],
                    f"cavity {s['cavity_matches_skin']['cavity_outer_mm3']/1000:.0f} "
                    f"vs windowed skin "
                    f"{s['cavity_matches_skin']['skin_out_mm3']/1000:.0f} cm3 "
                    f"(ratio {s['cavity_matches_skin']['ratio']:.3f})"),
                   ("former: window pillars bridge wall to pattern",
                    s["pillars"]["ok"],
                    f"{s['pillars']['formed_vol_frac']*100:.0f} % of bore volume "
                    f"forms a pillar ({s['pillars']['n']} of "
                    f"{s['pillars']['n_raw']} bodies); halves in "
                    f"{s['pillars']['bodies_lower']}/{s['pillars']['bodies_upper']} "
                    f"pieces"),
                   ("former: both halves open off the cured skin",
                    s["pour_shell_release_ok"],
                    "straight-pull sweep against the as-cast skin, and the control "
                    "against an unperforated one does interfere: "
                    f"{s['pour_shell_control_interferes']}"),
                   ("former: wall meets the deflection target",
                    s["pour_shell_wall_ok"],
                    f"{s['pour_shell_wall_mm']:.0f} mm, "
                    f"{s['pour_shell_deflection_mm']:.3f} mm under the silicone head"),
                   ("former: volume balance closes", s["pour_shell_balance_exact"],
                    "")]
    else:
        checks += [("mould wall meets deflection target", s["wall_meets_target"],
                    f"{s['wall_deflection_mm']:.3f} mm at the design pressure"),
                   ("keys mate one way only", s["keys_chiral"],
                    "tested against the flange's measured symmetry group")]
    st.dataframe(pd.DataFrame([{"check": c, "pass": bool(p), "detail": n}
                               for c, p, n in checks]),
                 hide_index=True, width="stretch")

    st.markdown("#### Parts and downloads")
    if not rec["manufacturable"]:
        st.error(
            "**This former has not passed its own manufacturability checks**, so the "
            "files below describe an assembly the generator has measured as unbuildable"
            ":\n\n" + "\n".join(f"- {b}" for b in rec["blockers"]) +
            "\n\nThe download is still offered — the parts are real geometry and you "
            "may want them anyway — but printing the pattern, the former and the "
            "jacket is many hours and several hundred grams of filament.")
    if kind == "silicone":
        st.markdown(PROCEDURE_MD.format(**s))
        pil = s["pillars"]
        if pil["open_area_formed_frac"] < 0.98:
            st.warning(
                f"**The former casts {pil['open_area_formed_frac']*100:.0f} % of the "
                f"designed window area.** Only bores running along the draw can be "
                f"formed — a peg across the pull shears through the cured rubber when "
                f"the halves open — so the transverse windows, and the fill gate, have "
                f"to be punched by hand to `skin*.stl`. **The cemented fraction above "
                f"was solved on the full window set**, so treat it as the figure for a "
                f"skin whose remaining windows have been punched, not for one straight "
                f"out of the former."
                + (" The hollow core's lining is cast unperforated for the same "
                   "reason — a pillar inside the silhouette has no former wall to "
                   "attach to — and its windows are hand-punched too."
                   if s["cavity_matches_skin"]["core_lining_windows_unformed"] else ""))
    st.dataframe(pd.DataFrame(
        [{"part": p["part"],
          "you": "3D print" if p["role"] == "print" else "cast in silicone",
          "note": PART_NOTE.get(p["part"],
                                "the flexible part — do NOT print this"
                                if p["role"] == "cast_silicone" else "")}
         for p in rec["parts"]]),
        hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"⬇ Download all {rec['n_files']} STLs "
            f"({len(rec['zip'])/1e6:.1f} MB zip)", rec["zip"],
            file_name=rec["zip_name"], mime="application/zip", type="primary",
            width="stretch", on_click="ignore")
        st.caption(
            f"**{rec['n_print']} files to print** in `{E.ROLE_DIR['print']}/`"
            + (f", **{rec['n_cast']} to cast in silicone** in "
               f"`{E.ROLE_DIR['cast_silicone']}/`" if rec["n_cast"] else "")
            + ". MANIFEST.txt carries the fabrication order, the open-faced cure "
              "requirement and the disassembly order.")
    with c2:
        st.download_button("Decisions + verification (JSON)",
                           lambda: json.dumps(s, indent=1, default=str),
                           file_name=rec["zip_name"].replace(".zip", ".json"),
                           mime="application/json", width="stretch",
                           on_click="ignore")
        st.caption(
            "Every number on this page, at full precision. The whole set for all "
            "three typologies is `PYTHONPATH=. python examples/regenerate_moulds.py`.")
    st.dataframe(pd.DataFrame(rec["manifest"])[
        ["file", "role", "volume_cm3", "watertight", "bbox_mm"]],
        hide_index=True, width="stretch")


def main():
    st.title("Bio-concrete design studio")
    st.caption(
        "Generate cast geometries for Bacillus subtilis bio-cementation of construction "
        "waste, and score how likely each is to solidify completely. Physics and rules "
        "come from the validated `biocast` engine — this interface adds no assumptions "
        "of its own.")

    mix, proc, jam_ratio, n_mc = sidebar_process()

    w = E.feasible_window(mix["d_max"], proc["cure_days"], proc["rh_pct"], jam_ratio)
    if not w["open"]:
        t_open = E.min_cure_to_open(mix["d_max"], proc["rh_pct"], jam_ratio)
        sieve = E.max_dmax_to_open(proc["cure_days"], proc["rh_pct"], jam_ratio)
        msg = (f"**No feasible section thickness at these settings.** Castability needs "
               f"≥ {w['floor_mm']:.0f} mm but drying only drains to "
               f"{w['ceiling_mm']:.0f} mm.")
        if t_open:
            msg += (f" Cure **{t_open:.0f} days** at {proc['rh_pct']:.0f} % RH, or "
                    f"sieve below **{sieve:.1f} mm**.")
        else:
            msg += (f" No cure up to 120 days opens it at this humidity — sieve below "
                    f"**{sieve:.1f} mm** or lower the RH.")
        st.warning(msg + "  See the **Process window** tab.")

    typology = st.radio("Typology", ["shell", "block", "tile"], horizontal=True,
                        format_func=lambda s: {"shell": "Hollow vessel",
                                               "block": "Hollow-core block",
                                               "tile": "Relief tile"}[s])

    t1, t2, t3, t4 = st.tabs(["Design", "Mould", "Process window", "Explore"])
    with t1:
        tab_design(typology, mix, proc, jam_ratio, n_mc)
    with t2:
        tab_mould(typology, mix, proc, jam_ratio)
    with t3:
        tab_process(mix, proc, jam_ratio)
    with t4:
        tab_explore(typology, mix, proc, jam_ratio)

    st.markdown("---")
    st.caption(
        "The score ranks designs; it is not a calibrated probability — no pass/fail "
        "dataset was available to fit it. Intervals are wide, driven mainly by biofilm "
        "volume fraction (0.01-0.10, assumed). See methods_report.md for provenance "
        "and the full list of limits.")


if __name__ == "__main__":
    main()
