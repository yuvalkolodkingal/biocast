"""Bio-concrete design studio — interactive shape generator with live scoring.

Run:  PYTHONPATH=. streamlit run biocast/gui/app.py

Three tabs:
  Design    move sliders, see the mesh in 3D, the four subscores, and every
            constraint verdict update live; download STL.
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


def geom_controls(typology: str, d_max: float) -> dict:
    """Typology-specific sliders. Floors that depend on d_max are shown inline."""
    fillet_floor = 1.5 * d_max
    if typology == "shell":
        st.markdown("**Hollow ovoid vessel** — the typology that succeeded in the paper")
        c1, c2, c3 = st.columns(3)
        with c1:
            a = st.slider("Semi-axis a (mm)", 30.0, 100.0, 55.0, 1.0)
            b = st.slider("Semi-axis b (mm)", 30.0, 100.0, 55.0, 1.0)
            c = st.slider("Semi-axis c, long (mm)", 40.0, 150.0, 78.0, 1.0)
        with c2:
            n = st.slider("Superellipsoid exponent n", 2.0, 4.0, 2.4, 0.1,
                          help="2 = ellipsoid, higher = boxier")
            ovoid = st.slider("Egg taper", 0.0, 0.45, 0.28, 0.01)
            wall = st.slider("Wall thickness (mm)", 6.0, 45.0, 22.0, 0.5,
                             help=f"Castability floor here is {6*d_max:.0f} mm "
                                  f"at d_max = {d_max:g} mm")
        with c3:
            aperture_r = st.slider("Aperture radius (mm)", 0.0, 35.0, 16.0, 0.5,
                                   help="This is a feed passage, so the bore diameter "
                                        "must also clear the jamming limit")
            fillet_r = st.slider("Fillet radius (mm)", 2.0, 20.0,
                                 max(8.0, fillet_floor), 0.5,
                                 help=f"Your rule: >= 1.5 x d_max = {fillet_floor:.1f} mm")
        return dict(a=a, b=b, c=c, n=n, ovoid=ovoid, wall=wall,
                    aperture_r=aperture_r, fillet_r=fillet_r)

    if typology == "block":
        st.markdown("**Hollow-core masonry unit** — CMU module, 390 x 190 x 190 mm")
        c1, c2, c3 = st.columns(3)
        with c1:
            face_shell = st.slider("Face shell (mm)", 20.0, 55.0, 34.0, 1.0,
                                   help="ASTM C90 minimum is 32 mm for 8 in units")
            web = st.slider("Web (mm)", 15.0, 45.0, 30.0, 1.0,
                            help="C90 permits 19 mm; your notes say 25 mm")
        with c2:
            n_cores = st.selectbox("Cores", [2, 3], 0)
            core_taper = st.slider("Core draft (deg)", 0.0, 5.0, 2.0, 0.1,
                                   help="Needed for release of a fragile green body")
        with c3:
            fillet_r = st.slider("Fillet radius (mm)", 2.0, 20.0,
                                 max(8.0, fillet_floor), 0.5,
                                 help=f"Your rule: >= {fillet_floor:.1f} mm")
            groove_depth = st.slider("Face groove depth (mm)", 0.0, 20.0, 0.0, 0.5)
            groove_width = st.slider("Face groove width (mm)", 0.0, 50.0, 0.0, 1.0)
        return dict(face_shell=face_shell, web=web, n_cores=int(n_cores),
                    core_taper=core_taper, fillet_r=fillet_r,
                    groove_depth=groove_depth, groove_width=groove_width,
                    groove_count=2 if groove_depth > 0 else 0)

    st.markdown("**Relief paving tile** — Panot type, 200 x 200 mm")
    c1, c2, c3 = st.columns(3)
    with c1:
        t = st.slider("Thickness (mm)", 20.0, 60.0, 40.0, 1.0)
        pattern = st.selectbox("Pattern", ["grid", "diagonal", "flower", "radial"])
    with c2:
        groove_depth = st.slider("Relief depth (mm)", 1.0, 10.0, 3.0, 0.5,
                                 help="Panot practice is 2-3 mm; the regulated ceiling "
                                      "is 5 mm")
        groove_width = st.slider("Channel width (mm)", 5.0, 45.0, 24.0, 1.0,
                                 help=f"Jamming floor is {6*d_max:.0f} mm at "
                                      f"d_max = {d_max:g} mm")
        groove_pitch = st.slider("Channel pitch (mm)", 25.0, 90.0, 50.0, 1.0)
    with c3:
        fillet_r = st.slider("Fillet radius (mm)", 2.0, 18.0,
                             max(8.0, fillet_floor), 0.5)
        joint = st.slider("Joint gap (mm)", 2.0, 15.0, 6.0, 0.5)
    return dict(t=t, pattern=pattern, groove_depth=groove_depth,
                groove_width=groove_width, groove_pitch=groove_pitch,
                fillet_r=fillet_r, joint=joint, thick_tile=t >= 35.0)


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
    """Mid-plane oxygen field — where the design does and does not cement."""
    import matplotlib.pyplot as plt
    ox = r.get("_oxygen")
    if ox is None:
        return None
    C, occ, pitch = ox["C"], ox["occ"], ox["pitch"]
    j = occ.shape[1] // 2
    sl_occ = occ[:, j, :].T
    sl_C = np.where(sl_occ, C[:, j, :].T, np.nan)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
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
    """The feasible window: castability floor against drying ceiling."""
    import matplotlib.pyplot as plt
    dmx = np.linspace(1.0, 8.0, 200)
    floor = jam_ratio * dmx
    w_now = E.feasible_window(mix["d_max"], proc["cure_days"], proc["rh_pct"], jam_ratio)
    ceiling = w_now["ceiling_mm"]

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
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
        r = E.evaluate(typology, geom_kw, mix, proc, jam_ratio=jam_ratio,
                       n_mc=n_mc, with_field=show_field)

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
    stl = io.BytesIO(); mesh.export(stl, file_type="stl"); stl.seek(0)
    tag = f"{typology}_dmax{mix['d_max']:g}_{proc['cure_days']:.0f}d{proc['rh_pct']:.0f}rh_score{r['score']:.3f}"
    c1.download_button("Download STL", stl, file_name=f"{tag}.stl",
                       mime="model/stl", width="stretch")
    rec = {k: v for k, v in r.items() if not k.startswith("_")}
    rec.update({"geom": geom_kw, "mix": mix, "proc": proc})
    c2.download_button("Download report (JSON)",
                       json.dumps(rec, indent=1, default=str),
                       file_name=f"{tag}.json", mime="application/json",
                       width="stretch")
    if typology == "shell" and proc["split_mould"]:
        from biocast.grammars import shell as sh
        try:
            lo, up = sh.split_halves(mesh)
            buf = io.BytesIO(); lo.export(buf, file_type="stl"); buf.seek(0)
            c3.download_button("Download lower half", buf,
                               file_name=f"{tag}_lower.stl", mime="model/stl",
                               width="stretch")
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
    st.markdown(
        "Randomised search inside the current mix and cure settings. Each candidate "
        "is meshed, diagnosed and scored with the same engine as the Design tab.")
    c1, c2 = st.columns([1, 3])
    n = c1.select_slider("Candidates", [12, 24, 48, 96], 24)
    seed = c1.number_input("Seed", 0, 9999, 0, 1)
    go = c1.button("Search", type="primary", width="stretch")
    if not go:
        st.info("Set a count and press **Search**. 24 candidates take roughly half a minute.")
        return

    rng = np.random.default_rng(int(seed))
    rows, prog = [], st.progress(0.0, text="scoring…")
    for i in range(int(n)):
        if typology == "shell":
            g = dict(a=rng.uniform(40, 90), b=rng.uniform(40, 90), c=rng.uniform(60, 140),
                     n=rng.uniform(2.0, 4.0), ovoid=rng.uniform(0, 0.45),
                     wall=rng.uniform(8, 42), aperture_r=rng.uniform(0, 30),
                     fillet_r=rng.uniform(max(4.0, 1.5*mix["d_max"]), 16))
        elif typology == "block":
            g = dict(face_shell=rng.uniform(25, 50), web=rng.uniform(19, 42),
                     n_cores=int(rng.choice([2, 3])),
                     fillet_r=rng.uniform(max(4.0, 1.5*mix["d_max"]), 16),
                     core_taper=rng.uniform(0.5, 4.0))
        else:
            g = dict(t=rng.uniform(25, 60), pattern=str(rng.choice(
                        ["grid", "diagonal", "flower", "radial"])),
                     groove_depth=rng.uniform(1.5, 8), groove_width=rng.uniform(8, 42),
                     groove_pitch=rng.uniform(30, 80),
                     fillet_r=rng.uniform(max(4.0, 1.5*mix["d_max"]), 14),
                     joint=rng.uniform(3, 12), thick_tile=True)
        try:
            r = E.evaluate(typology, g, mix, proc, jam_ratio=jam_ratio, n_mc=150)
        except Exception:
            continue
        rows.append({**{k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in g.items()},
                     "score": round(r["score"], 3),
                     "lo": round(r["score_lo"], 3), "hi": round(r["score_hi"], 3),
                     "feasible": r["feasible"],
                     "limiting": r["dominant_failure_mode"],
                     "section_mm": round(r["min_section_measured_mm"], 1),
                     "vol_cm3": round(r["volume_mm3"] / 1000, 1),
                     "failed": ", ".join(r["failed_rules"])})
        prog.progress((i + 1) / int(n), text=f"scored {i+1}/{int(n)}")
    prog.empty()

    if not rows:
        st.warning("No candidate built successfully. Try a different typology or mix.")
        return
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    nf = int(df.feasible.sum())
    st.write(f"**{nf} of {len(df)} feasible.** Best score {df.score.max():.3f}. "
             f"Most common limiter: **{df.limiting.mode()[0]}**.")
    st.dataframe(df, hide_index=True, width="stretch", height=380)
    st.download_button("Download results (CSV)", df.to_csv(index=False),
                       file_name=f"explore_{typology}.csv", mime="text/csv")
    if nf and (df[df.feasible].hi - df[df.feasible].lo).median() > 0.4:
        st.caption("Median interval among feasible designs exceeds 0.4 — designs whose "
                   "intervals overlap should be treated as tied, not ranked.")


# ----------------------------------------------------------------------------- main
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
        "and again per candidate window pitch. It is faster on a workstation.")
    if not st.button(f"Generate {kind} mould", type="primary"):
        return

    # Fall back to the grammar defaults rather than another typology's parameters:
    # geom_controls only runs when the Design tab renders, and passing a shell's
    # `wall` to a tile would either raise or silently build the wrong body.
    geom_kw = (st.session_state.get("geom_kw", {})
               if st.session_state.get("geom_kw_typology") == typology else {})
    if not geom_kw:
        st.info("Using the grammar's default parameters — open the **Design** tab "
                "first to mould the shape you have dialled in.")

    with st.spinner("Generating and verifying every part…"):
        res = E.generate_mould(typology, geom_kw, mix, proc, kind=kind,
                               skin_t=skin_t, deflect_target_mm=deflect)
    s = res["summary"]
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
        st.dataframe(a, hide_index=True, use_container_width=True,
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
        a = E.mould_aeration(proc, mould_res=res)
        st.dataframe(pd.DataFrame([
            {"boundary condition": "demoulded body (reference)",
             "cemented fraction": a["demoulded"]["cemented_fraction"]},
            {"boundary condition": "closed rigid mould (all faces sealed)",
             "cemented fraction": a["enclosed"]["cemented_fraction"]},
            {"boundary condition": "split mould, parting face open",
             "cemented fraction": a["open_faces_only"]["cemented_fraction"]},
        ]), hide_index=True, use_container_width=True,
            column_config={"cemented fraction":
                           st.column_config.NumberColumn(format="%.3f")})
        st.caption(
            f"A rigid mould face is no-flux too. Curing the halves **open-faced** is "
            f"what makes the difference — assembling them early converts the parting "
            f"face from an oxygen source into a sealed interface and reproduces the "
            f"source paper's solid-cast failure. Drained depth L_dry = "
            f"{a['L_dry_mm']:.1f} mm at {proc['cure_days']:.0f} d / "
            f"{proc['rh_pct']:.0f} % RH. These are drained-depth values, not the "
            f"field solve.")

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
                   ("pour shell can be filled", s["pour_shell_pourable"], "")]
    else:
        checks += [("mould wall meets deflection target", s["wall_meets_target"],
                    f"{s['wall_deflection_mm']:.3f} mm at the design pressure"),
                   ("keys mate one way only", s["keys_chiral"],
                    "tested against the flange's measured symmetry group")]
    st.dataframe(pd.DataFrame([{"check": c, "pass": bool(p), "detail": n}
                               for c, p, n in checks]),
                 hide_index=True, use_container_width=True)

    st.markdown("#### Parts and downloads")
    roles = {p: E.PART_ROLE.get(p, "print") for p in s["parts"]}
    n_print = sum(1 for r in roles.values() if r == "print")
    st.dataframe(pd.DataFrame(
        [{"part": p,
          "you": "3D print" if r == "print" else "cast in silicone",
          "note": ("the former you pour silicone into" if p.startswith("pour_shell")
                   else "carries mix pressure; draft lives here"
                   if p.startswith("jacket")
                   else "the flexible part — do NOT print this"
                   if r == "cast_silicone" else "")}
         for p, r in roles.items()]),
        hide_index=True, use_container_width=True)

    if kind == "silicone":
        st.caption(
            "Two families are printed and one is cast. **The skin is not a printable "
            "part** — a rigid copy of it fits the jacket and releases nothing. Print "
            "the pour shell, cast the skin in it, print the jacket.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"Prepare printable STLs ({n_print} parts)"):
            with st.spinner("Meshing parts — marching cubes on every one…"):
                blob, man = E.mould_stl_bundle(res, prefix=f"{kind}_{typology}",
                                              roles=("print",))
            st.session_state[f"stl_{kind}_{typology}"] = (blob, man)
        key = f"stl_{kind}_{typology}"
        if key in st.session_state:
            blob, man = st.session_state[key]
            st.download_button(
                f"Download printable STLs ({len(blob)/1e6:.0f} MB)", blob,
                file_name=f"mould_{kind}_{typology}_printable.zip",
                mime="application/zip", type="primary")
            st.caption("Includes MANIFEST.txt with the fabrication order, the "
                       "open-faced cure requirement, and the disassembly order.")
            st.dataframe(pd.DataFrame(man)[["file","volume_cm3","bbox_mm"]],
                         hide_index=True, use_container_width=True)
    with c2:
        if kind == "silicone" and st.button("Prepare silicone parts too (reference)"):
            with st.spinner("Meshing the cast parts…"):
                blob2, man2 = E.mould_stl_bundle(res, prefix=f"{kind}_{typology}",
                                                roles=("print", "cast_silicone"))
            st.download_button(
                f"Download all parts ({len(blob2)/1e6:.0f} MB)", blob2,
                file_name=f"mould_{kind}_{typology}_all.zip", mime="application/zip")
            st.caption("The skin meshes are for checking fit and estimating rubber "
                       "volume — not for printing.")
        st.download_button("Decisions + verification (JSON)",
                           json.dumps(_jsonable(s), indent=1),
                           file_name=f"mould_{kind}_{typology}.json",
                           mime="application/json")
    st.caption(
        "Meshing is deferred to this button because it is the expensive step and "
        "most sessions only want the numbers. The full set for all typologies is "
        "`PYTHONPATH=. python examples/regenerate_moulds.py`.")


def _jsonable(o):
    """Strip numpy scalars and arrays so the summary serialises."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return f"<array shape={o.shape}>"
    return o


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
