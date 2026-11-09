"""Streamlit interface for the urban-problem FCM.  Run:  streamlit run app.py"""
from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import fcm_model as m
from context import exposure_from_context, fetch_osm

st.set_page_config(page_title="Urban Problem Prioritisation — Bern", layout="wide")

BERN = (46.9480, 7.4474)
LEVEL_COLORS = {"Very high": "#b2182b", "High": "#ef8a62", "Medium": "#e0b000",
                "Low": "#67a9cf", "Very low": "#2166ac"}
TERM_LABEL = {t: t.replace("_", " ") for t in m.INPUT_TERM_NAMES}
INPUT_HELP = {
    "Severity": "How serious is the physical or operational defect?",
    "Deterioration": "Is the problem getting worse over time?",
    "Population_Exposure": "How many people are exposed? Use 'Fetch context' to get a suggestion from OpenStreetMap.",
    "Workaround_Availability": "Is there a safe, accessible alternative (detour, other sidewalk, other bin)? High = good alternative.",
}

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_osm(lat, lon):
    return fetch_osm(lat, lon)


# ------------------------------------------------------------------ state
ss = st.session_state
if "W" not in ss:
    ss.elicitation = m.load_elicitation()
    ss.edges = m.aggregate_weights(ss.elicitation)
    ss.W = m.weight_matrix(ss.edges)
    ss.reports = [dict(r) for r in m.EXAMPLE_REPORTS]
    ss.loc = BERN
    ss.ctx = None
    ss.lam = 1.0


def W():
    return ss.W


def report_dict():
    return {r["name"]: r["terms"] for r in ss.reports}


# ------------------------------------------------------------------ helpers
def fcm_graph(final: pd.Series | None, Wm: pd.DataFrame) -> str:
    """Graphviz DOT of the map; node fill shows activation, edge colour shows sign."""
    lines = ["digraph G {", 'rankdir=LR; bgcolor="transparent";',
             'node [shape=ellipse style=filled fontname="Helvetica" fontsize=11];',
             'edge [fontname="Helvetica" fontsize=9];']
    for c in m.CONCEPTS:
        if final is not None:
            v = float(final[c])
            r, g, b = (int(255 - v * (255 - 178)), int(255 - v * (255 - 24)), int(255 - v * (255 - 43)))
            fill = f"#{r:02x}{g:02x}{b:02x}"
            font = "white" if v > 0.6 else "black"
            label = f"{m.LABELS[c]}\\n{v:.2f}"
        else:
            fill, font, label = "#eeeeee", "black", m.LABELS[c]
        shape = "box" if c in m.INPUTS else ("doubleoctagon" if c == m.OUTPUT else "ellipse")
        lines.append(f'"{c}" [label="{label}" fillcolor="{fill}" fontcolor="{font}" shape={shape}];')
    for s in m.CONCEPTS:
        for t in m.CONCEPTS:
            w = float(Wm.loc[s, t])
            if w:
                col = "#2b8cbe" if w > 0 else "#d7301f"
                lines.append(f'"{s}" -> "{t}" [label="{w:+.2f}" color="{col}" penwidth={0.5 + 3 * abs(w):.1f}];')
    lines.append("}")
    return "\n".join(lines)


def score_card(a: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Priority score", f"{100 * a['score']:.0f} / 100")
    c2.markdown(f"**Level**<br><span style='font-size:1.6rem;color:{LEVEL_COLORS[a['label']]}'>"
                f"{a['label']}</span>", unsafe_allow_html=True)
    c3.metric("Plausible range", f"{100 * a['low']:.0f} – {100 * a['high']:.0f}",
              help="Score when each linguistic input is read at the low/high end of its fuzzy number (α-cut 0.5).")
    st.progress(min(1.0, a["score"]))


def effects_chart(terms):
    eff = m.input_effects(terms, W(), ss.lam) * 100
    df = pd.DataFrame({"Input": [m.LABELS[c] for c in eff.index], "Effect on score (points)": eff.values})
    st.bar_chart(df, x="Input", y="Effect on score (points)", horizontal=True, height=220)
    st.caption("Points this input adds (+) or removes (−) compared with assessing it as 'medium'.")


# ------------------------------------------------------------------ layout
st.title("Prioritising citizen-reported urban problems — Bern")
st.caption("Fuzzy linguistic assessment + Fuzzy Cognitive Map. Seminar of Urban Computing AS2026. "
           "Example reports are synthetic.")

tab_report, tab_queue, tab_model, tab_method = st.tabs(
    ["① New report", "② Priority queue", "③ Model & weights", "④ Method"])

# ------------------------------------------------------------------ ① new report
with tab_report:
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Where is the problem?")
        st.caption("Click on the map to place the report.")
        fmap = folium.Map(location=ss.loc, zoom_start=14, tiles="OpenStreetMap")
        folium.Marker(ss.loc, tooltip="Report location").add_to(fmap)
        folium.Circle(ss.loc, radius=200, color="#2b8cbe", fill=False, weight=1).add_to(fmap)
        out = st_folium(fmap, height=380, use_container_width=True, key="pick_map",
                        returned_objects=["last_clicked"])
        if out and out.get("last_clicked"):
            new = (round(out["last_clicked"]["lat"], 5), round(out["last_clicked"]["lng"], 5))
            if new != ss.loc:
                ss.loc, ss.ctx = new, None
                st.rerun()
        st.write(f"Location: `{ss.loc[0]:.5f}, {ss.loc[1]:.5f}`")

        if st.button("Fetch context from OpenStreetMap", help="Queries schools, hospitals, transit stops and street type around the point. Can take ~30 s."):
            with st.spinner("Querying OpenStreetMap (Overpass API)…"):
                try:
                    ctx = cached_osm(*ss.loc)
                    ss.ctx = {"raw": ctx, "exp": exposure_from_context(ctx)}
                except Exception as exc:  # network / server busy
                    ss.ctx = {"error": str(exc)}
        if ss.ctx:
            if "error" in ss.ctx:
                st.warning(f"Could not reach OpenStreetMap ({ss.ctx['error']}). Set exposure manually.")
            else:
                raw, exp = ss.ctx["raw"], ss.ctx["exp"]
                st.success(f"Suggested exposure: **{TERM_LABEL[exp['term']]}** ({exp['value']:.2f})")
                with st.expander("Why? (fuzzy rules)"):
                    st.write(f"Schools / kindergartens / hospitals / care homes within 120 m: {len(raw['vulnerable'])}")
                    for v in raw["vulnerable"]:
                        st.write(f"- {v}")
                    st.write(f"Transit stops within 200 m: {raw['n_stops']}")
                    st.write(f"Street types within 30 m: {', '.join(sorted(set(raw['roads']))) or 'none found'}")
                    st.table(pd.DataFrame(exp["rules"], columns=["Rule", "Firing strength"]))
                    st.caption("Zero-order Sugeno inference: exposure = Σ strength·output / Σ strength.")

    with right:
        st.subheader("What is the problem?")
        group = st.selectbox("Category", list(m.CATEGORIES))
        ptype = st.selectbox("Problem type", list(m.CATEGORIES[group]))
        preset = m.CATEGORIES[group][ptype]
        name = st.text_input("Short description", f"{ptype} at {ss.loc[0]:.4f}, {ss.loc[1]:.4f}")

        st.markdown("**Linguistic assessment** (pre-filled from the problem type — adjust it)")
        terms = {}
        for c in m.INPUTS:
            if c == "Population_Exposure":
                default = ss.ctx["exp"]["term"] if ss.ctx and "exp" in ss.ctx else "medium"
            else:
                default = preset[c]
            terms[c] = st.select_slider(m.LABELS[c], options=m.INPUT_TERM_NAMES, value=default,
                                        format_func=TERM_LABEL.get, help=INPUT_HELP[c],
                                        key=f"in_{c}_{group}_{ptype}_{default}")

        a = m.assess(terms, W(), ss.lam)
        st.divider()
        score_card(a)
        effects_chart(terms)
        if st.button("➕ Add to priority queue", type="primary"):
            ss.reports.append({"name": name, "group": group, "type": ptype,
                               "lat": ss.loc[0], "lon": ss.loc[1], "terms": terms})
            st.success(f"Added. Queue now has {len(ss.reports)} reports.")

    st.subheader("How the map reasons about this report")
    g1, g2 = st.columns([1.4, 1])
    g1.graphviz_chart(fcm_graph(a["final"], W()), width="stretch")
    with g2:
        st.caption(f"Converged in {a['iterations']} iterations. Boxes = inputs (clamped), "
                   "octagon = priority output. Darker = higher activation.")
        traj = a["trajectory"].rename(columns=m.LABELS)
        st.line_chart(traj, height=280)

# ------------------------------------------------------------------ ② queue
with tab_queue:
    if not ss.reports:
        st.info("No reports yet.")
    else:
        rk = m.rank_reports(report_dict(), W(), ss.lam)
        meta = pd.DataFrame(ss.reports).rename(columns={"name": "Report"})
        rk = rk.merge(meta[["Report", "group", "type", "lat", "lon"]], on="Report", how="left")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("Ranked queue")
            st.dataframe(rk[["Rank", "Report", "Score", "Range", "Level", "type"]],
                         hide_index=True, width="stretch")
            remove = st.multiselect("Remove reports", rk["Report"].tolist())
            if remove and st.button("Remove selected"):
                ss.reports = [r for r in ss.reports if r["name"] not in remove]
                st.rerun()
        with c2:
            st.subheader("Map")
            qmap = folium.Map(location=BERN, zoom_start=12, tiles="OpenStreetMap")
            for _, r in rk.iterrows():
                folium.CircleMarker((r.lat, r.lon), radius=6 + r.Score / 8, color=LEVEL_COLORS[r.Level],
                                    fill=True, fill_opacity=0.8,
                                    tooltip=f"#{r.Rank} {r.Report} — {r.Score:.0f} ({r.Level})").add_to(qmap)
            st_folium(qmap, height=420, use_container_width=True, key="queue_map", returned_objects=[])

        st.subheader("Is the ranking robust to the weights?")
        st.caption("Monte Carlo sensitivity: every non-zero weight is perturbed at random by ±noise "
                   "and the queue is re-ranked.")
        n1, n2 = st.columns(2)
        noise = n1.slider("Weight noise (±)", 0.05, 0.30, 0.10, 0.05)
        runs = n2.slider("Runs", 50, 500, 200, 50)
        if st.button("Run sensitivity analysis") and len(ss.reports) > 1:
            with st.spinner("Re-ranking under perturbed weights…"):
                rb = m.robustness(report_dict(), W(), noise=noise, n=runs, lam=ss.lam)
            s1, s2 = st.columns(2)
            s1.metric("Mean Spearman ρ vs. base ranking", f"{rb['spearman'].mean():.3f}")
            s2.metric("Runs with identical ranking", f"{np.mean(rb['spearman'] > 0.9999):.0%}")
            st.dataframe(pd.DataFrame({"Keeps its rank": (rb["same_rank_share"] * 100).round(0).astype(int).astype(str) + " %"}),
                         width="stretch")

# ------------------------------------------------------------------ ③ model & weights
with tab_model:
    st.subheader("Where the weights come from")
    st.markdown(
        "Each expert rates every causal link with a **linguistic term**. Terms are triangular fuzzy "
        "numbers; the experts' numbers are averaged and defuzzified with the centroid. "
        "Add one column per expert to `expert_weights.csv` (or upload it here) — "
        f"allowed terms: `{', '.join(m.WEIGHT_TERMS)}`.")
    up = st.file_uploader("Upload an elicitation CSV", type="csv")
    if up is not None:
        try:
            el = m.load_elicitation(up)
            ss.elicitation, ss.edges = el, m.aggregate_weights(el)
            ss.W = m.weight_matrix(ss.edges)
            st.success(f"Loaded {len(m.expert_columns(el))} expert column(s).")
        except Exception as exc:
            st.error(str(exc))
    st.write("**Experts' ratings**")
    st.dataframe(ss.elicitation, hide_index=True, width="stretch")
    st.write("**Aggregated weights** (disagreement = spread between experts' centroids)")
    st.dataframe(ss.edges.drop(columns=["rationale"]), hide_index=True, width="stretch")

    with st.expander("What-if: edit the weights by hand"):
        st.caption("For exploring sensitivity in class. Reset restores the elicited values.")
        edited = ss.edges[["source", "target", "weight"]].copy()
        edited = st.data_editor(edited, hide_index=True, disabled=["source", "target"],
                                column_config={"weight": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.05)},
                                key="w_editor")
        b1, b2 = st.columns(2)
        if b1.button("Apply edited weights"):
            ss.W = m.weight_matrix(edited)
            st.rerun()
        if b2.button("Reset to elicited weights"):
            ss.W = m.weight_matrix(ss.edges)
            st.rerun()
        ss.lam = st.slider("Sigmoid steepness λ", 0.5, 3.0, ss.lam, 0.1)

    st.subheader("Current weight matrix (row → column)")
    st.dataframe(W().round(2), width="stretch")
    st.graphviz_chart(fcm_graph(None, W()), width="stretch")

# ------------------------------------------------------------------ ④ method
with tab_method:
    st.markdown("""
### Pipeline
1. **Citizen report (participatory sensing).** Category, location and a linguistic description.
2. **Spatial context (urban data).** OpenStreetMap features around the point (schools, hospitals,
   care homes, transit stops, street type) → a small fuzzy rule base suggests *Population exposure*.
3. **Fuzzification.** Every input is a linguistic term (*very low … very high*) represented as a
   triangular fuzzy number. Its centroid is the activation; its α-cut gives the plausible range.
4. **Fuzzy Cognitive Map.** Eight concepts linked by expert-elicited causal weights. The four inputs
   (Severity, Deterioration, Population exposure, Workaround availability) are clamped; Safety risk,
   Service disruption, Residual public impact and Response urgency are inferred with the rescaled
   Kosko rule  A(t+1) = f((2A(t)−1) + (2A(t)−1)·W),  f = sigmoid, until convergence.
5. **Priority.** Final *Response urgency* × 100, with a linguistic level and range.

### Why each piece
| Piece | Why |
|---|---|
| Linguistic terms | Citizen reports and expert judgements are vague; fuzzy numbers keep that vagueness explicit. |
| FCM | Factors influence each other (severity → safety → impact → urgency); the map is transparent and explainable. |
| Spatial context | The same defect matters more near a school or a busy stop than on an empty street. |
| Expert elicitation | No ground truth exists for weights; the standard FCM practice is to elicit and aggregate them from experts. |
| Sensitivity analysis | Shows how much the ranking depends on any specific weight value. |

### Limitations
- Weights are a team draft until rated by additional experts (e.g. City of Bern civil engineering office, Tiefbauamt).
- Example reports are synthetic. Real validation could compare category rankings with resolution times
  in Zurich's open *Züri wie neu* data (Bern does not publish its reports).
- OSM coverage is good in Bern but not complete; exposure can always be overridden.

### References
- Kosko, B. (1986). Fuzzy cognitive maps. *Int. J. Man-Machine Studies* 24, 65–75.
- Stylios, C. D. & Groumpos, P. P. (2004). Modeling complex systems using fuzzy cognitive maps.
  *IEEE Trans. SMC — Part A* 34(1), 155–162.
- Papageorgiou, E. I. (2011). A new methodology for decisions in medical informatics using fuzzy
  cognitive maps based on fuzzy rule-extraction techniques. *Applied Soft Computing* 11(1), 500–513.
- Zheng, Y., Capra, L., Wolfson, O. & Yang, H. (2014). Urban computing: concepts, methodologies,
  and applications. *ACM TIST* 5(3).
- Takagi, T. & Sugeno, M. (1985). Fuzzy identification of systems. *IEEE Trans. SMC* 15(1), 116–132.
""")
