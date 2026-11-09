"""Fuzzy Cognitive Map (FCM) for prioritising citizen-reported urban problems.

Pipeline
--------
1. Weight elicitation  : each expert rates every causal link with a linguistic
   term ("weak", "strong", ...). Terms are triangular fuzzy numbers (TFN);
   the experts' TFNs are averaged and defuzzified with the centroid
   (Stylios & Groumpos 2004; Papageorgiou et al.).
2. Input fuzzification : a report gives four linguistic assessments
   (Severity, Deterioration, Population_Exposure, Workaround_Availability).
   Each term is a TFN; its centroid is the activation, its bounds give an
   optimistic / pessimistic range.
3. FCM inference       : the four input concepts are clamped; the other
   concepts are computed with the rescaled Kosko rule
       A(t+1) = f( (2A(t)-1) + (2A(t)-1) W )         f = sigmoid(lambda * x)
   (Papageorgiou 2011), iterated to a fixed point.
4. Output              : the activation of Response_Urgency is the priority.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONCEPTS = [
    "Severity",
    "Safety_Risk",
    "Service_Disruption",
    "Population_Exposure",
    "Deterioration",
    "Workaround_Availability",
    "Residual_Public_Impact",
    "Response_Urgency",
]
LABELS = {
    "Severity": "Severity",
    "Safety_Risk": "Safety risk",
    "Service_Disruption": "Service disruption",
    "Population_Exposure": "Population exposure",
    "Deterioration": "Deterioration",
    "Workaround_Availability": "Workaround availability",
    "Residual_Public_Impact": "Residual public impact",
    "Response_Urgency": "Response urgency",
}
# Concepts assessed from the report + city data. They are clamped during inference.
INPUTS = ["Severity", "Deterioration", "Population_Exposure", "Workaround_Availability"]
OUTPUT = "Response_Urgency"

# ---------------------------------------------------------------- linguistic scales
# Causal strength of a link (magnitude, sign is stored separately). 7-term scale.
WEIGHT_TERMS = {
    "none": (0.0, 0.0, 0.0),
    "very_weak": (0.0, 0.0, 0.17),
    "weak": (0.0, 0.17, 0.33),
    "moderately_weak": (0.17, 0.33, 0.50),
    "medium": (0.33, 0.50, 0.67),
    "moderately_strong": (0.50, 0.67, 0.83),
    "strong": (0.67, 0.83, 1.0),
    "very_strong": (0.83, 1.0, 1.0),
}
# Level of an input concept. 5-term scale.
INPUT_TERMS = {
    "very_low": (0.0, 0.0, 0.25),
    "low": (0.0, 0.25, 0.50),
    "medium": (0.25, 0.50, 0.75),
    "high": (0.50, 0.75, 1.0),
    "very_high": (0.75, 1.0, 1.0),
}
INPUT_TERM_NAMES = list(INPUT_TERMS)


def centroid(tfn) -> float:
    a, b, c = tfn
    return (a + b + c) / 3.0


def nearest_term(value: float, terms=INPUT_TERMS) -> str:
    return min(terms, key=lambda t: abs(centroid(terms[t]) - value))


# ---------------------------------------------------------------- weight elicitation
DEFAULT_WEIGHTS_CSV = Path(__file__).with_name("expert_weights.csv")
META_COLS = {"source", "target", "sign", "rationale"}


def load_elicitation(path=DEFAULT_WEIGHTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"source", "target", "sign"} - set(df.columns)
    if missing:
        raise ValueError(f"Elicitation file is missing columns: {missing}")
    for col in expert_columns(df):
        bad = set(df[col].dropna()) - set(WEIGHT_TERMS)
        if bad:
            raise ValueError(f"Unknown term(s) in column '{col}': {bad}. Use one of {list(WEIGHT_TERMS)}")
    unknown = (set(df.source) | set(df.target)) - set(CONCEPTS)
    if unknown:
        raise ValueError(f"Unknown concept(s): {unknown}")
    return df


def expert_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def aggregate_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Average the experts' TFNs per link and defuzzify (centroid)."""
    experts = expert_columns(df)
    rows = []
    for _, r in df.iterrows():
        tfns = np.array([WEIGHT_TERMS[r[e]] for e in experts if pd.notna(r[e])])
        agg = tfns.mean(axis=0)
        crisp = centroid(agg) * (-1 if str(r["sign"]).strip() == "-" else 1)
        spread = (tfns.mean(axis=1).max() - tfns.mean(axis=1).min()) if len(tfns) > 1 else 0.0
        rows.append({
            "source": r["source"], "target": r["target"], "sign": r["sign"],
            "n_experts": len(tfns), "tfn": tuple(np.round(agg, 3)),
            "weight": round(float(crisp), 3), "disagreement": round(float(spread), 3),
            "rationale": r.get("rationale", ""),
        })
    return pd.DataFrame(rows)


def weight_matrix(edges: pd.DataFrame) -> pd.DataFrame:
    W = pd.DataFrame(0.0, index=CONCEPTS, columns=CONCEPTS)
    for _, e in edges.iterrows():
        W.loc[e["source"], e["target"]] = float(e["weight"])
    return W


def default_matrix() -> pd.DataFrame:
    return weight_matrix(aggregate_weights(load_elicitation()))


# ---------------------------------------------------------------- inference
def sigmoid(x, lam=1.0):
    return 1.0 / (1.0 + np.exp(-lam * np.clip(x, -50, 50)))


def run_fcm(inputs: dict, W: pd.DataFrame, lam: float = 1.0, max_iter: int = 200,
            tol: float = 1e-5) -> pd.DataFrame:
    """Return the activation trajectory (one row per iteration).

    `inputs` must give a value in [0,1] for every concept in INPUTS.
    Non-input concepts start at 0 and are computed by the map.
    """
    missing = [c for c in INPUTS if c not in inputs]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    M = W.loc[CONCEPTS, CONCEPTS].to_numpy(float)
    idx = [CONCEPTS.index(c) for c in INPUTS]
    x = np.zeros(len(CONCEPTS))
    for c in INPUTS:
        v = float(inputs[c])
        if not 0 <= v <= 1:
            raise ValueError(f"{c} must be in [0,1], got {v}")
        x[CONCEPTS.index(c)] = v
    clamp = x[idx].copy()
    hist = [x.copy()]
    for _ in range(max_iter):
        s = 2 * x - 1
        nxt = sigmoid(s + s @ M, lam)
        nxt[idx] = clamp
        hist.append(nxt.copy())
        if np.max(np.abs(nxt - x)) < tol:
            break
        x = nxt
    return pd.DataFrame(hist, columns=CONCEPTS)


def priority(inputs: dict, W: pd.DataFrame, lam: float = 1.0) -> float:
    return float(run_fcm(inputs, W, lam).iloc[-1][OUTPUT])


def priority_label(score: float) -> str:
    if score >= 0.75:
        return "Very high"
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Medium"
    if score >= 0.25:
        return "Low"
    return "Very low"


def terms_to_inputs(terms: dict, which: str = "centroid", alpha: float = 0.5) -> dict:
    """Linguistic terms -> activations. which = centroid | low | high.

    'low'/'high' take the ends of each term's alpha-cut (default alpha = 0.5),
    i.e. the least/most urgent plausible reading of the term.
    Workaround is reversed because it lowers impact.
    """
    out = {}
    for c in INPUTS:
        a, b, cc = INPUT_TERMS[terms[c]]
        if which == "centroid":
            out[c] = centroid((a, b, cc))
        else:
            reverse = c == "Workaround_Availability"
            pick_low = (which == "low") != reverse
            out[c] = a + alpha * (b - a) if pick_low else cc - alpha * (cc - b)
    return out


def assess(terms: dict, W: pd.DataFrame, lam: float = 1.0) -> dict:
    """Full assessment of one report given its linguistic input terms."""
    traj = run_fcm(terms_to_inputs(terms), W, lam)
    final = traj.iloc[-1]
    score = float(final[OUTPUT])
    lo = priority(terms_to_inputs(terms, "low"), W, lam)
    hi = priority(terms_to_inputs(terms, "high"), W, lam)
    return {
        "score": score, "label": priority_label(score), "low": lo, "high": hi,
        "final": final, "trajectory": traj, "iterations": len(traj) - 1,
    }


def input_effects(terms: dict, W: pd.DataFrame, lam: float = 1.0) -> pd.Series:
    """How much each input moves urgency vs. a neutral ('medium') value.

    Positive = this input raises the priority of this report.
    """
    base = priority(terms_to_inputs(terms), W, lam)
    eff = {}
    for c in INPUTS:
        t = dict(terms)
        t[c] = "medium"
        eff[c] = base - priority(terms_to_inputs(t), W, lam)
    return pd.Series(eff).sort_values()


def rank_reports(reports: dict, W: pd.DataFrame, lam: float = 1.0) -> pd.DataFrame:
    rows = []
    for name, terms in reports.items():
        a = assess(terms, W, lam)
        rows.append({"Report": name, "Score": round(100 * a["score"], 1),
                     "Range": f"{100 * a['low']:.0f}–{100 * a['high']:.0f}",
                     "Level": a["label"], "Iterations": a["iterations"]})
    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def robustness(reports: dict, W: pd.DataFrame, noise: float = 0.10, n: int = 300,
               lam: float = 1.0, seed: int = 0) -> dict:
    """Monte Carlo sensitivity: perturb every non-zero weight by U(-noise, +noise).

    Returns the Spearman correlation of each perturbed ranking with the base
    ranking and, per report, how often it keeps its base rank.
    """
    rng = np.random.default_rng(seed)
    names = list(reports)
    inputs = [terms_to_inputs(reports[k]) for k in names]
    base = pd.Series([priority(i, W, lam) for i in inputs], index=names)
    base_rank = base.rank(ascending=False)
    mask = W.to_numpy() != 0
    rhos, same = [], pd.Series(0, index=names)
    for _ in range(n):
        Wp = W.copy()
        arr = Wp.to_numpy()
        arr[mask] = np.clip(arr[mask] + rng.uniform(-noise, noise, mask.sum()), -1, 1)
        Wp.loc[:, :] = arr
        s = pd.Series([priority(i, Wp, lam) for i in inputs], index=names)
        r = s.rank(ascending=False)
        rhos.append(base_rank.corr(r) if len(names) > 1 else 1.0)  # Pearson on ranks = Spearman
        same += (r == base_rank).astype(int)
    return {"spearman": np.array(rhos), "same_rank_share": same / n, "base": base}


# ---------------------------------------------------------------- report categories
# Starting suggestions for the linguistic inputs per problem type (team judgement;
# the person triaging adjusts them). Groups follow the mid-term presentation.
CATEGORIES = {
    "Road maintenance": {
        "Pothole / damaged pavement": dict(Severity="high", Deterioration="high", Workaround_Availability="medium"),
        "Damaged or missing sign": dict(Severity="medium", Deterioration="low", Workaround_Availability="medium"),
        "Blocked or damaged sidewalk": dict(Severity="medium", Deterioration="medium", Workaround_Availability="low"),
    },
    "Waste and cleanliness": {
        "Overflowing public bin": dict(Severity="low", Deterioration="medium", Workaround_Availability="high"),
        "Illegal dumping": dict(Severity="medium", Deterioration="medium", Workaround_Availability="medium"),
        "Littering / dog waste": dict(Severity="very_low", Deterioration="low", Workaround_Availability="high"),
    },
    "Utilities and lighting": {
        "Broken streetlight": dict(Severity="medium", Deterioration="low", Workaround_Availability="low"),
        "Water leak": dict(Severity="high", Deterioration="very_high", Workaround_Availability="medium"),
        "Power outage (public equipment)": dict(Severity="high", Deterioration="low", Workaround_Availability="low"),
    },
    "Public safety and environment": {
        "Blocked drain": dict(Severity="medium", Deterioration="high", Workaround_Availability="medium"),
        "Overgrown vegetation": dict(Severity="low", Deterioration="medium", Workaround_Availability="high"),
        "Noise pollution": dict(Severity="low", Deterioration="low", Workaround_Availability="medium"),
    },
}

# Synthetic example reports located in Bern (illustrative, not real cases).
EXAMPLE_REPORTS = [
    {"name": "Pothole with no way around, Bolligen", "group": "Road maintenance",
     "type": "Pothole / damaged pavement", "lat": 46.9757, "lon": 7.4972,
     "terms": dict(Severity="high", Deterioration="high", Population_Exposure="medium", Workaround_Availability="very_low")},
    {"name": "Overflowing bin, Bundesplatz", "group": "Waste and cleanliness",
     "type": "Overflowing public bin", "lat": 46.9466, "lon": 7.4440,
     "terms": dict(Severity="low", Deterioration="medium", Population_Exposure="very_high", Workaround_Availability="high")},
    {"name": "Broken streetlight, quiet street in Bümpliz", "group": "Utilities and lighting",
     "type": "Broken streetlight", "lat": 46.9440, "lon": 7.3930,
     "terms": dict(Severity="medium", Deterioration="low", Population_Exposure="low", Workaround_Availability="low")},
    {"name": "Blocked sidewalk near school, Länggasse", "group": "Road maintenance",
     "type": "Blocked or damaged sidewalk", "lat": 46.9545, "lon": 7.4290,
     "terms": dict(Severity="medium", Deterioration="medium", Population_Exposure="very_high", Workaround_Availability="very_low")},
    {"name": "Water leak, Breitenrain", "group": "Utilities and lighting",
     "type": "Water leak", "lat": 46.9590, "lon": 7.4560,
     "terms": dict(Severity="high", Deterioration="very_high", Population_Exposure="medium", Workaround_Availability="medium")},
    {"name": "Overgrown hedge on footpath, Bethlehem", "group": "Public safety and environment",
     "type": "Overgrown vegetation", "lat": 46.9480, "lon": 7.3880,
     "terms": dict(Severity="low", Deterioration="medium", Population_Exposure="low", Workaround_Availability="high")},
]
