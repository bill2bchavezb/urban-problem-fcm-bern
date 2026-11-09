# Prioritising citizen-reported urban problems in Bern (Fuzzy logic + FCM)

Seminar of Urban Computing AS2026.

**Start here:** [REPORT.md](REPORT.md) explains the project, the method, the fixes and likely presentation questions.

| File | What it is |
|---|---|
| `fcm_model.py` | The model: linguistic terms (triangular fuzzy numbers), weight elicitation + aggregation, FCM inference, explanation, sensitivity analysis |
| `context.py` | Spatial context: OpenStreetMap features around a location → suggested *Population exposure* (fuzzy Sugeno rules) |
| `expert_weights.csv` | Linguistic ratings of every causal link. **Add one column per expert**, then re-run |
| `app.py` | Interactive interface (Streamlit) |
| `Urban_Problem_Prioritization_FCM_v2.ipynb` | Report notebook: method, fix of v1, ranking, explanation, sensitivity |

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt            # app only
.venv/bin/pip install -r requirements-notebook.txt   # + notebook
.venv/bin/streamlit run app.py
```

Opens at http://localhost:8501.

## Adding experts

In `expert_weights.csv` add a column (e.g. `expert_lecturer`) and fill every row with one of:
`none, very_weak, weak, moderately_weak, medium, moderately_strong, strong, very_strong`.
The sign (+/−) is in the `sign` column. The app (tab ③) also accepts an uploaded CSV.

## Notes

- Example reports are synthetic.
- "Fetch context" uses the public Overpass API, which is sometimes overloaded (502 / timeout);
  then just set the exposure by hand.
