# Per-Contact Vaccine Efficacy Dashboard

A Streamlit app that Monte Carlo-simulates a two-arm (vaccinated vs. placebo)
prevention trial to explore how **per-contact vaccine efficacy (V)** —
the reduction in transmission probability on a single sexual contact —
translates into the trial-level efficacy actually observed
(**VE = 1 − CIR**), under a heterogeneous distribution of partner counts.

For the full model write-up (assumptions, design decisions, session log),
see [`notebook/lab-notebook.md`](notebook/lab-notebook.md).

## Directory structure

```
.
├── app.py                     # The entire app: simulation logic + Streamlit UI
├── requirements.txt           # Python dependencies
├── notebook/
│   └── lab-notebook.md        # Lab notebook — model structure, assumptions,
│                               # design decisions, and running session log
└── README.md                  # This file
```

## Files

- **[`app.py`](app.py)** — Everything the app does lives in this one file:
  the per-contact/per-period infection model, the Monte Carlo simulation
  loop, the Streamlit sidebar and per-arm input panels, and all three
  result figures plus summary tables. Heavily commented inline — read it
  directly for exact mechanics.
- **[`requirements.txt`](requirements.txt)** — Python dependencies
  (`streamlit`, `numpy`, `pandas`, `matplotlib`). Install with
  `pip install -r requirements.txt`.
- **[`notebook/lab-notebook.md`](notebook/lab-notebook.md)** — The
  project's running lab notebook: model structure and formulas, the full
  list of modeling assumptions, the reasoning behind each design decision
  (e.g. why Streamlit over Shiny, why CIR over IRR, why medians over
  means), deployment notes, and a dated log of what changed each session.
  New entries are added under its "Session Log" section at the start of
  each work session.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io) from
this repository's `main` branch. See the notebook's "Infrastructure and
Deployment" section for details on managing/sleeping the deployed app.
