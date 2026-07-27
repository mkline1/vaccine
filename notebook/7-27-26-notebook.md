# Per-Contact Vaccine Efficacy Dashboard — Lab Notebook

**Project start:** July 23, 2026
**Last updated:** July 24, 2026
**Platform:** Python / Streamlit
**Repository:** Public GitHub repo with GitHub Pages for documentation

---

## 1. Project Purpose

Built a Streamlit dashboard to explore how **per-contact vaccine efficacy** (VE) translates into observable **trial-level efficacy** (VE = 1 − CIR) when the study population has a heterogeneous distribution of sexual partner counts.

The core question:

> If a vaccine reduces per-contact transmission probability by some amount V, how does that manifest in cumulative incidence — and how sensitive is that estimate to the distribution of sexual-activity risk buckets in each arm?

---

## 2. Model Structure

### Population

- Total N = 1,000 (default); split into vaccinated and placebo arms (500 / 500 default)
- Follow-up: 4 six-month periods (2 years total)
- Each person assigned to a fixed behavioural risk bucket at baseline

### Partner Count Buckets

| Bucket | Default Proportion | Sampling |
|---|---|---|
| 0–1 partners | 0.01 | Discrete Uniform{0, 1} |
| 2–5 partners | 0.15 | Discrete Uniform{2, 3, 4, 5} |
| 6–10 partners | 0.20 | Discrete Uniform{6 … 10} |
| 11–50 partners | 0.50 | Discrete Uniform{11 … 50} |
| >50 partners | 0.10 | Discrete Uniform{lower … upper}; default 51–100; user-configurable |
| Missing | 0.04 | Not sampled directly — redistributed into active buckets |

### Transmission Model

Per-contact probability of infection:

    p_per_contact (placebo)    = p_contact_infected x p_transmission
    p_per_contact (vaccinated) = p_contact_infected x p_transmission x (1 - V)

| Parameter | Symbol | Default |
|---|---|---|
| Probability contact is infected | p_ci | 0.03 |
| Probability of transmission given infected contact | p_t | 0.50 |
| Per-contact vaccine efficacy | V | 0.00 |

### Per-Period Infection Probability

For each still-uninfected person who draws k partners in a given period:

    P(infected this period) = 1 - (1 - p_per_contact)^k

A single Bernoulli draw is made. If infected, the person exits the at-risk pool permanently for all remaining periods.

### Trial-Level VE

    CIR = CI_vaccinated / CI_placebo
    VE  = 1 - CIR

Because each person is a binary (ever-infected) outcome with no reinfections, CIR is equivalent to a risk ratio — the standard VE estimator in efficacy trials.

---

## 3. Model Assumptions

1. **1 partner = 1 sexual contact.** Partner count is used directly as the number of independent exposure events per period.
2. **p_contact_infected is fixed and homogeneous.** Every contact has the same probability of being infected, regardless of time, location, or partner identity.
3. **Each contact is independent.** No network effects, no partnership duration, no repeated contacts with the same partner within a period.
4. **Bucket assignment is fixed for the entire 2-year study.** A person's behavioural risk category does not change across the four periods.
5. **Partner counts are redrawn independently each period** from Discrete Uniform within the person's fixed bucket.
6. **No reinfections.** Once infected a person exits the at-risk pool permanently. VE is estimated from binary (ever-infected) outcomes.
7. **VE acts multiplicatively** on per-contact transmission probability: p_vax = p_ci x p_t x (1 - V).
8. **No waning immunity.** Vaccine efficacy V is constant across all four periods.
9. **The >50 partners bucket** is modelled as Discrete Uniform between user-specified bounds (default 51–100).
10. **Missing data redistribution** is applied once at baseline and treated as fixed for all four periods.

---

## 4. Design Decisions

### Platform

- **Chose Streamlit (Python)** over R/Shiny
- Streamlit Community Cloud is free with no active-hours cap; shinyapps.io free tier caps at 25 hours/month
- Deployment: push to GitHub, connect to share.streamlit.io, share URL

### Terminology

- Used **"per-contact"** rather than "per-act" throughout — more precise epidemiological language
- Used **"cumulative incidence"** and **"proportion infected"** rather than "attack rate" throughout — more precise and matches what is actually computed

### The >50 Bucket

- Modelled as Discrete Uniform with **user-configurable lower and upper bounds** (default 51–100)
- Upper bound matters substantially for cumulative incidence in this bucket so made it explicit and interactive rather than hidden

### Per-Arm Distributions

- Two arms have **completely independent distributions** — allows exploration of differential missing-data assumptions and potential imbalance in behavioural risk across arms
- Inputs are **count-based** (integer people, step = ±1) rather than proportions — user thinks in terms of moving one person from one bucket to another
- Live proportion table and running total update immediately as counts change
- Run button is blocked until both arms sum exactly to their target N
- Reset button per arm recalculates defaults for whatever N is currently set

### Missing Data

Three strategies selectable **per arm independently:**

1. Redistribute evenly across all five buckets (default)
2. Assign all to highest-risk (>50) — conservative assumption; people who did not report partner counts may have had more partners
3. Assign all to lowest-risk (0–1) — optimistic assumption

Having separate selectors per arm implicitly covers the "split by group" scenario without needing a separate mode.

### Bucket Assignment

- **Fixed for the full two-year study**
- Reflects the assumption that behavioural risk category is a stable individual characteristic, not something that changes period to period

### Reinfections

- **No reinfections modelled**
- Once infected a person is removed from the at-risk pool permanently
- Avoids needing to model duration of infection, recovery, or timing of events within a period

### CIR vs IRR

- Used **CIR (= risk ratio)** rather than incidence rate ratio
- IRR would require person-time at risk, which is not modelled here because there is no within-period timing information
- CIR is the correct and simpler estimator for this setup

### Summary Statistics

- **Medians throughout** — top-line metrics, histogram reference lines, bar charts, summary table
- With N = 500 per arm the CIR distribution can be right-skewed in some settings, making the median more robust than the mean
- Mean and SD also shown in the summary table for completeness

### Number of Simulation Runs

- Default 100; user can set 1–2,000
- With N = 1,000 and default parameters, 100 runs is near-instant
- 500–1,000 runs give smoother distributions for presentations

---

## 5. Dashboard Outputs

### Top-Line Metrics (medians across simulation runs)

- Median cumulative incidence — vaccinated arm, with IQR
- Median cumulative incidence — placebo arm, with IQR
- Median CIR, with IQR
- Median VE (1 − CIR), with 2.5th–97.5th percentile simulation interval

### Figure 1 — Simulation Outcome Distributions

- **Left:** histogram of VE across runs; median and 2.5/97.5 percentile lines marked
- **Centre:** overlaid histograms of cumulative incidence for both arms; median lines for each arm marked
- **Right:** histogram of CIR; null line at CIR = 1 and median marked

### Figure 2 — Infections by Partner Count Bucket

- **Left:** median absolute infections per bucket, vaccinated vs. placebo
- **Centre:** median cumulative incidence per bucket — proportion of people in each bucket who become infected
- **Right:** share of total infections attributable to each bucket — illustrates that high-activity buckets drive a disproportionate fraction of all events even when they are a small fraction of the population
- All bars show IQR as error bars

### Figure 3 — Within >50 Bucket Detail

- **Left:** distribution of total partners accumulated across all four periods for a person in the >50 bucket — generated analytically via 200,000 draws from the sum of four independent Discrete Uniforms; uses fixed seed 0 so the plot is deterministic
- **Right:** theoretical (exact, not simulation-based) cumulative incidence curve as a function of total partner count, for both arms; median total partner count marked by dashed line
- Key insight: even within the >50 bucket there is a wide range of total contacts (e.g. 204–400 with default bounds) and CI rises steeply across that range — which is why the choice of upper bound matters

### Summary Table

Median, 25th percentile, 75th percentile, 2.5th percentile, 97.5th percentile, mean, and SD for: CI vaccinated, CI placebo, CIR, and VE.

### Expandable Details

- **Per-bucket infection detail** — median N, median infections, median cumulative incidence, and median % of all infections for each bucket in each arm
- **Effective bucket proportions** — post-redistribution proportions actually used in the run, after missing data has been handled

---

## 6. Infrastructure and Deployment

### Repository Structure

    repo/
    ├── app.py            # Streamlit app — all simulation logic and UI
    ├── requirements.txt  # Python dependencies
    ├── index.html        # Documentation page served via GitHub Pages
    └── notebook.md       # This file — lab notebook and session log

### Dependencies

    streamlit
    numpy
    pandas
    matplotlib

### GitHub Pages

- `index.html` served publicly via GitHub Pages
- Contains full methodology documentation, assumptions, and design decisions
- To update: edit the file, push to GitHub; Pages rebuilds automatically within ~60 seconds
- Hard refresh browser to see changes (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)

### Streamlit Community Cloud

- App deployed at share.streamlit.io on the free tier
- Put app to sleep when not in use: click **Manage app** in the bottom-right corner of the app, then select **Put app to sleep**
- App wakes automatically when someone visits the URL

### Privacy Notes

- Repo and Pages site are currently **public** — appropriate for current content (methodology, formulas, design decisions only)
- If real trial data or unpublished results are ever added, move to a **separate private repo**
- On the free GitHub plan, GitHub Pages remains public even if the repo is private; private Pages requires GitHub Team plan (~$4/month)
- For private documentation, use Markdown `.md` files in a private repo — GitHub renders them automatically without Pages, and they stay private when the repo is private

---

## 7. Session Log

### July 23, 2026 — Session 1

- Designed model structure, population parameters, bucket definitions, transmission formula, and VE estimator
- Chose Streamlit over R/Shiny for deployment reasons
- Decided on CIR over IRR; no reinfections; fixed bucket assignment across periods
- Built initial app with shared proportion-based distribution inputs
- Added per-arm independent distributions with count-based integer inputs
- Added live proportion table, validation gate, and Reset button per arm
- Added three missing data strategies selectable per arm independently
- Added 100-run Monte Carlo simulation with progress bar and reproducible random seed
- Added Figure 1 (VE / CI / CIR histograms)
- Added Figure 2 (infections by bucket — absolute counts, cumulative incidence, share of total)
- Added Figure 3 (within >50 bucket — partner distribution and theoretical CI curve)
- Switched all summary statistics from means to medians throughout
- Replaced "attack rate" with "cumulative incidence" / "proportion infected" throughout
- Created `index.html` documentation page and deployed via GitHub Pages

### July 24, 2026 — Session 2

- Renamed "per-act" to "per-contact" throughout app and documentation
- Attempted to add a save-to-GitHub feature (JSON export via GitHub Contents API) — encountered a SyntaxError caused by triple-quoted strings containing backtick fences, and separately ran into the Streamlit Community Cloud resource limit; decided to remove the feature entirely and keep things simple
- Reverted to clean app without saving feature; removed `requests` from requirements.txt
- Updated `index.html` to remove all references to saving runs and the GitHub token setup
- Added a note about resource management to the documentation (Manage app → Put app to sleep)
- Discussed GitHub Pages privacy model; current public setup is appropriate for current content
- Discussed future plan: if repo goes private, switch to Markdown-based lab notebook instead of HTML for documentation (renders automatically on GitHub, private when repo is private, no Pages needed)
- Created this lab notebook (`notebook.md`)

---

*Add a new dated entry under Section 7 at the start of each session.*
