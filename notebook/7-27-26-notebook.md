# Per-Contact Vaccine Efficacy Dashboard — Lab Notebook

**Project start:** July 23, 2026
**Last updated:** July 24, 2026
**Platform:** Python / Streamlit
**Repository:** Public GitHub repo with GitHub Pages for documentation

---

## 1. Project Purpose

Built a Streamlit dashboard to explore how **per-contact vaccine efficacy** (VE)
translates into observable **trial-level efficacy** (VE = 1 − CIR) when the study
population has a heterogeneous distribution of sexual partner counts.

The core question:
> If a vaccine reduces per-contact transmission probability by some amount V,
> how does that manifest in cumulative incidence — and how sensitive is that
> estimate to the distribution of sexual-activity risk buckets in each arm?

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
| >50 partners | 0.10 | Discrete Uniform{lower … upper}; default 51–100 |
| Missing | 0.04 | Not sampled — redistributed |

### Transmission Model


p_per_contact (placebo)    = p_contact_infected × p_transmission
p_per_contact (vaccinated) = p_contact_infected × p_transmission × (1 − V)

sql

Copy code

| Parameter | Symbol | Default |
|---|---|---|
| P(contact is infected) | p_ci | 0.03 |
| P(transmission given contact infected) | p_t | 0.50 |
| Per-contact vaccine efficacy | V | 0.00 |

### Per-period infection probability

For each uninfected person with k partners drawn that period:


P(infected this period) = 1 − (1 − p_per_contact)^k

python

Copy code

A single Bernoulli draw is made. If infected, the person exits the at-risk
pool permanently (no reinfections).

### Trial-level VE


CIR = CI_vaccinated / CI_placebo
VE  = 1 − CIR

vbnet

Copy code

Because each person is a binary (ever-infected) outcome, this is equivalent
to a risk ratio — the standard VE estimator in efficacy trials.

---

## 3. Model Assumptions

1. **1 partner = 1 sexual contact.** Partner count is used directly as the
   number of independent exposure events per period.
2. **p_contact_infected is fixed and homogeneous.** Every contact has the same
   probability of being infected, regardless of time, location, or partner identity.
3. **Each contact is independent.** No network effects, no partnership duration,
   no repeated contacts with the same partner within a period.
4. **Bucket assignment is fixed for the entire 2-year study.** A person's
   behavioural risk category does not change across the four periods.
5. **Partner counts are redrawn independently each period** from Discrete
   Uniform within the person's fixed bucket.
6. **No reinfections.** Once infected a person exits the at-risk pool
   permanently. VE is estimated from binary (ever-infected) outcomes.
7. **VE acts multiplicatively** on per-contact transmission probability:
   p_vax = p_ci × p_t × (1 − V).
8. **No waning immunity.** Vaccine efficacy V is constant across all four periods.
9. **The >50 partners bucket** is modelled as Discrete Uniform between
   user-specified bounds (default 51–100).
10. **Missing data redistribution** is applied once at baseline and treated
    as fixed for all four periods.

---

## 4. Design Decisions

### Platform
- **Chose Streamlit (Python)** over R/Shiny
- Streamlit Community Cloud is free with no active-hours cap
- shinyapps.io free tier caps at 25 hours/month
- Deploy by pushing to GitHub and connecting to share.streamlit.io

### Terminology
- Used **"per-contact"** throughout rather than "per-act"
- More precise epidemiological language for this type of transmission model
- Used **"cumulative incidence"** and **"proportion infected"** rather than
  "attack rate" throughout — more precise and matches what is actually computed

### >50 Bucket
- Modelled as Discrete Uniform with **user-configurable lower and upper bounds**
  (default 51–100)
- The upper bound matters substantially for cumulative incidence in that bucket
  so made it explicit and interactive rather than hidden

### Per-arm distributions
- Two arms have **completely independent distributions**
- Allows exploration of differential missing-data assumptions and potential
  imbalance in behavioural risk across arms
- Inputs are **count-based** (integer people, step = ±1) rather than proportions
  so the user thinks in terms of moving one person from one bucket to another
- Live proportion table and running total update immediately
- Run button blocked until both arms sum exactly to their target N
- Reset button per arm recalculates defaults for whatever N is currently set

### Missing Data
- Three strategies selectable **per arm independently**:
  1. Redistribute evenly across all five buckets (default)
  2. Assign all to highest-risk (>50) — conservative assumption
  3. Assign all to lowest-risk (0–1) — optimistic assumption
- Having separate selectors per arm implicitly covers the "split by group"
  scenario without needing a separate mode

### Bucket Assignment
- **Fixed for the full two-year study**
- Reflects the assumption that a person's behavioural risk category is a
  stable characteristic, not something that changes period to period

### Reinfections
- **No reinfections**
- Once infected a person is removed from the at-risk pool permanently
- Avoids needing to model duration of infection or timing of events

### CIR vs IRR
- Used **CIR (= risk ratio)** rather than incidence rate ratio
- Because infections are binary and there is no within-period timing
  information, IRR would require person-time at risk which is not modelled
- CIR is the correct and simpler estimator for this setup

### Summary statistics
- **Medians throughout** — top-line metrics, histograms, bar charts, summary table
- With N = 500 per arm the CIR distribution can be right-skewed in some
  settings, making median more robust
- Mean and SD also shown in the summary table for completeness

### Number of simulation runs
- Default 100; user can set 1–2,000
- With N = 1,000 and default parameters, 100 runs is near-instant
- For presentations, 500–1,000 runs give smoother distributions

---

## 5. Dashboard Outputs

### Top-line metrics
- Median cumulative incidence — vaccinated arm (+ IQR)
- Median cumulative incidence — placebo arm (+ IQR)
- Median CIR (+ IQR)
- Median VE (1 − CIR) with 2.5th–97.5th percentile simulation interval

### Figure 1 — Simulation outcome distributions
- **Left:** histogram of VE across runs with median and 2.5/97.5 percentile lines
- **Centre:** overlaid histograms of cumulative incidence for both arms with
  median lines
- **Right:** histogram of CIR with null line (CIR = 1) and median

### Figure 2 — Infections by partner count bucket
- **Left:** median absolute infections per bucket (vaccinated vs. placebo)
- **Centre:** median cumulative incidence per bucket — proportion of people
  in each bucket who become infected
- **Right:** share of total infections attributable to each bucket — illustrates
  that high-activity buckets drive a disproportionate fraction of events
- All bars show IQR as error bars

### Figure 3 — Within >50 bucket detail
- **Left:** distribution of total partners accumulated across all four periods
  for a person in the >50 bucket — analytical via 200,000 draws from the sum
  of four independent Discrete Uniforms; deterministic with fixed seed 0
- **Right:** theoretical (exact, not simulation-based) cumulative incidence
  curve as a function of total partner count for both arms; median total
  partner count marked by dashed line
- This figure highlights that even within the >50 bucket there is a wide range
  of total contacts and the CI rises steeply across that range — which is why
  the choice of upper bound matters

### Summary table
- Median, 25th, 75th, 2.5th, 97.5th percentile, mean, and SD for:
  CI vaccinated, CI placebo, CIR, and VE

### Expandable details
- **Per-bucket infection detail** — median N, infections, cumulative incidence,
  and % of all infections for each bucket in each arm
- **Effective bucket proportions** — post-redistribution proportions actually
  used in the run

---

## 6. Infrastructure and Deployment

### Repository structure


repo/
├── app.py               # Streamlit app — all simulation logic and UI
├── requirements.txt     # Python dependencies
├── index.html           # Documentation page served via GitHub Pages
└── notebook.md          # This file — lab notebook and session log

shell

Copy code

### Dependencies


streamlit
numpy
pandas
matplotlib

sql

Copy code

### GitHub Pages
- `index.html` served publicly via GitHub Pages
- Contains full methodology documentation, assumptions, and design decisions
- Updated each session with changes made
- To update: edit the file and push to GitHub; Pages rebuilds automatically
  within ~60 seconds; hard refresh browser to see changes (Cmd+Shift+R on Mac,
  Ctrl+Shift+R on Windows)

### Streamlit Community Cloud
- App deployed at share.streamlit.io
- Free tier — put app to sleep when not in use via the
  **Manage app** button in the bottom-right corner of the app
- App wakes automatically when someone visits the URL

### Privacy notes
- Repo and Pages site are currently **public**
- Current content (methodology, formulas, design decisions) is not sensitive
- If real trial data or unpublished results are ever involved, move to a
  **separate private repo**
- A private repo hides code but GitHub Pages remains public on the free plan
- To have both private repo and private Pages requires GitHub Team plan (~$4/month)
- Alternative for private documentation: use Markdown `.md` files in a
  private repo — GitHub renders Markdown automatically without needing Pages,
  and the file is private when the repo is private

---

## 7. Session Log

### July 23, 2026 — Session 1

- Designed model structure, population parameters, bucket definitions,
  transmission formula, and VE estimator
- Chose Streamlit over R/Shiny for deployment reasons
- Decided on CIR over IRR, no reinfections, fixed bucket assignment
- Built initial app with shared distribution inputs (proportion-based)
- Added per-arm independent distributions with count-based integer inputs
- Added live proportion table, validation gate, and Reset button per arm
- Added missing data handling (three strategies, per arm independently)
- Added 100-run Monte Carlo simulation with progress bar
- Added Figure 1 (VE / CI / CIR histograms)
- Added Figure 2 (infections by bucket — absolute, CI, share)
- Added Figure 3 (within >50 bucket detail — partner distribution and
  theoretical CI curve)
- Switched all summary statistics from means to medians
- Replaced "attack rate" with "cumulative incidence" / "proportion infected"
  throughout
- Created `index.html` documentation page and deployed via GitHub Pages

### July 24, 2026 — Session 2

- Renamed "per-act" to "per-contact" throughout app and documentation
- Attempted to add save-to-GitHub feature (JSON export via GitHub Contents API)
  — encountered SyntaxError from triple-quoted string containing backtick fences
  and also ran into Streamlit resource limit; decided to remove the feature
  and keep things simple for now
- Reverted to clean app without saving feature
- Removed `requests` from requirements.txt
- Updated `index.html` to remove all references to saving runs
- Added resource management note to documentation (Manage app → sleep)
- Discussed GitHub Pages privacy model — current public setup appropriate
  for current content; plan to switch to private repo with Markdown-based
  lab notebook if sensitive content is ever added
- Created this Markdown lab notebook (`notebook.md`)

---

*Add a new dated entry under Section 7 at the start of each session.*
