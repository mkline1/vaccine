# Per-Contact Vaccine Efficacy Dashboard — Lab Notebook

**Project start:** July 23, 2026
**Last updated:** July 29, 2026
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
- Each person assigned to a fixed behavioral risk bucket at baseline

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

**Zero-placebo-infection runs are excluded from summaries (2026-07-29):** if a run/replicate happens to have zero placebo infections, `CI_placebo = 0`, making `CIR = CI_vaccinated / 0` undefined. That run's `cir`/`ve` are recorded as `NaN` rather than a computed number, and are excluded from every downstream summary (medians, IQRs, percentiles, and the Figures 1–3 plots) via `np.nanmedian`/`np.nanquantile`-style functions and `.dropna()`. The app surfaces this explicitly with a warning banner ("N run(s) had zero placebo infections → CIR/VE undefined. Excluded from summaries and plots.") rather than silently dropping them.

---

## 3. Model Assumptions

1. **1 partner = 1 sexual contact.** Partner count is used directly as the number of independent exposure events per period.
2. **p_contact_infected is fixed and homogeneous.** Every contact has the same probability of being infected, regardless of time, location, or partner identity.
3. **Each contact is independent.** No network effects, no partnership duration, no repeated contacts with the same partner within a period.
4. **Bucket assignment is fixed for the entire 2-year study.** A person's behavioral risk category does not change across the four periods.
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

- Two arms have **completely independent distributions** — allows exploration of differential missing-data assumptions and potential imbalance in behavioral risk across arms
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

**Note on "Redistribute evenly" (2026-07-29):** "evenly" means *equally* — the missing count is split 5 ways and the same flat `miss / 5` is added to every bucket, regardless of that bucket's existing size. It is **not proportional** to each bucket's current share (i.e. the `11–50` bucket, which already holds the most people by default, doesn't get a proportionally larger share of the missing count than `0–1` does). In effect this strategy nudges the overall mix slightly toward the smaller buckets rather than preserving the pre-redistribution ratios exactly.

### Bucket Assignment

- **Fixed for the full two-year study**
- Reflects the assumption that behavioral risk category is a stable individual characteristic, not something that changes period to period

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
- **Center:** overlaid histograms of cumulative incidence for both arms; median lines for each arm marked
- **Right:** histogram of CIR; null line at CIR = 1 and median marked

### Figure 2 — Infections by Partner Count Bucket

- **Left:** median absolute infections per bucket, vaccinated vs. placebo
- **Center:** median cumulative incidence per bucket — proportion of people in each bucket who become infected
- **Right:** share of total infections attributable to each bucket — illustrates that high-activity buckets drive a disproportionate fraction of all events even when they are a small fraction of the population
- All bars show IQR as error bars

### Figure 3 — Within >50 Bucket Detail

**Neither panel in this figure comes from the main Monte Carlo simulation (Figures 1–2).** Both are idealized/theoretical reference calculations that describe the `>50` bucket's exposure range and the risk it implies, in isolation — they ignore the real simulation's *informative dropout*: in the actual trial, once a person is infected they exit the at-risk pool and stop accumulating partners for any remaining periods (notebook assumption 6), so a real simulated person's total partner count depends on *when, if ever,* they got infected. Neither panel here accounts for that.

- **Left:** distribution of total partners accumulated across all four periods for a hypothetical person in the >50 bucket, **assuming they survive uninfected through all 4 periods** (not derived from any simulated trial outcomes) — generated by drawing 200,000 samples of the sum of four independent Discrete Uniforms and plotting their histogram; uses a fixed seed (0), independent of the sidebar's simulation seed, so the plot is always identical regardless of what seed is used for the actual trial simulation.
- **Right:** an exact closed-form curve, `CI(t) = 1 − (1 − p_per_contact)^t`, evaluated directly as a function of total partner count `t` — no sampling or randomness of any kind. This formula treats all `t` total partners as if they occurred as one lump exposure (the standard "at least one infection across t independent contacts" formula), rather than reflecting the period-by-period structure with early exit on infection. The median total partner count from the left panel is marked with a dashed line, to show where a "typical" bucket member would land on this risk curve.
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
    ├── app.py                  # Streamlit app — all simulation logic and UI
    ├── requirements.txt        # Python dependencies
    ├── README.md               # Directory-level overview
    └── notebook/
        └── lab-notebook.md     # This file — lab notebook and session log

### Dependencies

    streamlit
    numpy
    pandas
    matplotlib

### Streamlit Community Cloud

- App deployed at share.streamlit.io on the free tier
- Put app to sleep when not in use: click **Manage app** in the bottom-right corner of the app, then select **Put app to sleep**
- App wakes automatically when someone visits the URL

### Privacy Notes

- Repo is currently **public** — appropriate for current content (methodology, formulas, design decisions only)
- **Repo visibility does not control Streamlit's own access.** Streamlit Community Cloud needs read access to this repo to build/deploy the app regardless of whether it's public or private — flipping the repo to private only hides it from the general public, it does not additionally protect content from Streamlit's service. (Confirmed July 28, 2026: briefly set the repo to private, which broke the deployed app until reverted, since Streamlit's GitHub integration lost read access — see Session 3 log below.)
- **The actual safety boundary for future sensitive content is a separate repo.** If real trial data or unpublished results are ever added, they must go in a **new, separate private repo that is never connected to Streamlit (or any other deployment service)** — not just a private flag on *this* repo.
- Avoid granting Streamlit's GitHub App "all repositories" access on this account; if private-repo hosting is needed later, prefer its "select repositories" option scoped to just the specific repo being deployed (least privilege — an app permission granted to "all repos" also covers unrelated private repos on the account).
- On the free GitHub plan, GitHub Pages remains public even if the repo is private; private Pages requires GitHub Team plan (~$4/month)
- For private documentation, use Markdown `.md` files in a private repo — GitHub renders them automatically without Pages, and they stay private when the repo is private

---

## 7. Modeling Insights & Open Questions

### High-prevalence "ceiling effect" can mask real per-contact efficacy

- **Observation:** In high-prevalence settings — where `P(contact infected)` (`p_ci`) is high — it's easy to see **no apparent vaccine effect at the trial level even when per-contact efficacy V is genuinely 30%** (or higher).
- **Why:** Per-period infection probability is `1 - (1 - p_per_contact)^partners`. When `p_per_contact = p_ci × p_t` is already high, this quantity saturates toward 1 very quickly as `partners` grows — and a 30% reduction in `p_per_contact` (the vaccinated arm's `× (1 − V)` factor) barely moves a probability that's already close to 1 back down. For people with many partners per period (the `11–50` and especially `>50` buckets), **both arms end up with cumulative incidence near 1** within one or a few periods regardless of vaccination status, so `CIR ≈ 1` and observed `VE ≈ 0` — even though the underlying per-contact protection is real.
- **Where this shows up:** Most pronounced in the high-activity buckets, but since those buckets already drive a disproportionate share of total infections (see Figure 2 / §5), high enough `p_ci` can pull the *overall* trial-level VE down toward zero too, not just the >50 bucket's own estimate.
- **Implication:** A trial's ability to *detect* a real per-contact effect depends heavily on the background prevalence/transmission environment, not just on whether V is truly nonzero — worth keeping in mind when interpreting "no observed effect" results from either simulation mode, separately from the chance-imbalance question below.

### ~~Detection threshold ε — open question~~ — Resolved July 29, 2026

- **Original concern (2026-07-28):** the "counter-intuitive replicate" classifier used a fixed **ε threshold on the point-estimate VE** (default 0.05) to decide whether a single replicate "showed an effect" — a fixed point-estimate cutoff that ignored the precision/uncertainty behind each estimate.
- **Resolution (Session 6):** replaced ε entirely with an **interquartile-range (IQR) rule**, and restructured the simulation into a proper nested design to support it:
    - **Randomizations** (outer loop, new "Number of randomizations" parameter): each is one population draw + one arm shuffle-split, held fixed.
    - **Runs per randomization** (inner loop, the existing "Runs" parameter): repeats of the infection process under that same fixed split, producing a distribution of VE estimates — median, 25th, and 75th percentile — for that one randomization.
    - **Verdict per randomization:** does its VE's 25th–75th percentile interval cross zero? If true `V = 0`, the interval crossing zero is *expected*; if true `V > 0`, the interval should stay entirely above zero — a randomization is "counter-intuitive" whenever the interval's relationship to zero contradicts what the true `V` implies.
- **Emergent finding from testing this:** even at **N = 500/arm** (large), true `V = 0`, and 100 runs per randomization, roughly **35–40% of randomizations were still counter-intuitive** by this rule. This is not a bug — it reflects that the per-run VE estimator's noise (SD ≈ 0.04 in this setting) and the between-randomization spread from chance arm imbalance are of *comparable magnitude* even at N = 500, so a 100-run sample's IQR misses zero by chance fairly often. Increasing "Runs per randomization" narrows each randomization's IQR and should reduce this rate; this is a real, useful operating characteristic of the design to be aware of, not an artifact.
    - **Exact parameters (reproducible):** N_vax = 500, N_placebo = 500, true V = 0.0, `p_ci` = 0.03, `p_t` = 0.5, >50 bucket bounds = 51–100 (default), population distribution = default proportions `[0.01, 0.15, 0.20, 0.50, 0.10, 0.04]` with missing redistributed evenly, Number of randomizations = 40, Runs per randomization = 100, seed = 42. Result: 15 of 40 randomizations (37.5%) had a VE interquartile range that did not cross zero despite V truly being 0.

---

## 8. Known Issues

### ~~`Vaccinated N_vax` crashes when `Total N` is reduced below ~501~~ — Fixed July 29, 2026

- **Symptom:** Setting **Total N** below 501 (e.g. to explore small-N behavior) crashed the app with `StreamlitValueAboveMaxError: The value 500 is greater than the max_value <N_tot - 1>`.
- **Root cause:** `st.number_input("Vaccinated N_vax", value=500, min_value=1, max_value=N_tot - 1, step=50)` had no explicit `key=`. Streamlit auto-derives a widget's identity from a hash of *all* its parameters when no key is given — including `max_value`. Changing `Total N` changed `max_value`, so Streamlit treated it as a brand-new widget with no memory of what the user previously typed, and fell back to the literal `value=500` in the code. That hardcoded default then exceeded the new (smaller) `max_value`, crashing instead of clamping.
- **Discovered:** July 28, 2026, while browser-testing the Randomized Single Population mode at small N (Session 4). Confirmed pre-existing — reproduced identically in Manual mode.
- **Fix applied (Session 6):** gave the widget an explicit `key="n_vax"` so its identity persists across reruns independent of `max_value`. `N_vax` now defaults to `N_tot // 2` on first load, and is only clamped down (never forcibly reset) if a later reduction in `Total N` would put it out of bounds — a deliberate uneven split you've set is preserved otherwise.

---

## 9. Session Log

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

### July 28, 2026 — Session 3

- Added thorough inline comments and docstrings throughout `app.py` explaining exactly what each function and section does, and referencing `requirements.txt`, `notebook/lab-notebook.md`, and `README.md` where relevant
- Created `README.md` — directory-level overview of what each file is and how they fit together
- Committed and pushed both changes to GitHub; adopted a new project rule: every commit in this repo notes it was made with Claude Code
- Tried to view the deployed Streamlit app and found it erroring
- Diagnosed the cause: the repo had been switched to private (for safety), which broke Streamlit Community Cloud's read access to pull the code — its push webhook still delivered successfully, but that's a separate mechanism from read access, so the deploy failed
- Discussed the tradeoffs of making the repo public vs. keeping it private and granting Streamlit's GitHub App access to only this repo vs. to all repos (public and private) on the account
- Key realization: repo visibility (public/private) and Streamlit's read access are separate axes — Streamlit needs read access either way to deploy, so a private flag on *this* repo does not protect any sensitive content added to *this* repo from Streamlit's service. It only hides the repo from the general public.
- **Decision: reverted the repo back to public.** Given the above, keeping it private bought no extra protection for future sensitive content in this same repo, so simplicity won out.
- **New policy adopted:** any real trial data or unpublished/sensitive results added in the future must live in a separate, private repo that is never connected to Streamlit or any other deployment service — not just toggled private within this repo. Also decided against ever granting Streamlit's GitHub App "all repositories" access, to avoid exposing unrelated private repos on the account for the sake of one deployed app.
- Updated the notebook's "Privacy Notes" section (§6) to capture this reasoning for future reference
- Renamed this file from `7-27-26-notebook.md` to `lab-notebook.md`, to make clear it's a single running notebook rather than a per-date file

### July 28, 2026 — Session 4

- Designed and implemented a new **Randomized Single Population** simulation mode in `app.py`, alongside the existing Manual Arm Distributions mode (toggle at the top of the page):
    - One pooled population partner-count distribution is defined instead of two independent per-arm distributions
    - New `run_one_simulation_randomized()`: draws every person's bucket from the pooled distribution, then shuffles and splits the population — the first `N_vax` people become vaccinated, the rest placebo (block randomization) — so arm sizes are always exact but bucket *composition* can differ by chance, especially at small N
    - Refactored the existing Figures 1–3 / summary table / per-bucket detail code out of the old monolithic run block into a shared `render_results()` function used by both modes, to avoid duplicating ~150 lines
    - Added a new "Chance Imbalance Between Arms" section (Randomized mode only): a combined or per-bucket imbalance metric (selectable, including the `>50` bucket specifically), a "% counter-intuitive replicates" statistic, a scatter plot of imbalance vs. observed VE, and a diagnostic table listing exact per-bucket breakdowns for flagged replicates
    - "Counter-intuitive" is defined as: whether a replicate shows a detectable effect (VE > ε, ε adjustable) disagrees with what the true V implies (V > 0 should show an effect; V = 0 should not) — one rule that covers both the "vaccine works but trial shows nothing" and "no effect but trial shows one" failure modes from the same run
- **Verified the new logic directly in Python** (bypassing the UI, which hit the small-N crash described in §8): at N=25/25 with true V=0.3, 30.3% of replicates were counter-intuitive; at N=500/500, only 3.6% were — confirming chance imbalance shrinks with N as expected. Arm sizes were always exact across all replicates.
- Discovered the pre-existing small-N crash bug in `Vaccinated N_vax` while testing (see §8, Known Issues) — confirmed unrelated to this session's changes (reproduces identically in the old Manual mode) and left unfixed pending a decision on priority
- Did not get a live browser confirmation that "Run Simulation" renders the new section end-to-end in Randomized mode — repeated automated-browser click attempts failed, but a control test showed the *same, pre-existing* Run button also fails to trigger via automation in Manual mode, indicating a session-specific browser-automation limitation rather than a bug in the new code
- Committed and pushed the new mode to GitHub

### July 28, 2026 — Session 5

- Ran the new Randomized Single Population mode for the first time in the actual (not just Python-tested) app
- Flagged the ε detection-threshold parameter as not obviously well-motivated — recorded as an open question in §7, "Detection threshold ε — open question," rather than resolved on the spot
- Noted an important modeling insight from exploring the app: in high-prevalence settings (`p_ci` high), it's easy to see no apparent vaccine effect at the trial level even with a real 30% per-contact efficacy, due to a ceiling effect in the per-period infection probability formula — recorded in §7, "High-prevalence 'ceiling effect' can mask real per-contact efficacy"
- Added new notebook section "§7. Modeling Insights & Open Questions" to hold observations like these going forward, separate from the "Known Issues" bug log (renumbered to §8) and Session Log (renumbered to §9)

### July 29, 2026 — Session 6

- **Fixed the `Vaccinated N_vax` crash** (§8): gave the widget an explicit `key="n_vax"`, defaulted it to `N_tot // 2` on first load, and made later reductions in `Total N` only clamp `N_vax` down (never forcibly re-split it) if it would otherwise no longer fit — preserves a deliberately uneven split.
- **Restructured Randomized Single Population mode into a proper nested design**, replacing the flat "n_runs total replicates" loop:
    - Split the old `run_one_simulation_randomized` into two reusable pieces: `assign_randomized_arms()` (one population draw + one arm shuffle-split — "one randomization") and a new shared `run_infection_process()` (the per-period infection mechanics, callable repeatedly against the SAME arm assignment — "runs per randomization"). `run_one_simulation` (Manual mode) was refactored to call the same shared `run_infection_process()`, removing what would otherwise have been duplicated physics code in three places.
    - New sidebar parameter "Number of randomizations" (outer loop count); the existing "Runs" parameter is now specifically "Runs per randomization" (inner loop count) in this mode.
    - Figures 1–3 / summary table (via `render_results()`) now summarize the **pooled** results across every (randomization, run) pair — the trial design's overall operating characteristics, mixing both randomization-imbalance noise and infection-process noise, same as a real trial would experience both at once.
- **Replaced the ε-threshold classifier with the user-specified IQR-crosses-zero rule** (§7, resolved): for each randomization, take the median VE and 25th/75th percentile across its runs; a randomization is counter-intuitive if whether its IQR crosses zero disagrees with what the true V implies. The "Arm Imbalance vs. Per-Randomization VE" scatter plot and diagnostic table were reworked accordingly, one point/row per randomization (median VE with an IQR error bar) rather than one per individual run.
- **Verified all of the above directly in Python** (bypassing the UI, consistent with prior sessions): confirmed the same fixed arm assignment produces different outcomes across repeated `run_infection_process()` calls (infection noise only), confirmed Manual mode still works after the refactor, and ran the full nested randomizations × runs structure at both N=25/arm and N=500/arm.
- **Notable emergent finding**, not a bug: even at N=500/arm with true V=0 and 100 runs per randomization, ~35–40% of randomizations were still counter-intuitive by the IQR rule — the per-run VE estimator's noise and the between-randomization imbalance spread are comparable in magnitude at this N. Recorded in §7.

---

*Add a new dated entry under Section 9 at the start of each session.*
