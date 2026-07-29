"""
app.py — Per-Contact Vaccine Efficacy (VE) Dashboard
=====================================================

A Streamlit web app that Monte Carlo-simulates a two-arm (vaccinated vs.
placebo) HIV/STI-style prevention trial, to explore how a per-contact
vaccine efficacy (V) — the reduction in transmission probability on any
single sexual contact — translates into the trial-level efficacy that
would actually be observed (VE = 1 - CIR, where CIR is the cumulative
incidence ratio between arms).

Two simulation modes are offered (toggle at the top of the page): setting
each arm's partner-count distribution independently ("Manual Arm
Distributions", to explore deliberate/hypothetical imbalance), or defining
one pooled population distribution and letting the app randomly split it
into arms each replicate ("Randomized Single Population", to explore how
much chance imbalance from randomization noise alone — with no deliberate
difference between arms — can affect the observed trial result).

This file contains all simulation logic and all UI code for the app; there
is no separate model/view split. Related files in this directory:

    requirements.txt            Python dependencies needed to run this app
                                 (streamlit, numpy, pandas, matplotlib).
    notebook/lab-notebook.md     Lab notebook: full write-up of the model
                                 structure, every modeling assumption, the
                                 reasoning behind each design decision below,
                                 and a session-by-session change log. Read
                                 it for the "why" behind anything that looks
                                 like an arbitrary choice in this file.
    README.md                   Directory-level overview: what each file is
                                 and how they fit together.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Page setup — must be the first Streamlit call in the script.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Per-Contact VE Dashboard", layout="wide")
st.title("🔬 Per-Contact Vaccine Efficacy Dashboard")
st.markdown(
    "Simulate **VE = 1 − CIR** over 4 six-month periods (2 years). "
    "Set partner-count distributions per arm, then hit **▶ Run Simulation**."
)

# Mode is read once, near the top, before the sidebar and distribution
# panels below — both branch on this same value for the rest of the script.
mode = st.radio(
    "Simulation Mode",
    ["Manual Arm Distributions", "Randomized Single Population"],
    horizontal=True,
    help=(
        "Manual: set each arm's partner-count distribution independently, to "
        "explore hypothetical or deliberate imbalance between arms. "
        "Randomized: define one population-level distribution, then let the "
        "app randomly split that population into arms — like the "
        "randomization step of a real trial — so the two arms' realized "
        "compositions can differ purely by chance, especially at small N."
    ),
)
if mode == "Randomized Single Population":
    st.caption(
        "🎲 One pooled population distribution is defined below. Each "
        "simulation replicate draws every person's partner-count bucket from "
        "that distribution, shuffles the whole population, and assigns the "
        "first N_vax people to the vaccinated arm (the rest become placebo) "
        "— mirroring how a real trial randomizes recruits into arms."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# All six partner-count buckets, including "Missing" — used for the per-arm
# input panels where a person must be assigned to exactly one of these.
BUCKET_LABELS = ["0–1", "2–5", "6–10", "11–50", ">50", "Missing"]
# The five buckets people are actually simulated in, after "Missing" has been
# redistributed away (see get_adjusted_props below). Used for plot axis labels.
ACTIVE_LABELS = ["0–1", "2–5", "6–10", "11–50", ">50"]
# Trial duration: 4 six-month follow-up periods = 2 years total.
N_PERIODS     = 4
# Default population split across the six buckets above, used to seed the
# per-arm count inputs the first time the app loads (or on Reset). These
# mirror the "Default Proportion" column in notebook/lab-notebook.md §2.
DEFAULT_PROPS = [0.01, 0.15, 0.20, 0.50, 0.10, 0.04]
# The three strategies a user can pick (independently per arm) for what to do
# with people who have no reported partner-count bucket. See notebook §4
# "Missing Data" for the rationale for offering all three.
MISSING_OPTS  = [
    "Redistribute evenly across buckets",
    "All → highest risk (>50)",
    "All → lowest risk (0–1)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────
def default_counts(n, props=None):
    """
    Convert a set of bucket proportions into integer head-counts that sum
    exactly to n (used to populate/reset the per-arm number inputs).

    Rounding each proportion independently can leave the total off by a
    person or two, so any leftover/deficit (`diff`) is dumped entirely into
    whichever of the five *active* buckets (indices 0-4; "Missing" at index
    5 is deliberately excluded) already has the largest count — this keeps
    the correction small in relative terms since it lands on the biggest
    bucket.
    """
    if props is None:
        props = DEFAULT_PROPS
    counts = [int(round(n * p)) for p in props]
    diff   = n - sum(counts)
    if diff:
        idx = max(range(5), key=lambda i: counts[i])
        counts[idx] += diff
    return counts


def get_adjusted_props(counts_6, mode):
    """
    Fold the "Missing" bucket's head-count into the five active buckets
    according to the chosen strategy, then normalize to proportions that
    sum to 1. This is what run_one_simulation actually samples from — the
    simulation never draws anyone into "Missing" directly.

    mode: one of the three strings in MISSING_OPTS.
        "Redistribute evenly across buckets" -> split missing count 5 ways
        "All → highest risk (>50)"           -> all missing added to bucket 4
        "All → lowest risk (0–1)"            -> all missing added to bucket 0

    Note: "Redistribute evenly across buckets" means EQUALLY — miss / 5 is
    added to every bucket regardless of that bucket's existing size — not
    PROPORTIONALLY to each bucket's current share. A bucket that already
    has many more people than another still only gets the same flat miss/5
    addition, so this strategy shifts the overall mix slightly toward the
    smaller buckets rather than preserving the pre-redistribution ratios.
    """
    c    = [float(x) for x in counts_6[:5]]
    miss = float(counts_6[5])
    # miss / 5 is a flat per-bucket addition (equal), not weighted by each
    # bucket's current count (proportional) — see docstring note above.
    if   mode == "Redistribute evenly across buckets": c = [x + miss / 5 for x in c]
    elif mode == "All → highest risk (>50)":           c[4] += miss
    else:                                              c[0] += miss
    total = sum(c)
    return [x / total for x in c] if total else [0.2] * 5


def run_infection_process(
    vax_b, pbo_b, bucket_ranges, p_ci, p_t, V,
    n_periods=N_PERIODS, rng=None,
):
    """
    Simulate the per-period infection process over n_periods six-month
    periods for a FIXED arm assignment (each person's bucket-index already
    decided — see the `vax_b`/`pbo_b` arguments) and return per-arm
    infection totals plus per-bucket breakdowns.

    This is the shared core both simulation modes use: in every period,
    every person still uninfected draws a fresh partner count uniformly
    from their bucket's [low, high] range and faces one Bernoulli trial for
    infection with probability 1 - (1 - p_per_contact)^partners. Once
    infected, a person is permanently removed from the at-risk pool for all
    later periods (no reinfection is modeled — see notebook §3,
    assumption 6).

    p_per_contact is p_ci * p_t for placebo and p_ci * p_t * (1 - V) for
    vaccinated, i.e. V multiplicatively reduces per-contact transmission
    risk (notebook §3, assumption 7).

    vax_b, pbo_b: per-person bucket-index arrays for each arm (length
        N_vax / N_placebo). Manual mode draws these fresh every call from
        each arm's own fixed proportions; Randomized mode draws them once
        per randomization (see assign_randomized_arms) and can call this
        function repeatedly against the SAME vax_b/pbo_b to characterize
        outcome variability from infection-process noise alone, holding
        the randomization itself fixed.

    Returns:
        (total infected in vax arm, total infected in placebo arm,
         infections-by-bucket for vax, infections-by-bucket for placebo,
         population-by-bucket for vax, population-by-bucket for placebo)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_b = len(bucket_ranges)
    bl  = np.array([r[0] for r in bucket_ranges])   # per-bucket lower bound
    bh  = np.array([r[1] for r in bucket_ranges])   # per-bucket upper bound
    bs  = bh - bl + 1                                # per-bucket range width

    N_vax     = len(vax_b)
    N_placebo = len(pbo_b)
    vax_inf = np.zeros(N_vax,     dtype=bool)   # ever-infected flag, vax arm
    pbo_inf = np.zeros(N_placebo, dtype=bool)   # ever-infected flag, placebo arm

    for _ in range(n_periods):
        # Process both arms identically each period, just with a different
        # per-contact infection probability (vaccinated gets the (1-V) factor).
        for inf_arr, b_arr, p_per in [
            (vax_inf, vax_b, p_ci * p_t * (1.0 - V)),
            (pbo_inf, pbo_b, p_ci * p_t),
        ]:
            mask = ~inf_arr          # only still-at-risk people participate
            n    = int(mask.sum())
            if not n:
                continue
            b        = b_arr[mask]
            # Fresh Discrete Uniform[bl, bh] partner-count draw for this period
            # (notebook §3, assumption 5): bl + floor(U(0,1) * range_width).
            partners = bl[b] + (rng.random(n) * bs[b]).astype(int)
            p_inf    = 1.0 - (1.0 - p_per) ** partners
            inf_arr[mask] = rng.random(n) < p_inf   # single Bernoulli draw per person

    # Tally outcomes by bucket for the per-bucket figures/tables.
    vax_bi = np.array([np.sum(vax_inf & (vax_b == k)) for k in range(n_b)])  # infections per bucket, vax
    pbo_bi = np.array([np.sum(pbo_inf & (pbo_b == k)) for k in range(n_b)])  # infections per bucket, placebo
    vax_bn = np.bincount(vax_b, minlength=n_b)   # population per bucket, vax
    pbo_bn = np.bincount(pbo_b, minlength=n_b)   # population per bucket, placebo

    return int(vax_inf.sum()), int(pbo_inf.sum()), vax_bi, pbo_bi, vax_bn, pbo_bn


def run_one_simulation(
    N_vax, N_placebo, vax_props, pbo_props,
    bucket_ranges, p_ci, p_t, V,
    n_periods=N_PERIODS, rng=None,
):
    """
    Manual-mode replicate: draw each arm's bucket assignment independently
    from that arm's own fixed proportions (notebook §3, assumption 4), then
    run the shared infection process. See run_infection_process for the
    return shape and the per-period mechanics.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_b   = len(bucket_ranges)
    vax_b = rng.choice(n_b, size=N_vax,     p=vax_props)
    pbo_b = rng.choice(n_b, size=N_placebo, p=pbo_props)
    return run_infection_process(vax_b, pbo_b, bucket_ranges, p_ci, p_t, V, n_periods, rng)


def assign_randomized_arms(N_vax, N_placebo, pop_props, bucket_ranges, rng=None):
    """
    Perform ONE randomization for Randomized-Population mode: draw every
    person in the pooled population a partner-count bucket from the single
    population-level distribution `pop_props`, then shuffle-and-split —
    the first N_vax people (in shuffled order) become vaccinated, the rest
    become placebo — mirroring block randomization in a real trial.

    Unlike run_one_simulation, the two arms' bucket compositions are not
    drawn independently: they emerge from randomly partitioning one shared
    population, so by chance they can differ, especially at small N.

    Callers run the shared infection process (run_infection_process)
    repeatedly against the SAME vax_b/pbo_b returned here to characterize
    outcome variability under this one randomization, holding the arm
    assignment itself fixed — see the nested "randomizations × runs"
    structure in the Randomized-mode run block below.

    Returns: (vax_b, pbo_b) — per-person bucket-index arrays for each arm.
    """
    if rng is None:
        rng = np.random.default_rng()

    N_tot = N_vax + N_placebo
    n_b   = len(bucket_ranges)

    # One pooled bucket draw for the whole population, from the single
    # population-level distribution.
    pop_b = rng.choice(n_b, size=N_tot, p=pop_props)

    # Shuffle-and-split: randomly permute the population, then the first
    # N_vax become vaccinated and the remainder become placebo. This is
    # the only mechanism by which the two arms' realized bucket
    # compositions can differ from one randomization to the next.
    perm  = rng.permutation(N_tot)
    vax_b = pop_b[perm[:N_vax]]
    pbo_b = pop_b[perm[N_vax:]]
    return vax_b, pbo_b


def bar_with_iqr(ax, x_pos, medians, q25, q75, color, label, width=0.35):
    """
    Draw one grouped bar series (e.g. one arm) on `ax` at x-positions
    `x_pos`, bar height = median across simulation runs, with an error bar
    spanning the 25th-75th percentile (IQR) across runs. Used throughout
    Figure 2 to show run-to-run variability alongside the central estimate.
    """
    ax.bar(x_pos, medians, width, color=color, alpha=0.85, label=label)
    ax.errorbar(
        x_pos, medians,
        # np.clip guards against floating-point cases where q25/q75 could
        # fall marginally on the wrong side of the median.
        yerr=[np.clip(medians - q25, 0, None), np.clip(q75 - medians, 0, None)],
        fmt="none", color=color, capsize=3, lw=1.2, alpha=0.9,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-state init (once on first load)
# ─────────────────────────────────────────────────────────────────────────────
# Seed Streamlit's session_state with default per-bucket counts (500 each for
# the vax/placebo arms, 1000 for the pooled population) so the number_input
# widgets below have starting values on first render. This only runs once
# per session — subsequent reruns (e.g. from moving a slider) keep whatever
# the user has typed, since the keys already exist in session_state.
for _pfx, _default_n in (("vax", 500), ("pbo", 500), ("pop", 1000)):
    if f"{_pfx}_0" not in st.session_state:
        for _i, _c in enumerate(default_counts(_default_n)):
            st.session_state[f"{_pfx}_{_i}"] = _c


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — all simulation-wide parameters live here
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Population")
    # Total trial size and the vaccinated/placebo split; placebo N is derived
    # so the two always sum to the total.
    N_tot = int(st.number_input("Total N", value=1000, min_value=2, step=100))

    # N_vax must stay within [1, N_tot - 1]. It's given an explicit key so
    # its remembered value persists across reruns independent of max_value
    # -- without a key, Streamlit derives the widget's identity from ALL its
    # parameters (including max_value), so shrinking N_tot would silently
    # reset it to a hardcoded default that could exceed the new max_value
    # and crash. Streamlit ignores the `value=` argument once a keyed
    # widget already has a session_state entry, so the default (first load:
    # split evenly in two) and the clamp (later: only if it no longer fits,
    # never forcibly re-split) both have to happen here, before the widget
    # is instantiated.
    if "n_vax" not in st.session_state:
        st.session_state.n_vax = N_tot // 2
    elif st.session_state.n_vax > N_tot - 1:
        st.session_state.n_vax = N_tot - 1

    N_vax     = int(st.number_input("Vaccinated N_vax", min_value=1,
                                     max_value=N_tot - 1, step=50, key="n_vax"))
    N_placebo = N_tot - N_vax
    st.caption(f"Placebo: **{N_placebo:,}**")

    st.divider()

    st.header(">50 Bucket Bounds")
    # The highest-activity bucket has no natural upper bound, so its range
    # is user-configurable rather than hardcoded (notebook §4, "The >50
    # Bucket") — the choice of upper bound has an outsized effect on that
    # bucket's cumulative incidence (see Figure 3).
    sc1, sc2 = st.columns(2)
    upper_lo = int(sc1.number_input("Lower", value=51,  min_value=51))
    upper_hi = int(sc2.number_input("Upper", value=100, min_value=upper_lo + 1, step=10))
    # (low, high) ranges for all 5 active buckets, in bucket-index order —
    # consumed directly by run_one_simulation.
    bucket_ranges = [(0, 1), (2, 5), (6, 10), (11, 50), (upper_lo, upper_hi)]

    st.divider()

    st.header("Transmission Parameters")
    # p_ci: probability a given contact is infected.
    # p_t:  probability of transmission given contact with an infected person.
    # V:    per-contact vaccine efficacy — multiplicatively reduces the
    #       vaccinated arm's per-contact transmission probability (p_ci * p_t)
    #       by a factor of (1 - V).
    p_ci = st.slider("P(contact infected)",         0.0, 1.0, 0.03, 0.01, format="%.2f")
    p_t  = st.slider("P(transmission | infected)",  0.0, 1.0, 0.50, 0.05, format="%.2f")
    V    = st.slider("VE per contact (V)",           0.0, 1.0, 0.00, 0.05, format="%.2f")
    st.caption(f"Placebo per-contact P: **{p_ci * p_t:.4f}**")
    st.caption(f"Vax per-contact P:     **{p_ci * p_t * (1 - V):.4f}**")

    if mode == "Randomized Single Population":
        st.divider()
        st.header("Randomized Population Analysis")
        # How many independent randomizations (population draw + arm
        # shuffle-split) to try. For EACH one, the infection process below
        # is repeated "Runs per randomization" times to build that
        # randomization's own VE distribution (median + IQR) — see the
        # nested "randomizations x runs" loop in the run block below.
        n_rand = int(st.number_input(
            "Number of randomizations", value=30, min_value=2, max_value=500, step=10,
            help=(
                "How many independent random population draws + arm splits "
                "to try. Each one gets its own VE distribution (from "
                "'Runs per randomization' below) and its own counter-"
                "intuitive verdict — this is what the 'Chance Imbalance' "
                "analysis reports a rate across."
            ),
        ))
        # Which per-randomization imbalance metric drives the scatter plot below.
        imbalance_view = st.selectbox(
            "Imbalance metric to plot",
            ["Combined (mean partners/period)"] + ACTIVE_LABELS,
            help=(
                "Combined: difference in each arm's overall mean partners "
                "per period. Per-bucket: difference in the proportion of "
                "each arm that landed in one specific bucket (e.g. >50) — "
                "useful for testing whether imbalance in a single "
                "high-activity bucket alone can flip the trial's conclusion."
            ),
        )

    st.divider()

    st.header("Simulation")
    # How many independent trial replicates to run (or, in Randomized mode,
    # how many to run PER randomization — see above), and the RNG seed
    # shared across all of them (for reproducibility given identical inputs).
    runs_label = ("Runs per randomization" if mode == "Randomized Single Population"
                  else "Runs")
    n_runs  = int(st.number_input(runs_label,    value=100, min_value=1, max_value=2000, step=50))
    seed    = int(st.number_input("Random seed", value=42,  min_value=0))
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-arm distribution panel
# ─────────────────────────────────────────────────────────────────────────────
def dist_panel(prefix, title, n_target, miss_key):
    """
    Render one arm's editable partner-count distribution panel: six integer
    inputs (one per bucket in BUCKET_LABELS), a live running total with a
    validity check against n_target, a proportion table, a Reset-to-defaults
    button, and the missing-data-handling selector for this arm.

    prefix:   "vax" or "pbo" — used as the session_state key prefix so each
              arm's inputs are independent widgets.
    title:    panel heading shown to the user.
    n_target: the head-count this arm's buckets must sum to (N_vax or
              N_placebo) before the Run button will accept it.
    miss_key: session_state key for this arm's missing-data selectbox.

    Returns: (counts, miss_mode, is_valid) — the six raw bucket counts, the
    chosen missing-data strategy, and whether counts currently sum to
    n_target.
    """
    hcol, bcol = st.columns([5, 1])
    hcol.subheader(title)
    if bcol.button("↺ Reset", key=f"reset_{prefix}",
                   help=f"Reset to defaults for N = {n_target:,}"):
        # Overwrite this arm's session_state entries with fresh defaults
        # sized to the current target N, then force a rerun so the number
        # inputs immediately reflect the reset values.
        for i, c in enumerate(default_counts(n_target)):
            st.session_state[f"{prefix}_{i}"] = c
        st.rerun()

    st.caption(f"Target: **{n_target:,}** · step ±1 or type · all 6 must sum to target")

    # One number_input per bucket; each is bound to its own session_state
    # key so edits persist across reruns (e.g. when the other arm changes).
    cols   = st.columns(6)
    counts = [
        int(cols[i].number_input(lbl, min_value=0, step=1, key=f"{prefix}_{i}"))
        for i, lbl in enumerate(BUCKET_LABELS)
    ]

    total    = sum(counts)
    diff     = total - n_target
    is_valid = diff == 0

    # Immediate feedback on whether this arm's counts are usable — the Run
    # button is gated on both arms being valid (see "if run_btn:" below).
    if is_valid:
        st.success(f"Total: {total:,} / {n_target:,}  ✓")
    elif diff > 0:
        st.error(f"Total: {total:,} / {n_target:,}  — {diff} over  ← fix before running")
    else:
        st.error(f"Total: {total:,} / {n_target:,}  — {abs(diff)} under  ← fix before running")

    # Live proportion table (raw counts / raw total — this is *before* the
    # missing-data redistribution applied by get_adjusted_props).
    denom   = total if total else 1
    prop_df = pd.DataFrame({
        "Bucket":     BUCKET_LABELS,
        "Count":      counts,
        "Proportion": [round(c / denom, 4) for c in counts],
    })
    st.dataframe(prop_df, hide_index=True, use_container_width=True, height=245)

    miss_mode = st.selectbox("Handle missing data →", MISSING_OPTS, key=miss_key)
    return counts, miss_mode, is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Distribution panel(s) — two independent arm panels in Manual mode, or one
# pooled population panel (split into arms at simulation time) in Randomized
# mode.
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Manual Arm Distributions":
    st.header("Partner Count Distributions")
    left, right = st.columns(2)
    with left:
        vax_counts, vax_miss, vax_ok = dist_panel("vax", "💉 Vaccinated", N_vax, "miss_vax")
    with right:
        pbo_counts, pbo_miss, pbo_ok = dist_panel("pbo", "🧪 Placebo / Unvaccinated", N_placebo, "miss_pbo")
else:
    st.header("Population Partner Count Distribution")
    pop_counts, pop_miss, pop_ok = dist_panel(
        "pop", "🌐 Total Population (randomly split into arms at run time)",
        N_tot, "miss_pop",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Assumptions — static reference table shown to the user in-app.
# Mirrors notebook/lab-notebook.md §3 "Model Assumptions" verbatim;
# update both places together if an assumption changes.
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📌 Model Assumptions"):
    st.markdown("""
| # | Assumption |
|---|---|
| 1 | **1 partner = 1 sexual contact.** Partner count is used directly as the number of independent exposure events per period. |
| 2 | **p_contact_infected is fixed and homogeneous.** Every contact has the same probability of being infected, regardless of time, location, or partner identity. |
| 3 | **Each contact is independent.** No network effects, partnership duration, or repeated contacts with the same partner. |
| 4 | **Bucket assignment is fixed** for the full 2-year study — behavioral risk category does not change across periods. |
| 5 | **Partner counts are redrawn each period** from Discrete Uniform within the person's fixed bucket. |
| 6 | **No reinfections.** Once infected a person exits the at-risk pool permanently; VE is estimated from binary (ever-infected) outcomes. |
| 7 | **VE is multiplicative:** p_vax = p_ci × p_t × (1−V);  p_placebo = p_ci × p_t. |
| 8 | **No waning immunity.** V is constant across all 4 periods. |
| 9 | **>50 bucket** is Discrete Uniform[lower, upper] with user-specified bounds (default 51–100). |
| 10 | **Missing data redistribution** is applied once at baseline and fixed for all 4 periods. |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Shared results renderer — everything below is common to both simulation
# modes (Figures 1-3, top-line metrics, summary table, per-bucket detail).
# Mode-specific outputs (the "proportions used" expander, and the
# Randomized-mode imbalance analysis) are rendered by the caller afterward.
# ─────────────────────────────────────────────────────────────────────────────
def render_results(df, vax_bi, pbo_bi, vax_bn, pbo_bn, bucket_ranges, upper_lo, upper_hi, p_ci, p_t, V):
    """
    Render every results output common to both simulation modes: the
    zero-placebo-infections warning, top-line metrics, Figures 1-3, the
    summary statistics table, and the per-bucket infection detail expander.

    df:      one row per replicate, columns n_vax, n_pbo, ci_v, ci_p, cir, ve.
             In Randomized mode this is pooled across every (randomization,
             run) pair, so it reflects the trial design's overall operating
             characteristics (randomization noise and infection-process
             noise combined) rather than any single randomization.
    vax_bi/pbo_bi/vax_bn/pbo_bn: shape (n_runs, n_buckets) infection/
             population counts per bucket per replicate — see
             run_one_simulation / run_infection_process.
    bucket_ranges, upper_lo, upper_hi, p_ci, p_t, V: passed straight through
             to Figure 3, which is mode-agnostic (it only depends on the
             >50 bucket's bounds and the transmission parameters, not on how
             arms were assigned).

    Draws directly to the page; returns nothing.
    """
    # Derived per-bucket metrics, computed across all replicates at once.
    # np.errstate + np.where(... > 0, ...) avoids 0/0 warnings for buckets
    # that happen to be empty in a given replicate.
    with np.errstate(divide="ignore", invalid="ignore"):
        vax_ci_bucket = np.where(vax_bn > 0, vax_bi / vax_bn, np.nan)   # cum. incidence within bucket, vax
        pbo_ci_bucket = np.where(pbo_bn > 0, pbo_bi / pbo_bn, np.nan)   # cum. incidence within bucket, placebo
        vax_tot = vax_bi.sum(axis=1, keepdims=True)                    # total infections that replicate, vax
        pbo_tot = pbo_bi.sum(axis=1, keepdims=True)                    # total infections that replicate, placebo
        vax_pct = np.where(vax_tot > 0, vax_bi / vax_tot, np.nan)      # bucket's share of all vax infections
        pbo_pct = np.where(pbo_tot > 0, pbo_bi / pbo_tot, np.nan)      # bucket's share of all placebo infections

    # A replicate where the placebo arm had zero infections makes CIR/VE
    # undefined (division by zero) — flag and exclude rather than silently
    # dropping via NaN-aware functions with no explanation.
    n_nan = int(df.ve.isna().sum())
    ve_lo = float(df.ve.quantile(0.025))
    ve_hi = float(df.ve.quantile(0.975))

    if n_nan:
        st.warning(
            f"{n_nan} run(s) had zero placebo infections → CIR/VE undefined. "
            "Excluded from summaries and plots."
        )

    # ── Top-line metrics ──────────────────────────────────────────────────────
    # Medians (not means) are used throughout as the headline statistic,
    # since CIR can be right-skewed at N=500/arm — see notebook §4
    # "Summary Statistics" for the rationale.
    st.header("Results")

    def iqr_str(s):
        return f"IQR: [{s.quantile(0.25):.4f}, {s.quantile(0.75):.4f}]"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median CI — vaccinated", f"{df.ci_v.median():.3f}")
    m1.caption(iqr_str(df.ci_v))
    m2.metric("Median CI — placebo",    f"{df.ci_p.median():.3f}")
    m2.caption(iqr_str(df.ci_p))
    m3.metric("Median CIR",             f"{df.cir.median():.3f}")
    m3.caption(iqr_str(df.cir))
    m4.metric("Median VE (1 − CIR)",    f"{df.ve.median():.3f}")
    m4.caption(f"95% sim. interval: [{ve_lo:.3f}, {ve_hi:.3f}]")

    # ── Figure 1: Simulation outcome distributions ─────────────────────────────
    # Three histograms, each summarizing one metric's spread across the
    # n_runs replicates: VE, both arms' cumulative incidence overlaid, CIR.
    st.subheader("Simulation Outcome Distributions")
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

    # Left panel: distribution of VE across replicates, with median and the
    # 2.5th/97.5th percentile simulation interval marked.
    ax = axes1[0]
    ax.hist(df.ve.dropna(), bins=30, color="steelblue", edgecolor="white", lw=0.5)
    ax.axvline(df.ve.median(), color="red",    ls="--", lw=1.5,
               label=f"Median {df.ve.median():.3f}")
    ax.axvline(ve_lo,          color="orange", ls=":",  lw=1.2,
               label=f"2.5%   {ve_lo:.3f}")
    ax.axvline(ve_hi,          color="orange", ls=":",  lw=1.2,
               label=f"97.5%  {ve_hi:.3f}")
    ax.set_title("VE (1 − CIR)")
    ax.set_xlabel("Vaccine Efficacy"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Center panel: cumulative incidence for both arms overlaid, each with
    # its own median line, to visualize the separation CIR is computed from.
    ax = axes1[1]
    ax.hist(df.ci_v, bins=30, color="steelblue", alpha=0.7,
            edgecolor="white", label="Vaccinated")
    ax.hist(df.ci_p, bins=30, color="tomato",    alpha=0.7,
            edgecolor="white", label="Placebo")
    ax.axvline(df.ci_v.median(), color="steelblue", ls="--", lw=1.5,
               label=f"Vax median {df.ci_v.median():.3f}")
    ax.axvline(df.ci_p.median(), color="tomato",    ls="--", lw=1.5,
               label=f"Pbo median {df.ci_p.median():.3f}")
    ax.set_title("Cumulative Incidence")
    ax.set_xlabel("Proportion Infected"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Right panel: CIR distribution with the null (CIR=1, no vaccine effect)
    # and the observed median marked for reference.
    ax = axes1[2]
    ax.hist(df.cir.dropna(), bins=30, color="mediumseagreen",
            edgecolor="white", lw=0.5)
    ax.axvline(1.0,             color="black", ls="--", lw=1.2, alpha=0.5,
               label="Null (CIR = 1)")
    ax.axvline(df.cir.median(), color="red",   ls="--", lw=1.5,
               label=f"Median {df.cir.median():.3f}")
    ax.set_title("CIR (vax CI / placebo CI)")
    ax.set_xlabel("CIR"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)   # release the figure so repeated runs don't leak memory

    # ── Figure 2: Infections by bucket ─────────────────────────────────────────
    # Shows, per active bucket, how infections are distributed across the
    # risk buckets — illustrating that high-activity buckets can drive a
    # disproportionate share of infections even as a small share of N.
    st.subheader("Infections by Partner Count Bucket")
    st.caption("Medians across simulation runs; error bars = IQR (25th–75th percentile).")

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(5)
    w = 0.35

    # Left: raw infection counts per bucket (vax vs. placebo, grouped bars).
    ax = axes2[0]
    bar_with_iqr(
        ax, x - w/2, np.median(vax_bi, 0),
        np.quantile(vax_bi, 0.25, 0), np.quantile(vax_bi, 0.75, 0),
        "steelblue", "Vaccinated",
    )
    bar_with_iqr(
        ax, x + w/2, np.median(pbo_bi, 0),
        np.quantile(pbo_bi, 0.25, 0), np.quantile(pbo_bi, 0.75, 0),
        "tomato", "Placebo",
    )
    ax.set_xticks(x); ax.set_xticklabels(ACTIVE_LABELS)
    ax.set_xlabel("Partner Count Bucket"); ax.set_ylabel("Infections (n)")
    ax.set_title("Median Infections per Bucket"); ax.legend()

    # Center: cumulative incidence *within* each bucket (infections / bucket
    # population) — shows risk concentration independent of bucket size.
    ax = axes2[1]
    bar_with_iqr(
        ax, x - w/2, np.nanmedian(vax_ci_bucket, 0),
        np.nanquantile(vax_ci_bucket, 0.25, 0), np.nanquantile(vax_ci_bucket, 0.75, 0),
        "steelblue", "Vaccinated",
    )
    bar_with_iqr(
        ax, x + w/2, np.nanmedian(pbo_ci_bucket, 0),
        np.nanquantile(pbo_ci_bucket, 0.25, 0), np.nanquantile(pbo_ci_bucket, 0.75, 0),
        "tomato", "Placebo",
    )
    ax.set_xticks(x); ax.set_xticklabels(ACTIVE_LABELS)
    ax.set_xlabel("Partner Count Bucket"); ax.set_ylabel("Proportion Infected")
    ax.set_title("Median Cumulative Incidence per Bucket"); ax.legend()

    # Right: each bucket's share of that arm's *total* infections — the key
    # illustration that a small high-activity bucket can account for a large
    # fraction of all trial infections.
    ax = axes2[2]
    bar_with_iqr(
        ax, x - w/2, np.nanmedian(vax_pct, 0),
        np.nanquantile(vax_pct, 0.25, 0), np.nanquantile(vax_pct, 0.75, 0),
        "steelblue", "Vaccinated",
    )
    bar_with_iqr(
        ax, x + w/2, np.nanmedian(pbo_pct, 0),
        np.nanquantile(pbo_pct, 0.25, 0), np.nanquantile(pbo_pct, 0.75, 0),
        "tomato", "Placebo",
    )
    ax.set_xticks(x); ax.set_xticklabels(ACTIVE_LABELS)
    ax.set_xlabel("Partner Count Bucket"); ax.set_ylabel("Share of All Infections")
    ax.set_title("Share of Total Infections per Bucket")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Figure 3: Within >50 bucket detail ────────────────────────────────────
    # Zooms into the user-configurable >50 bucket to show why its upper
    # bound matters: total contacts across the 4 periods vary widely even
    # within this one bucket, and cumulative incidence rises steeply with it.
    st.subheader(">50 Partner Bucket: Within-Bucket Detail")
    st.caption(
        f"**Left:** distribution of total partners across all 4 periods for a person "
        f"in the >50 bucket (Discrete Uniform {upper_lo}–{upper_hi} per period; "
        f"200k analytical draws). "
        f"**Right:** theoretical cumulative incidence as a function of total partners."
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    # Fixed seed (0) here is intentional and independent of the sidebar
    # `seed` — this panel is an analytical/illustrative view of the >50
    # bucket in isolation, not part of the n_runs Monte Carlo batch above,
    # so it stays deterministic regardless of what seed the user picks.
    rng_vis  = np.random.default_rng(0)
    n_vis    = 200_000
    span_b4  = upper_hi - upper_lo + 1
    # Sum of 4 independent Discrete Uniform[upper_lo, upper_hi] draws — the
    # total partners a >50-bucket person would accumulate over the full
    # 2-year trial (ignoring the possibility they were infected partway
    # through, which is what makes this "theoretical" rather than
    # simulation-derived).
    tot_part = np.sum(
        upper_lo + np.floor(rng_vis.random((n_vis, N_PERIODS)) * span_b4).astype(int),
        axis=1,
    )
    med_tot = int(np.median(tot_part))

    # Left: density histogram of that total-partners distribution.
    ax = axes3[0]
    ax.hist(tot_part, bins=min(100, span_b4 * N_PERIODS),
            color="mediumpurple", edgecolor="white", lw=0.3, density=True)
    ax.axvline(med_tot, color="black", ls="--", lw=1.5,
               label=f"Median: {med_tot} partners")
    ax.set_xlabel("Total Partners Across 4 Periods"); ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Total Partners\n(>50 bucket: {upper_lo}–{upper_hi} per period)")
    ax.legend(fontsize=8)

    # Right: exact (closed-form, not sampled) cumulative incidence curve
    # 1 - (1 - p_per_contact)^t as a function of total partner count t, for
    # both arms — shows how steeply risk climbs across the >50 bucket's
    # possible total-contact range.
    t_range = np.arange(N_PERIODS * upper_lo, N_PERIODS * upper_hi + 1)
    ax = axes3[1]
    ax.plot(t_range, 1.0 - (1.0 - p_ci * p_t) ** t_range,
            color="tomato",    lw=2, label="Placebo")
    ax.plot(t_range, 1.0 - (1.0 - p_ci * p_t * (1.0 - V)) ** t_range,
            color="steelblue", lw=2, label="Vaccinated")
    ax.axvline(med_tot, color="black", ls="--", lw=1.5, alpha=0.7,
               label=f"Median: {med_tot} partners")
    ax.set_xlabel("Total Partners Across 4 Periods")
    ax.set_ylabel("Proportion Infected")
    ax.set_title(f"Cumulative Incidence by Total Partners\n(>50 bucket: {upper_lo}–{upper_hi} per period)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Summary table ──────────────────────────────────────────────────────────
    # Full numeric summary (median, quartiles, 95% simulation interval,
    # mean, SD) for the four headline metrics, across all replicates.
    st.subheader("Summary Statistics (across simulation runs)")
    cols_o = ("ci_v", "ci_p", "cir", "ve")
    summary = pd.DataFrame({
        "Metric":  ["CI vaccinated", "CI placebo", "CIR", "VE"],
        "Median":  [df[c].median()        for c in cols_o],
        "25%":     [df[c].quantile(0.25)  for c in cols_o],
        "75%":     [df[c].quantile(0.75)  for c in cols_o],
        "2.5%":    [df[c].quantile(0.025) for c in cols_o],
        "97.5%":   [df[c].quantile(0.975) for c in cols_o],
        "Mean":    [df[c].mean()          for c in cols_o],
        "SD":      [df[c].std()           for c in cols_o],
    })
    st.dataframe(
        summary.set_index("Metric").style.format("{:.4f}"),
        use_container_width=True,
    )

    # Detail table for users who want the raw per-bucket numbers behind
    # Figure 2. (The "proportions actually used" expander is mode-specific —
    # manual mode has one fixed vector per arm, randomized mode has one
    # pooled population vector — so the caller renders that separately.)
    with st.expander("Per-bucket infection detail (medians across runs)"):
        bk_df = pd.DataFrame({
            "Bucket":                        ACTIVE_LABELS,
            "Range":                         [f"{r[0]}–{r[1]}" for r in bucket_ranges],
            "Vax N (med)":                   np.median(vax_bn, 0).astype(int),
            "Vax Infections (med)":          np.median(vax_bi, 0).round(1),
            "Vax Cum. Incidence (med)":      np.nanmedian(vax_ci_bucket, 0).round(4),
            "Vax % of All Infections (med)": (np.nanmedian(vax_pct, 0) * 100).round(1),
            "Pbo N (med)":                   np.median(pbo_bn, 0).astype(int),
            "Pbo Infections (med)":          np.median(pbo_bi, 0).round(1),
            "Pbo Cum. Incidence (med)":      np.nanmedian(pbo_ci_bucket, 0).round(4),
            "Pbo % of All Infections (med)": (np.nanmedian(pbo_pct, 0) * 100).round(1),
        })
        st.dataframe(bk_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Run simulation — everything below only executes after the button is clicked
# (Streamlit reruns the whole script top-to-bottom on every interaction, so
# `run_btn` is only True on the rerun triggered by that specific click). The
# two branches share render_results() above for all mode-agnostic output;
# each branch adds its own mode-specific setup and follow-up sections.
# ─────────────────────────────────────────────────────────────────────────────
if run_btn and mode == "Manual Arm Distributions":
    # Gate: both arms' bucket counts must sum exactly to their target N
    # before we can build valid sampling proportions.
    if not vax_ok or not pbo_ok:
        if not vax_ok: st.error("❌ Vaccinated group counts do not sum to N_vax")
        if not pbo_ok: st.error("❌ Placebo group counts do not sum to N_placebo")
        st.stop()

    # Fold each arm's "Missing" bucket into the five active buckets per that
    # arm's chosen strategy, producing the 5-length probability vectors that
    # run_one_simulation samples bucket assignment from.
    vax_props = get_adjusted_props(vax_counts, vax_miss)
    pbo_props = get_adjusted_props(pbo_counts, pbo_miss)

    # Single shared RNG, seeded once, threaded through every replicate below
    # so the whole batch of n_runs simulations is reproducible from `seed`.
    rng     = np.random.default_rng(seed)
    results = []
    vax_bi_all, pbo_bi_all = [], []
    vax_bn_all, pbo_bn_all = [], []

    # ── Monte Carlo loop: run the trial n_runs independent times ───────────
    prog = st.progress(0, text="Running simulations…")
    for i in range(n_runs):
        nvi, npi, vbi, pbi, vbn, pbn = run_one_simulation(
            N_vax, N_placebo, vax_props, pbo_props,
            bucket_ranges, p_ci, p_t, V, rng=rng,
        )
        ci_v = nvi / N_vax                       # cumulative incidence, vax arm
        ci_p = npi / N_placebo                   # cumulative incidence, placebo arm
        cir  = (ci_v / ci_p) if ci_p > 0 else np.nan   # cumulative incidence ratio
        results.append(dict(
            n_vax=nvi, n_pbo=npi,
            ci_v=ci_v, ci_p=ci_p,
            cir=cir,   ve=1.0 - cir,             # VE = 1 - CIR, this replicate's estimate
        ))
        vax_bi_all.append(vbi);  pbo_bi_all.append(pbi)
        vax_bn_all.append(vbn);  pbo_bn_all.append(pbn)
        prog.progress((i + 1) / n_runs)
    prog.empty()

    # One row per replicate: n_vax, n_pbo, ci_v, ci_p, cir, ve.
    df     = pd.DataFrame(results)
    # Each array below is shaped (n_runs, n_buckets) — one row per replicate.
    vax_bi = np.array(vax_bi_all)   # infections per bucket, per replicate, vax
    pbo_bi = np.array(pbo_bi_all)   # infections per bucket, per replicate, placebo
    vax_bn = np.array(vax_bn_all)   # population per bucket, per replicate, vax
    pbo_bn = np.array(pbo_bn_all)   # population per bucket, per replicate, placebo

    render_results(df, vax_bi, pbo_bi, vax_bn, pbo_bn, bucket_ranges, upper_lo, upper_hi, p_ci, p_t, V)

    with st.expander("Effective bucket proportions used in this run"):
        st.dataframe(
            pd.DataFrame({
                "Bucket":  ACTIVE_LABELS,
                "Range":   [f"{r[0]}–{r[1]}" for r in bucket_ranges],
                "Vax":     [f"{p:.4f}" for p in vax_props],
                "Placebo": [f"{p:.4f}" for p in pbo_props],
            }),
            hide_index=True,
            use_container_width=True,
        )


elif run_btn and mode == "Randomized Single Population":
    # Gate: the pooled population's bucket counts must sum exactly to N_tot.
    if not pop_ok:
        st.error("❌ Population counts do not sum to Total N")
        st.stop()

    # One shared distribution the whole population is drawn from for each
    # randomization; arm membership itself is randomized at simulation time
    # (see assign_randomized_arms), not fixed here.
    pop_props = get_adjusted_props(pop_counts, pop_miss)
    rng       = np.random.default_rng(seed)

    # ── Nested Monte Carlo: n_rand randomizations, each run n_runs times ────
    # Outer loop: one randomization = one population draw + one arm
    # shuffle-split, held FIXED across its inner runs. Inner loop: repeat
    # just the infection process against that same arm assignment, to
    # characterize outcome variability from infection-process noise alone,
    # holding the randomization's chance imbalance fixed.
    #
    # Two things are collected in parallel:
    #  - the POOLED results across every (randomization, run) pair, fed to
    #    render_results() for the same Figures 1-3 / summary table as
    #    Manual mode — the trial design's overall operating characteristics,
    #    mixing both noise sources, exactly as a single real trial would
    #    experience both at once.
    #  - one row PER RANDOMIZATION (its own median VE + IQR from its n_runs
    #    inner runs, plus its arm-imbalance metrics) for the chance-
    #    imbalance analysis below, which needs the two noise sources kept
    #    separate to relate imbalance to outcome.
    pooled_results = []
    pooled_vax_bi, pooled_pbo_bi = [], []
    pooled_vax_bn, pooled_pbo_bn = [], []
    rand_rows = []

    total_iters = n_rand * n_runs
    done = 0
    prog = st.progress(0, text="Running simulations…")
    for r in range(n_rand):
        vax_b, pbo_b = assign_randomized_arms(N_vax, N_placebo, pop_props, bucket_ranges, rng=rng)
        # Bucket composition is fixed for this randomization.
        n_b       = len(bucket_ranges)
        vax_bn_r  = np.bincount(vax_b, minlength=n_b)
        pbo_bn_r  = np.bincount(pbo_b, minlength=n_b)

        ve_this_rand = []
        for _ in range(n_runs):
            nvi, npi, vbi, pbi, vbn, pbn = run_infection_process(
                vax_b, pbo_b, bucket_ranges, p_ci, p_t, V, rng=rng,
            )
            ci_v = nvi / N_vax
            ci_p = npi / N_placebo
            cir  = (ci_v / ci_p) if ci_p > 0 else np.nan
            ve   = 1.0 - cir
            ve_this_rand.append(ve)
            pooled_results.append(dict(n_vax=nvi, n_pbo=npi, ci_v=ci_v, ci_p=ci_p, cir=cir, ve=ve))
            pooled_vax_bi.append(vbi);  pooled_pbo_bi.append(pbi)
            pooled_vax_bn.append(vbn);  pooled_pbo_bn.append(pbn)
            done += 1
            prog.progress(done / total_iters)

        ve_arr   = np.array(ve_this_rand)
        valid_ve = ve_arr[~np.isnan(ve_arr)]
        if len(valid_ve):
            rand_rows.append(dict(
                randomization=r,
                median_ve=np.median(valid_ve),
                q25=np.quantile(valid_ve, 0.25),
                q75=np.quantile(valid_ve, 0.75),
                vax_bn=vax_bn_r, pbo_bn=pbo_bn_r,
            ))
        else:
            rand_rows.append(dict(
                randomization=r, median_ve=np.nan, q25=np.nan, q75=np.nan,
                vax_bn=vax_bn_r, pbo_bn=pbo_bn_r,
            ))
    prog.empty()

    df     = pd.DataFrame(pooled_results)
    vax_bi = np.array(pooled_vax_bi)
    pbo_bi = np.array(pooled_pbo_bi)
    vax_bn = np.array(pooled_vax_bn)
    pbo_bn = np.array(pooled_pbo_bn)

    render_results(df, vax_bi, pbo_bi, vax_bn, pbo_bn, bucket_ranges, upper_lo, upper_hi, p_ci, p_t, V)

    with st.expander("Population proportions used in this run"):
        st.dataframe(
            pd.DataFrame({
                "Bucket":     ACTIVE_LABELS,
                "Range":      [f"{r[0]}–{r[1]}" for r in bucket_ranges],
                "Population": [f"{p:.4f}" for p in pop_props],
            }),
            hide_index=True,
            use_container_width=True,
        )

    # ── Chance-imbalance analysis (per randomization) ───────────────────────
    # Because arm membership is randomized from a single pooled population,
    # the two arms' realized partner-count mixes can differ purely by
    # chance — especially at small N. This checks, across the n_rand
    # randomizations tried, how often that chance imbalance was large
    # enough to produce a VE distribution whose interquartile range (IQR)
    # implies a conclusion that contradicts the true V set in the sidebar.
    st.header("Chance Imbalance Between Arms")
    st.caption(
        "Each point below is **one randomization**: the population is split "
        "into arms once, then the infection process is repeated "
        f"{n_runs} times ('Runs per randomization') under that same split "
        "to get that randomization's median VE and interquartile range "
        "(IQR, 25th–75th percentile). A randomization's VE interval "
        f"\"crosses zero\" if its 25th percentile ≤ 0 ≤ its 75th percentile — "
        f"since the true V = {V:.2f} set in the sidebar implies "
        + ("a real effect (IQR should stay entirely above zero)."
           if V > 0 else
           "no effect (IQR should cross zero).")
    )

    rand_df       = pd.DataFrame([{k: v for k, v in row.items() if k not in ("vax_bn", "pbo_bn")}
                                   for row in rand_rows])
    rand_vax_bn   = np.array([row["vax_bn"] for row in rand_rows])   # shape (n_rand, 5)
    rand_pbo_bn   = np.array([row["pbo_bn"] for row in rand_rows])

    # Bucket midpoints turn each randomization's realized bucket counts into
    # one summary number: mean partners-per-period for that arm.
    bucket_mid         = np.array([(r[0] + r[1]) / 2 for r in bucket_ranges])
    vax_mean_partners  = (rand_vax_bn * bucket_mid).sum(axis=1) / N_vax
    pbo_mean_partners  = (rand_pbo_bn * bucket_mid).sum(axis=1) / N_placebo
    imbalance_combined = vax_mean_partners - pbo_mean_partners   # shape (n_rand,)

    # Per-bucket proportion imbalance: vax arm's share of that bucket minus
    # placebo arm's share, one column per active bucket, per randomization.
    vax_prop_bucket  = rand_vax_bn / N_vax
    pbo_prop_bucket  = rand_pbo_bn / N_placebo
    bucket_imbalance = vax_prop_bucket - pbo_prop_bucket        # shape (n_rand, 5)

    # A randomization's IQR "crosses zero" if its 25th percentile is <= 0
    # and its 75th percentile is >= 0.
    valid             = rand_df.median_ve.notna()
    iqr_crosses_zero  = (rand_df.q25 <= 0) & (rand_df.q75 >= 0)
    expected_effect    = V > 0
    # V > 0 implies the IQR should NOT cross zero -> counter-intuitive if it does.
    # V = 0 implies the IQR SHOULD cross zero -> counter-intuitive if it doesn't.
    counter_intuitive = (iqr_crosses_zero if expected_effect else ~iqr_crosses_zero) & valid

    n_valid  = int(valid.sum())
    n_flag   = int(counter_intuitive.sum())
    pct_flag = (n_flag / n_valid * 100) if n_valid else float("nan")

    st.metric(
        "Counter-intuitive randomizations",
        f"{pct_flag:.1f}%  ({n_flag} of {n_valid})",
        help=(
            f"Randomizations whose VE interquartile range "
            f"{'crossed' if expected_effect else 'did not cross'} zero, "
            f"contrary to what V = {V:.2f} implies."
        ),
    )

    # Pick which imbalance metric drives the scatter's x-axis, per the
    # sidebar selector.
    if imbalance_view == "Combined (mean partners/period)":
        imb_x   = imbalance_combined
        x_label = "Vax − Placebo mean partners/period"
    else:
        k       = ACTIVE_LABELS.index(imbalance_view)
        imb_x   = bucket_imbalance[:, k]
        x_label = f"Vax − Placebo proportion in {imbalance_view} bucket"

    # ── Figure 4: Arm imbalance vs. per-randomization VE (median + IQR) ────
    st.subheader("Arm Imbalance vs. Per-Randomization VE")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    for mask, color, label in [
        (valid & ~counter_intuitive, "steelblue", "Expected direction"),
        (valid & counter_intuitive,  "crimson",   "Counter-intuitive"),
    ]:
        med = rand_df.median_ve[mask].to_numpy()
        q25 = rand_df.q25[mask].to_numpy()
        q75 = rand_df.q75[mask].to_numpy()
        ax4.errorbar(
            imb_x[mask], med,
            yerr=[np.clip(med - q25, 0, None), np.clip(q75 - med, 0, None)],
            fmt="o", color=color, alpha=0.75, capsize=3, lw=1.2, label=label,
        )
    ax4.axhline(0, color="black", ls="--", lw=1, alpha=0.5, label="VE = 0")
    ax4.axvline(0, color="gray", ls=":", lw=1, alpha=0.6)
    ax4.set_xlabel(x_label)
    ax4.set_ylabel("Median VE, this randomization (error bar = IQR)")
    ax4.set_title("Does chance arm imbalance predict a flipped conclusion?")
    ax4.legend(fontsize=8)
    st.pyplot(fig4)
    plt.close(fig4)

    # Diagnostic table: for just the flagged randomizations, show exactly
    # which bucket(s) were imbalanced between arms in that randomization.
    with st.expander(f"Counter-intuitive randomization details ({n_flag} of {n_valid})"):
        if n_flag == 0:
            st.write("No counter-intuitive randomizations in this batch.")
        else:
            flagged_idx = np.where(counter_intuitive)[0]
            diag_rows = []
            for i in flagged_idx:
                row = {
                    "Randomization":       int(i),
                    "Median VE":           round(float(rand_df.median_ve.iloc[i]), 4),
                    "25%":                 round(float(rand_df.q25.iloc[i]), 4),
                    "75%":                 round(float(rand_df.q75.iloc[i]), 4),
                    "Imbalance (combined)": round(float(imbalance_combined[i]), 3),
                }
                for bi, lbl in enumerate(ACTIVE_LABELS):
                    row[f"Vax % {lbl}"] = round(float(vax_prop_bucket[i, bi]) * 100, 2)
                    row[f"Pbo % {lbl}"] = round(float(pbo_prop_bucket[i, bi]) * 100, 2)
                diag_rows.append(row)
            st.dataframe(pd.DataFrame(diag_rows), hide_index=True, use_container_width=True)
