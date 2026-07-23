import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Per-Act VE Dashboard", layout="wide")

st.title("🔬 Per-Act Vaccine Efficacy Dashboard")
st.markdown(
    "Simulate cumulative incidence and **VE = 1 − CIR** across a heterogeneous "
    "population over 4 six-month periods (2 years). "
    "Adjust parameters in the sidebar, then click **▶ Run Simulation**."
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BUCKET_LABELS = ["0–1", "2–5", "6–10", "11–50", ">50"]
N_PERIODS = 4

MISSING_OPTS = [
    "Redistribute evenly across all buckets",
    "All → highest risk (>50)",
    "All → lowest risk (0–1)",
    "Split by group",
]

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Population ────────────────────────────────────────────────────────────
    st.header("Population")
    N_tot = int(st.number_input("Total N (N_tot)", value=1000, min_value=2, step=100))
    N_vax = int(st.number_input(
        "Vaccinated (N_vax)", value=500,
        min_value=1, max_value=N_tot - 1, step=50
    ))
    N_placebo = N_tot - N_vax
    st.caption(f"Placebo / unvaccinated: **{N_placebo:,}**")

    st.divider()

    # ── Partner count distribution ────────────────────────────────────────────
    st.header("Partner Count Distribution")
    st.caption("All six values (5 buckets + missing) must sum to **1.00**")

    c1, c2 = st.columns(2)
    with c1:
        p0_1   = st.number_input("0–1",     value=0.01, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        p2_5   = st.number_input("2–5",     value=0.15, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        p6_10  = st.number_input("6–10",    value=0.20, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    with c2:
        p11_50 = st.number_input("11–50",   value=0.50, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        p50p   = st.number_input(">50",     value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        p_miss = st.number_input("Missing", value=0.04, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    base_props = [p0_1, p2_5, p6_10, p11_50, p50p]
    prop_total = sum(base_props) + p_miss

    if abs(prop_total - 1.0) > 0.005:
        st.error(f"Proportions sum to {prop_total:.3f} — must equal 1.00")
    else:
        st.success(f"Proportions sum to {prop_total:.3f} ✓")

    # ── >50 bucket bounds ─────────────────────────────────────────────────────
    st.subheader(">50 Bucket Bounds")
    c1, c2 = st.columns(2)
    with c1:
        upper_lo = int(st.number_input("Lower bound", value=51,  min_value=51, step=1))
    with c2:
        upper_hi = int(st.number_input("Upper bound", value=100, min_value=upper_lo + 1, step=10))

    bucket_ranges = [(0, 1), (2, 5), (6, 10), (11, 50), (upper_lo, upper_hi)]

    # ── Missing data handling ─────────────────────────────────────────────────
    st.subheader("Missing Data Handling")
    missing_mode = st.selectbox("Strategy", MISSING_OPTS)

    if missing_mode == "Split by group":
        sub_opts = MISSING_OPTS[:3]   # exclude "Split by group" recursion
        miss_vax = st.selectbox("Vaccinated group", sub_opts, key="mv")
        miss_pbo = st.selectbox("Placebo group",    sub_opts, key="mp")
    else:
        miss_vax = miss_pbo = missing_mode

    st.divider()

    # ── Transmission parameters ───────────────────────────────────────────────
    st.header("Transmission Parameters")
    p_ci = st.slider("P(contact is infected)",             0.00, 1.0, 0.03, step=0.01, format="%.2f")
    p_t  = st.slider("P(transmission | contact infected)", 0.00, 1.0, 0.50, step=0.05, format="%.2f")
    V    = st.slider("Vaccine efficacy per contact (V)",   0.00, 1.0, 0.00, step=0.05, format="%.2f")

    st.caption(f"Baseline per-contact P(infection): **{p_ci * p_t:.4f}**")
    st.caption(f"Vaccinated per-contact P(infection): **{p_ci * p_t * (1 - V):.4f}**")

    st.divider()

    # ── Simulation controls ───────────────────────────────────────────────────
    st.header("Simulation")
    n_runs = int(st.number_input("Number of runs", value=100, min_value=1, max_value=2000, step=50))
    seed   = int(st.number_input("Random seed",    value=42,  min_value=0))

    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def adjust_props(base_props: list, missing_prop: float, mode: str) -> list:
    """
    Redistribute the missing proportion into the five active buckets.
    Returns a normalised list of five floats summing to 1.
    """
    props = list(base_props)
    if mode == "Redistribute evenly across all buckets":
        add = missing_prop / 5.0
        props = [p + add for p in props]
    elif mode == "All → highest risk (>50)":
        props[4] += missing_prop
    elif mode == "All → lowest risk (0–1)":
        props[0] += missing_prop
    else:                               # fallback (should not reach here)
        add = missing_prop / 5.0
        props = [p + add for p in props]
    total = sum(props)
    return [p / total for p in props]  # normalise for floating-point safety


def run_one_simulation(
    N_vax: int, N_placebo: int,
    vax_props: list, pbo_props: list,
    bucket_ranges: list,
    p_ci: float, p_t: float, V: float,
    n_periods: int = N_PERIODS,
    rng=None,
):
    """
    Simulate one trial.

    For each person:
      - Bucket is fixed for all periods (drawn once at the start).
      - Each period, draw partner count ~ Discrete Uniform[lo, hi].
      - P(infected this period) = 1 - (1 - p_per_contact) ^ n_partners
        where p_per_contact = p_ci * p_t * (1-V) for vaccinated,
                              p_ci * p_t           for placebo.
      - Once infected, the person leaves the at-risk pool.

    Returns
    -------
    (n_vax_infected, n_pbo_infected) : int, int
    """
    if rng is None:
        rng = np.random.default_rng()

    bl = np.array([r[0] for r in bucket_ranges])  # lower bounds
    bh = np.array([r[1] for r in bucket_ranges])  # upper bounds
    bs = bh - bl + 1                               # span (# of integers in each bucket)

    # Assign each person to a fixed bucket for the whole study
    vax_b = rng.choice(len(bucket_ranges), size=N_vax,     p=vax_props)
    pbo_b = rng.choice(len(bucket_ranges), size=N_placebo, p=pbo_props)

    vax_inf = np.zeros(N_vax,     dtype=bool)
    pbo_inf = np.zeros(N_placebo, dtype=bool)

    p_per_vax = p_ci * p_t * (1.0 - V)
    p_per_pbo = p_ci * p_t

    for _ in range(n_periods):
        for inf_arr, b_arr, p_per in (
            (vax_inf, vax_b, p_per_vax),
            (pbo_inf, pbo_b, p_per_pbo),
        ):
            mask = ~inf_arr          # currently uninfected
            n    = int(mask.sum())
            if n == 0:
                continue

            b = b_arr[mask]

            # Draw partner count: Discrete Uniform[bl[b], bh[b]]
            # rng.random() in [0,1) -> floor(* span) in {0,...,span-1} -> + lo
            partners = bl[b] + (rng.random(n) * bs[b]).astype(int)

            # P(at least one transmission across k independent contacts)
            p_inf = 1.0 - (1.0 - p_per) ** partners

            # Bernoulli draw for each person
            # Assignment is safe: mask selects only currently-False elements;
            # we write True (newly infected) or False (still uninfected).
            inf_arr[mask] = rng.random(n) < p_inf

    return int(vax_inf.sum()), int(pbo_inf.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Assumptions panel (always visible)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📌 Model Assumptions", expanded=False):
    st.markdown(
        """
| # | Assumption |
|---|-----------|
| 1 | **1 partner = 1 sexual contact.** Partner count is used directly as the number of independent exposure events per period. |
| 2 | **p_contact_infected is fixed and homogeneous.** Every contact carries the same probability of being infected, regardless of time, location, or partner identity. |
| 3 | **Each contact is independent.** No network effects, no partnership duration, no repeated contacts with the same partner. |
| 4 | **Bucket assignment is fixed for the entire 2-year study.** A person's behavioural risk category does not change across the 4 periods. |
| 5 | **Partner counts are redrawn independently each period** from a Discrete Uniform distribution within the person's fixed bucket. |
| 6 | **No reinfections.** Once infected, a person exits the at-risk pool permanently. Cumulative incidence is a binary (ever infected) outcome. |
| 7 | **VE acts multiplicatively on per-contact transmission probability:** p_vax = p_ci × p_t × (1 − V); p_placebo = p_ci × p_t. |
| 8 | **No waning immunity.** Vaccine efficacy V is constant over the full 2-year period. |
| 9 | **The >50 partners bucket** is modelled as Discrete Uniform between the user-specified lower and upper bounds (default 51–100). |
| 10 | **Missing data redistribution is fixed at baseline** and applied identically across all 4 periods. |
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run simulation and display results
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if abs(prop_total - 1.0) > 0.005:
        st.error("Fix the partner distribution proportions (must sum to 1.00) before running.")
        st.stop()

    vax_props_adj = adjust_props(base_props, p_miss, miss_vax)
    pbo_props_adj = adjust_props(base_props, p_miss, miss_pbo)

    rng = np.random.default_rng(seed)

    results = []
    prog = st.progress(0, text="Running simulations…")

    for i in range(n_runs):
        nvi, npi = run_one_simulation(
            N_vax, N_placebo,
            vax_props_adj, pbo_props_adj,
            bucket_ranges,
            p_ci, p_t, V,
            rng=rng,
        )
        ci_v = nvi / N_vax
        ci_p = npi / N_placebo
        cir  = ci_v / ci_p if ci_p > 0 else np.nan
        ve   = 1.0 - cir
        results.append(dict(n_vax=nvi, n_pbo=npi, ci_v=ci_v, ci_p=ci_p, cir=cir, ve=ve))
        prog.progress((i + 1) / n_runs, text=f"Run {i + 1} / {n_runs}")

    prog.empty()
    df = pd.DataFrame(results)

    n_nan = df.ve.isna().sum()
    if n_nan > 0:
        st.warning(
            f"{n_nan} run(s) had zero placebo infections → CIR/VE undefined (NaN). "
            "These runs are excluded from VE/CIR summaries and plots."
        )

    # ── Top-line metrics ──────────────────────────────────────────────────────
    st.header("Results")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("CI — vaccinated",  f"{df.ci_v.mean():.3f}")
        st.caption(f"SD across runs: {df.ci_v.std():.4f}")
    with m2:
        st.metric("CI — placebo",     f"{df.ci_p.mean():.3f}")
        st.caption(f"SD across runs: {df.ci_p.std():.4f}")
    with m3:
        st.metric("Mean CIR",         f"{df.cir.mean():.3f}")
        st.caption(f"SD across runs: {df.cir.std():.4f}")
    with m4:
        ve_lo = df.ve.quantile(0.025)
        ve_hi = df.ve.quantile(0.975)
        st.metric("Mean VE  (1 − CIR)", f"{df.ve.mean():.3f}")
        st.caption(f"Sim. 95% interval: [{ve_lo:.3f}, {ve_hi:.3f}]")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # VE histogram
    ax = axes[0]
    ax.hist(df.ve.dropna(), bins=30, color="steelblue", edgecolor="white", lw=0.5)
    ax.axvline(df.ve.mean(), color="red",    ls="--", lw=1.5, label=f"Mean  {df.ve.mean():.3f}")
    ax.axvline(ve_lo,        color="orange", ls=":",  lw=1.2, label=f"2.5%  {ve_lo:.3f}")
    ax.axvline(ve_hi,        color="orange", ls=":",  lw=1.2, label=f"97.5% {ve_hi:.3f}")
    ax.set_title("VE  (1 − CIR)")
    ax.set_xlabel("Vaccine Efficacy"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Cumulative incidence histogram
    ax = axes[1]
    ax.hist(df.ci_v, bins=30, color="steelblue", edgecolor="white", alpha=0.75, label="Vaccinated")
    ax.hist(df.ci_p, bins=30, color="tomato",    edgecolor="white", alpha=0.75, label="Placebo")
    ax.set_title("Cumulative Incidence")
    ax.set_xlabel("CI"); ax.set_ylabel("Count")
    ax.legend()

    # CIR histogram
    ax = axes[2]
    ax.hist(df.cir.dropna(), bins=30, color="mediumseagreen", edgecolor="white", lw=0.5)
    ax.axvline(1.0,           color="black", ls="--", lw=1.2, alpha=0.5, label="CIR = 1 (null)")
    ax.axvline(df.cir.mean(), color="red",   ls="--", lw=1.5,            label=f"Mean  {df.cir.mean():.3f}")
    ax.set_title("CIR  (vax CI / placebo CI)")
    ax.set_xlabel("CIR"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Summary Statistics (across simulation runs)")
    summary_df = pd.DataFrame({
        "Metric":  ["CI vaccinated", "CI placebo", "CIR", "VE"],
        "Mean":    [df.ci_v.mean(),  df.ci_p.mean(),  df.cir.mean(),  df.ve.mean()],
        "SD":      [df.ci_v.std(),   df.ci_p.std(),   df.cir.std(),   df.ve.std()],
        "2.5%":    [df.ci_v.quantile(0.025), df.ci_p.quantile(0.025),
                    df.cir.quantile(0.025),  df.ve.quantile(0.025)],
        "Median":  [df.ci_v.median(), df.ci_p.median(),
                    df.cir.median(),  df.ve.median()],
        "97.5%":   [df.ci_v.quantile(0.975), df.ci_p.quantile(0.975),
                    df.cir.quantile(0.975),  df.ve.quantile(0.975)],
    })
    st.dataframe(
        summary_df.set_index("Metric").style.format("{:.4f}"),
        use_container_width=True,
    )

    # ── Adjusted proportions used ─────────────────────────────────────────────
    with st.expander("Adjusted bucket proportions used in this run"):
        range_labels = [f"{r[0]}–{r[1]}" for r in bucket_ranges]
        prop_df = pd.DataFrame({
            "Bucket":        BUCKET_LABELS,
            "Range":         range_labels,
            "Vax (adj)":     [f"{p:.4f}" for p in vax_props_adj],
            "Placebo (adj)": [f"{p:.4f}" for p in pbo_props_adj],
        })
        st.dataframe(prop_df, hide_index=True, use_container_width=True)