import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Per-Act VE Dashboard", layout="wide")
st.title("🔬 Per-Act Vaccine Efficacy Dashboard")
st.markdown(
    "Simulate **VE = 1 − CIR** over 4 six-month periods (2 years). "
    "Set partner-count distributions per arm, then hit **▶ Run Simulation**."
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BUCKET_LABELS = ["0–1", "2–5", "6–10", "11–50", ">50", "Missing"]
ACTIVE_LABELS = ["0–1", "2–5", "6–10", "11–50", ">50"]
N_PERIODS     = 4
DEFAULT_PROPS = [0.01, 0.15, 0.20, 0.50, 0.10, 0.04]
MISSING_OPTS  = [
    "Redistribute evenly across buckets",
    "All → highest risk (>50)",
    "All → lowest risk (0–1)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────
def default_counts(n, props=None):
    """Round n × props to ints; absorb rounding error in the largest bucket."""
    if props is None:
        props = DEFAULT_PROPS
    counts = [int(round(n * p)) for p in props]
    diff   = n - sum(counts)
    if diff:
        idx = max(range(5), key=lambda i: counts[i])
        counts[idx] += diff
    return counts


def get_adjusted_props(counts_6, mode):
    """Redistribute missing count into 5 active buckets; return 5 normed props."""
    c    = [float(x) for x in counts_6[:5]]
    miss = float(counts_6[5])
    if   mode == "Redistribute evenly across buckets": c = [x + miss / 5 for x in c]
    elif mode == "All → highest risk (>50)":           c[4] += miss
    else:                                              c[0] += miss
    total = sum(c)
    return [x / total for x in c] if total else [0.2] * 5


def run_one_simulation(
    N_vax, N_placebo, vax_props, pbo_props,
    bucket_ranges, p_ci, p_t, V,
    n_periods=N_PERIODS, rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    n_b = len(bucket_ranges)
    bl  = np.array([r[0] for r in bucket_ranges])
    bh  = np.array([r[1] for r in bucket_ranges])
    bs  = bh - bl + 1

    vax_b = rng.choice(n_b, size=N_vax,     p=vax_props)
    pbo_b = rng.choice(n_b, size=N_placebo, p=pbo_props)

    vax_inf = np.zeros(N_vax,     dtype=bool)
    pbo_inf = np.zeros(N_placebo, dtype=bool)

    for _ in range(n_periods):
        for inf_arr, b_arr, p_per in [
            (vax_inf, vax_b, p_ci * p_t * (1.0 - V)),
            (pbo_inf, pbo_b, p_ci * p_t),
        ]:
            mask = ~inf_arr
            n    = int(mask.sum())
            if not n:
                continue
            b        = b_arr[mask]
            partners = bl[b] + (rng.random(n) * bs[b]).astype(int)
            p_inf    = 1.0 - (1.0 - p_per) ** partners
            inf_arr[mask] = rng.random(n) < p_inf

    # Per-bucket tallies
    vax_bi = np.array([np.sum(vax_inf & (vax_b == k)) for k in range(n_b)])
    pbo_bi = np.array([np.sum(pbo_inf & (pbo_b == k)) for k in range(n_b)])
    vax_bn = np.bincount(vax_b, minlength=n_b)
    pbo_bn = np.bincount(pbo_b, minlength=n_b)

    return int(vax_inf.sum()), int(pbo_inf.sum()), vax_bi, pbo_bi, vax_bn, pbo_bn


def bar_with_iqr(ax, x_pos, medians, q25, q75, color, label, width=0.35):
    """Grouped bar with IQR error bars."""
    ax.bar(x_pos, medians, width, color=color, alpha=0.85, label=label)
    ax.errorbar(
        x_pos, medians,
        yerr=[medians - q25, q75 - medians],
        fmt="none", color=color, capsize=3, lw=1.2, alpha=0.9,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-state init (once on first load)
# ─────────────────────────────────────────────────────────────────────────────
for _pfx in ("vax", "pbo"):
    if f"{_pfx}_0" not in st.session_state:
        for _i, _c in enumerate(default_counts(500)):
            st.session_state[f"{_pfx}_{_i}"] = _c


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Population")
    N_tot     = int(st.number_input("Total N",          value=1000, min_value=2, step=100))
    N_vax     = int(st.number_input("Vaccinated N_vax", value=500,  min_value=1,
                                     max_value=N_tot - 1, step=50))
    N_placebo = N_tot - N_vax
    st.caption(f"Placebo: **{N_placebo:,}**")

    st.divider()

    st.header(">50 Bucket Bounds")
    sc1, sc2 = st.columns(2)
    upper_lo = int(sc1.number_input("Lower", value=51,  min_value=51))
    upper_hi = int(sc2.number_input("Upper", value=100, min_value=upper_lo + 1, step=10))
    bucket_ranges = [(0, 1), (2, 5), (6, 10), (11, 50), (upper_lo, upper_hi)]

    st.divider()

    st.header("Transmission Parameters")
    p_ci = st.slider("P(contact infected)",         0.0, 1.0, 0.03, 0.01, format="%.2f")
    p_t  = st.slider("P(transmission | infected)",  0.0, 1.0, 0.50, 0.05, format="%.2f")
    V    = st.slider("VE per contact (V)",           0.0, 1.0, 0.00, 0.05, format="%.2f")
    st.caption(f"Placebo per-contact P: **{p_ci * p_t:.4f}**")
    st.caption(f"Vax per-contact P:     **{p_ci * p_t * (1 - V):.4f}**")

    st.divider()

    st.header("Simulation")
    n_runs  = int(st.number_input("Runs",        value=100, min_value=1, max_value=2000, step=50))
    seed    = int(st.number_input("Random seed", value=42,  min_value=0))
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-arm distribution panel
# ─────────────────────────────────────────────────────────────────────────────
def dist_panel(prefix, title, n_target, miss_key):
    hcol, bcol = st.columns([5, 1])
    hcol.subheader(title)
    if bcol.button("↺ Reset", key=f"reset_{prefix}",
                   help=f"Reset to defaults for N = {n_target:,}"):
        for i, c in enumerate(default_counts(n_target)):
            st.session_state[f"{prefix}_{i}"] = c
        st.rerun()

    st.caption(
        f"Target: **{n_target:,}** · step ±1 or type · all 6 must sum to target"
    )

    cols   = st.columns(6)
    counts = [
        int(cols[i].number_input(lbl, min_value=0, step=1, key=f"{prefix}_{i}"))
        for i, lbl in enumerate(BUCKET_LABELS)
    ]

    total    = sum(counts)
    diff     = total - n_target
    is_valid = diff == 0

    if is_valid:
        st.success(f"Total: {total:,} / {n_target:,}  ✓")
    elif diff > 0:
        st.error(f"Total: {total:,} / {n_target:,}  — {diff} over  ← fix before running")
    else:
        st.error(f"Total: {total:,} / {n_target:,}  — {abs(diff)} under  ← fix before running")

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
# Partner count distribution panels
# ─────────────────────────────────────────────────────────────────────────────
st.header("Partner Count Distributions")
left, right = st.columns(2)
with left:
    vax_counts, vax_miss, vax_ok = dist_panel(
        "vax", "💉 Vaccinated", N_vax, "miss_vax"
    )
with right:
    pbo_counts, pbo_miss, pbo_ok = dist_panel(
        "pbo", "🧪 Placebo / Unvaccinated", N_placebo, "miss_pbo"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Assumptions
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📌 Model Assumptions"):
    st.markdown("""
| # | Assumption |
|---|---|
| 1 | **1 partner = 1 sexual contact.** Partner count is used directly as the number of independent exposure events per period. |
| 2 | **p_contact_infected is fixed and homogeneous.** Every contact has the same probability of being infected, regardless of time, location, or partner identity. |
| 3 | **Each contact is independent.** No network effects, partnership duration, or repeated contacts with the same partner. |
| 4 | **Bucket assignment is fixed** for the full 2-year study — behavioural risk category does not change across periods. |
| 5 | **Partner counts are redrawn each period** from Discrete Uniform within the person's fixed bucket. |
| 6 | **No reinfections.** Once infected a person exits the at-risk pool permanently; VE is estimated from binary (ever-infected) outcomes. |
| 7 | **VE is multiplicative:** p_vax = p_ci × p_t × (1−V);  p_placebo = p_ci × p_t. |
| 8 | **No waning immunity.** V is constant across all 4 periods. |
| 9 | **>50 bucket** is Discrete Uniform[lower, upper] with user-specified bounds (default 51–100). |
| 10 | **Missing data redistribution** is applied once at baseline and fixed for all 4 periods. |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:

    if not vax_ok or not pbo_ok:
        if not vax_ok: st.error("❌ Vaccinated group counts do not sum to N_vax")
        if not pbo_ok: st.error("❌ Placebo group counts do not sum to N_placebo")
        st.stop()

    vax_props = get_adjusted_props(vax_counts, vax_miss)
    pbo_props = get_adjusted_props(pbo_counts, pbo_miss)

    rng = np.random.default_rng(seed)

    results                          = []
    vax_bi_all, pbo_bi_all           = [], []
    vax_bn_all, pbo_bn_all           = [], []

    prog = st.progress(0, text="Running simulations…")
    for i in range(n_runs):
        nvi, npi, vbi, pbi, vbn, pbn = run_one_simulation(
            N_vax, N_placebo, vax_props, pbo_props,
            bucket_ranges, p_ci, p_t, V, rng=rng,
        )
        ci_v = nvi / N_vax
        ci_p = npi / N_placebo
        cir  = (ci_v / ci_p) if ci_p > 0 else np.nan
        results.append(dict(
            n_vax=nvi, n_pbo=npi,
            ci_v=ci_v, ci_p=ci_p,
            cir=cir,   ve=1.0 - cir,
        ))
        vax_bi_all.append(vbi);  pbo_bi_all.append(pbi)
        vax_bn_all.append(vbn);  pbo_bn_all.append(pbn)
        prog.progress((i + 1) / n_runs)

    prog.empty()
    df = pd.DataFrame(results)

    # Bucket arrays: shape (n_runs, 5)
    vax_bi = np.array(vax_bi_all)
    pbo_bi = np.array(pbo_bi_all)
    vax_bn = np.array(vax_bn_all)
    pbo_bn = np.array(pbo_bn_all)

    # Attack rate per bucket per run
    with np.errstate(divide="ignore", invalid="ignore"):
        vax_ar = np.where(vax_bn > 0, vax_bi / vax_bn, np.nan)
        pbo_ar = np.where(pbo_bn > 0, pbo_bi / pbo_bn, np.nan)

    # Share of infections per bucket per run
    with np.errstate(divide="ignore", invalid="ignore"):
        vax_tot = vax_bi.sum(axis=1, keepdims=True)
        pbo_tot = pbo_bi.sum(axis=1, keepdims=True)
        vax_pct = np.where(vax_tot > 0, vax_bi / vax_tot, np.nan)
        pbo_pct = np.where(pbo_tot > 0, pbo_bi / pbo_tot, np.nan)

    n_nan = df.ve.isna().sum()
    if n_nan:
        st.warning(
            f"{n_nan} run(s) had zero placebo infections → CIR/VE undefined. "
            "Excluded from summaries and plots."
        )

    # ── Top-line metrics (medians) ─────────────────────────────────────────────
    st.header("Results")
    m1, m2, m3, m4 = st.columns(4)

    def iqr_str(s):
        return f"IQR: [{s.quantile(0.25):.4f}, {s.quantile(0.75):.4f}]"

    m1.metric("Median CI — vaccinated", f"{df.ci_v.median():.3f}")
    m1.caption(iqr_str(df.ci_v))

    m2.metric("Median CI — placebo",    f"{df.ci_p.median():.3f}")
    m2.caption(iqr_str(df.ci_p))

    m3.metric("Median CIR",             f"{df.cir.median():.3f}")
    m3.caption(iqr_str(df.cir))

    ve_lo, ve_hi = df.ve.quantile(0.025), df.ve.quantile(0.975)
    m4.metric("Median VE (1 − CIR)",    f"{df.ve.median():.3f}")
    m4.caption(f"95% sim. interval: [{ve_lo:.3f}, {ve_hi:.3f}]")

    # ── Row 1: VE / CI / CIR distributions ────────────────────────────────────
    st.subheader("Simulation Outcome Distributions")
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

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
    ax.set_xlabel("CI"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

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
    plt.close(fig1)

    # ── Row 2: Infections by bucket ────────────────────────────────────────────
    st.subheader("Infections by Partner Count Bucket")
    st.caption("All panels show medians across simulation runs; error bars = IQR (25th–75th percentile).")

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(5)
    w = 0.35

    # Panel A — absolute infections ──────────────────────────────────────────
    ax = axes2[0]
    bar_with_iqr(
        ax, x - w / 2,
        np.median(vax_bi, 0),
        np.quantile(vax_bi, 0.25, 0), np.quantile(vax_bi, 0.75, 0),
        "steelblue", "Vaccinated",
    )
    bar_with_iqr(
        ax, x + w / 2,
        np.median(pbo_bi, 0),
        np.quantile(pbo_bi, 0.25, 0), np.quantile(pbo_bi, 0.75, 0),
        "tomato", "Placebo",
    )
    ax.set_xticks(x); ax.set_xticklabels(ACTIVE_LABELS)
    ax.set_xlabel("Partner Count Bucket"); ax.set_ylabel("Infections (n)")
    ax.set_title("Median Infections per Bucket")
    ax.legend()

    # Panel B — attack rate ───────────────────────────────────────────────────
    ax = axes2[1]
    bar_with_iqr(
        ax, x - w / 2,
        np.nanmedian(vax_ar, 0),
        np.nanquantile(vax_ar, 0.25, 0), np.nanquantile(vax_ar, 0.75, 0),
        "steelblue", "Vaccinated",
    )
    bar_with_iqr(
        ax, x + w / 2,
        np.nanmedian(pbo_ar, 0),
        np.nanquantile(pbo_ar, 0.25, 0), np.nanquantile(pbo_ar, 0.75, 0),
        "tomato", "Placebo",
    )
    ax.set_xticks(x); ax.set_xticklabels(ACTIVE_LABELS)
    ax.set_xlabel("Partner Count Bucket"); ax.set_ylabel("Attack Rate")
    ax.set_title("Median Attack Rate per Bucket")
    ax.legend()

    # Panel C — share of all infections ──────────────────────────────────────
    ax = axes2[2]
    bar_with_iqr(
        ax, x - w / 2,
        np.nanmedian(vax_pct, 0),
        np.nanquantile(vax_pct, 0.25, 0), np.nanquantile(vax_pct, 0.75, 0),
        "steelblue", "Vaccinated",
    )
    bar_with_iqr(
        ax, x + w / 2,
        np.nanmedian(pbo_pct, 0),
        np.nanquantile(pbo_pct, 0.25, 0), np.nanquantile(pbo_pct, 0.75, 0),
        "tomato", "Placebo",
    )
    ax.set_xticks(x); ax.set_xticklabels(ACTIVE_LABELS)
    ax.set_xlabel("Partner Count Bucket"); ax.set_ylabel("Share of All Infections")
    ax.set_title("Share of Total Infections per Bucket")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Summary table ──────────────────────────────────────────────────────────
    st.subheader("Summary Statistics (across simulation runs)")
    cols_o = ("ci_v", "ci_p", "cir", "ve")
    summary = pd.DataFrame({
        "Metric":  ["CI vaccinated", "CI placebo", "CIR", "VE"],
        "Median":  [df[c].median()          for c in cols_o],
        "25%":     [df[c].quantile(0.25)     for c in cols_o],
        "75%":     [df[c].quantile(0.75)     for c in cols_o],
        "2.5%":    [df[c].quantile(0.025)    for c in cols_o],
        "97.5%":   [df[c].quantile(0.975)    for c in cols_o],
        "Mean":    [df[c].mean()             for c in cols_o],
        "SD":      [df[c].std()              for c in cols_o],
    })
    st.dataframe(
        summary.set_index("Metric").style.format("{:.4f}"),
        use_container_width=True,
    )

    # ── Per-bucket detail (expander) ───────────────────────────────────────────
    with st.expander("Per-bucket infection detail (medians across runs)"):
        bk_df = pd.DataFrame({
            "Bucket":                ACTIVE_LABELS,
            "Range":                 [f"{r[0]}–{r[1]}" for r in bucket_ranges],
            "Vax N (med)":           np.median(vax_bn, 0).astype(int),
            "Vax Infections (med)":  np.median(vax_bi, 0).round(1),
            "Vax Attack Rate (med)": np.nanmedian(vax_ar, 0).round(4),
            "Vax % of Inf (med)":    (np.nanmedian(vax_pct, 0) * 100).round(1),
            "Pbo N (med)":           np.median(pbo_bn, 0).astype(int),
            "Pbo Infections (med)":  np.median(pbo_bi, 0).round(1),
            "Pbo Attack Rate (med)": np.nanmedian(pbo_ar, 0).round(4),
            "Pbo % of Inf (med)":    (np.nanmedian(pbo_pct, 0) * 100).round(1),
        })
        st.dataframe(bk_df, hide_index=True, use_container_width=True)

    # ── Effective proportions used ─────────────────────────────────────────────
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
