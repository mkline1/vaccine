import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import base64
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Per-Contact VE Dashboard", layout="wide")
st.title("🔬 Per-Contact Vaccine Efficacy Dashboard")
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
    if props is None:
        props = DEFAULT_PROPS
    counts = [int(round(n * p)) for p in props]
    diff   = n - sum(counts)
    if diff:
        idx = max(range(5), key=lambda i: counts[i])
        counts[idx] += diff
    return counts


def get_adjusted_props(counts_6, mode):
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

    vax_bi = np.array([np.sum(vax_inf & (vax_b == k)) for k in range(n_b)])
    pbo_bi = np.array([np.sum(pbo_inf & (pbo_b == k)) for k in range(n_b)])
    vax_bn = np.bincount(vax_b, minlength=n_b)
    pbo_bn = np.bincount(pbo_b, minlength=n_b)

    return int(vax_inf.sum()), int(pbo_inf.sum()), vax_bi, pbo_bi, vax_bn, pbo_bn


def bar_with_iqr(ax, x_pos, medians, q25, q75, color, label, width=0.35):
    ax.bar(x_pos, medians, width, color=color, alpha=0.85, label=label)
    ax.errorbar(
        x_pos, medians,
        yerr=[np.clip(medians - q25, 0, None), np.clip(q75 - medians, 0, None)],
        fmt="none", color=color, capsize=3, lw=1.2, alpha=0.9,
    )


def save_to_github(run_record, token, owner, repo, branch, filepath):
    """Write a new file to a GitHub repo via the Contents API."""
    content = json.dumps(run_record, indent=2)
    encoded = base64.b64encode(content.encode()).decode()
    url     = f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}"
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
    }
    payload = {
        "message": f"Add VE simulation run {filepath.split('/')[-1].replace('.json','')}",
        "content": encoded,
        "branch":  branch,
    }
    try:
        r = requests.put(url, headers=headers, json=payload, timeout=15)
        return r.status_code, r.json()
    except Exception as e:
        return None, {"error": str(e)}


def safe_float(x):
    """Return Python float or None (for NaN), safe for JSON serialisation."""
    try:
        f = float(x)
        return None if (f != f) else f   # f != f is True only for NaN
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GitHub credentials from Streamlit secrets (optional)
# ─────────────────────────────────────────────────────────────────────────────
try:
    GH_TOKEN  = st.secrets["GITHUB_TOKEN"]
    GH_OWNER  = st.secrets["GITHUB_OWNER"]
    GH_REPO   = st.secrets["GITHUB_REPO"]
    GH_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
    github_configured = True
except Exception:
    github_configured = False


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
    n_runs    = int(st.number_input("Runs",        value=100, min_value=1, max_value=2000, step=50))
    seed      = int(st.number_input("Random seed", value=42,  min_value=0))
    run_label = st.text_input(
        "Run label / notes",
        placeholder="e.g. 'Base case, V=0.5'",
        help="Included in saved JSON output so you can tell runs apart.",
    )
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

    st.caption(f"Target: **{n_target:,}** · step ±1 or type · all 6 must sum to target")

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
# Distribution panels
# ─────────────────────────────────────────────────────────────────────────────
st.header("Partner Count Distributions")
left, right = st.columns(2)
with left:
    vax_counts, vax_miss, vax_ok = dist_panel("vax", "💉 Vaccinated", N_vax, "miss_vax")
with right:
    pbo_counts, pbo_miss, pbo_ok = dist_panel("pbo", "🧪 Placebo / Unvaccinated", N_placebo, "miss_pbo")


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
# Run simulation — store all results in session_state so they persist
# across button-click re-renders (needed for the save buttons to work)
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not vax_ok or not pbo_ok:
        if not vax_ok: st.error("❌ Vaccinated group counts do not sum to N_vax")
        if not pbo_ok: st.error("❌ Placebo group counts do not sum to N_placebo")
        st.stop()

    vax_props = get_adjusted_props(vax_counts, vax_miss)
    pbo_props = get_adjusted_props(pbo_counts, pbo_miss)

    rng     = np.random.default_rng(seed)
    results = []
    vax_bi_all, pbo_bi_all = [], []
    vax_bn_all, pbo_bn_all = [], []

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

    df     = pd.DataFrame(results)
    vax_bi = np.array(vax_bi_all)
    pbo_bi = np.array(pbo_bi_all)
    vax_bn = np.array(vax_bn_all)
    pbo_bn = np.array(pbo_bn_all)

    with np.errstate(divide="ignore", invalid="ignore"):
        vax_ci_bucket = np.where(vax_bn > 0, vax_bi / vax_bn, np.nan)
        pbo_ci_bucket = np.where(pbo_bn > 0, pbo_bi / pbo_bn, np.nan)
        vax_tot = vax_bi.sum(axis=1, keepdims=True)
        pbo_tot = pbo_bi.sum(axis=1, keepdims=True)
        vax_pct = np.where(vax_tot > 0, vax_bi / vax_tot, np.nan)
        pbo_pct = np.where(pbo_tot > 0, pbo_bi / pbo_tot, np.nan)

    n_nan = int(df.ve.isna().sum())
    ve_lo = float(df.ve.quantile(0.025))
    ve_hi = float(df.ve.quantile(0.975))

    # ── Store everything so results section renders on every re-render ────────
    st.session_state["sim_results"] = {
        "df":            df,
        "vax_bi":        vax_bi,
        "pbo_bi":        pbo_bi,
        "vax_bn":        vax_bn,
        "pbo_bn":        pbo_bn,
        "vax_ci_bucket": vax_ci_bucket,
        "pbo_ci_bucket": pbo_ci_bucket,
        "vax_pct":       vax_pct,
        "pbo_pct":       pbo_pct,
        "n_nan":         n_nan,
        "ve_lo":         ve_lo,
        "ve_hi":         ve_hi,
        "params": {
            "N_tot":         N_tot,
            "N_vax":         N_vax,
            "N_placebo":     N_placebo,
            "upper_lo":      upper_lo,
            "upper_hi":      upper_hi,
            "bucket_ranges": bucket_ranges,
            "p_ci":          p_ci,
            "p_t":           p_t,
            "V":             V,
            "n_runs":        n_runs,
            "seed":          seed,
            "run_label":     run_label,
            "vax_counts":    vax_counts,
            "pbo_counts":    pbo_counts,
            "vax_miss":      vax_miss,
            "pbo_miss":      pbo_miss,
            "vax_props":     vax_props,
            "pbo_props":     pbo_props,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Render results (reads from session_state — persists across all re-renders)
# ─────────────────────────────────────────────────────────────────────────────
if "sim_results" in st.session_state:
    res           = st.session_state["sim_results"]
    df            = res["df"]
    vax_bi        = res["vax_bi"]
    pbo_bi        = res["pbo_bi"]
    vax_bn        = res["vax_bn"]
    pbo_bn        = res["pbo_bn"]
    vax_ci_bucket = res["vax_ci_bucket"]
    pbo_ci_bucket = res["pbo_ci_bucket"]
    vax_pct       = res["vax_pct"]
    pbo_pct       = res["pbo_pct"]
    n_nan         = res["n_nan"]
    ve_lo         = res["ve_lo"]
    ve_hi         = res["ve_hi"]
    p             = res["params"]

    if n_nan:
        st.warning(
            f"{n_nan} run(s) had zero placebo infections → CIR/VE undefined. "
            "Excluded from summaries and plots."
        )

    # ── Top-line metrics ──────────────────────────────────────────────────────
    st.header("Results")
    if p["run_label"]:
        st.caption(f"Showing results for run: **{p['run_label']}**")

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
    ax.set_xlabel("Proportion Infected"); ax.set_ylabel("Count")
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

    # ── Figure 2: Infections by bucket ─────────────────────────────────────────
    st.subheader("Infections by Partner Count Bucket")
    st.caption("Medians across simulation runs; error bars = IQR (25th–75th percentile).")

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(5)
    w = 0.35

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
    _ulo = p["upper_lo"];  _uhi = p["upper_hi"]
    _pci = p["p_ci"];      _pt  = p["p_t"];     _V = p["V"]

    st.subheader(">50 Partner Bucket: Within-Bucket Detail")
    st.caption(
        f"**Left:** distribution of total partners across all 4 periods for a person in the "
        f">50 bucket (Discrete Uniform {_ulo}–{_uhi} per period; 200k analytical draws). "
        f"**Right:** theoretical cumulative incidence as a function of total partners."
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    rng_vis = np.random.default_rng(0)
    n_vis   = 200_000
    span_b4 = _uhi - _ulo + 1
    tot_part = np.sum(
        _ulo + np.floor(rng_vis.random((n_vis, N_PERIODS)) * span_b4).astype(int),
        axis=1,
    )
    med_tot = int(np.median(tot_part))

    ax = axes3[0]
    ax.hist(tot_part, bins=min(100, span_b4 * N_PERIODS),
            color="mediumpurple", edgecolor="white", lw=0.3, density=True)
    ax.axvline(med_tot, color="black", ls="--", lw=1.5,
               label=f"Median: {med_tot} partners")
    ax.set_xlabel("Total Partners Across 4 Periods"); ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Total Partners\n(>50 bucket: {_ulo}–{_uhi} per period)")
    ax.legend(fontsize=8)

    t_range = np.arange(N_PERIODS * _ulo, N_PERIODS * _uhi + 1)
    ax = axes3[1]
    ax.plot(t_range, 1.0 - (1.0 - _pci * _pt) ** t_range,
            color="tomato",    lw=2, label="Placebo")
    ax.plot(t_range, 1.0 - (1.0 - _pci * _pt * (1.0 - _V)) ** t_range,
            color="steelblue", lw=2, label="Vaccinated")
    ax.axvline(med_tot, color="black", ls="--", lw=1.5, alpha=0.7,
               label=f"Median: {med_tot} partners")
    ax.set_xlabel("Total Partners Across 4 Periods")
    ax.set_ylabel("Proportion Infected")
    ax.set_title(f"Cumulative Incidence by Total Partners\n(>50 bucket: {_ulo}–{_uhi} per period)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Summary table ──────────────────────────────────────────────────────────
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

    with st.expander("Per-bucket infection detail (medians across runs)"):
        bk_df = pd.DataFrame({
            "Bucket":                        ACTIVE_LABELS,
            "Range":                         [f"{r[0]}–{r[1]}" for r in p["bucket_ranges"]],
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

    with st.expander("Effective bucket proportions used in this run"):
        st.dataframe(
            pd.DataFrame({
                "Bucket":  ACTIVE_LABELS,
                "Range":   [f"{r[0]}–{r[1]}" for r in p["bucket_ranges"]],
                "Vax":     [f"{prop:.4f}" for prop in p["vax_props"]],
                "Placebo": [f"{prop:.4f}" for prop in p["pbo_props"]],
            }),
            hide_index=True,
            use_container_width=True,
        )

    # ── Save results ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Save This Run")
    st.caption(
        "The JSON file includes all parameters and summary statistics for this run. "
        "Add a label in the sidebar before running to keep scenarios organised."
    )

    # Build JSON-safe run record
    run_record = {
        "timestamp": datetime.now().isoformat(),
        "label":     p["run_label"] or "(unlabelled)",
        "parameters": {
            "N_tot":              p["N_tot"],
            "N_vax":              p["N_vax"],
            "N_placebo":          p["N_placebo"],
            "vax_counts":         p["vax_counts"],
            "pbo_counts":         p["pbo_counts"],
            "vax_missing_mode":   p["vax_miss"],
            "pbo_missing_mode":   p["pbo_miss"],
            "upper_lo":           p["upper_lo"],
            "upper_hi":           p["upper_hi"],
            "p_contact_infected": p["p_ci"],
            "p_transmission":     p["p_t"],
            "V_per_contact":      p["V"],
            "n_runs":             p["n_runs"],
            "seed":               p["seed"],
            "bucket_labels":      BUCKET_LABELS,
        },
        "results": {
            "median_ci_vax":      safe_float(df.ci_v.median()),
            "median_ci_pbo":      safe_float(df.ci_p.median()),
            "median_cir":         safe_float(df.cir.median()),
            "median_ve":          safe_float(df.ve.median()),
            "q25_ci_vax":         safe_float(df.ci_v.quantile(0.25)),
            "q75_ci_vax":         safe_float(df.ci_v.quantile(0.75)),
            "q25_ci_pbo":         safe_float(df.ci_p.quantile(0.25)),
            "q75_ci_pbo":         safe_float(df.ci_p.quantile(0.75)),
            "q25_cir":            safe_float(df.cir.quantile(0.25)),
            "q75_cir":            safe_float(df.cir.quantile(0.75)),
            "q25_ve":             safe_float(df.ve.quantile(0.25)),
            "q75_ve":             safe_float(df.ve.quantile(0.75)),
            "ve_2_5pct":          safe_float(ve_lo),
            "ve_97_5pct":         safe_float(ve_hi),
            "mean_ci_vax":        safe_float(df.ci_v.mean()),
            "mean_ci_pbo":        safe_float(df.ci_p.mean()),
            "mean_cir":           safe_float(df.cir.mean()),
            "mean_ve":            safe_float(df.ve.mean()),
            "sd_ci_vax":          safe_float(df.ci_v.std()),
            "sd_ci_pbo":          safe_float(df.ci_p.std()),
            "sd_cir":             safe_float(df.cir.std()),
            "sd_ve":              safe_float(df.ve.std()),
            "n_runs_nan_ve":      n_nan,
        },
    }
    json_str = json.dumps(run_record, indent=2)
    ts_now   = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_msg     = st.empty()   # placeholder for success / error feedback
    scol1, scol2 = st.columns(2)

    # Download button — always available, no secrets needed
    with scol1:
        st.download_button(
            label="⬇️  Download results as JSON",
            data=json_str,
            file_name=f"ve_run_{ts_now}.json",
            mime="application/json",
            use_container_width=True,
            help="Downloads parameters + summary statistics as a JSON file.",
        )

    # GitHub save — only if secrets are configured
    with scol2:
        if github_configured:
            if st.button("📤  Save to GitHub repo", use_container_width=True,
                         help="Saves a timestamped JSON to the results/ folder of your repo."):
                gh_filename = f"results/run_{ts_now}.json"
                status_code, resp_data = save_to_github(
                    run_record, GH_TOKEN, GH_OWNER, GH_REPO, GH_BRANCH, gh_filename
                )
                if status_code == 201:
                    save_msg.success(f"✅ Saved to GitHub: `{gh_filename}`")
                else:
                    err = resp_data.get("message", "") if isinstance(resp_data, dict) else str(resp_data)
                    save_msg.error(
                        f"❌ GitHub save failed "
                        f"(HTTP {status_code if status_code else 'connection error'}) — {err}. "
                        "Check token permissions and repo name in secrets."
                    )
        else:
            with st.expander("ℹ️ Enable GitHub saving — setup instructions"):
                st.markdown("""
**Local development** — create `.streamlit/secrets.toml` (add this file to `.gitignore`):
```toml
GITHUB_TOKEN  = "ghp_xxxxxxxxxxxxxxxxxxxx"
GITHUB_OWNER  = "your-github-username"
GITHUB_REPO   = "your-repo-name"
GITHUB_BRANCH = "main"
