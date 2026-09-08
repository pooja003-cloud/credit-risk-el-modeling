"""Streamlit credit-risk monitoring dashboard.

Run from the repository root:

    streamlit run dashboard/app.py

The dashboard reads the scored test population and the result tables written by
``python -m src.run_pipeline``. Everything on screen recomputes live: move the LGD, the
credit conversion factor or the PD stress and the expected-loss figures follow, so the
weight of each assumption is visible rather than buried.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
CURRENCY = "NT$"

st.set_page_config(
    page_title="Consumer credit risk monitor",
    page_icon="\N{CHART WITH UPWARDS TREND}",
    layout="wide",
)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@st.cache_data
def load_all():
    if not (TABLES / "scored_test_accounts.csv").exists():
        return None
    data = {
        "accounts": pd.read_csv(TABLES / "scored_test_accounts.csv"),
        "results": json.loads((TABLES / "results.json").read_text()),
        "model_comparison": pd.read_csv(TABLES / "model_comparison.csv"),
        "deciles": pd.read_csv(TABLES / "risk_deciles.csv"),
        "calibration": pd.read_csv(TABLES / "calibration_champion.csv"),
        "scenarios": pd.read_csv(TABLES / "scenario_summary.csv"),
        "segment_perf": pd.read_csv(TABLES / "segment_performance.csv"),
        "margin_sens": pd.read_csv(TABLES / "cutoff_margin_sensitivity.csv"),
        "feature_sets": pd.read_csv(TABLES / "feature_set_comparison.csv"),
    }
    for name in ("permutation_importance", "shap_global_importance", "logistic_odds_ratios"):
        path = TABLES / f"{name}.csv"
        if path.exists():
            data[name] = pd.read_csv(path)
    return data


DATA = load_all()
if DATA is None:
    st.error(
        "No model output found. Run `python -m src.run_pipeline` from the repository root "
        "first - the dashboard reads the tables it writes into `outputs/tables/`."
    )
    st.stop()

accounts_all = DATA["accounts"]
results = DATA["results"]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def stress_pd(p: np.ndarray, multiplier: float) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    odds = p / (1 - p) * multiplier
    return odds / (1 + odds)


def money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{CURRENCY}{x / 1e9:,.2f}bn"
    if abs(x) >= 1e6:
        return f"{CURRENCY}{x / 1e6:,.1f}m"
    return f"{CURRENCY}{x:,.0f}"


def shade(styler, subset):
    """Apply a background gradient if matplotlib is available, otherwise return unchanged.

    pandas delegates ``Styler.background_gradient`` to matplotlib. The shading is a reading
    aid, not information - the numbers are already in the table - so a missing optional
    dependency should not take the dashboard down with it.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return styler
    return styler.background_gradient(cmap="Blues", subset=subset)


def base_layout(fig, height=360, ylab=None, xlab=None):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, title_text=xlab, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.22)", title_text=ylab, zeroline=False)
    return fig


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.title("Risk settings")
st.sidebar.caption(
    "PD comes from the model. LGD and the credit conversion factor are assumptions - "
    "move them and watch how much of the loss number they carry."
)

scenario_names = DATA["scenarios"]["scenario"].tolist()
preset = st.sidebar.selectbox("Scenario preset", scenario_names, index=0)
preset_row = DATA["scenarios"].set_index("scenario").loc[preset]

pd_mult = st.sidebar.slider(
    "PD stress - multiplier on the odds of default",
    1.0, 4.0, float(preset_row["pd_odds_multiplier"]), 0.05,
    help="Applied on the log-odds scale, so PDs stay inside (0, 1) and ranking is preserved.",
)
lgd = st.sidebar.slider("Loss given default (assumed)", 0.20, 1.00,
                        float(preset_row["lgd_assumed"]), 0.05)
ccf = st.sidebar.slider("Credit conversion factor on the undrawn line (assumed)", 0.0, 1.0,
                        float(preset_row["ccf_assumed"]), 0.05)
cutoff = st.sidebar.slider(
    "Decline cutoff on predicted PD", 0.05, 0.95,
    float(results["thresholds"]["policy_cutoff_from_validation"]), 0.01,
    help="The saved policy cutoff was chosen on the validation fold to hold the approved "
         "book's default rate at or below the target risk appetite.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Population filter")
seg_filters = {}
for col, label in [
    ("limit_band", "Credit-limit quintile"),
    ("age_band", "Age band"),
    ("education_label", "Education"),
    ("marriage_label", "Marital status"),
    ("sex_label", "Sex"),
]:
    options = sorted(accounts_all[col].dropna().unique().tolist())
    chosen = st.sidebar.multiselect(label, options, default=options)
    seg_filters[col] = chosen

mask = np.ones(len(accounts_all), dtype=bool)
for col, chosen in seg_filters.items():
    mask &= accounts_all[col].isin(chosen).to_numpy()
acc = accounts_all[mask].copy()

if acc.empty:
    st.warning("No accounts match the current filter.")
    st.stop()

# Recompute exposure and expected loss under the chosen assumptions.
acc["pd_stressed"] = stress_pd(acc["pd"].to_numpy(), pd_mult)
drawn = acc["BILL_AMT1"].clip(lower=0).clip(upper=acc["LIMIT_BAL"])
acc["ead_live"] = drawn + ccf * (acc["LIMIT_BAL"] - drawn).clip(lower=0)
acc["el_live"] = acc["pd_stressed"] * lgd * acc["ead_live"]
acc["declined"] = acc["pd_stressed"] >= cutoff

total_ead = float(acc["ead_live"].sum())
total_el = float(acc["el_live"].sum())
approved = ~acc["declined"]


# --------------------------------------------------------------------------------------
# Header and KPI row
# --------------------------------------------------------------------------------------
st.title("Consumer credit risk monitor")
st.caption(
    f"Champion model: **{results['champion_model']}**  ·  scored population: "
    f"{len(acc):,} held-out accounts  ·  pipeline run {results['generated_on']}  ·  "
    "UCI Default of Credit Card Clients (Taiwan, 2005), amounts in New Taiwan dollars."
)

k = st.columns(5)
k[0].metric("Exposure at default", money(total_ead),
            help="Drawn balance plus the assumed drawn share of the undrawn line.")
k[1].metric("Expected loss", money(total_el),
            f"{total_el / total_ead:.2%} of exposure" if total_ead else "-")
k[2].metric("Exposure-weighted PD", f"{np.average(acc['pd_stressed'], weights=acc['ead_live']):.1%}")
k[3].metric("Approval rate at cutoff", f"{approved.mean():.1%}",
            f"{int((~approved).sum()):,} accounts declined", delta_color="off")
k[4].metric(
    "Default rate of approved book",
    f"{acc.loc[approved, 'default'].mean():.1%}" if approved.any() else "-",
    f"{acc.loc[approved, 'default'].mean() - acc['default'].mean():+.1%} vs no cutoff"
    if approved.any() else None,
    delta_color="inverse",
)

if abs(pd_mult - 1.0) > 1e-6 or abs(lgd - float(DATA['scenarios'].iloc[0]['lgd_assumed'])) > 1e-6:
    st.info(
        f"Stressed view: default odds x{pd_mult:.2f}, LGD {lgd:.0%}, CCF {ccf:.0%}. "
        "LGD and CCF are assumptions, not estimates from this dataset - no recovery or "
        "collection data exists in it.",
        icon="\N{WARNING SIGN}",
    )

tabs = st.tabs(
    ["Portfolio", "Model performance", "Expected loss", "Scenarios", "Drivers", "Assumptions"]
)


# --------------------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Risk distribution")
        band = acc.groupby("risk_decile", observed=True).agg(
            accounts=("pd", "size"),
            observed_default_rate=("default", "mean"),
            mean_pd=("pd_stressed", "mean"),
            exposure=("ead_live", "sum"),
            expected_loss=("el_live", "sum"),
        ).reset_index()

        fig = go.Figure()
        fig.add_bar(
            x=band["risk_decile"], y=band["observed_default_rate"],
            marker_color=[SEQUENTIAL[min(int(i * len(SEQUENTIAL) / 10), len(SEQUENTIAL) - 1)]
                          for i in range(len(band))],
            name="Observed default rate",
            text=[f"{v:.0%}" for v in band["observed_default_rate"]],
            textposition="outside",
            hovertemplate="Decile %{x}<br>Observed %{y:.1%}<extra></extra>",
        )
        fig.add_scatter(
            x=band["risk_decile"], y=band["mean_pd"], mode="lines+markers",
            line=dict(color=SERIES[1], width=2.5), marker=dict(size=8),
            name="Mean predicted PD",
            hovertemplate="Decile %{x}<br>Predicted %{y:.1%}<extra></extra>",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(base_layout(fig, 380, "Default rate", "Predicted-risk decile"),
                        width="stretch")
        st.caption(
            "Bars are what happened; the line is what the model said would happen. "
            "A monotone climb is the evidence that the model rank-orders risk."
        )

    with right:
        st.subheader("Delinquency profile")
        prof = acc.copy()
        prof["worst_arrears"] = prof["dpd_max"].clip(upper=4)
        labels = {0: "Never late", 1: "1 month", 2: "2 months", 3: "3 months", 4: "4+ months"}
        g = prof.groupby("worst_arrears", observed=True).agg(
            accounts=("pd", "size"), default_rate=("default", "mean"),
            exposure=("ead_live", "sum"),
        ).reset_index()
        fig = go.Figure()
        fig.add_bar(
            x=[labels[v] for v in g["worst_arrears"]], y=g["default_rate"],
            marker_color=SEQUENTIAL[3],
            text=[f"{v:.0%}" for v in g["default_rate"]], textposition="outside",
            hovertemplate="%{x}<br>Default rate %{y:.1%}<extra></extra>",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(base_layout(fig, 380, "Default rate next month",
                                    "Worst arrears in the last six months"),
                        width="stretch")
        st.caption("The strongest single signal in the data, and the one a collections team acts on.")

    st.subheader("Segment monitoring")
    seg_choice = st.selectbox(
        "Segment", ["limit_band", "age_band", "education_label", "marriage_label", "sex_label"],
        format_func=lambda c: {
            "limit_band": "Credit-limit quintile", "age_band": "Age band",
            "education_label": "Education", "marriage_label": "Marital status",
            "sex_label": "Sex",
        }[c],
    )
    seg = acc.groupby(seg_choice, observed=True).agg(
        accounts=("pd", "size"),
        observed_default_rate=("default", "mean"),
        mean_pd=("pd_stressed", "mean"),
        exposure=("ead_live", "sum"),
        expected_loss=("el_live", "sum"),
    ).reset_index()
    seg["share_of_exposure"] = seg["exposure"] / seg["exposure"].sum()
    seg["share_of_expected_loss"] = seg["expected_loss"] / seg["expected_loss"].sum()
    seg["loss_concentration"] = seg["share_of_expected_loss"] / seg["share_of_exposure"]

    fig = go.Figure()
    fig.add_bar(x=seg[seg_choice].astype(str), y=seg["share_of_exposure"],
                name="Share of exposure", marker_color=SERIES[0],
                text=[f"{v:.0%}" for v in seg["share_of_exposure"]], textposition="outside")
    fig.add_bar(x=seg[seg_choice].astype(str), y=seg["share_of_expected_loss"],
                name="Share of expected loss", marker_color=SERIES[1],
                text=[f"{v:.0%}" for v in seg["share_of_expected_loss"]], textposition="outside")
    fig.update_layout(barmode="group", bargap=0.28)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(base_layout(fig, 340, "Share of portfolio"), width="stretch")
    st.caption("A segment whose loss share exceeds its exposure share is concentrating risk.")

    st.dataframe(
        seg.style.format({
            "observed_default_rate": "{:.2%}", "mean_pd": "{:.2%}",
            "exposure": "{:,.0f}", "expected_loss": "{:,.0f}",
            "share_of_exposure": "{:.1%}", "share_of_expected_loss": "{:.1%}",
            "loss_concentration": "{:.2f}x",
        }),
        width="stretch", hide_index=True,
    )


# --------------------------------------------------------------------------------------
# Model performance
# --------------------------------------------------------------------------------------
with tabs[1]:
    head = results["headline"]
    m = st.columns(5)
    m[0].metric("ROC-AUC", f"{head['champion_roc_auc']:.3f}", f"Gini {head['champion_gini']:.3f}")
    m[1].metric("PR-AUC", f"{head['champion_pr_auc']:.3f}",
                f"base rate {head['test_default_rate']:.1%}", delta_color="off")
    m[2].metric("KS", f"{head['champion_ks']:.3f}")
    m[3].metric("Brier score", f"{head['champion_brier']:.4f}", "lower is better",
                delta_color="off")
    m[4].metric("Calibration error", f"{head['champion_ece']:.4f}",
                "mean predicted vs observed", delta_color="off")

    st.subheader("Model comparison")
    st.caption(
        "Selected on the validation fold, reported on the test fold. Accuracy is deliberately "
        "not a headline: approving everyone scores 78% accurate on this portfolio."
    )
    comp = DATA["model_comparison"]
    show = comp[["model", "val_auc", "roc_auc", "pr_auc", "ks", "brier", "ece",
                 "at_cutoff_precision", "at_cutoff_recall", "at_cutoff_f1"]].rename(columns={
        "model": "Model", "val_auc": "Validation ROC-AUC", "roc_auc": "Test ROC-AUC",
        "pr_auc": "PR-AUC", "ks": "KS", "brier": "Brier", "ece": "Calibration error",
        "at_cutoff_precision": "Precision at cutoff", "at_cutoff_recall": "Recall at cutoff",
        "at_cutoff_f1": "F1 at cutoff",
    })
    st.dataframe(
        shade(show.style.format({c: "{:.4f}" for c in show.columns if c != "Model"}),
              subset=["Test ROC-AUC", "PR-AUC"]),
        width="stretch", hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Calibration")
        cal = DATA["calibration"]
        fig = go.Figure()
        fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers",
                        line=dict(color=SERIES[0], width=2.5), marker=dict(size=9),
                        name="Champion model",
                        hovertemplate="predicted %{x:.1%}<br>observed %{y:.1%}<extra></extra>")
        lim = float(max(cal["predicted"].max(), cal["observed"].max())) * 1.05
        fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect calibration",
                        line=dict(color="#7f7f7f", width=1.5, dash="dash"))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(base_layout(fig, 340, "Observed default rate", "Mean predicted PD"),
                        width="stretch")
        st.caption(
            "Expected loss multiplies PD *levels*, so points must sit on the diagonal - "
            "ranking correctly is not enough."
        )
    with c2:
        st.subheader("Decision outcomes at the chosen cutoff")
        y = acc["default"].to_numpy().astype(bool)
        d = acc["declined"].to_numpy()
        cells = pd.DataFrame(
            {
                "Approved": [int((~d & ~y).sum()), int((~d & y).sum())],
                "Declined": [int((d & ~y).sum()), int((d & y).sum())],
            },
            index=["Actually paid", "Actually defaulted"],
        )
        fig = go.Figure(
            go.Heatmap(
                z=cells.to_numpy(), x=cells.columns, y=cells.index,
                colorscale=[[0, SEQUENTIAL[0]], [1, SEQUENTIAL[-1]]], showscale=False,
                text=[[f"{v:,}" for v in row] for row in cells.to_numpy()],
                texttemplate="%{text}", textfont=dict(size=20),
                hovertemplate="%{y} / %{x}: %{z:,} accounts<extra></extra>",
            )
        )
        st.plotly_chart(base_layout(fig, 340), width="stretch")
        st.caption(
            "Top-right: good customers turned away. Bottom-left: defaults let through. "
            "The second costs roughly ten times the first under the assumptions in the sidebar."
        )

    st.subheader("Does the model work everywhere?")
    st.caption(
        "Portfolio-level AUC can hide a model that works on one population and fails on "
        "another. Segments with fewer than 50 accounts or 5 defaults are omitted."
    )
    sp = DATA["segment_perf"]
    st.dataframe(
        shade(sp.style.format({
            "observed_default_rate": "{:.2%}", "mean_predicted_pd": "{:.2%}",
            "roc_auc": "{:.3f}", "brier": "{:.4f}", "calibration_gap": "{:+.3%}",
        }), subset=["roc_auc"]),
        width="stretch", hide_index=True,
    )


# --------------------------------------------------------------------------------------
# Expected loss
# --------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Expected loss = PD x LGD x EAD")
    st.caption(
        f"Live under the sidebar assumptions: PD odds x{pd_mult:.2f}, LGD {lgd:.0%}, CCF {ccf:.0%}."
    )

    band = acc.groupby("risk_decile", observed=True).agg(
        accounts=("pd", "size"), mean_pd=("pd_stressed", "mean"),
        exposure=("ead_live", "sum"), expected_loss=("el_live", "sum"),
        observed_default_rate=("default", "mean"),
    ).reset_index()
    band["el_rate_on_exposure"] = band["expected_loss"] / band["exposure"]
    band["share_of_expected_loss"] = band["expected_loss"] / band["expected_loss"].sum()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=band["risk_decile"], y=band["expected_loss"] / 1e6,
                    marker_color=[SEQUENTIAL[min(int(i * len(SEQUENTIAL) / 10), len(SEQUENTIAL) - 1)]
                                  for i in range(len(band))],
                    text=[f"{v / 1e6:,.0f}" for v in band["expected_loss"]],
                    textposition="outside",
                    hovertemplate="Decile %{x}<br>EL " + CURRENCY + "%{y:,.1f}m<extra></extra>")
        st.plotly_chart(base_layout(fig, 340, f"Expected loss ({CURRENCY} m)",
                                    "Predicted-risk decile"), width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=band["risk_decile"], y=band["el_rate_on_exposure"],
                    marker_color=[SEQUENTIAL[min(int(i * len(SEQUENTIAL) / 10), len(SEQUENTIAL) - 1)]
                                  for i in range(len(band))],
                    text=[f"{v:.0%}" for v in band["el_rate_on_exposure"]],
                    textposition="outside",
                    hovertemplate="Decile %{x}<br>EL rate %{y:.1%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(base_layout(fig, 340, "Expected loss / exposure",
                                    "Predicted-risk decile"), width="stretch")

    st.dataframe(
        band.style.format({
            "mean_pd": "{:.2%}", "observed_default_rate": "{:.2%}",
            "exposure": "{:,.0f}", "expected_loss": "{:,.0f}",
            "el_rate_on_exposure": "{:.2%}", "share_of_expected_loss": "{:.1%}",
        }),
        width="stretch", hide_index=True,
    )

    st.subheader("Where the cutoff sits")
    grid = np.quantile(acc["pd_stressed"], np.linspace(0.02, 0.99, 120))
    rows = []
    for t in grid:
        ap = acc["pd_stressed"] < t
        if ap.sum() < 30:
            continue
        rows.append({
            "cutoff": float(t),
            "approval_rate": float(ap.mean()),
            "bad_rate_of_approved": float(acc.loc[ap, "default"].mean()),
            "expected_loss_retained": float(acc.loc[ap, "el_live"].sum()),
        })
    frontier = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=frontier["approval_rate"], y=frontier["bad_rate_of_approved"],
                    mode="lines", line=dict(color=SERIES[0], width=2.5), name="Frontier",
                    hovertemplate="approve %{x:.0%}<br>bad rate %{y:.1%}<extra></extra>")
    here = frontier.iloc[(frontier["cutoff"] - cutoff).abs().argmin()]
    fig.add_scatter(x=[here["approval_rate"]], y=[here["bad_rate_of_approved"]],
                    mode="markers+text", marker=dict(size=14, color=SERIES[1]),
                    text=[f"  cutoff {cutoff:.0%}"], textposition="middle right",
                    name="Current cutoff")
    fig.add_hline(y=float(acc["default"].mean()), line_dash="dash", line_color="#7f7f7f",
                  annotation_text=f"approve everyone: {acc['default'].mean():.1%}")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(base_layout(fig, 360, "Default rate of approved book", "Approval rate"),
                    width="stretch")
    st.caption(
        "Retrospective on held-out accounts, not a claim about live lending decisions: it shows "
        "what this cutoff would have done to this population."
    )


# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Scenario analysis")
    st.caption(
        "Each scenario moves PD, LGD and the credit conversion factor together, because in a "
        "downturn they do not move independently. The magnitudes are judgemental and are not "
        "calibrated to a published macroeconomic model."
    )

    live_rows = []
    for _, r in DATA["scenarios"].iterrows():
        p = stress_pd(acc["pd"].to_numpy(), float(r["pd_odds_multiplier"]))
        dr = acc["BILL_AMT1"].clip(lower=0).clip(upper=acc["LIMIT_BAL"])
        ead = dr + float(r["ccf_assumed"]) * (acc["LIMIT_BAL"] - dr).clip(lower=0)
        el = float((p * float(r["lgd_assumed"]) * ead).sum())
        live_rows.append({
            "Scenario": r["scenario"],
            "PD odds multiplier": float(r["pd_odds_multiplier"]),
            "LGD assumed": float(r["lgd_assumed"]),
            "CCF assumed": float(r["ccf_assumed"]),
            "Average PD": float(p.mean()),
            "Exposure": float(ead.sum()),
            "Expected loss": el,
            "EL rate on exposure": el / float(ead.sum()),
            "Narrative": r["narrative"],
        })
    live = pd.DataFrame(live_rows)
    live["vs baseline"] = live["Expected loss"] / live["Expected loss"].iloc[0] - 1

    fig = go.Figure()
    fig.add_bar(
        x=live["Scenario"], y=live["Expected loss"] / 1e6,
        marker_color=[SERIES[0], "#eda100", SERIES[7]][: len(live)],
        text=[f"{CURRENCY}{v / 1e6:,.0f}m" + ("" if i == 0 else f"<br>+{live['vs baseline'].iloc[i]:.0%}")
              for i, v in enumerate(live["Expected loss"])],
        textposition="outside",
        hovertemplate="%{x}<br>EL " + CURRENCY + "%{y:,.1f}m<extra></extra>",
    )
    st.plotly_chart(base_layout(fig, 380, f"Expected loss ({CURRENCY} m)"),
                    width="stretch")

    st.dataframe(
        live.drop(columns="Narrative").style.format({
            "PD odds multiplier": "{:.2f}x", "LGD assumed": "{:.0%}", "CCF assumed": "{:.0%}",
            "Average PD": "{:.2%}", "Exposure": "{:,.0f}", "Expected loss": "{:,.0f}",
            "EL rate on exposure": "{:.2%}", "vs baseline": "{:+.0%}",
        }),
        width="stretch", hide_index=True,
    )
    for _, r in live.iterrows():
        st.markdown(f"**{r['Scenario']}** - {r['Narrative']}")

    st.subheader("How much of the answer is the LGD assumption?")
    mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    lgds = [0.45, 0.55, 0.65, 0.75, 0.85]
    dr = acc["BILL_AMT1"].clip(lower=0).clip(upper=acc["LIMIT_BAL"])
    ead_fixed = (dr + ccf * (acc["LIMIT_BAL"] - dr).clip(lower=0)).to_numpy()
    z = [[float((stress_pd(acc["pd"].to_numpy(), m) * l * ead_fixed).sum()) / 1e6 for m in mults]
         for l in lgds]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{m:g}x" for m in mults], y=[f"{l:.0%}" for l in lgds],
        colorscale=[[0, SEQUENTIAL[0]], [0.5, SEQUENTIAL[4]], [1, SEQUENTIAL[-1]]],
        texttemplate="%{z:,.0f}", textfont=dict(size=11),
        colorbar=dict(title=f"EL ({CURRENCY} m)"),
        hovertemplate="PD stress %{x}<br>LGD %{y}<br>EL " + CURRENCY + "%{z:,.0f}m<extra></extra>",
    ))
    st.plotly_chart(base_layout(fig, 340, "Assumed LGD", "PD stress (odds multiplier)"),
                    width="stretch")


# --------------------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("What drives the score")
    c1, c2 = st.columns(2)

    with c1:
        if "shap_global_importance" in DATA:
            t = DATA["shap_global_importance"].head(15).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=t["mean_abs_shap"], y=t["feature"], orientation="h",
                marker_color=SERIES[0],
                hovertemplate="%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>",
            ))
            st.plotly_chart(base_layout(fig, 520, None, "Mean |SHAP| (log-odds)"),
                            width="stretch")
            st.caption("Global SHAP importance for the champion model, 2,000-account sample.")
        elif "permutation_importance" in DATA:
            t = DATA["permutation_importance"].head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=t["auc_drop_mean"], y=t["feature"], orientation="h",
                                   marker_color=SERIES[0]))
            st.plotly_chart(base_layout(fig, 520, None, "ROC-AUC drop when permuted"),
                            width="stretch")

    with c2:
        if "logistic_odds_ratios" in DATA:
            t = DATA["logistic_odds_ratios"].head(15).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=t["odds_ratio"] - 1, base=1, y=t["feature"], orientation="h",
                marker_color=[SERIES[7] if v > 1 else SERIES[0] for v in t["odds_ratio"]],
                hovertemplate="%{y}<br>odds ratio %{x:.2f}<extra></extra>",
            ))
            fig.add_vline(x=1, line_color="#7f7f7f", line_width=1)
            st.plotly_chart(base_layout(fig, 520, None, "Odds ratio (>1 raises default odds)"),
                            width="stretch")
            st.caption(
                "Logistic-regression effects, per standard deviation of the input or versus the "
                "reference category. The scorecard view a credit committee can audit line by line."
            )

    st.subheader("How much of the model is repayment history?")
    fs = DATA["feature_sets"].copy()
    fs["feature_set"] = fs["feature_set"].map({
        "behavioural": "Behavioural (six months of history)",
        "origination": "Origination only (limit + demographics)",
    })
    st.dataframe(
        fs.style.format({"roc_auc": "{:.4f}", "gini": "{:.4f}", "pr_auc": "{:.4f}",
                         "ks": "{:.4f}", "brier": "{:.4f}"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        "The behavioural model is not an application scorecard. At origination none of the "
        "repayment history exists, and the same model class loses most of its power - which is "
        "the leakage boundary this project is careful about."
    )


# --------------------------------------------------------------------------------------
# Assumptions
# --------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("What is estimated, and what is assumed")
    st.markdown(
        """
| Component | Source | Status |
| --- | --- | --- |
| **PD** | Gradient-boosting model, isotonic-calibrated on the validation fold | **Estimated** from data |
| **EAD** | Drawn balance plus a credit conversion factor on the undrawn line | Balance and line **observed**; CCF **assumed** |
| **LGD** | Sidebar setting | **Assumed** - the dataset has no recovery, collection or charge-off data |
| **Margin on a good account** | Sidebar / config | **Assumed** - used only to test a cutoff, never reported as revenue |
| **Scenario severity** | Config | **Judgemental** - not calibrated to a published macroeconomic model |
"""
    )
    st.warning(
        "No loss given default can be estimated from this dataset. Every loss figure in this "
        "dashboard is a PD estimate multiplied by an assumed LGD, and moves one-for-one with "
        "that assumption.",
        icon="\N{WARNING SIGN}",
    )

    st.subheader("Cutoff economics are sensitive to the margin assumption")
    st.dataframe(
        DATA["margin_sens"].style.format({
            "assumed_margin_on_good_account": "{:.0%}", "implied_cutoff": "{:.1%}",
            "approval_rate": "{:.1%}", "bad_rate_of_approved": "{:.2%}", "net_cost": "{:,.0f}",
        }),
        width="stretch", hide_index=True,
    )
    st.caption(
        "A one-period loss weighed against one period of margin will always recommend an "
        "implausibly tight policy. That is why the deployed cutoff is set by risk appetite - a "
        "target default rate on the approved book - and the cost calculation is shown as a "
        "sensitivity rather than as the answer."
    )

    st.subheader("Known limitations")
    st.markdown(
        """
- **One cohort, one month.** Every account is observed over the same April-September 2005
  window with an October 2005 outcome, so no out-of-time validation is possible and the split
  is random rather than time-based.
- **Taiwan, 2005.** The portfolio sits in the aftermath of a domestic card-debt crisis; a
  22% default rate is not a normal through-the-cycle rate.
- **Behavioural, not application.** The champion model needs six months of repayment history.
- **No macro variables.** Unemployment and interest rates enter only through the scenario
  multipliers, not as model inputs.
- **Demographic inputs.** Sex, education and marital status are in the dataset and are used
  here for segment monitoring. Several of them are prohibited or restricted inputs for credit
  decisions in many jurisdictions; a deployable model would exclude them and be tested for
  disparate impact.
"""
    )

st.markdown("---")
st.caption(
    "Built from the UCI Default of Credit Card Clients dataset (Taiwan, 2005). "
    "Portfolio project - not investment, credit or financial advice."
)
