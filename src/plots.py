"""Figure generation for the notebook, the memo and the dashboards.

Charting conventions used throughout, so the figures read as one set:

* one measure per axis - no chart in this project has two y-scales;
* categorical hues assigned in a fixed order from a colour-vision-deficiency
  validated palette, never cycled;
* a single-hue light-to-dark ramp wherever colour encodes magnitude (risk deciles,
  sensitivity grid) rather than identity;
* direct value labels instead of a number on every gridline, recessive axes and grid;
* every figure is also available as a CSV table in ``outputs/tables``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C

plt.rcParams.update(C.PLOT_STYLE)
plt.rcParams["axes.edgecolor"] = C.PALETTE["grid"]
plt.rcParams["grid.color"] = C.PALETTE["grid"]
plt.rcParams["text.color"] = C.PALETTE["text"]
plt.rcParams["axes.labelcolor"] = C.PALETTE["text_secondary"]
plt.rcParams["xtick.color"] = C.PALETTE["text_secondary"]
plt.rcParams["ytick.color"] = C.PALETTE["text_secondary"]
plt.rcParams["figure.facecolor"] = C.PALETTE["surface"]
plt.rcParams["axes.facecolor"] = C.PALETTE["surface"]

BLUE_RAMP = LinearSegmentedColormap.from_list("risk_blue", C.SEQUENTIAL)


def _save(fig, name: str) -> str:
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path, facecolor=C.PALETTE["surface"])
    plt.close(fig)
    return str(path)


def _title(ax, title: str, subtitle: str | None = None):
    pad = 26 if subtitle else 12
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=C.PALETTE["text"], pad=pad)
    if subtitle:
        ax.text(
            0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.8,
            color=C.PALETTE["text_secondary"], va="bottom",
        )


def _pct(x, _=None):
    return f"{x:.0%}"


# --------------------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------------------
def roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], aucs: dict[str, float], name="roc_curves"):
    """One ROC curve per model. Legend plus AUC in the label - identity is never colour alone."""
    fig, ax = plt.subplots()
    for i, (label, (y, p)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, color=C.SERIES[i % len(C.SERIES)],
                label=f"{label}  (AUC {aucs[label]:.3f})")
    ax.plot([0, 1], [0, 1], lw=1.5, ls="--", color=C.PALETTE["neutral"], label="No skill (AUC 0.500)")
    ax.set_xlabel("False positive rate - good borrowers wrongly flagged")
    ax.set_ylabel("True positive rate - defaults caught")
    ax.xaxis.set_major_formatter(_pct)
    ax.yaxis.set_major_formatter(_pct)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    _title(ax, "Model discrimination on the held-out test fold",
           "Higher and further left is better. The diagonal is a coin flip.")
    return _save(fig, name)


def pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], aps: dict[str, float], prevalence: float, name="pr_curves"):
    """Precision-recall curves - the honest view when only 22% of accounts default."""
    fig, ax = plt.subplots()
    for i, (label, (y, p)) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, lw=2, color=C.SERIES[i % len(C.SERIES)],
                label=f"{label}  (PR-AUC {aps[label]:.3f})")
    ax.axhline(prevalence, lw=1.5, ls="--", color=C.PALETTE["neutral"],
               label=f"No skill = base rate ({prevalence:.1%})")
    ax.set_xlabel("Recall - share of all defaults identified")
    ax.set_ylabel("Precision - share of flagged accounts that default")
    ax.xaxis.set_major_formatter(_pct)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    _title(ax, "Precision-recall trade-off on the test fold",
           "Accuracy is uninformative here; this is the curve that matters for a rare event.")
    return _save(fig, name)


def model_comparison_bars(table: pd.DataFrame, name="model_comparison"):
    """Grouped bars for the three headline discrimination measures."""
    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("ks", "KS")]
    models = table["model"].tolist()
    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for j, (col, label) in enumerate(metrics):
        vals = table[col].to_numpy()
        pos = x + (j - 1) * (width + 0.02)
        ax.bar(pos, vals, width, color=C.SERIES[j], label=label,
               edgecolor=C.PALETTE["surface"], linewidth=2)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                    color=C.PALETTE["text_secondary"])
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=8.5)
    ax.set_ylim(0, max(0.9, table[[m[0] for m in metrics]].to_numpy().max() + 0.12))
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    _title(ax, "Model comparison on the test fold",
           "All three measures are threshold-free. Values are direct-labelled above each bar.")
    return _save(fig, name)


def score_distributions(y_true: np.ndarray, y_prob: np.ndarray, name="score_distribution"):
    """Predicted-PD distribution split by realised outcome - the visual form of KS."""
    fig, ax = plt.subplots()
    bins = np.linspace(0, max(0.95, float(np.quantile(y_prob, 0.999))), 45)
    ax.hist(y_prob[y_true == 0], bins=bins, color=C.SERIES[0], alpha=0.85,
            label="Paid (no default)", edgecolor=C.PALETTE["surface"], linewidth=0.6)
    ax.hist(y_prob[y_true == 1], bins=bins, color=C.SERIES[1], alpha=0.85,
            label="Defaulted", edgecolor=C.PALETTE["surface"], linewidth=0.6)
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Accounts")
    ax.xaxis.set_major_formatter(_pct)
    ax.legend(frameon=False)
    _title(ax, "Predicted PD by realised outcome",
           "Separation between the two distributions is what the KS statistic measures.")
    return _save(fig, name)


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def calibration_plot(tables: dict[str, pd.DataFrame], name="calibration"):
    """Predicted versus observed default rate. The 45-degree line is perfect calibration."""
    fig, ax = plt.subplots()
    lim = 0.0
    for i, (label, t) in enumerate(tables.items()):
        ax.plot(t["predicted"], t["observed"], marker="o", ms=6, lw=2,
                color=C.SERIES[i % len(C.SERIES)], label=label)
        lim = max(lim, float(t[["predicted", "observed"]].to_numpy().max()))
    lim = min(1.0, lim * 1.1)
    ax.plot([0, lim], [0, lim], ls="--", lw=1.5, color=C.PALETTE["neutral"],
            label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted PD in bin")
    ax.set_ylabel("Observed default rate in bin")
    ax.xaxis.set_major_formatter(_pct)
    ax.yaxis.set_major_formatter(_pct)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _title(ax, "Calibration on the test fold",
           "Expected loss multiplies PD levels, so points must sit on the diagonal, not just rank correctly.")
    return _save(fig, name)


def decile_default_rates(table: pd.DataFrame, name="decile_default_rates"):
    """Observed default rate by predicted-risk decile - colour encodes magnitude."""
    fig, ax = plt.subplots()
    vals = table["observed_default_rate"].to_numpy()
    colors = BLUE_RAMP(np.linspace(0.25, 0.95, len(vals)))
    ax.bar(table["decile"].astype(int), vals, color=colors,
           edgecolor=C.PALETTE["surface"], linewidth=2)
    ax.plot(table["decile"].astype(int), table["mean_predicted_pd"], marker="o", ms=6,
            lw=2, color=C.SERIES[1], label="Mean predicted PD")
    tall = vals.max() * 0.45
    for d, v in zip(table["decile"].astype(int), vals):
        if v > tall:  # keep the label clear of the predicted-PD line
            ax.text(d, v - 0.018, f"{v:.0%}", ha="center", va="top", fontsize=8.5,
                    color="white", fontweight="bold")
        else:
            ax.text(d, v + 0.014, f"{v:.0%}", ha="center", va="bottom", fontsize=8.5,
                    color=C.PALETTE["text_secondary"])
    ax.set_xlabel("Predicted-risk decile  (1 = safest, 10 = riskiest)")
    ax.set_ylabel("Default rate")
    ax.yaxis.set_major_formatter(_pct)
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, loc="upper left")
    _title(ax, "Observed default rate by predicted-risk decile",
           "Bars are the realised outcome; the line is what the model predicted.")
    return _save(fig, name)


def confusion_matrix_plot(m: dict, name="confusion_matrix"):
    """A 2x2 confusion matrix at the chosen cutoff, labelled in business terms."""
    grid = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(grid / grid.sum(), cmap=BLUE_RAMP, vmin=0, vmax=0.75)
    labels = [
        ["Correctly approved", "Declined but would have paid\n(false positive)"],
        ["Approved and defaulted\n(false negative)", "Correctly declined"],
    ]
    for i in range(2):
        for j in range(2):
            share = grid[i, j] / grid.sum()
            ax.text(j, i - 0.13, f"{int(grid[i, j]):,}", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if share > 0.4 else C.PALETTE["text"])
            ax.text(j, i + 0.17, labels[i][j], ha="center", va="center", fontsize=8,
                    color="white" if share > 0.4 else C.PALETTE["text_secondary"])
    ax.set_xticks([0, 1], ["Approved by model", "Declined by model"])
    ax.set_yticks([0, 1], ["Actually paid", "Actually defaulted"])
    ax.grid(False)
    _title(ax, f"Decision outcomes at a {m['threshold']:.1%} PD cutoff",
           "The two error cells have very different costs; that asymmetry sets the cutoff.")
    return _save(fig, name)


def cutoff_curve(curve: pd.DataFrame, cost_min: float, policy: float, base_margin: float,
                 portfolio_bad_rate: float, margins=(0.06, 0.15, 0.30, 0.60),
                 name="cutoff_net_cost"):
    """Two views of the approve / decline decision.

    Left: net cost across cutoffs under four margin assumptions. The forgone-margin term
    scales linearly with the assumed margin, so the whole family is obtained by rescaling
    that component of the base curve. Under a single-period 6% margin the curve has no
    interior minimum at all - the arithmetic simply says "lend less" - which is the honest
    reason a cutoff cannot be read off a cost calculation alone.

    Right: the risk-return frontier the business actually negotiates - how far the bad
    rate of the approved book falls as approvals are tightened.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    loss = curve["loss_on_approved"].to_numpy()
    forgone_base = curve["forgone_margin"].to_numpy()
    for i, m in enumerate(margins):
        net = (loss + forgone_base * (m / base_margin)) / 1e6
        ax.plot(curve["threshold"], net, lw=2, color=C.SERIES[i % len(C.SERIES)],
                label=f"margin {m:.0%}")
        j = int(np.argmin(net))
        ax.scatter([curve["threshold"].iloc[j]], [net[j]], s=55, zorder=5,
                   color=C.SERIES[i % len(C.SERIES)],
                   edgecolor=C.PALETTE["surface"], linewidth=1.6)
        ax.annotate(f"{curve['threshold'].iloc[j]:.0%}",
                    xy=(curve["threshold"].iloc[j], net[j]), xytext=(4, -11),
                    textcoords="offset points", fontsize=8,
                    color=C.SERIES[i % len(C.SERIES)])
    ax.set_xlabel("Decline cutoff on predicted PD")
    ax.set_ylabel(f"Net cost ({C.CURRENCY} millions)")
    ax.xaxis.set_major_formatter(_pct)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2,
              title="Assumed margin on a good account", title_fontsize=8.5)
    _title(ax, "Cutoff economics depend on the margin assumed",
           "Dots mark each curve's minimum. At a 6% margin there is none in range.")

    ax = axes[1]
    ax.plot(curve["approval_rate"], curve["bad_rate_of_approved"], lw=2, color=C.SERIES[0])
    ax.axhline(portfolio_bad_rate, ls="--", lw=1.4, color=C.PALETTE["neutral"])
    ax.text(0.03, portfolio_bad_rate - 0.006, f"approve everyone: {portfolio_bad_rate:.1%} bad rate",
            fontsize=8.5, va="top", color=C.PALETTE["text_secondary"])
    row = curve.iloc[(curve["threshold"] - policy).abs().argmin()]
    ax.scatter([row["approval_rate"]], [row["bad_rate_of_approved"]], s=95, zorder=5,
               color=C.SERIES[1], edgecolor=C.PALETTE["surface"], linewidth=2)
    ax.annotate(
        f"chosen policy: decline PD >= {policy:.0%}\napprove {row['approval_rate']:.0%}, "
        f"bad rate {row['bad_rate_of_approved']:.1%}",
        xy=(row["approval_rate"], row["bad_rate_of_approved"]),
        xytext=(-150, -46), textcoords="offset points", fontsize=8.5,
        color=C.PALETTE["text_secondary"],
        arrowprops=dict(arrowstyle="-", color=C.PALETTE["neutral"], lw=1),
    )
    ax.set_xlabel("Approval rate")
    ax.set_ylabel("Default rate of the approved book")
    ax.xaxis.set_major_formatter(_pct)
    ax.yaxis.set_major_formatter(_pct)
    _title(ax, "The risk-return frontier",
           "Each point is one cutoff. Appetite picks the point; the model sets the curve.")
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------------------
# Expected loss and scenarios
# --------------------------------------------------------------------------------------
def el_by_decile(table: pd.DataFrame, name="el_by_decile"):
    """Expected loss and exposure share by risk decile, as small multiples (one scale each)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    d = table["risk_decile"].astype(int)

    colors = BLUE_RAMP(np.linspace(0.25, 0.95, len(d)))
    axes[0].bar(d, table["expected_loss"] / 1e6, color=colors,
                edgecolor=C.PALETTE["surface"], linewidth=2)
    for xi, v in zip(d, table["expected_loss"] / 1e6):
        axes[0].text(xi, v * 1.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5,
                     color=C.PALETTE["text_secondary"])
    axes[0].set_ylabel(f"Expected loss ({C.CURRENCY} millions)")
    axes[0].set_xlabel("Predicted-risk decile")
    axes[0].set_xticks(range(1, 11))
    _title(axes[0], "Expected loss concentration", "Baseline scenario, test fold")

    axes[1].bar(d, table["el_rate_on_ead"], color=colors,
                edgecolor=C.PALETTE["surface"], linewidth=2)
    for xi, v in zip(d, table["el_rate_on_ead"]):
        axes[1].text(xi, v * 1.02, f"{v:.1%}", ha="center", va="bottom", fontsize=7.5,
                     color=C.PALETTE["text_secondary"])
    axes[1].set_ylabel("Expected loss as a share of exposure")
    axes[1].set_xlabel("Predicted-risk decile")
    axes[1].set_xticks(range(1, 11))
    axes[1].yaxis.set_major_formatter(_pct)
    _title(axes[1], "Loss rate on exposure", "Same deciles, expressed as a rate")
    fig.tight_layout()
    return _save(fig, name)


def scenario_comparison(table: pd.DataFrame, name="scenario_comparison"):
    """Portfolio expected loss under each scenario, direct-labelled."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = table["total_expected_loss"].to_numpy() / 1e6
    colors = [C.SERIES[0], C.PALETTE["warn"], C.SERIES[7]]
    ax.bar(table["scenario"], vals, color=colors[: len(vals)], width=0.55,
           edgecolor=C.PALETTE["surface"], linewidth=2)
    base = vals[0]
    for i, (s, v) in enumerate(zip(table["scenario"], vals)):
        delta = "" if i == 0 else f"\n+{(v / base - 1):.0%} vs baseline"
        ax.text(i, v * 1.01, f"{C.CURRENCY}{v:,.0f}m{delta}", ha="center", va="bottom",
                fontsize=9.5, color=C.PALETTE["text_secondary"])
    ax.set_ylabel(f"Portfolio expected loss ({C.CURRENCY} millions)")
    ax.set_ylim(0, vals.max() * 1.25)
    _title(ax, "Expected loss under three scenarios",
           "Illustrative stresses. PD, LGD and the credit conversion factor all move together.")
    return _save(fig, name)


def sensitivity_heatmap(grid: pd.DataFrame, name="sensitivity_heatmap"):
    """PD stress x LGD assumption grid - magnitude encoded on one hue."""
    piv = grid.pivot(index="lgd", columns="pd_odds_multiplier", values="total_expected_loss") / 1e6
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    im = ax.imshow(piv.to_numpy(), cmap=BLUE_RAMP, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}x" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{r:.0%}" for r in piv.index])
    vmax = piv.to_numpy().max()
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 * vmax else C.PALETTE["text"])
    ax.set_xlabel("PD stress (multiplier on the odds of default)")
    ax.set_ylabel("Assumed loss given default")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=f"Expected loss ({C.CURRENCY} m)", fraction=0.035)
    _title(ax, "How much of the answer is the assumption?",
           "Portfolio expected loss across PD stresses and LGD assumptions. Cells are labelled.")
    return _save(fig, name)


def el_by_segment(table: pd.DataFrame, segment_label: str, name: str):
    """Exposure share versus expected-loss share for a borrower segment."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    idx = np.arange(len(table))
    w = 0.38
    ax.bar(idx - w / 2, table["share_of_ead"], w, color=C.SERIES[0], label="Share of exposure",
           edgecolor=C.PALETTE["surface"], linewidth=2)
    ax.bar(idx + w / 2, table["share_of_el"], w, color=C.SERIES[1], label="Share of expected loss",
           edgecolor=C.PALETTE["surface"], linewidth=2)
    for i, (a, b) in enumerate(zip(table["share_of_ead"], table["share_of_el"])):
        ax.text(i - w / 2, a + 0.004, f"{a:.0%}", ha="center", va="bottom", fontsize=8,
                color=C.PALETTE["text_secondary"])
        ax.text(i + w / 2, b + 0.004, f"{b:.0%}", ha="center", va="bottom", fontsize=8,
                color=C.PALETTE["text_secondary"])
    ax.set_xticks(idx, [str(v) for v in table.iloc[:, 0]], fontsize=9)
    ax.set_ylabel("Share of portfolio")
    ax.yaxis.set_major_formatter(_pct)
    ax.legend(frameon=False, ncol=2)
    _title(ax, f"Exposure versus expected loss by {segment_label}",
           "A segment whose loss share exceeds its exposure share is concentrating risk.")
    return _save(fig, name)


# --------------------------------------------------------------------------------------
# Drivers and portfolio description
# --------------------------------------------------------------------------------------
def importance_bars(table: pd.DataFrame, value_col: str, title: str, subtitle: str,
                    xlabel: str, name: str, top_n: int = 15):
    """Horizontal importance bars, single hue - one series, so no legend is needed."""
    t = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(t) + 1.8))
    ax.barh(t["feature"], t[value_col], color=C.SERIES[0], height=0.68,
            edgecolor=C.PALETTE["surface"], linewidth=1.5)
    span = float(t[value_col].max())
    for y, v in zip(range(len(t)), t[value_col]):
        ax.text(v + span * 0.012, y, f"{v:.4f}".rstrip("0").rstrip("."), va="center",
                fontsize=8, color=C.PALETTE["text_secondary"])
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, span * 1.16)
    ax.grid(axis="y", visible=False)
    _title(ax, title, subtitle)
    return _save(fig, name)


def odds_ratio_plot(table: pd.DataFrame, name="logistic_odds_ratios", top_n: int = 14):
    """Logistic-regression odds ratios, diverging around 1.0."""
    t = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(t) + 1.8))
    colors = [C.SERIES[7] if v > 0 else C.SERIES[0] for v in t["coefficient_log_odds"]]
    ax.barh(t["feature"], t["odds_ratio"] - 1.0, left=1.0, color=colors, height=0.66,
            edgecolor=C.PALETTE["surface"], linewidth=1.5)
    ax.axvline(1.0, color=C.PALETTE["neutral"], lw=1.2)
    for y, v in zip(range(len(t)), t["odds_ratio"]):
        off = 0.012 if v >= 1 else -0.012
        ax.text(v + off, y, f"{v:.2f}x", va="center",
                ha="left" if v >= 1 else "right", fontsize=8,
                color=C.PALETTE["text_secondary"])
    ax.set_xlabel("Odds ratio  (>1 increases the odds of default, <1 reduces them)")
    ax.grid(axis="y", visible=False)
    _title(ax, "Scorecard effects: logistic-regression odds ratios",
           "Per one standard deviation of the input, or versus the reference category.")
    return _save(fig, name)


def default_rate_by_category(df: pd.DataFrame, col: str, label: str, name: str):
    """Observed default rate across a borrower attribute, with account counts labelled."""
    g = df.groupby(col, observed=True)[C.TARGET].agg(["mean", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = BLUE_RAMP(np.linspace(0.3, 0.9, len(g)))
    ax.bar(g[col].astype(str), g["mean"], color=colors, width=0.6,
           edgecolor=C.PALETTE["surface"], linewidth=2)
    overall = df[C.TARGET].mean()
    ax.axhline(overall, ls="--", lw=1.4, color=C.SERIES[1])
    ax.text(len(g) - 0.4, overall + 0.004, f"portfolio average {overall:.1%}",
            ha="right", fontsize=8.5, color=C.SERIES[1])
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m + 0.006, f"{m:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=8,
                color=C.PALETTE["text_secondary"])
    ax.set_ylabel("Default rate")
    ax.set_ylim(0, g["mean"].max() * 1.35)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_xlabel(label)
    _title(ax, f"Default rate by {label.lower()}", "Observed on the full cleaned dataset.")
    return _save(fig, name)


def delinquency_profile(df: pd.DataFrame, name="delinquency_profile"):
    """Default rate by the worst delinquency observed in the six-month window."""
    t = df.copy()
    t["worst_dpd"] = t["dpd_max"].clip(upper=5)
    g = t.groupby("worst_dpd", observed=True)[C.TARGET].agg(["mean", "size"]).reset_index()
    labels = {0: "Never late", 1: "1 month", 2: "2 months", 3: "3 months", 4: "4 months", 5: "5+ months"}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = BLUE_RAMP(np.linspace(0.25, 0.95, len(g)))
    ax.bar([labels[v] for v in g["worst_dpd"]], g["mean"], color=colors, width=0.6,
           edgecolor=C.PALETTE["surface"], linewidth=2)
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m + 0.012, f"{m:.0%}\nn={n:,}", ha="center", va="bottom", fontsize=8.5,
                color=C.PALETTE["text_secondary"])
    ax.set_ylabel("Default rate next month")
    ax.set_ylim(0, min(1.0, g["mean"].max() * 1.3))
    ax.yaxis.set_major_formatter(_pct)
    ax.set_xlabel("Worst delinquency in the six-month observation window")
    _title(ax, "Delinquency is the dominant risk signal",
           "Default rate in October 2005 by the worst arrears status observed April-September.")
    return _save(fig, name)
