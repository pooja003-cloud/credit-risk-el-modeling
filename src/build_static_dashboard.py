"""Build the self-contained static risk dashboard (``dashboard/dashboard.html``).

The Streamlit app is the interactive tool; this file is the snapshot that opens in any
browser with nothing installed, which is what makes it useful to send to someone or to
host on GitHub Pages. Everything - styles, tables and figures - is inlined, so the single
file is the whole dashboard.

Figures are embedded as PNGs on a fixed light plate. The page itself follows the reader's
light or dark theme; the chart plates deliberately do not, because a rendered raster
cannot re-theme itself and inverting it would wreck the palette.
"""

from __future__ import annotations

import base64
import json
from datetime import date

import pandas as pd

from . import config as C

FIG_ORDER = [
    ("delinquency_profile", "Delinquency is the dominant risk signal",
     "Default rate in October 2005 by the worst arrears status observed over the previous six months."),
    ("decile_default_rates", "Rank-ordering by predicted-risk decile",
     "Bars are realised outcomes on the held-out test fold; the line is the model's prediction."),
    ("roc_curves", "Discrimination",
     "ROC curves for every model in the ladder, scored on the test fold."),
    ("pr_curves", "Precision and recall",
     "The relevant curve for a 22% base rate, where accuracy is uninformative."),
    ("calibration", "Calibration",
     "Predicted PD against observed default rate. Expected loss multiplies levels, so the points must sit on the diagonal."),
    ("confusion_matrix", "Decision outcomes at the policy cutoff",
     "What the chosen cutoff would have done to the held-out population."),
    ("cutoff_net_cost", "Setting the cutoff",
     "Cost economics under four margin assumptions, and the risk-return frontier the business negotiates."),
    ("el_by_decile", "Expected-loss concentration",
     "Baseline scenario. Both panels use one measure per axis."),
    ("scenario_comparison", "Scenario analysis",
     "Portfolio expected loss under baseline, moderate and severe deterioration."),
    ("sensitivity_heatmap", "How much of the answer is the assumption?",
     "Portfolio expected loss across PD stresses and assumed loss-given-default."),
    ("el_by_limit_band", "Exposure versus expected loss by credit-limit quintile",
     "A segment whose loss share exceeds its exposure share is concentrating risk."),
    ("shap_importance", "What the champion model relies on",
     "Global SHAP importance across a 2,000-account sample of the test fold."),
    ("logistic_odds_ratios", "The scorecard view",
     "Logistic-regression odds ratios, per standard deviation of the input or versus the reference category."),
]


def _b64(name: str) -> str | None:
    path = C.FIGURES / f"{name}.png"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def _money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{C.CURRENCY}{x / 1e9:,.2f}bn"
    if abs(x) >= 1e6:
        return f"{C.CURRENCY}{x / 1e6:,.1f}m"
    return f"{C.CURRENCY}{x:,.0f}"


def _table_html(df: pd.DataFrame, formats: dict[str, str] | None = None,
                rename: dict[str, str] | None = None) -> str:
    d = df.copy()
    if formats:
        for col, fmt in formats.items():
            if col in d.columns:
                d[col] = d[col].map(lambda v: fmt.format(v) if pd.notna(v) else "-")
    if rename:
        d = d.rename(columns=rename)
    d.columns = [str(c).replace("_", " ") for c in d.columns]
    head = "".join(f"<th>{c}</th>" for c in d.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in d.astype(str).to_numpy()
    )
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Sans:wght@400;500;600;700&'
    'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
)

CSS = """
:root {
  color-scheme: light;
  --surface: #f7f8fa;
  --card: #ffffff;
  --border: #dfe3e9;
  --ink: #0d1520;
  --ink-2: #4a5666;
  --ink-3: #778393;
  --accent: #2a78d6;
  --accent-soft: #eaf2fd;
  --warn: #eda100;
  --bad: #e34948;
  --good: #1baf7a;
  --plate: #fcfcfb;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface: #0e1319;
    --card: #161c25;
    --border: #2a323d;
    --ink: #f2f5f9;
    --ink-2: #b3bdcb;
    --ink-3: #7d8896;
    --accent: #3987e5;
    --accent-soft: #1b2a3d;
    --warn: #c98500;
    --bad: #e66767;
    --good: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #0e1319;
  --card: #161c25;
  --border: #2a323d;
  --ink: #f2f5f9;
  --ink-2: #b3bdcb;
  --ink-3: #7d8896;
  --accent: #3987e5;
  --accent-soft: #1b2a3d;
  --warn: #c98500;
  --bad: #e66767;
  --good: #199e70;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.6 "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.value, .tile .sub, td, th, code {
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1;
}
.value, code { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 22px 80px; }
header { border-bottom: 1px solid var(--border); padding-bottom: 22px; margin-bottom: 30px; }
h1 { font-size: 32px; line-height: 1.15; margin: 0 0 10px; letter-spacing: -0.025em;
     font-weight: 600; text-wrap: balance; }
h2 { font-size: 20px; margin: 42px 0 6px; letter-spacing: -0.01em; }
h3 { font-size: 15px; margin: 0 0 4px; }
p.lede { color: var(--ink-2); margin: 0; max-width: 68ch; }
.meta { color: var(--ink-3); font-size: 13px; margin-top: 10px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 12px; margin: 26px 0 8px; }
.tile { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-3); }
.tile .value { font-size: 25px; font-weight: 650; margin-top: 6px; letter-spacing: -0.02em; }
.tile .sub { font-size: 12.5px; color: var(--ink-2); margin-top: 3px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px 18px 14px; margin: 16px 0; }
.card > h3 { font-size: 16px; margin-bottom: 3px; }
.card > .sub { color: var(--ink-2); font-size: 13.5px; margin: 0 0 12px; }
.plate { background: var(--plate); border: 1px solid var(--border); border-radius: 8px; padding: 8px; overflow-x: auto; }
.plate img { display: block; width: 100%; height: auto; }
.tablewrap { overflow-x: auto; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--ink-3); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em; }
td:first-child, th:first-child { text-align: left; }
tbody tr:hover { background: var(--accent-soft); }
.note { border-left: 3px solid var(--warn); background: var(--card); padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 18px 0; color: var(--ink-2); font-size: 14px; }
.note strong { color: var(--ink); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; }
ul { color: var(--ink-2); padding-left: 20px; }
li { margin: 6px 0; }
footer { margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--border); color: var(--ink-3); font-size: 13px; }
code { background: var(--accent-soft); padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px;
        font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
        padding: 3px 9px; border-radius: 999px; border: 1px solid currentColor; }
.chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%;
                background: currentColor; }
.chip-base { color: var(--good); }
.chip-mod  { color: var(--warn); }
.chip-sev  { color: var(--bad); }
.chips { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0 2px; }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def build() -> str:
    results = json.loads((C.TABLES / "results.json").read_text())
    head = results["headline"]
    thresholds = results["thresholds"]
    decision = results["decision_rule"]

    comparison = pd.read_csv(C.TABLES / "model_comparison.csv")
    deciles = pd.read_csv(C.TABLES / "risk_deciles.csv")
    scen = pd.read_csv(C.TABLES / "scenario_summary.csv")
    featureset = pd.read_csv(C.TABLES / "feature_set_comparison.csv")
    cutoffs = pd.read_csv(C.TABLES / "cutoff_comparison.csv")
    el_limit = pd.read_csv(C.TABLES / "el_by_limit_band.csv")

    tiles = [
        ("Exposure at default", _money(head["portfolio_ead"]),
         f"{head['test_accounts']:,} held-out accounts"),
        ("Expected loss, baseline", _money(head["portfolio_el"]),
         f"{head['portfolio_el_rate']:.2%} of exposure"),
        ("Expected loss, severe", _money(head["severe_el"]),
         f"+{head['severe_el_uplift_pct']:.0%} versus baseline"),
        ("Model ROC-AUC", f"{head['champion_roc_auc']:.3f}",
         f"Gini {head['champion_gini']:.3f} · KS {head['champion_ks']:.3f}"),
        ("Riskiest decile", f"{head['top_decile_default_rate']:.0%}",
         f"{head['top_decile_lift']:.1f}x the portfolio rate"),
        ("Approved book at cutoff", f"{decision['bad_rate_of_approved']:.1%}",
         f"from {decision['bad_rate_no_cutoff']:.1%}, approving {decision['approval_rate']:.0%}"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="label">{a}</div><div class="value">{b}</div>'
        f'<div class="sub">{c}</div></div>'
        for a, b, c in tiles
    )

    figs_html = []
    for name, title, sub in FIG_ORDER:
        data = _b64(name)
        if data is None:
            continue
        figs_html.append(
            f'<div class="card"><h3>{title}</h3><p class="sub">{sub}</p>'
            f'<div class="plate"><img alt="{title}" src="data:image/png;base64,{data}"></div></div>'
        )

    comp_html = _table_html(
        comparison[["model", "val_auc", "roc_auc", "pr_auc", "ks", "brier", "ece"]],
        formats={c: "{:.4f}" for c in ["val_auc", "roc_auc", "pr_auc", "ks", "brier", "ece"]},
        rename={"model": "Model", "val_auc": "Validation ROC-AUC", "roc_auc": "Test ROC-AUC",
                "pr_auc": "PR-AUC", "ks": "KS", "brier": "Brier", "ece": "Calibration error"},
    )
    dec_html = _table_html(
        deciles[["decile", "accounts", "defaults", "observed_default_rate", "mean_predicted_pd",
                 "lift", "share_of_exposure", "cumulative_defaults_captured"]],
        formats={"observed_default_rate": "{:.1%}", "mean_predicted_pd": "{:.1%}",
                 "lift": "{:.2f}x", "share_of_exposure": "{:.1%}",
                 "cumulative_defaults_captured": "{:.0%}", "accounts": "{:,.0f}",
                 "defaults": "{:,.0f}"},
    )
    scen_html = _table_html(
        scen[["scenario", "pd_odds_multiplier", "lgd_assumed", "ccf_assumed", "average_pd",
              "total_ead", "total_expected_loss", "el_rate_on_ead", "el_vs_baseline_pct"]],
        formats={"pd_odds_multiplier": "{:.2f}x", "lgd_assumed": "{:.0%}", "ccf_assumed": "{:.0%}",
                 "average_pd": "{:.2%}", "total_ead": "{:,.0f}", "total_expected_loss": "{:,.0f}",
                 "el_rate_on_ead": "{:.2%}", "el_vs_baseline_pct": "{:+.0%}"},
        rename={"scenario": "Scenario", "el_vs_baseline_pct": "vs baseline"},
    )
    fs_html = _table_html(
        featureset,
        formats={c: "{:.4f}" for c in ["roc_auc", "gini", "pr_auc", "ks", "brier"]},
        rename={"feature_set": "Feature set", "model": "Model", "n_features": "Features"},
    )
    cut_html = _table_html(
        cutoffs[["cutoff_rule", "threshold", "flagged_rate", "precision", "recall", "f1"]],
        formats={"threshold": "{:.1%}", "flagged_rate": "{:.1%}", "precision": "{:.3f}",
                 "recall": "{:.3f}", "f1": "{:.3f}"},
        rename={"cutoff_rule": "Cutoff rule", "threshold": "PD cutoff",
                "flagged_rate": "Declined share"},
    )
    el_html = _table_html(
        el_limit,
        formats={"mean_pd": "{:.2%}", "observed_default_rate": "{:.2%}", "total_ead": "{:,.0f}",
                 "expected_loss": "{:,.0f}", "el_rate_on_ead": "{:.2%}",
                 "share_of_ead": "{:.1%}", "share_of_el": "{:.1%}",
                 "el_concentration": "{:.2f}x", "accounts": "{:,.0f}"},
    )

    html = f"""<meta charset="utf-8">
<title>Consumer Credit Risk Monitor</title>
{FONTS}
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>Consumer credit risk monitor</h1>
  <p class="lede">Probability-of-default modelling and expected-loss analysis on the UCI
  Default of Credit Card Clients dataset (Taiwan, 2005). Every figure below is computed on a
  held-out test fold of {head['test_accounts']:,} accounts that no model saw during training,
  selection or calibration.</p>
  <p class="meta">Champion model: <strong>{head['champion']}</strong> &middot; pipeline run
  {results['generated_on']} &middot; page generated {date.today().isoformat()} &middot; amounts in
  New Taiwan dollars &middot; static snapshot; the Streamlit app in <code>dashboard/app.py</code>
  is the interactive version.</p>
</header>

<div class="tiles">{tiles_html}</div>

<div class="note"><strong>What is estimated and what is assumed.</strong> The probability of
default is estimated from the data. Loss given default is <em>not</em>: this dataset contains no
recovery, collection or charge-off information, so every loss figure on this page is a modelled
PD multiplied by an assumed LGD of {C.LGD_BASELINE:.0%} and moves one-for-one with that
assumption. Exposure at default combines the observed balance with an assumed
{C.CCF_BASELINE:.0%} credit conversion factor on the undrawn line.</div>

<h2>Headline results</h2>
<p class="lede">The champion model separates risk well and is calibrated well enough for its
output to be multiplied into an expected loss: mean predicted PD across the test fold is
{results['expected_loss_baseline']['average_pd']:.2%} against an
observed default rate of {head['test_default_rate']:.2%}, and the expected calibration error is
{head['champion_ece']:.4f}. The gain over the logistic-regression scorecard is real but modest
&mdash; {head['auc_gain_over_logreg']:+.4f} ROC-AUC &mdash; which is itself a useful finding when
a simpler, fully interpretable model is on the table.</p>

<div class="card"><h3>Model comparison</h3>
<p class="sub">Selected on the validation fold, reported on the test fold. Accuracy is
deliberately absent from the headline: approving every account scores
{1 - head['test_default_rate']:.0%} accurate on this portfolio and identifies no defaults at all.</p>
{comp_html}</div>

<div class="card"><h3>Behavioural history versus origination-only inputs</h3>
<p class="sub">The champion model is a behavioural scorecard: it needs six months of repayment
history. Rebuilt on only what is known when an account is opened, the same model class loses most
of its power. That gap is the reason the leakage boundary matters.</p>
{fs_html}</div>

<h2>Risk ranking and decision rule</h2>
<div class="card"><h3>Predicted-risk deciles</h3>
<p class="sub">The riskiest decile defaults at {head['top_decile_default_rate']:.0%} against
{head['bottom_decile_default_rate']:.0%} in the safest, and the top two deciles contain
{head['top_two_deciles_capture']:.0%} of all defaults in the test fold.</p>
{dec_html}</div>

<div class="card"><h3>Candidate cutoffs</h3>
<p class="sub">The deployed cutoff of {thresholds['policy_cutoff_from_validation']:.1%} was chosen
on the validation fold as the loosest rule holding the approved book's default rate at or below
{thresholds['policy_target_bad_rate']:.0%}. Applied to the test fold it declines
{decision['decline_rate']:.0%} of accounts and cuts the realised default rate of the approved book
from {decision['bad_rate_no_cutoff']:.1%} to {decision['bad_rate_of_approved']:.1%}. This is a
retrospective evaluation on held-out data, not a claim about live lending decisions.</p>
{cut_html}</div>

<h2>Expected loss and scenarios</h2>
<div class="card"><h3>Scenario summary</h3>
<div class="chips">
  <span class="chip chip-base">Baseline {_money(scen['total_expected_loss'].iloc[0])}</span>
  <span class="chip chip-mod">Moderate {_money(scen['total_expected_loss'].iloc[1])} &middot; {scen['el_vs_baseline_pct'].iloc[1]:+.0%}</span>
  <span class="chip chip-sev">Severe {_money(scen['total_expected_loss'].iloc[2])} &middot; {scen['el_vs_baseline_pct'].iloc[2]:+.0%}</span>
</div>
<p class="sub">Each scenario moves PD, LGD and the credit conversion factor together. Severities
are judgemental and are not calibrated to a published macroeconomic model.</p>
{scen_html}</div>

<div class="card"><h3>Expected loss by credit-limit quintile</h3>
<p class="sub">Baseline scenario. The concentration column is the ratio of a segment's share of
expected loss to its share of exposure.</p>
{el_html}</div>

<h2>Charts</h2>
{''.join(figs_html)}

<h2>Limitations</h2>
<ul>
<li><strong>One cohort, one outcome month.</strong> Every account is observed over the same
April&ndash;September 2005 window with an October 2005 outcome, so no out-of-time validation is
possible and the split is stratified-random rather than time-based.</li>
<li><strong>Taiwan, 2005.</strong> The portfolio sits in the aftermath of a domestic card-debt
crisis. A {head['test_default_rate']:.0%} default rate is not a through-the-cycle rate.</li>
<li><strong>Behavioural, not application.</strong> The champion model requires six months of
repayment history and cannot be used at origination.</li>
<li><strong>No macroeconomic variables.</strong> Unemployment and interest rates enter only
through the scenario multipliers, not as model inputs.</li>
<li><strong>Demographic inputs.</strong> Sex, education and marital status appear in the dataset
and are used here for segment monitoring. Several are prohibited or restricted inputs for credit
decisions in many jurisdictions; a deployable model would exclude them and be tested for
disparate impact.</li>
</ul>

<footer>Built from the UCI Default of Credit Card Clients dataset. Portfolio project &mdash; not
investment, credit or financial advice. Regenerate with
<code>python -m src.build_static_dashboard</code>.</footer>
</div>
"""
    # Escape every non-ASCII character so the page renders correctly no matter what
    # encoding a browser guesses for a local file.
    html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    out = C.ROOT / "dashboard" / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    print(build())
