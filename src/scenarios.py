"""Scenario analysis: how expected loss moves when conditions deteriorate.

Three scenarios are run - baseline, moderate deterioration and severe deterioration -
and each one moves all three EL components at once, because in a real downturn they do
not move independently: defaults rise, recoveries fall, and distressed borrowers draw
more of their remaining credit line before they stop paying.

**How the PD stress is applied.** The multiplier acts on the *odds* of default, not on
the probability:

    odds' = m x odds,   PD' = odds' / (1 + odds')

A flat multiplier on the probability itself would push high-risk accounts above 1.0 and
would move a 1% account and a 40% account by wildly different amounts in log-odds terms.
Stressing the odds is the standard scorecard treatment: it is equivalent to shifting
every score by a constant amount, ``log(m)``, on the log-odds scale, so the ranking is
preserved and the stress is proportionate across the risk spectrum.

The scenario magnitudes are judgemental. They are not calibrated to a published
macroeconomic model, and are labelled as illustrative wherever they are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .expected_loss import account_level_el, el_by_group, portfolio_summary

EPS = 1e-9


def stress_pd(pd_values: np.ndarray, odds_multiplier: float) -> np.ndarray:
    """Multiply the odds of default by ``odds_multiplier`` and convert back to a PD."""
    p = np.clip(np.asarray(pd_values, dtype=float), EPS, 1 - EPS)
    odds = p / (1 - p)
    stressed = odds * float(odds_multiplier)
    return stressed / (1.0 + stressed)


def implied_log_odds_shift(odds_multiplier: float) -> float:
    """The equivalent additive shift on the log-odds (score) scale."""
    return float(np.log(odds_multiplier))


def run_scenario(
    frame: pd.DataFrame,
    base_pd: np.ndarray,
    scenario_name: str,
    spec: dict,
) -> tuple[pd.DataFrame, dict]:
    """Apply one scenario and return the account-level table plus a portfolio summary."""
    stressed = stress_pd(base_pd, spec["pd_odds_multiplier"])
    el = account_level_el(frame, stressed, lgd=spec["lgd"], ccf=spec["ccf"])
    summary = portfolio_summary(el)
    summary.update(
        {
            "scenario": scenario_name,
            "pd_odds_multiplier": spec["pd_odds_multiplier"],
            "log_odds_shift": implied_log_odds_shift(spec["pd_odds_multiplier"]),
            "narrative": spec["narrative"],
        }
    )
    return el, summary


def run_all_scenarios(
    frame: pd.DataFrame,
    base_pd: np.ndarray,
    scenarios: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run every scenario and return a comparison table plus each account-level frame."""
    scenarios = scenarios or C.SCENARIOS
    summaries, frames = [], {}
    for name, spec in scenarios.items():
        el, summary = run_scenario(frame, base_pd, name, spec)
        frames[name] = el
        summaries.append(summary)

    table = pd.DataFrame(summaries)
    baseline_el = float(table.loc[table["scenario"] == "Baseline", "total_expected_loss"].iloc[0])
    table["el_vs_baseline_abs"] = table["total_expected_loss"] - baseline_el
    table["el_vs_baseline_pct"] = table["total_expected_loss"] / baseline_el - 1.0

    ordered = [
        "scenario",
        "pd_odds_multiplier",
        "log_odds_shift",
        "lgd_assumed",
        "ccf_assumed",
        "average_pd",
        "exposure_weighted_pd",
        "total_ead",
        "total_expected_loss",
        "el_rate_on_ead",
        "el_vs_baseline_abs",
        "el_vs_baseline_pct",
        "narrative",
    ]
    return table[ordered], frames


def scenario_by_segment(
    frames: dict[str, pd.DataFrame], group_col: str
) -> pd.DataFrame:
    """Expected loss by segment under every scenario, in long form."""
    parts = []
    for name, el in frames.items():
        part = el_by_group(el, group_col)
        part.insert(0, "scenario", name)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def sensitivity_grid(
    frame: pd.DataFrame,
    base_pd: np.ndarray,
    pd_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5),
    lgds: tuple[float, ...] = (0.45, 0.55, 0.65, 0.75, 0.85),
    ccf: float = C.CCF_BASELINE,
) -> pd.DataFrame:
    """A PD-stress x LGD grid of portfolio expected loss.

    Because LGD is assumed rather than estimated, the honest way to present expected loss
    is as a surface rather than a single number: this grid shows how much of the answer is
    driven by the assumption instead of by the model.
    """
    rows = []
    for m in pd_multipliers:
        stressed = stress_pd(base_pd, m)
        ead = account_level_el(frame, stressed, lgd=1.0, ccf=ccf)["ead"].to_numpy()
        pd_ead = float((stressed * ead).sum())
        for lgd in lgds:
            rows.append(
                {
                    "pd_odds_multiplier": m,
                    "lgd": lgd,
                    "total_ead": float(ead.sum()),
                    "total_expected_loss": pd_ead * lgd,
                    "el_rate_on_ead": pd_ead * lgd / float(ead.sum()),
                }
            )
    return pd.DataFrame(rows)
