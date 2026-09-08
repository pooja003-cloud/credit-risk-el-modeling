"""Regression tests on the risk arithmetic.

These are deliberately aimed at the parts where a silent error would be most damaging and
least visible: exposure construction, the expected-loss identity, the PD stress transform,
and the leakage boundary between the two feature sets. A model that is slightly worse is a
disappointment; an expected-loss number that is silently wrong is a different kind of
problem.

Run with ``pytest -q`` from the repository root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import evaluate as E
from src import scenarios as S
from src.data_prep import (
    BEHAVIOURAL,
    ORIGINATION,
    clean,
    engineer_features,
    load_raw,
    split_data,
    validate_raw,
)
from src.expected_loss import account_level_el, exposure_at_default, expected_loss


# --------------------------------------------------------------------------------------
# Data integrity
# --------------------------------------------------------------------------------------
def test_raw_file_matches_published_uci_statistics():
    checks = validate_raw(load_raw())
    assert checks["rows"] == 30_000
    assert checks["defaults"] == 6_636
    assert checks["missing_values"] == 0
    assert checks["duplicate_ids"] == 0


def test_cleaning_removes_duplicates_and_keeps_every_other_row():
    raw = load_raw()
    cleaned, log = clean(raw)
    assert log["rows_out"] == log["rows_in"] - log["duplicate_rows_dropped"]
    assert cleaned.drop(columns=[C.ID_COL]).duplicated().sum() == 0
    assert set(cleaned["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(cleaned["MARRIAGE"].unique()) <= {1, 2, 3}
    assert "PAY_1" in cleaned.columns and "PAY_0" not in cleaned.columns


def test_splits_are_disjoint_and_stratified():
    cleaned, _ = clean(load_raw())
    splits = split_data(engineer_features(cleaned))
    ids = {k: set(v[C.ID_COL]) for k, v in splits.items()}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    assert sum(len(v) for v in splits.values()) == len(cleaned)
    rates = [v[C.TARGET].mean() for v in splits.values()]
    assert max(rates) - min(rates) < 1e-6  # stratification is exact here


# --------------------------------------------------------------------------------------
# Leakage boundary
# --------------------------------------------------------------------------------------
def test_origination_feature_set_contains_no_repayment_history():
    forbidden = ("dpd", "pay_ratio", "util", "bill", "revolver", "full_payer",
                 "inactive", "months_", "paid_", "available_credit")
    for feature in ORIGINATION.all:
        assert not any(feature.startswith(f) or f in feature for f in forbidden), feature


def test_behavioural_set_excludes_the_identifier_and_raw_pay_codes():
    assert C.ID_COL not in BEHAVIOURAL.all
    assert not any(f.startswith("PAY_") for f in BEHAVIOURAL.all)
    assert C.TARGET not in BEHAVIOURAL.all


# --------------------------------------------------------------------------------------
# Exposure at default
# --------------------------------------------------------------------------------------
def test_ead_is_balance_plus_ccf_times_undrawn():
    ead = exposure_at_default(limit=[100_000.0], balance=[40_000.0], ccf=0.35)
    assert ead[0] == pytest.approx(40_000 + 0.35 * 60_000)


def test_ead_bounds():
    limit, balance = np.array([100_000.0]), np.array([40_000.0])
    assert exposure_at_default(limit, balance, ccf=0.0)[0] == pytest.approx(40_000)
    assert exposure_at_default(limit, balance, ccf=1.0)[0] == pytest.approx(100_000)


def test_ead_treats_a_credit_balance_as_zero_drawn():
    """A negative bill is an overpayment, not a negative exposure."""
    assert exposure_at_default([50_000.0], [-3_000.0], ccf=0.5)[0] == pytest.approx(25_000)


def test_ead_never_exceeds_the_limit_when_overdrawn():
    assert exposure_at_default([50_000.0], [70_000.0], ccf=0.4)[0] == pytest.approx(50_000)


def test_ead_is_monotone_in_ccf():
    limit, balance = np.array([80_000.0]), np.array([20_000.0])
    values = [exposure_at_default(limit, balance, ccf=c)[0] for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(values, values[1:]))


# --------------------------------------------------------------------------------------
# Expected loss
# --------------------------------------------------------------------------------------
def test_expected_loss_identity():
    el = expected_loss(np.array([0.10, 0.50]), 0.60, np.array([10_000.0, 20_000.0]))
    assert el[0] == pytest.approx(600.0)
    assert el[1] == pytest.approx(6_000.0)


def test_expected_loss_scales_linearly_with_lgd():
    pd_, ead = np.array([0.2, 0.4]), np.array([1_000.0, 2_000.0])
    assert expected_loss(pd_, 0.8, ead).sum() == pytest.approx(
        2 * expected_loss(pd_, 0.4, ead).sum()
    )


def test_expected_loss_is_zero_when_any_component_is_zero():
    assert expected_loss(np.array([0.0]), 0.65, np.array([1e6]))[0] == 0.0
    assert expected_loss(np.array([0.5]), 0.0, np.array([1e6]))[0] == 0.0
    assert expected_loss(np.array([0.5]), 0.65, np.array([0.0]))[0] == 0.0


def test_account_level_el_frame_is_internally_consistent():
    frame = pd.DataFrame({
        C.ID_COL: [1, 2, 3],
        "LIMIT_BAL": [100_000.0, 50_000.0, 200_000.0],
        "BILL_AMT1": [50_000.0, 0.0, 180_000.0],
        C.TARGET: [0, 1, 0],
    })
    out = account_level_el(frame, np.array([0.05, 0.30, 0.60]), lgd=0.65, ccf=0.35)
    assert np.allclose(out["expected_loss"], out["pd"] * 0.65 * out["ead"])
    assert (out["expected_loss"] <= out["ead"]).all()
    assert (out["el_rate_on_ead"] <= 1.0).all()


# --------------------------------------------------------------------------------------
# Scenario stressing
# --------------------------------------------------------------------------------------
def test_stress_multiplies_the_odds_not_the_probability():
    stressed = S.stress_pd(np.array([0.10]), 2.0)[0]
    # odds 0.1111 -> 0.2222 -> p = 0.1818, not 0.20
    assert stressed == pytest.approx(2 * (0.10 / 0.90) / (1 + 2 * (0.10 / 0.90)))
    assert stressed < 0.20


def test_stress_keeps_probabilities_inside_the_unit_interval():
    extreme = S.stress_pd(np.array([0.001, 0.5, 0.95, 0.999]), 10.0)
    assert (extreme > 0).all() and (extreme < 1).all()


def test_stress_preserves_ranking():
    base = np.array([0.02, 0.15, 0.4, 0.85])
    assert np.all(np.argsort(S.stress_pd(base, 3.0)) == np.argsort(base))


def test_stress_of_one_is_the_identity():
    base = np.array([0.02, 0.15, 0.4, 0.85])
    assert np.allclose(S.stress_pd(base, 1.0), base)


def test_stress_is_monotone_in_the_multiplier():
    base = np.array([0.2])
    values = [S.stress_pd(base, m)[0] for m in (1.0, 1.5, 2.0, 3.0)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_severe_scenario_produces_more_loss_than_baseline():
    frame = pd.DataFrame({
        C.ID_COL: range(100),
        "LIMIT_BAL": np.linspace(20_000, 500_000, 100),
        "BILL_AMT1": np.linspace(0, 300_000, 100),
        C.TARGET: np.zeros(100, dtype=int),
    })
    base_pd = np.linspace(0.01, 0.8, 100)
    table, _ = S.run_all_scenarios(frame, base_pd)
    losses = table["total_expected_loss"].to_numpy()
    assert all(b > a for a, b in zip(losses, losses[1:]))
    assert table["el_vs_baseline_pct"].iloc[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# PD floor and cap
# --------------------------------------------------------------------------------------
def test_pd_floor_and_cap_are_applied_to_saved_scores():
    """No account may carry a PD of exactly 0 or exactly 1.

    A zero PD would silently remove that account from the expected-loss calculation, which
    is the kind of error that never shows up in an AUC.
    """
    path = C.TABLES / "scored_test_accounts.csv"
    if not path.exists():
        pytest.skip("run `python -m src.run_pipeline` first")
    p = pd.read_csv(path)["pd"].to_numpy()
    assert p.min() >= C.PD_FLOOR - 1e-12
    assert p.max() <= C.PD_CAP + 1e-12
    assert (p > 0).all() and (p < 1).all()


def test_floored_model_preserves_ranking_and_normalisation():
    from src.models import FlooredProbabilityModel

    class Stub:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            p = np.asarray(X, dtype=float).ravel()
            return np.column_stack([1 - p, p])

    raw = np.array([0.0, 0.2, 0.6, 1.0])
    out = FlooredProbabilityModel(Stub(), floor=0.001, cap=0.999).predict_proba(raw)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 1] == pytest.approx(0.001)
    assert out[-1, 1] == pytest.approx(0.999)
    assert np.all(np.diff(out[:, 1]) >= 0)  # ranking preserved


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def test_perfect_and_random_scores_bound_the_metrics():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    perfect = y.astype(float) * 0.9 + 0.05
    assert E.core_metrics(y, perfect)["roc_auc"] == pytest.approx(1.0)
    constant = np.full(len(y), y.mean())
    m = E.core_metrics(y, constant)
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["ks"] == pytest.approx(0.0)


def test_ece_is_zero_for_a_perfectly_calibrated_constant_predictor():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.25, 20_000)
    assert E.expected_calibration_error(y, np.full(len(y), y.mean())) < 0.01


def test_decile_table_covers_every_account_exactly_once():
    rng = np.random.default_rng(1)
    p = rng.random(5_000)
    y = rng.binomial(1, p)
    table = E.decile_table(y, p)
    assert table["accounts"].sum() == 5_000
    assert table["defaults"].sum() == y.sum()
    assert len(table) == C.N_RISK_BANDS
    assert table["cumulative_defaults_captured"].max() == pytest.approx(1.0)


def test_target_bad_rate_threshold_respects_appetite():
    rng = np.random.default_rng(2)
    p = rng.beta(2, 6, 8_000)
    y = rng.binomial(1, p)
    t = E.threshold_for_target_bad_rate(y, p, target_bad_rate=0.10)
    approved = p < t
    assert y[approved].mean() <= 0.10 + 1e-9
    assert approved.mean() > 0.05  # a usable, not degenerate, policy


def test_tighter_cutoffs_never_raise_the_approved_bad_rate_materially():
    rng = np.random.default_rng(3)
    p = rng.beta(2, 5, 6_000)
    y = rng.binomial(1, p)
    rates = [y[p < t].mean() for t in (0.2, 0.4, 0.6, 0.8)]
    assert all(b >= a - 1e-9 for a, b in zip(rates, rates[1:]))
