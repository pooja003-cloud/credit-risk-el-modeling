"""Tests for the model-validation statistics.

Each test builds a case where the right answer is known in advance - a perfectly
calibrated model, a deliberately over-confident one, two identical models - and checks the
statistic says what it should. A validation test that cannot tell a good model from a bad
one is worse than none, because it produces false comfort.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import validation as V
from src.data_prep import load_splits


def _simulate(n=20_000, seed=0, overconfidence=1.0):
    """Outcomes drawn from true PDs; predictions optionally stretched on the logit scale."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-1.3, 1.1, n)
    true_p = 1 / (1 + np.exp(-true_logit))
    y = rng.binomial(1, true_p)
    pred = 1 / (1 + np.exp(-true_logit * overconfidence))
    return y, pred


# --- Hosmer-Lemeshow -------------------------------------------------------

def test_hl_does_not_reject_a_calibrated_model():
    y, p = _simulate(seed=1)
    assert V.hosmer_lemeshow(y, p)["p_value_out_of_sample_df"] > 0.01


def test_hl_rejects_a_badly_miscalibrated_model():
    y, p = _simulate(seed=2)
    assert V.hosmer_lemeshow(y, p * 0.6)["p_value_out_of_sample_df"] < 1e-6


def test_hl_group_table_accounts_for_every_observation():
    y, p = _simulate(n=5_000, seed=3)
    groups = V.hosmer_lemeshow(y, p)["groups"]
    assert groups["accounts"].sum() == 5_000
    assert groups["observed_defaults"].sum() == pytest.approx(y.sum())
    assert groups["expected_defaults"].sum() == pytest.approx(p.sum())


# --- Binomial backtest -----------------------------------------------------

def test_binomial_passes_almost_every_band_for_a_calibrated_model():
    y, p = _simulate(seed=4)
    table = V.binomial_by_decile(y, p)
    assert len(table) == 10
    assert table["passes"].sum() >= 9  # one miss in ten is allowed at this confidence


def test_binomial_fails_bands_when_pd_is_halved():
    y, p = _simulate(seed=5)
    assert V.binomial_by_decile(y, p * 0.5)["passes"].sum() <= 2


def test_binomial_interval_contains_the_expected_count():
    y, p = _simulate(n=6_000, seed=6)
    t = V.binomial_by_decile(y, p)
    assert ((t["interval_low"] <= t["expected_defaults"])
            & (t["expected_defaults"] <= t["interval_high"])).all()


# --- Calibration diagnostics -----------------------------------------------

def test_slope_is_near_one_for_a_calibrated_model():
    y, p = _simulate(seed=7)
    slope = V.calibration_diagnostics(y, p, np.ones_like(p))["calibration_slope"]
    assert slope == pytest.approx(1.0, abs=0.06)


def test_slope_falls_below_one_for_an_overconfident_model():
    y, p = _simulate(seed=8, overconfidence=1.5)
    slope = V.calibration_diagnostics(y, p, np.ones_like(p))["calibration_slope"]
    assert slope < 0.75


def test_exposure_weighting_can_reverse_the_direction_of_bias():
    """The F-03 mechanism: under-predicted accounts that are also the large ones."""
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    p = np.array([.1, .1, .1, .1, .9, .9, .9, .9])   # low band under, high band over
    ead = np.array([100, 100, 100, 100, 1, 1, 1, 1])  # low band carries the exposure
    d = V.calibration_diagnostics(y, p, ead)
    assert d["ae_ratio_count_weighted"] < 1    # conservative by account
    assert d["ae_ratio_exposure_weighted"] > 1  # optimistic by exposure


# --- Champion vs challenger ------------------------------------------------

def test_identical_models_have_zero_gap():
    _, p = _simulate(n=1_000, seed=9)
    cc = V.champion_vs_challenger(p, p, np.ones_like(p), lgd=0.65)
    assert cc["el_relative_gap"] == pytest.approx(0.0)
    assert cc["account_mean_abs_gap_pp"] == pytest.approx(0.0)
    assert cc["within_tolerance"]


def test_a_ten_percent_higher_challenger_breaches_tolerance():
    _, p = _simulate(n=1_000, seed=10)
    cc = V.champion_vs_challenger(p, np.clip(p * 1.10, 0, 1), np.ones_like(p), lgd=0.65)
    assert not cc["within_tolerance"]


# --- Data lineage ----------------------------------------------------------

def test_lineage_trace_matches_raw_file_exactly():
    result = V.data_lineage_trace(load_splits()["test"])
    assert result["records"] == 25
    assert result["all_pass"]
    assert result["worst_relative_discrepancy"] < 1e-9
