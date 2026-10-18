"""Sharpe probabilístico y deflactado."""
import numpy as np
import pandas as pd
from scipy import stats

from etfml.stats import (
    deflated_sharpe,
    expected_max_sharpe,
    expected_max_standard_normal,
    probabilistic_sharpe,
    sharpe_per_period,
)


def test_expected_max_matches_simulation():
    rng = np.random.default_rng(0)
    for n in (5, 10, 100):
        simulated = rng.standard_normal((20_000, n)).max(axis=1).mean()
        assert abs(expected_max_standard_normal(n) - simulated) / simulated < 0.05, n


def test_single_trial_has_no_luck_premium():
    assert expected_max_standard_normal(1) == 0.0
    assert expected_max_sharpe([0.2]) == 0.0
    assert expected_max_sharpe([]) == 0.0


def test_luck_premium_grows_with_trials_and_dispersion():
    rng = np.random.default_rng(1)
    few = rng.normal(0.15, 0.04, 5)
    assert expected_max_sharpe(few) < expected_max_sharpe(np.concatenate([few, few + 1e-6, few - 1e-6]))
    assert expected_max_sharpe([0.1, 0.12]) < expected_max_sharpe([0.0, 0.3])


def test_psr_is_one_half_at_the_threshold():
    r = pd.Series(np.random.default_rng(2).normal(0.01, 0.04, 240))
    assert abs(probabilistic_sharpe(r, sharpe_per_period(r)) - 0.5) < 1e-12


def test_psr_matches_closed_form_and_grows_with_history():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.01, 0.04, 240))
    sr = sharpe_per_period(r)
    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=False, bias=False)
    expected = stats.norm.cdf(sr * np.sqrt(len(r) - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    assert abs(probabilistic_sharpe(r, 0.0) - expected) < 1e-12
    assert probabilistic_sharpe(r.iloc[:60], 0.0) < probabilistic_sharpe(r, 0.0)


def test_deflated_threshold_is_benchmark_plus_luck():
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.008, 0.04, 240))
    strat = bench + rng.normal(0.002, 0.01, 240)
    trials = [0.15, 0.18, 0.2, 0.22, 0.25]
    d = deflated_sharpe(strat, bench, trials)
    assert abs(d.threshold - (d.benchmark_sharpe + d.luck_premium)) < 1e-15
    assert d.n_trials == 5
    # más pruebas con la misma dispersión exigen más
    d_more = deflated_sharpe(strat, bench, trials * 4)
    assert d_more.luck_premium > d.luck_premium
    assert d_more.probability < d.probability


def test_strategy_equal_to_benchmark_never_passes():
    r = pd.Series(np.random.default_rng(5).normal(0.01, 0.04, 300))
    d = deflated_sharpe(r, r, [0.2])
    assert abs(d.probability - 0.5) < 1e-12
    assert not d.passed
