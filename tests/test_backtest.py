"""Walk-forward: purga correcta, detecta señal real y no inventa señal en ruido puro."""
from itertools import pairwise

import numpy as np
import pandas as pd

from etfml.backtest import simulate
from etfml.config import CostConfig
from etfml.metrics import information_coefficients, newey_west_tstat, summarize


def test_training_never_sees_unrealized_returns(with_signal):
    cfg, _, panel, result = with_signal
    labeled = panel.labeled
    for fold in result.walkforward.folds:
        # el último mes de entrenamiento termina al menos 2 meses antes del primer mes de prueba:
        # su retorno se conoce recién al inicio del mes anterior a la predicción
        assert fold.train_last.to_period("M") <= fold.test_first.to_period("M") - 2
        train = labeled[labeled.index.get_level_values("date") <= fold.train_last]
        assert (train["realized_at"] <= fold.test_first).all()
        assert fold.train_months >= cfg.walkforward.min_train_months


def test_out_of_sample_covers_each_month_once(with_signal):
    _, _, _, result = with_signal
    pred = result.walkforward.predictions
    assert not pred.index.duplicated().any()
    folds = result.walkforward.folds
    for a, b in pairwise(folds):
        assert a.test_last < b.test_first


def test_models_find_planted_signal(with_signal):
    cfg, _, _, result = with_signal
    s = summarize(result, 0.0, cfg.strategy.top_k)
    m = s["models"]
    for name in ("ridge", "logistica", "lgbm", "ensamble"):
        assert m.loc[name, "ic_mean"] > 0.05, name
        assert m.loc[name, "ic_tstat"] > 3, name
    strat = s["strategy"]
    assert strat.loc["ensamble", "sharpe"] > strat.loc["equal_weight", "sharpe"]


def test_no_fake_alpha_on_pure_noise(pure_noise):
    cfg, _, _, result = pure_noise
    s = summarize(result, 0.0, cfg.strategy.top_k)
    m = s["models"]
    for name in ("ridge", "logistica", "lgbm", "ensamble"):
        assert abs(m.loc[name, "ic_mean"]) < 0.05, name
        assert abs(m.loc[name, "ic_tstat"]) < 3, name
        assert not s["verdicts"][name]["passed"], name


def test_information_coefficient_perfect_and_inverse():
    idx = pd.MultiIndex.from_product([pd.date_range("2020-01-31", periods=3, freq="ME"), list("ABCDE")],
                                     names=["date", "ticker"])
    excess = pd.Series(np.tile([0.03, 0.01, 0.0, -0.01, -0.03], 3), index=idx)
    assert np.allclose(information_coefficients(excess * 2, excess), 1.0)
    assert np.allclose(information_coefficients(-excess, excess), -1.0)


def test_newey_west_is_more_conservative_with_autocorrelation():
    rng = np.random.default_rng(1)
    noise = rng.normal(0.02, 0.1, 400)
    persistent = pd.Series(noise).rolling(6, min_periods=1).mean()  # muy autocorrelacionada
    naive = persistent.mean() / persistent.std() * np.sqrt(len(persistent))
    assert newey_west_tstat(persistent) < naive


COSTS = CostConfig(initial_capital_usd=10_000, commission_bps=0.5, min_commission_usd=1.0,
                   slippage_bps=5.0, rebalance_band_usd=100.0)


def _fwd(values: dict[str, list[float]]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=len(next(iter(values.values()))), freq="ME")
    return pd.DataFrame(values, index=dates)


def test_buy_and_hold_pays_costs_only_once():
    fwd = _fwd({"A": [0.01, -0.02, 0.03, 0.0]})
    weights = {d: pd.Series({"A": 1.0}) for d in fwd.index}
    sim = simulate(weights, fwd, COSTS)
    expected_first = (max(10_000 * 0.5e-4, 1.0) + 10_000 * 5e-4) / 10_000
    assert np.isclose(sim["cost"].iloc[0], expected_first)
    assert (sim["cost"].iloc[1:] == 0).all()
    assert np.isclose(sim["equity"].iloc[-1], 10_000 * (1 - expected_first) * 1.01 * 0.98 * 1.03)


def test_minimum_commission_applies_per_order():
    fwd = _fwd({"A": [0.0], "B": [0.0], "C": [0.0]})
    weights = {fwd.index[0]: pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})}
    sim = simulate(weights, fwd, COSTS)
    per_order = max(10_000 / 3 * 0.5e-4, 1.0)  # la proporcional (0,17 USD) es menor al mínimo
    expected = (3 * per_order + 10_000 * 5e-4) / 10_000
    assert np.isclose(sim["cost"].iloc[0], expected)


def test_small_adjustments_are_skipped_but_exits_are_not():
    fwd = _fwd({"A": [0.004, 0.0, 0.0], "B": [-0.004, 0.0, 0.0], "C": [0.0, 0.0, 0.0]})
    d = fwd.index
    weights = {
        d[0]: pd.Series({"A": 0.5, "B": 0.5}),
        d[1]: pd.Series({"A": 0.5, "B": 0.5}),   # deriva de 0,4 %: ajuste de ~US$ 20 < banda
        d[2]: pd.Series({"A": 0.5, "C": 0.5}),   # B sale y entra C
    }
    sim = simulate(weights, fwd, COSTS)
    assert sim["n_trades"].iloc[1] == 0
    assert sim["n_trades"].iloc[2] == 2
    assert sim["holdings"].iloc[2] == "A C"
