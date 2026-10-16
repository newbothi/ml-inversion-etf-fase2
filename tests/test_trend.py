"""Filtro de tendencia: señales correctas, sin mirar el futuro y con costos solo al cambiar."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from etfml.config import TrendConfig, load_config
from etfml.data import MarketData, synthetic_market_data
from etfml.trend import BENCHMARK, evaluate, monthly_closes, run_trend, trend_signals

TODAY = pd.Timestamp("2026-01-15")


@pytest.fixture(scope="module")
def setup():
    market = synthetic_market_data(seed=7)
    cfg = load_config().with_universe(list(market.close.columns), "SYN01")
    cfg = replace(cfg, trend=TrendConfig("SYN01", ("SYN02", "SYN03")))
    return cfg, market


def test_signals_follow_the_definitions():
    idx = pd.date_range("2020-01-31", periods=30, freq="ME")
    up = np.linspace(100, 200, 30)
    month_end = pd.DataFrame({"R": up, "C": np.linspace(100, 101, 30)}, index=idx)
    sig = trend_signals(month_end, "R", "C")
    assert sig["media10"].iloc[:9].isna().all() and sig["media10"].iloc[9:].eq(True).all()
    assert sig["mom12"].iloc[:12].isna().all() and sig["mom12"].iloc[12:].eq(True).all()
    down = month_end.assign(R=up[::-1])
    sig = trend_signals(down, "R", "C")
    assert sig["media10"].iloc[9:].eq(False).all()
    assert sig["mom12"].iloc[12:].eq(False).all()


def test_mom12_compares_against_cash():
    idx = pd.date_range("2020-01-31", periods=14, freq="ME")
    month_end = pd.DataFrame({"R": np.linspace(100, 103, 14), "C": np.linspace(100, 110, 14)}, index=idx)
    sig = trend_signals(month_end, "R", "C")
    assert sig["mom12"].iloc[12:].eq(False).all()  # sube, pero menos que la caja


def test_changing_the_future_does_not_change_the_past(setup):
    cfg, market = setup
    cut = pd.Timestamp("2015-06-30")  # último día hábil de junio: su señal no puede ver julio
    a = run_trend(cfg, market, TODAY)
    # un desplome y un salto: si la señal mirara el mes siguiente, alguno de los dos la daría vuelta
    for factor in (0.3, 3.0):
        shocked = market.close.copy()
        shocked.loc[shocked.index > cut] *= factor
        b = run_trend(cfg, MarketData(shocked, market.volume, None, "sintetico", close_raw=shocked), TODAY)
        past = a.risk_on.index <= cut
        assert past.sum() > 100
        pd.testing.assert_frame_equal(a.risk_on[past], b.risk_on[past])
        for key in a.simulations:
            sa, sb = a.simulations[key], b.simulations[key]
            # meses cuyo retorno terminó antes del corte (la señal de abril se vende el 1 de junio)
            done = sa.index[sa.index <= cut - pd.offsets.MonthEnd(2)]
            np.testing.assert_allclose(sa.loc[done, "net"], sb.loc[done, "net"])


def test_costs_only_when_switching(setup):
    cfg, market = setup
    res = run_trend(cfg, market, TODAY)
    for v in res.variants:
        sim = res.simulations[v.key]
        on = res.risk_on[v.signal]
        switches = (on != on.shift()).iloc[1:]
        assert sim["n_trades"].iloc[0] == 1                  # compra inicial
        assert (sim["n_trades"].iloc[1:][~switches.to_numpy()] == 0).all()
        assert (sim["n_trades"].iloc[1:][switches.to_numpy()] == 2).all()  # vende uno y compra el otro
        assert res.verdicts is None


def test_all_strategies_share_the_period_and_benchmark_is_fully_invested(setup):
    cfg, market = setup
    res = run_trend(cfg, market, TODAY)
    idx = res.simulations[BENCHMARK].index
    for sim in res.simulations.values():
        assert sim.index.equals(idx)
    assert (res.simulations[BENCHMARK]["holdings"] == "SYN01").all()
    _, fwd, _ = monthly_closes(market.close, TODAY)
    np.testing.assert_allclose(res.simulations[BENCHMARK]["gross"], fwd.loc[idx, "SYN01"])


def test_holdings_follow_the_signal(setup):
    cfg, market = setup
    res = run_trend(cfg, market, TODAY)
    for v in res.variants:
        expected = np.where(res.risk_on[v.signal], "SYN01", v.safe_asset)
        assert (res.simulations[v.key]["holdings"].to_numpy() == expected).all()


def test_verdict_levels_are_consistent(setup):
    cfg, market = setup
    res = run_trend(cfg, market, TODAY)
    evaluate(res, [0.1, 0.15, 0.2, 0.25])
    for v in res.variants:
        vd = res.verdicts[v.key]
        if vd["level"] == "pasa":
            assert vd["better"] and vd["consistent"] and vd["evidence"]
        elif vd["level"] == "prometedor":
            assert vd["better"] and vd["consistent"] and not vd["evidence"]
        else:
            assert not (vd["better"] and vd["consistent"])
