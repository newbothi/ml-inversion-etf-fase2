"""Que las features y el target no miren el futuro."""
import numpy as np
import pandas as pd

from etfml.data import MarketData, synthetic_market_data
from etfml.features import FEATURE_COLS, build_panel, rebalance_calendar

from .conftest import TODAY


def test_calendar_executes_after_signal():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    cal = rebalance_calendar(idx)
    for row in cal.dropna().itertuples():
        assert row.exec_date > row.signal_date
        # se ejecuta el día hábil inmediatamente siguiente
        assert idx.get_loc(row.exec_date) == idx.get_loc(row.signal_date) + 1
        assert row.realized_at > row.exec_date
    realized = cal["realized_at"].iloc[:-1].reset_index(drop=True)
    next_exec = cal["exec_date"].iloc[1:].reset_index(drop=True)
    assert realized.equals(next_exec)  # NaT al final en ambos: el último mes aún no se realiza


def test_current_month_is_excluded():
    idx = pd.bdate_range("2020-01-01", "2020-06-15")
    cal = rebalance_calendar(idx, today=pd.Timestamp("2020-06-15"))
    assert cal["signal_date"].max() == pd.Timestamp("2020-05-29")


def test_changing_the_future_does_not_change_the_past():
    market = synthetic_market_data(seed=11)
    cut = pd.Timestamp("2015-06-30")
    rng = np.random.default_rng(0)
    future = market.close.index > cut
    shocked_close = market.close.copy()
    shocked_close.loc[future] = shocked_close.loc[future] * rng.uniform(0.5, 1.5, shocked_close.loc[future].shape)
    shocked_volume = market.volume.copy()
    shocked_volume.loc[future] = shocked_volume.loc[future] * 7.0
    shocked = MarketData(shocked_close, shocked_volume, None, "sintetico")

    a = build_panel(market, min_assets=6, today=TODAY).data
    b = build_panel(shocked, min_assets=6, today=TODAY).data

    past = a.index.get_level_values("date") <= cut
    pa, pb = a[past], b.loc[a[past].index]
    np.testing.assert_allclose(pa[FEATURE_COLS].to_numpy(), pb[FEATURE_COLS].to_numpy())
    np.testing.assert_allclose(pa["raw_mom_12_1"].to_numpy(), pb["raw_mom_12_1"].to_numpy())

    known = a["realized_at"] <= cut
    np.testing.assert_allclose(a.loc[known, "fwd_ret"].to_numpy(), b.loc[a[known].index, "fwd_ret"].to_numpy())
    # y algo sí cambió después del corte (el test no es trivial)
    later = a.index.get_level_values("date") > cut
    assert not np.allclose(a.loc[later, "fwd_ret"].dropna(), b.loc[a[later].index, "fwd_ret"].dropna())


def test_ranks_are_within_each_month():
    market = synthetic_market_data(seed=5)
    data = build_panel(market, min_assets=6, today=TODAY).data
    for _, g in list(data.groupby(level="date"))[:20]:
        assert g[FEATURE_COLS].min().min() > -0.5 - 1e-9
        assert g[FEATURE_COLS].max().max() <= 0.5 + 1e-9


def test_excess_sums_to_zero_each_month():
    market = synthetic_market_data(seed=5)
    labeled = build_panel(market, min_assets=6, today=TODAY).labeled
    sums = labeled["excess"].groupby(level="date").sum()
    assert np.allclose(sums, 0.0, atol=1e-12)
