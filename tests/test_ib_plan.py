"""Plan de órdenes para IB (sin conectarse a nada)."""
import pandas as pd
import pytest

from etfml.ib_rebalance import BUY, HOLD, SELL, AccountSnapshot, check_account_allowed, plan_orders

UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "XLE"]
PRICES = pd.Series({"SPY": 500.0, "QQQ": 400.0, "TLT": 90.0, "GLD": 200.0, "XLE": 80.0})


def test_plan_from_empty_account():
    target = pd.Series({"SPY": 1 / 3, "QQQ": 1 / 3, "GLD": 1 / 3})
    plan = plan_orders(target, PRICES, pd.Series(dtype=float), 10_000, 0.0, 0.02, 100, UNIVERSE)
    assert set(plan["orden"]) == {BUY}
    row = plan.set_index("ticker").loc["SPY"]
    assert row["cantidad"] == int(10_000 * 0.98 / 3 // 500)
    assert (plan["monto_usd"].sum()) <= 10_000 * 0.98


def test_sells_come_first_and_exits_are_complete():
    target = pd.Series({"SPY": 0.5, "GLD": 0.5})
    positions = pd.Series({"SPY": 10, "TLT": 1})  # TLT vale 90 USD: menos que la banda, pero sale igual
    plan = plan_orders(target, PRICES, positions, 10_000, 0.0, 0.0, 100, UNIVERSE)
    assert plan.iloc[0]["orden"] == SELL
    tlt = plan.set_index("ticker").loc["TLT"]
    assert tlt["orden"] == SELL and tlt["cantidad"] == 1


def test_small_adjustment_is_skipped():
    target = pd.Series({"SPY": 0.5, "QQQ": 0.5})
    positions = pd.Series({"SPY": 9, "QQQ": 12})  # SPY objetivo 10: diferencia 1 acción = 500 USD
    plan = plan_orders(target, PRICES, positions, 10_000, 0.0, 0.0, 600, UNIVERSE).set_index("ticker")
    assert plan.loc["SPY", "orden"] == HOLD
    assert plan.loc["SPY", "nota"] != ""


def test_positions_outside_universe_are_ignored():
    target = pd.Series({"SPY": 1.0})
    positions = pd.Series({"AAPL": 100})
    plan = plan_orders(target, PRICES, positions, 30_000, 20_000, 0.0, 100, UNIVERSE)
    assert "AAPL" not in set(plan["ticker"])
    assert plan.set_index("ticker").loc["SPY", "cantidad"] == 20  # solo los 10.000 restantes


def test_missing_price_raises():
    with pytest.raises(ValueError):
        plan_orders(pd.Series({"XYZ": 1.0}), PRICES, pd.Series(dtype=float), 10_000, 0.0, 0.0, 100, UNIVERSE)


def test_live_account_is_blocked_by_default(cfg):
    snap = AccountSnapshot("U1234567", False, 10_000, "USD", pd.Series(dtype=float), pd.Series(dtype=float))
    with pytest.raises(RuntimeError):
        check_account_allowed(snap, cfg)
    paper = AccountSnapshot("DU1234567", True, 10_000, "USD", pd.Series(dtype=float), pd.Series(dtype=float))
    check_account_allowed(paper, cfg)
