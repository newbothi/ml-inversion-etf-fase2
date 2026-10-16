"""Señal en vivo y flujo completo de rebalanceo contra un IB falso (sin red)."""
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import etfml.__main__ as cli
import etfml.data as data_mod
import etfml.ib_rebalance as ibr
from etfml.config import load_config
from etfml.data import synthetic_market_data
from etfml.features import build_panel
from etfml.live_signal import compute_live_signal

from .conftest import TODAY


def _synthetic_cfg(tmp_path):
    market = synthetic_market_data(seed=7)
    tickers = list(market.close.columns)
    cfg = load_config().with_universe(tickers, tickers[0])
    cfg = replace(cfg, logs_dir=tmp_path / "registros", data_dir=tmp_path / "datos",
                  reports_dir=tmp_path / "reportes")
    return cfg, market


def test_live_signal_uses_last_complete_month(tmp_path):
    cfg, market = _synthetic_cfg(tmp_path)
    panel = build_panel(market, min_assets=6, today=TODAY)
    sig = compute_live_signal(panel, market, cfg, "ensamble", TODAY)
    assert sig.signal_date == pd.Timestamp("2025-12-31")
    assert len(sig.weights) == cfg.strategy.top_k
    assert abs(sig.weights.sum() - 1) < 1e-12
    assert set(sig.weights.index) <= set(sig.scores.index)
    # entrena solo con meses cuyo retorno ya se conocía
    assert sig.train_months == panel.labeled.index.get_level_values("date").nunique()
    assert not sig.warnings


class FakeTrade:
    def __init__(self, contract, order):
        self.contract, self.order = contract, order
        self.orderStatus = SimpleNamespace(status="Filled", filled=order.totalQuantity, avgFillPrice=100.0)

    def isDone(self):
        return True


class FakeIB:
    def __init__(self, account="DU999", positions=None, net_liq=50_000.0):
        self.account, self.net_liq = account, net_liq
        self.positions = positions or {}
        self.placed = []
        self.disconnected = False

    def managedAccounts(self):
        return [self.account]

    def accountSummary(self, account=""):
        return [SimpleNamespace(tag="NetLiquidation", value=str(self.net_liq), currency="USD")]

    def portfolio(self, account=""):
        return [SimpleNamespace(contract=SimpleNamespace(symbol=s, secType="STK"), position=q, marketValue=q * 100.0)
                for s, q in self.positions.items()]

    def qualifyContracts(self, *contracts):
        for i, c in enumerate(contracts):
            c.conId = 1000 + i
        return list(contracts)

    def placeOrder(self, contract, order):
        self.placed.append((contract.symbol, order.action, order.totalQuantity, order.orderType))
        order.orderId = order.permId = len(self.placed)
        return FakeTrade(contract, order)

    def reqExecutions(self, execFilter=None):
        now = datetime.now(timezone.utc)
        return [SimpleNamespace(
            contract=SimpleNamespace(symbol=sym, secType="STK"),
            execution=SimpleNamespace(execId=f"X{i}", time=now, acctNumber=self.account,
                                      side="BOT" if act == "BUY" else "SLD", shares=qty, price=100.0,
                                      permId=i + 1, orderId=i + 1),
            commissionReport=SimpleNamespace(execId=f"X{i}", commission=1.0),
        ) for i, (sym, act, qty, _) in enumerate(self.placed)]

    def sleep(self, _):
        pass

    def disconnect(self):
        self.disconnected = True


@pytest.fixture
def wired(tmp_path, monkeypatch):
    cfg, market = _synthetic_cfg(tmp_path)
    fake = FakeIB(positions={"SYN01": 50, "AAPL": 10})
    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(cli, "_today", lambda: TODAY)
    monkeypatch.setattr(data_mod, "load_market_data", lambda c, refresh=False: market)
    monkeypatch.setattr(ibr, "connect", lambda c, readonly: fake)
    return cfg, fake


def test_dry_run_sends_nothing(wired):
    cfg, fake = wired
    assert cli.main(["rebalancear"]) == 0
    assert fake.placed == []
    assert fake.disconnected
    assert list(cfg.logs_dir.glob("ordenes_*_simulacion.csv"))
    assert (cfg.logs_dir / "senales.csv").exists()


def test_execute_requires_typing_si(wired, monkeypatch):
    cfg, fake = wired
    monkeypatch.setattr("builtins.input", lambda _: "si, dale")
    assert cli.main(["rebalancear", "--ejecutar"]) == 1
    assert fake.placed == []


def test_execute_sells_before_buying_and_ignores_outside_positions(wired, monkeypatch):
    cfg, fake = wired
    monkeypatch.setattr("builtins.input", lambda _: "SI")
    assert cli.main(["rebalancear", "--ejecutar"]) == 0
    assert fake.placed, "debería haber órdenes"
    actions = [a for _, a, _, _ in fake.placed]
    assert actions == sorted(actions, key=lambda a: 0 if a == "SELL" else 1)
    assert "AAPL" not in {s for s, *_ in fake.placed}
    assert all(t == "MKT" for *_, t in fake.placed)
    assert list(cfg.logs_dir.glob("ordenes_*.csv"))

    # deja todo lo que necesita el seguimiento
    records = list((cfg.logs_dir / "ejecuciones").glob("*.json"))
    assert len(records) == 1
    rec = json.loads(records[0].read_text(encoding="utf-8"))
    assert rec["cuenta"] == "DU999" and rec["patrimonio_antes"] == 50_000.0
    assert rec["fuera_universo"] == 1_000.0                      # AAPL: 10 acciones x 100
    assert rec["posiciones_antes"] == {"SYN01": {"acciones": 50.0, "valor": 5_000.0}}
    assert len(rec["ordenes"]) == len(fake.placed)
    assert abs(sum(rec["pesos_objetivo"].values()) - 1) < 1e-9
    fills = (cfg.logs_dir / "llenados.csv").read_text(encoding="utf-8").splitlines()
    assert len(fills) == 1 + len(fake.placed)

    # el seguimiento corre (queda pendiente: los datos sintéticos no llegan a la fecha de hoy)
    assert cli.main(["seguimiento"]) == 0
    assert (cfg.logs_dir / "seguimiento.csv").exists()


def test_llenados_command_saves_todays_fills(wired):
    cfg, fake = wired
    fake.placed = [("SYN01", "BUY", 5, "MKT"), ("SYN02", "SELL", 3, "MKT")]
    assert cli.main(["llenados"]) == 0
    assert cli.main(["llenados"]) == 0  # correrlo dos veces no duplica
    lines = (cfg.logs_dir / "llenados.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_seguimiento_without_records(wired, capsys):
    assert cli.main(["seguimiento"]) == 0
    assert "Todavía no hay ejecuciones" in capsys.readouterr().out


def test_live_account_blocks_execution(wired, monkeypatch):
    cfg, fake = wired
    fake.account = "U1234567"
    monkeypatch.setattr("builtins.input", lambda _: "SI")
    assert cli.main(["rebalancear", "--ejecutar"]) == 2
    assert fake.placed == []
    assert fake.disconnected


def test_yahoo_format_is_parsed_and_cached(tmp_path, monkeypatch):
    """Simula la respuesta de yfinance (columnas MultiIndex Price x Ticker) sin internet."""
    cfg, market = _synthetic_cfg(tmp_path)
    tickers = list(cfg.universe.tickers)
    dates = market.close.index.tz_localize("America/New_York")
    raw_close = market.close * 1.02  # sin ajustar: algo más alto que el ajustado por dividendos
    fields = {"Adj Close": market.close, "Close": raw_close, "Volume": market.volume, "Open": raw_close,
              "High": raw_close, "Low": raw_close}
    raw = pd.concat({k: v.set_axis(dates) for k, v in fields.items()}, axis=1)
    raw.columns.names = ["Price", "Ticker"]
    monkeypatch.setattr(data_mod, "_yf_download", lambda t, start: raw[[c for c in raw.columns if c[1] in t]])

    got = data_mod.load_market_data(cfg, refresh=True)
    assert list(got.close.columns) == tickers
    assert got.close.index.tz is None
    pd.testing.assert_frame_equal(got.close, data_mod.clean_prices(market.close, market.volume)[0],
                                  check_freq=False)
    pd.testing.assert_frame_equal(got.close_raw, raw_close, check_freq=False)
    cached = data_mod.load_cache(cfg)
    assert cached is not None
    pd.testing.assert_frame_equal(cached.close, got.close, check_freq=False, check_names=False)
    pd.testing.assert_frame_equal(cached.close_raw, got.close_raw, check_freq=False, check_names=False)


def test_aligned_portfolio_still_registers_the_month(wired, monkeypatch):
    cfg, fake = wired
    monkeypatch.setattr("builtins.input", lambda _: "SI")
    assert cli.main(["rebalancear", "--ejecutar"]) == 0
    first_orders = len(fake.placed)
    # la cuenta queda con lo que se compró: un segundo rebalanceo no tiene nada que hacer
    held = {}
    for sym, act, qty, _ in fake.placed:
        held[sym] = held.get(sym, 0) + (qty if act == "BUY" else -qty)
    fake.positions = {t: q + fake.positions.get(t, 0) for t, q in held.items()} | {"AAPL": 10}
    fake.positions = {t: q for t, q in fake.positions.items() if q}
    for path in (cfg.logs_dir / "ejecuciones").glob("*.json"):
        path.unlink()
    assert cli.main(["rebalancear", "--ejecutar"]) == 0
    assert len(fake.placed) == first_orders
    assert len(list((cfg.logs_dir / "ejecuciones").glob("*.json"))) == 1
