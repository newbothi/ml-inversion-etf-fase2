"""Registro de experimentos y comandos de la fase 3 de punta a punta (sin internet)."""
from dataclasses import replace

import pandas as pd
import pytest

import etfml.__main__ as cli
import etfml.data as data_mod
from etfml.config import TrendConfig, load_config
from etfml.data import restrict, synthetic_market_data
from etfml.registry import load_registry, record_trials

TODAY = pd.Timestamp("2026-01-15")


def _row(trial_id, sharpe=0.5):
    return {"id": trial_id, "experimento": "tendencia", "variante": trial_id, "descripcion": trial_id,
            "desde": "2005-02", "hasta": "2025-11", "meses": 250, "retorno_anual": 0.08, "volatilidad": 0.15,
            "sharpe_anual": sharpe, "sharpe_mensual": sharpe / 12 ** 0.5, "max_caida": -0.3, "costo_anual": 0.001,
            "datos_hasta": "2025-12-31", "registrado": "2026-01-15 10:00", "parametros": "{}"}


def test_same_trial_updates_instead_of_adding(tmp_path):
    cfg = replace(load_config(), experiments_dir=tmp_path)
    record_trials(cfg, [_row("a"), _row("b")])
    reg = record_trials(cfg, [_row("a", sharpe=0.9), _row("c")])
    assert sorted(reg["id"]) == ["a", "b", "c"]
    assert load_registry(cfg).set_index("id").loc["a", "sharpe_anual"] == 0.9


@pytest.fixture
def wired(tmp_path, monkeypatch):
    market = synthetic_market_data(seed=7)
    tickers = list(market.close.columns)
    cfg = load_config().with_universe(tickers, tickers[0])
    cfg = replace(cfg, trend=TrendConfig("SYN01", ("SYN02", "SYN03")), data_dir=tmp_path / "datos",
                  reports_dir=tmp_path / "reportes", logs_dir=tmp_path / "registros",
                  experiments_dir=tmp_path / "experimentos")
    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(cli, "_today", lambda: TODAY)
    # los comandos creen que son datos reales de Yahoo: así se prueba el registro
    real_like = data_mod.MarketData(market.close, market.volume, None, "yahoo", close_raw=market.close_raw)
    monkeypatch.setattr(data_mod, "load_market_data",
                        lambda c, refresh=False, tickers=None: restrict(real_like, list(tickers or c.universe.tickers)))
    return cfg


@pytest.mark.slow
def test_tendencia_registers_its_variants_and_warns_without_phase1(wired, capsys):
    assert cli.main(["tendencia"]) == 0
    out = capsys.readouterr().out
    assert "no tiene las pruebas de la fase 1" in out
    reg = load_registry(wired)
    assert len(reg) == 4 and set(reg["experimento"]) == {"tendencia"}
    reports = list(wired.reports_dir.glob("*_tendencia/reporte.html"))
    assert len(reports) == 1
    html = reports[0].read_text(encoding="utf-8")
    assert "Registro de experimentos (4 pruebas)" in html


@pytest.mark.slow
def test_full_flow_counts_every_trial_once(wired, capsys):
    assert cli.main(["backtest"]) == 0
    assert len(load_registry(wired)) == 5           # 4 modelos + momentum
    assert cli.main(["backtest"]) == 0
    assert len(load_registry(wired)) == 5           # repetir no suma
    assert cli.main(["tendencia"]) == 0
    out = capsys.readouterr().out
    assert "9 pruebas en el registro" in out
    assert "no tiene las pruebas de la fase 1" not in out
    assert cli.main(["experimentos"]) == 0
    assert "9 prueba(s) registradas" in capsys.readouterr().out


@pytest.mark.slow
def test_synthetic_runs_never_register(wired):
    assert cli.main(["backtest", "--sintetico"]) == 0
    assert cli.main(["tendencia", "--sintetico"]) == 0
    assert load_registry(wired).empty


@pytest.mark.slow
def test_changing_costs_counts_as_a_new_trial(wired, monkeypatch):
    assert cli.main(["tendencia"]) == 0
    pricier = replace(wired, costs=replace(wired.costs, slippage_bps=10.0))
    monkeypatch.setattr(cli, "load_config", lambda path=None: pricier)
    assert cli.main(["tendencia"]) == 0
    assert len(load_registry(wired)) == 8


def test_download_includes_experiment_tickers_and_tolerates_missing_ones(tmp_path, monkeypatch, capsys):
    market = synthetic_market_data(seed=7)
    tickers = list(market.close.columns)
    cfg = load_config().with_universe(tickers, tickers[0])
    cfg = replace(cfg, trend=TrendConfig("SYN01", ("SYN02", "XXX")), data_dir=tmp_path)
    fields = {"Adj Close": market.close, "Close": market.close, "Volume": market.volume}
    raw = pd.concat(fields, axis=1)
    asked = []

    def fake_download(t, start):
        asked.append(list(t))
        return raw[[c for c in raw.columns if c[1] in t]]

    monkeypatch.setattr(data_mod, "_yf_download", fake_download)
    got = data_mod.load_market_data(cfg, refresh=True)
    assert asked[-1] == tickers + ["XXX"]                     # pide el universo y el extra
    assert list(got.close.columns) == tickers                 # pero devuelve solo el universo
    assert "sin datos para XXX" in capsys.readouterr().out    # el extra que falta solo avisa
    trend = data_mod.load_market_data(cfg, tickers=["SYN01", "SYN02"])
    assert list(trend.close.columns) == ["SYN01", "SYN02"]    # sale del archivo guardado
    with pytest.raises(RuntimeError, match="XXX"):
        data_mod.load_market_data(cfg, tickers=["SYN01", "XXX"])
