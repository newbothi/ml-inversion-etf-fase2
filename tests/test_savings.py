"""Simulador de aportes: cuentas exactas en casos simples, pesos de hoy, escenarios, meta y el comando."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import etfml.__main__ as cli
import etfml.data as data_mod
import etfml.external as ext
from etfml.config import DEFAULT_SAVINGS, load_config
from etfml.data import synthetic_market_data
from etfml.savings import (PORTFOLIOS, Scenario, Scheme, goal_table, ib_commission, investment_cost,
                           long_scenarios, money_weighted_return, monthly_returns, pesos_scenario,
                           required_contribution, run_windows, scenario_comparison, simulate_window,
                           to_real_usd, usd_scenario)

FREE = replace(DEFAULT_SAVINGS, fx_spread=0.0, dividend_drag=0.0, btg_order_fee=0.0, btg_min_order=0.0,
               ib_swift_fee=0.0, ib_min_fee=0.0, ib_per_share=0.0, ib_max_pct=0.0)
GLOBAL = PORTFOLIOS["global"]
BTG = Scheme("btg_1", "BTG mensual", "btg", 1)


def _months(n, start="2004-02"):
    return pd.period_range(start, periods=n, freq="M")


# --------------------------------------------------------------------------- motor

def test_no_returns_no_costs_gives_back_what_you_put_in():
    final, costs, values, contributed = simulate_window(np.zeros(24), 100.0, BTG, GLOBAL, FREE)
    assert final == pytest.approx(2400.0)
    assert costs == 0.0
    np.testing.assert_allclose(values, contributed)
    assert money_weighted_return(100.0, 24, final) == pytest.approx(0.0, abs=1e-6)


def test_constant_return_matches_the_annuity_formula():
    r, n, c = 0.01, 60, 100.0
    final, _, _, _ = simulate_window(np.full(n, r), c, BTG, GLOBAL, FREE)
    assert final == pytest.approx(c * ((1 + r) ** n - 1) / r * (1 + r))      # aporte al inicio de cada mes
    assert money_weighted_return(c, n, final) == pytest.approx((1 + r) ** 12 - 1, rel=1e-6)


def test_btg_charges_per_order_plus_fx():
    cfg = replace(FREE, fx_spread=0.01, btg_order_fee=1.0)
    assert investment_cost(100.0, BTG, GLOBAL, cfg)["total"] == pytest.approx(2.0)
    assert investment_cost(100.0, BTG, PORTFOLIOS["conservadora"], cfg)["commission"] == pytest.approx(2.0)


def test_ib_commission_rules():
    assert ib_commission(50.0, DEFAULT_SAVINGS) == pytest.approx(0.50)          # tope de 1 % de la orden
    assert ib_commission(500.0, DEFAULT_SAVINGS) == pytest.approx(1.00)         # mínimo US$1
    assert ib_commission(130 * 1000, DEFAULT_SAVINGS) == pytest.approx(5.00)    # US$0,005 por acción


def test_ib_sends_every_k_months_and_pays_swift_each_time():
    ib3 = Scheme("ib_3", "IB trimestral", "ib", 3)
    final, costs, _, _ = simulate_window(np.zeros(12), 100.0, ib3, GLOBAL, replace(FREE, ib_swift_fee=25.0))
    assert costs == pytest.approx(4 * 25.0)
    assert final == pytest.approx(1200.0 - 100.0)


def test_money_waiting_is_not_invested():
    ib3 = Scheme("ib_3", "IB trimestral", "ib", 3)
    rets = np.array([0.10, 0.0, 0.0])
    assert simulate_window(rets, 100.0, BTG, GLOBAL, FREE)[0] == pytest.approx(310.0)
    assert simulate_window(rets, 100.0, ib3, GLOBAL, FREE)[0] == pytest.approx(300.0)
    assert simulate_window(rets, 100.0, ib3, GLOBAL, replace(FREE, waiting_rate=0.12))[0] > 300.0


def test_btg_minimum_order_makes_small_amounts_wait():
    final, _, values, _ = simulate_window(np.full(3, 0.10), 10.0, BTG, GLOBAL, replace(FREE, btg_min_order=20.0))
    assert values[0] == pytest.approx(10.0)
    assert final == pytest.approx(20 * 1.1 * 1.1 + 10.0)


def test_costs_in_pesos_are_converted_through_dollars():
    # aporte de $100.000 con dólar a $1.000 = US$100: 1 % de cambio + US$1 = US$2 = $2.000
    cfg = replace(FREE, fx_spread=0.01, btg_order_fee=1.0)
    _, costs, _, _ = simulate_window(np.zeros(1), 100_000.0, BTG, GLOBAL, cfg, unit_to_usd=1 / 1000)
    assert costs == pytest.approx(2_000.0)


def test_windows_cover_every_start():
    rets = pd.Series(np.full(100, 0.005), index=_months(100))
    ws = run_windows(usd_scenario(rets, "x"), 5, BTG, GLOBAL, FREE, 100.0)
    assert len(ws) == 100 - 60 + 1
    assert ws.starts[0] == rets.index[0] and ws.starts[-1] == rets.index[40]
    np.testing.assert_allclose(ws.final, ws.final[0])                          # retornos iguales, ventanas iguales


def test_etf_returns_skip_the_current_month():
    market = synthetic_market_data(seed=7)
    close = market.close.rename(columns={"SYN01": "SPY", "SYN02": "EFA", "SYN03": "EEM", "SYN04": "IEF"})
    assert str(monthly_returns(close, GLOBAL, pd.Timestamp("2025-12-15")).index[-1]) == "2025-11"


# --------------------------------------------------------------------------- pesos de hoy y dólares de hoy

def test_pesos_of_today_follow_dollar_and_uf():
    idx = _months(40, "2004-01")
    usd = pd.Series(0.01, index=idx[1:])
    flat = pd.DataFrame({"dolar": 800.0, "uf": 20000.0}, index=idx)
    same = pesos_scenario(usd, flat, 800.0, "x")
    np.testing.assert_allclose(same.returns, 0.01)                            # sin cambios de dólar ni UF
    moving = pd.DataFrame({"dolar": 800.0 * 1.005 ** np.arange(40), "uf": 20000.0 * 1.003 ** np.arange(40)}, index=idx)
    scn = pesos_scenario(usd, moving, 800.0, "x")
    np.testing.assert_allclose(scn.returns, 1.01 * 1.005 / 1.003 - 1)
    assert scn.unit == "CLP" and scn.real and scn.unit_to_usd == pytest.approx(1 / 800)


def test_real_dollars_discount_us_inflation():
    idx = _months(5)
    cpi = pd.Series(100 * 1.002 ** np.arange(6), index=_months(6, "2004-01"))
    real = to_real_usd(pd.Series(0.01, index=idx), cpi)
    np.testing.assert_allclose(real, 1.01 / 1.002 - 1)


def test_long_scenarios_use_market_and_bills():
    _, long_df = ext.synthetic_external()
    etf = pd.Series(0.005, index=_months(200))
    scns = {s.key: s for s in long_scenarios(long_df, PORTFOLIOS["conservadora"], etf)}
    assert set(scns) == {"etf", "eeuu", "japon"}
    assert "letras" in scns["eeuu"].label and scns["japon"].returns.index[0] >= pd.Period("1990-07", "M")
    m = pd.Period("1950-01", "M")
    nominal = 0.7 * long_df.loc[m, "eeuu_mercado"] + 0.3 * long_df.loc[m, "eeuu_rf"]
    infl = long_df.loc[m, "ipc_eeuu"] / long_df.loc[m - 1, "ipc_eeuu"]
    assert scns["eeuu"].returns.loc[m] == pytest.approx((1 + nominal) / infl - 1)


def test_comparison_counts_independent_periods():
    rets = pd.Series(0.004, index=_months(25 * 12, "1990-01"))
    scn = Scenario("x", "X", "USD", True, rets, 1.0)
    cfg = replace(FREE, monthly_clp=100_000, usd_clp=1000)
    c = scenario_comparison([scn], [5, 10, 20], GLOBAL, cfg).set_index("anios")
    assert list(c["independientes"]) == [5, 2, 1]
    assert (c["pct_perdida"] == 0).all() and (c["mediana_x"] > 1).all()


# --------------------------------------------------------------------------- meta

def test_required_contribution_inverts_the_annuity():
    r, years = 0.005, 10
    n = years * 12
    scn = Scenario("x", "X", "CLP", True, pd.Series(r, index=_months(n + 30)), 1 / 1000)
    target = 20_000_000.0
    expected = target / (((1 + r) ** n - 1) / r * (1 + r))
    for level in (0.5, 0.9, 1.0):                                               # todas las ventanas son iguales
        assert required_contribution(scn, years, target, level, GLOBAL, FREE) == pytest.approx(expected, rel=1e-6)


def test_goal_table_is_in_pesos_and_ordered_by_safety():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.006, 0.04, 300), index=_months(300))
    clp = Scenario("principal", "Pesos", "CLP", True, rets, 1 / 1000)
    usd = Scenario("eeuu", "Dólares", "USD", True, rets, 1.0)
    cfg = replace(DEFAULT_SAVINGS, monthly_clp=100_000, usd_clp=1000)
    g = goal_table([clp, usd], 10, 20_000_000, 0.9, GLOBAL, cfg).set_index("clave")
    for key in ("principal", "eeuu"):
        assert g.loc[key, "tipico"] < g.loc[key, "confianza"] < g.loc[key, "peor"]
    # mismos retornos: en pesos o en dólares (convertidos a pesos) el aporte necesario es casi igual
    assert g.loc["eeuu", "tipico"] == pytest.approx(g.loc["principal", "tipico"], rel=0.01)
    assert 0 <= g.loc["principal", "prob_con_aporte_actual"] <= 1


# --------------------------------------------------------------------------- comando

@pytest.fixture
def wired(tmp_path, monkeypatch):
    cfg = replace(load_config(), reports_dir=tmp_path / "reportes", data_dir=tmp_path / "datos")
    market = synthetic_market_data(seed=7)
    close = market.close.rename(columns={"SYN01": "SPY", "SYN02": "EFA", "SYN03": "EEM", "SYN04": "IEF"})
    fake = data_mod.MarketData(close, market.volume, None, "yahoo", close_raw=close)
    chile, long_df = ext.synthetic_external()
    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(cli, "_today", lambda: pd.Timestamp("2026-01-15"))
    monkeypatch.setattr(data_mod, "load_market_data",
                        lambda c, refresh=False, tickers=None: data_mod.restrict(fake, list(tickers)))
    monkeypatch.setattr(ext, "load_chile_indicators", lambda c, refresh=False: chile)
    monkeypatch.setattr(ext, "load_long_history", lambda c, refresh=False: long_df)
    return cfg


def test_simulador_in_pesos_with_scenarios_and_goal(wired, capsys):
    assert cli.main(["simulador", "--aporte", "50000", "--anios", "10", "--meta", "15000000"]) == 0
    out = capsys.readouterr().out
    assert "Resultados en pesos de hoy" in out
    assert "Dólar observado al" in out
    assert "Japón" in out and "EE.UU. 1926" in out
    assert "Meta: $15.000.000 de hoy en 10 años" in out
    report = next(wired.reports_dir.glob("*_simulador/reporte.html"))
    html = report.read_text(encoding="utf-8")
    assert "Tu meta: $15.000.000 de hoy en 10 años" in html
    assert "¿Y si el futuro se parece a otra época?" in html
    for name in ("escenarios.csv", "meta.csv", "ventanas.csv"):
        assert (report.parent / name).exists()


def test_simulador_without_external_data_falls_back_to_dollars(wired, monkeypatch, capsys):
    monkeypatch.setattr(ext, "load_chile_indicators", lambda c, refresh=False: None)
    monkeypatch.setattr(ext, "load_long_history", lambda c, refresh=False: None)
    assert cli.main(["simulador", "--dolar", "950"]) == 0
    out = capsys.readouterr().out
    assert "Resultados en dólares" in out
    assert "Sin dólar observado ni UF" in out and "Sin historia larga" in out


def test_simulador_rejects_bad_input(wired):
    assert cli.main(["simulador", "--aporte", "0"]) == 2
    assert cli.main(["simulador", "--anios", "40"]) == 2
    assert cli.main(["simulador", "--meta", "-5"]) == 2
    assert cli.main(["simulador", "--confianza", "1.5"]) == 2
