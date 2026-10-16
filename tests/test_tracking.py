"""Seguimiento: una cuenta paper simulada día a día contra lo que reconstruye el comando."""
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from etfml.config import load_config
from etfml.data import MarketData, synthetic_market_data
from etfml.ib_rebalance import FILL_COLUMNS, fills_to_frame, load_fills, save_fills
from etfml.tracking import format_report, rebalances_from_records, track

ACCOUNT = "DU1234567"
MONTHS = ["2025-07", "2025-08", "2025-09", "2025-10"]
TODAY = pd.Timestamp("2025-12-31")
MID_OCTOBER = pd.Timestamp("2025-10-20")  # el rebalanceo de octubre sigue abierto y al día
TARGETS = [["SYN01", "SYN02", "SYN03"], ["SYN02", "SYN04", "SYN05"], ["SYN04", "SYN05", "SYN06"],
           ["SYN01", "SYN06", "SYN07"]]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_market(dividends: bool) -> MarketData:
    m = synthetic_market_data(seed=7)
    if not dividends:
        return m
    # raw = adj / F: el precio sin ajustar cae 1 % extra cada 63 días hábiles (dividendo)
    idx = m.close.index
    ex_dates = idx[63::63]
    remaining = np.array([(ex_dates > t).sum() for t in idx])
    factor = pd.Series(0.99 ** remaining, index=idx)
    raw = m.close.div(factor, axis=0)
    return MarketData(m.close, m.volume, None, "sintetico", close_raw=raw)


def assumed_commission(amount: float, cfg) -> float:
    return max(amount * cfg.costs.commission_bps / 1e4, cfg.costs.min_commission_usd)


def simulate_account(market: MarketData, cfg, *, delay_days: int = 0, slip_bps: float | None = None,
                     commission_mult: float = 1.0, start_cash: float = 100_000.0, months=MONTHS):
    """Opera una cuenta ficticia como lo haría el rebalanceo real y devuelve registros + llenados."""
    adj, raw = market.close, market.close_raw
    idx = adj.index
    slip = (cfg.costs.slippage_bps if slip_bps is None else slip_bps) / 1e4
    shares: dict[str, int] = {}
    cash = start_cash
    records, fills = [], []
    last_d = None
    exec_id = 0
    for i, month in enumerate(months):
        start = pd.Period(month).to_timestamp()
        signal = idx[idx < start][-1]
        first = idx[idx >= start][0]
        d = idx[idx.get_loc(first) + delay_days]

        # dividendos cobrados en caja desde el último rebalanceo
        if last_d is not None:
            ratio = (adj / raw).loc[last_d:d]
            for t_prev, t in zip(ratio.index[:-1], ratio.index[1:]):
                for tk, q in shares.items():
                    f_prev, f_now = ratio.at[t_prev, tk], ratio.at[t, tk]
                    if not np.isclose(f_prev, f_now):
                        delta = 1 - f_prev / f_now
                        cash += q * delta * raw.at[t, tk] / (1 - delta)
        last_d = d

        px = raw.loc[d]
        before = {t: q for t, q in shares.items() if q}
        value_before = {t: q * px[t] for t, q in before.items()}
        nl = sum(value_before.values()) + cash
        target = {t: 1 / 3 for t in TARGETS[i % len(TARGETS)]}
        investable = nl * (1 - cfg.ib.cash_buffer)
        goal = {t: int(np.floor(w * investable / px[t])) for t, w in target.items()}
        orders = []
        for t in sorted(set(goal) | set(shares)):
            delta = goal.get(t, 0) - shares.get(t, 0)
            if delta:
                orders.append((t, delta))
        orders.sort(key=lambda o: o[1] > 0)  # ventas primero
        exec_orders = []
        for t, delta in orders:
            fill = px[t] * (1 + slip if delta > 0 else 1 - slip)
            amount = abs(delta) * fill
            commission = assumed_commission(amount, cfg) * commission_mult
            cash -= delta * fill + commission
            shares[t] = shares.get(t, 0) + delta
            exec_id += 1
            fills.append({
                "exec_id": f"E{exec_id:04d}", "fecha_hora_utc": f"{d.date()} 19:45:00", "fecha_ny": str(d.date()),
                "cuenta": ACCOUNT, "ticker": t, "lado": "COMPRA" if delta > 0 else "VENTA",
                "acciones": float(abs(delta)), "precio": float(fill), "comision_usd": float(commission),
                "perm_id": exec_id, "order_id": exec_id,
            })
            exec_orders.append({"ticker": t, "orden": "BUY" if delta > 0 else "SELL", "cantidad": abs(delta),
                                "estado": "Filled", "llenadas": abs(delta), "precio_promedio": float(fill)})
        records.append({
            "version": 1, "fecha_hora_ny": f"{d.date()} 15:45:00", "fecha_ny": str(d.date()), "cuenta": ACCOUNT,
            "es_paper": True, "modelo": "ensamble", "fecha_senal": str(signal.date()), "tipo_orden": "MKT",
            "patrimonio_antes": nl, "fuera_universo": 0.0, "colchon_caja": cfg.ib.cash_buffer,
            "posiciones_antes": {t: {"acciones": float(q), "valor": float(value_before[t])}
                                 for t, q in before.items()},
            "pesos_objetivo": target, "plan": [], "ordenes": exec_orders,
        })
        shares = {t: q for t, q in shares.items() if q}
    return records, pd.DataFrame(fills, columns=FILL_COLUMNS)


def scenario(cfg, dividends=False, **kw):
    market = make_market(dividends)
    records, fills = simulate_account(market, cfg, **kw)
    return market, records, fills


def test_perfect_execution_matches_backtest(cfg):
    market, records, fills = scenario(cfg)
    res = track(records, fills, market, cfg, MID_OCTOBER)
    p = res.periods
    assert list(p["completo"]) == [True, True, True, False]
    done = p[p["completo"]]
    comps = done[["caja_redondeo", "fecha", "precio", "comisiones"]].sum(axis=1)
    np.testing.assert_allclose(done["diferencia"], comps, atol=1e-12)
    np.testing.assert_allclose(done["fecha"], 0.0, atol=1e-12)
    np.testing.assert_allclose(done["comisiones"], 0.0, atol=1e-12)
    np.testing.assert_allclose(done["precio"], 0.0, atol=1e-6)
    # sin dividendos, la reconstrucción coincide con el patrimonio de la cuenta
    np.testing.assert_allclose(done["real"], done["segun_ib"], atol=1e-9)
    # la única diferencia es el colchón de caja y el redondeo, que es chica
    assert (done["diferencia"].abs() < 0.01).all()
    assert all(c["ok"] for c in res.checks), [c["texto"] for c in res.checks if not c["ok"]]
    assert res.passed
    assert "Puedes pasar a la siguiente fase" in format_report(res)


def test_dividends_use_unadjusted_prices(cfg):
    market, records, fills = scenario(cfg, dividends=True)
    res = track(records, fills, market, cfg, MID_OCTOBER)
    done = res.periods[res.periods["completo"]]
    # los llenados se comparan con el cierre sin ajustar: el "precio" no se infla por dividendos
    np.testing.assert_allclose(done["precio"], 0.0, atol=1e-6)
    np.testing.assert_allclose(done["precio_bps"], cfg.costs.slippage_bps, atol=0.01)
    # el retorno reconstruido incluye dividendos, igual que la cuenta
    np.testing.assert_allclose(done["real"], done["segun_ib"], atol=2e-3)
    assert res.passed


def test_late_execution_is_flagged(cfg):
    market, records, fills = scenario(cfg, delay_days=2)
    res = track(records, fills, market, cfg, TODAY)
    p = res.periods
    assert not p["fecha_correcta"].any()
    done = p[p["completo"]]
    np.testing.assert_allclose(done["diferencia"], done[["caja_redondeo", "fecha", "precio", "comisiones"]].sum(axis=1),
                               atol=1e-12)
    assert (done["fecha"].abs() > 1e-6).any()
    assert not res.passed


def test_skipped_month_is_flagged(cfg):
    market, records, fills = scenario(cfg, months=["2025-07", "2025-09", "2025-10"])
    res = track(records, fills, market, cfg, TODAY)
    p = res.periods.set_index("mes")
    assert not p.loc["2025-07", "sin_saltos"]
    assert p.loc["2025-09", "sin_saltos"]
    assert any("falta el de 2025-08" in n for n in res.notes)
    dates = res.checks[2]
    assert not dates["ok"] and "2025-07" in dates["accion"]


def test_open_period_compares_same_window(cfg):
    market, records, fills = scenario(cfg)
    res = track(records, fills, market, cfg, TODAY)
    last = res.periods.iloc[-1]
    # el último rebalanceo es de octubre: se compara hasta el rebalanceo teórico de noviembre, no hasta hoy
    assert str(last["hasta"]) == "2025-11-03"
    assert abs(last["fecha"]) < 1e-12
    assert any("2025-11" in n for n in res.notes)


def test_high_commissions_fail(cfg):
    market, records, fills = scenario(cfg, commission_mult=3.0)
    res = track(records, fills, market, cfg, TODAY)
    check = [c for c in res.checks if c["texto"].startswith("Comisiones")][0]
    assert not check["ok"]
    assert (res.periods.loc[res.periods["real"].notna(), "comisiones"] < 0).all()


def test_high_slippage_fails(cfg):
    market, records, fills = scenario(cfg, slip_bps=60)
    res = track(records, fills, market, cfg, TODAY)
    check = [c for c in res.checks if c["texto"].startswith("Precio")][0]
    assert not check["ok"]


def test_missing_fills_are_flagged(cfg):
    market, records, fills = scenario(cfg)
    second = records[1]["fecha_ny"]
    res = track(records, fills[fills["fecha_ny"] != second], market, cfg, TODAY)
    p = res.periods.set_index("mes")
    assert not p.loc["2025-08", "llenado_completo"]
    assert not res.passed


def test_execution_day_without_close_is_pending(cfg):
    market, records, fills = scenario(cfg)
    last = pd.Timestamp(records[-1]["fecha_ny"])
    res = track(records, fills, market, cfg, last)  # hoy = día de la ejecución: aún no hay cierre
    assert res.periods.iloc[-1]["estado"].startswith("pendiente")
    assert res.periods.iloc[-2]["estado"].startswith("en curso")


def test_one_month_is_not_enough(cfg):
    market, records, fills = scenario(cfg, months=MONTHS[:2])
    res = track(records, fills, market, cfg, TODAY)
    assert res.totals["meses_completos"] == 1
    assert not res.checks[0]["ok"]


def test_same_day_records_are_merged():
    base = {"cuenta": ACCOUNT, "modelo": "ensamble", "fecha_senal": "2025-09-30", "tipo_orden": "MKT",
            "patrimonio_antes": 1000.0, "fuera_universo": 0.0, "posiciones_antes": {},
            "pesos_objetivo": {"SYN01": 1.0}, "plan": []}
    a = {**base, "fecha_hora_ny": "2025-10-01 10:00:00", "fecha_ny": "2025-10-01",
         "ordenes": [{"ticker": "SYN01", "orden": "BUY", "cantidad": 3}]}
    b = {**base, "fecha_hora_ny": "2025-10-01 11:00:00", "fecha_ny": "2025-10-01", "patrimonio_antes": 999.0,
         "ordenes": [{"ticker": "SYN01", "orden": "BUY", "cantidad": 2}]}
    rbs, notes = rebalances_from_records([a, b])
    assert len(rbs) == 1
    assert rbs[0].ordered["SYN01"] == 5
    assert rbs[0].equity_before == 1000.0  # foto de la primera ejecución
    assert notes


def _fill(exec_id, when, side, shares, price, commission, account=ACCOUNT, report_id=None):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol="SPY", secType="STK"),
        execution=SimpleNamespace(execId=exec_id, time=when, acctNumber=account, side=side, shares=shares,
                                  price=price, permId=7, orderId=3),
        commissionReport=SimpleNamespace(execId=exec_id if report_id is None else report_id, commission=commission),
    )


def test_fills_to_frame_converts_ib_fills():
    late_utc = datetime(2025, 11, 4, 1, 0, tzinfo=timezone.utc)  # 20:00 del 3 de nov. en Nueva York
    df = fills_to_frame([
        _fill("A", datetime(2025, 11, 3, 20, 30, tzinfo=timezone.utc), "BOT", 10, 500.0, 1.0),
        _fill("B", late_utc, "SLD", 5, 501.0, 0.0, report_id=""),           # comisión aún no informada
        _fill("C", late_utc, "BOT", 1, 1.0, 1.7976931348623157e308),        # valor "sin dato" de IB
        _fill("D", late_utc, "BOT", 1, 1.0, 1.0, account="U999"),            # otra cuenta
    ], ACCOUNT)
    assert list(df["exec_id"]) == ["A", "B", "C"]
    assert list(df["lado"]) == ["COMPRA", "VENTA", "COMPRA"]
    assert list(df["fecha_ny"]) == ["2025-11-03", "2025-11-03", "2025-11-03"]
    assert df["comision_usd"].iloc[0] == 1.0
    assert df["comision_usd"].iloc[1:].isna().all()


def test_save_fills_dedupes_and_updates_commission(tmp_path, cfg):
    c = replace(cfg, logs_dir=tmp_path)
    when = datetime(2025, 11, 3, 20, 30, tzinfo=timezone.utc)
    first = fills_to_frame([_fill("A", when, "BOT", 10, 500.0, 0.0, report_id="")], ACCOUNT)
    assert save_fills(c, first) == 1
    later = fills_to_frame([_fill("A", when, "BOT", 10, 500.0, 1.25),
                            _fill("B", when, "SLD", 5, 501.0, 1.0)], ACCOUNT)
    assert save_fills(c, later) == 1
    stored = load_fills(c)
    assert list(stored["exec_id"]) == ["A", "B"]
    assert stored.set_index("exec_id").loc["A", "comision_usd"] == 1.25
