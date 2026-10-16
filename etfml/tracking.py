"""Seguimiento de la cuenta paper: lo ejecutado contra lo que asume el backtest.

Para cada rebalanceo ejecutado se reconstruye el mes de la cartera con los llenados reales y los
precios de Yahoo, y se compara con el "teórico": la misma señal simulada con los supuestos del
backtest (compra al cierre del día hábil siguiente a la señal, costos de config.toml).

La diferencia se descompone en cuatro partes que suman exacto:
  caja y redondeo   pesos reales distintos a los objetivo (colchón de caja, acciones enteras)
  fecha             ejecutar otro día que el que asume el backtest
  precio            precio de llenado contra el cierre del día, frente al deslizamiento supuesto
  comisiones        comisiones reales frente a las supuestas
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .data import MarketData
from .features import rebalance_calendar

MIN_COMPLETED_MONTHS = 2
COMMISSION_TOLERANCE = 1.5      # comisiones reales hasta 1,5 veces las supuestas
PRICE_TOLERANCE_BPS = 20.0      # precio vs cierre hasta 20 bps sobre el deslizamiento supuesto
IB_GAP_WARNING = 0.015          # aviso si la reconstrucción y el patrimonio de IB difieren más de 1,5 pp


@dataclass
class Rebalance:
    date: pd.Timestamp
    account: str
    model: str
    signal_date: pd.Timestamp
    order_type: str
    equity_before: float           # patrimonio de la estrategia antes de operar (sin posiciones fuera del universo)
    shares_before: pd.Series
    value_before: pd.Series
    target: pd.Series
    ordered: pd.Series             # cantidad neta pedida por ticker (+ compra, - venta)
    merged_records: int = 1


@dataclass
class TrackingResult:
    periods: pd.DataFrame
    checks: list[dict]
    totals: dict
    passed: bool
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- lectura de registros

def _ordered_from_record(record: dict) -> pd.Series:
    sign = {"BUY": 1, "SELL": -1, "COMPRAR": 1, "VENDER": -1}
    rows = record.get("ordenes") or []
    if rows:
        q = {}
        for o in rows:
            q[o["ticker"]] = q.get(o["ticker"], 0.0) + sign[o["orden"]] * float(o["cantidad"])
        return pd.Series(q, dtype=float)
    plan = [p for p in record.get("plan", []) if p["orden"] in ("COMPRAR", "VENDER")]
    return pd.Series({p["ticker"]: sign[p["orden"]] * float(p["cantidad"]) for p in plan}, dtype=float)


def rebalances_from_records(records: list[dict]) -> tuple[list[Rebalance], list[str]]:
    """Agrupa los registros por cuenta y día. Si hubo varias ejecuciones el mismo día se suman."""
    notes: list[str] = []
    if not records:
        return [], notes
    account = records[-1]["cuenta"]
    others = sorted({r["cuenta"] for r in records} - {account})
    if others:
        notes.append(f"Se ignoran registros de otras cuentas: {', '.join(others)}.")
    own = [r for r in records if r["cuenta"] == account]
    by_day: dict[str, list[dict]] = {}
    for r in own:
        by_day.setdefault(r["fecha_ny"], []).append(r)

    out = []
    for day in sorted(by_day):
        group = sorted(by_day[day], key=lambda r: r["fecha_hora_ny"])
        first, last = group[0], group[-1]
        before = first.get("posiciones_antes", {})
        ordered = pd.Series(dtype=float)
        for r in group:
            ordered = ordered.add(_ordered_from_record(r), fill_value=0.0)
        if len(group) > 1:
            notes.append(f"El {day} hubo {len(group)} ejecuciones; se tratan como un solo rebalanceo.")
        out.append(Rebalance(
            date=pd.Timestamp(day),
            account=account,
            model=last["modelo"],
            signal_date=pd.Timestamp(last["fecha_senal"]),
            order_type=last.get("tipo_orden", "MKT"),
            equity_before=float(first["patrimonio_antes"]) - float(first.get("fuera_universo", 0.0)),
            shares_before=pd.Series({t: v["acciones"] for t, v in before.items()}, dtype=float),
            value_before=pd.Series({t: v["valor"] for t, v in before.items()}, dtype=float),
            target=pd.Series(last["pesos_objetivo"], dtype=float),
            ordered=ordered[ordered != 0],
            merged_records=len(group),
        ))
    return out, notes


# --------------------------------------------------------------------------- cálculo

def _assumed_costs(trades: pd.DataFrame, cfg: Config) -> tuple[float, float]:
    """Comisión y deslizamiento que el backtest habría supuesto para estas mismas órdenes."""
    if trades.empty:
        return 0.0, 0.0
    amount = trades.groupby(["ticker", "lado"])["monto"].sum()
    c = cfg.costs
    commission = float(np.maximum(amount * c.commission_bps / 1e4, c.min_commission_usd).sum())
    slippage = float((amount * c.slippage_bps / 1e4).sum())
    return commission, slippage


def _next_trading_day(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(date, side="right")
    return index[pos] if pos < len(index) else None


def track(records: list[dict], fills: pd.DataFrame, market: MarketData, cfg: Config,
          today: pd.Timestamp) -> TrackingResult:
    rebalances, notes = rebalances_from_records(records)
    if market.close_raw is None:
        raise ValueError("Faltan los precios sin ajustar. Corre el comando con --actualizar.")

    today = pd.Timestamp(today).normalize()
    adj = market.close[market.close.index < today]
    raw = market.close_raw.reindex_like(adj)
    index = adj.index
    cal = rebalance_calendar(index).set_index("signal_date")

    fills = fills.copy()
    if not fills.empty:
        fills["fecha_ny"] = pd.to_datetime(fills["fecha_ny"])
        fills["signo"] = np.where(fills["lado"] == "COMPRA", 1.0, -1.0)

    rows = []
    for i, rb in enumerate(rebalances):
        nxt = rebalances[i + 1] if i + 1 < len(rebalances) else None
        row = {"mes": str(rb.date.to_period("M")), "ejecutado": rb.date.date(), "senal": rb.signal_date.date(),
               "modelo": rb.model, "tipo_orden": rb.order_type, "estado": "", "completo": False}

        # ---- llenados de este rebalanceo
        f = fills[(fills["cuenta"] == rb.account) & (fills["fecha_ny"] == rb.date)] if not fills.empty else fills
        filled = (f["signo"] * f["acciones"]).groupby(f["ticker"]).sum() if len(f) else pd.Series(dtype=float)
        filled = filled[filled != 0]
        tickers = sorted(set(rb.ordered.index) | set(filled.index))
        missing_fill = [t for t in tickers if not np.isclose(filled.get(t, 0.0), rb.ordered.get(t, 0.0))]
        missing_commission = bool(len(f)) and bool(f["comision_usd"].isna().any())
        row["llenado_completo"] = not missing_fill
        row["comisiones_completas"] = not missing_commission

        # ---- fecha y señal que asume el backtest
        month_start = rb.date.to_period("M").to_timestamp()
        prev = index[index < month_start]
        expected_signal = prev[-1] if len(prev) else None
        theo_start = cal["exec_date"].get(rb.signal_date) if rb.signal_date in cal.index else None
        theo_start = None if theo_start is None or pd.isna(theo_start) else pd.Timestamp(theo_start)
        row["senal_correcta"] = expected_signal is not None and rb.signal_date == expected_signal
        row["fecha_correcta"] = theo_start is not None and rb.date == theo_start
        row["fecha_teorica"] = theo_start.date() if theo_start is not None else None

        if rb.date not in index:
            row["estado"] = f"pendiente: falta el cierre del {rb.date.date()} (corre con --actualizar después del cierre)"
            rows.append(row)
            continue
        if theo_start is None:
            row["estado"] = "pendiente: la fecha de la señal no es un fin de mes conocido o faltan precios"
            rows.append(row)
            continue

        # ---- fin del período: próximo rebalanceo, o último cierre disponible si sigue abierto
        last_date = index[-1]
        theo_end_cal = cal["realized_at"].get(rb.signal_date)
        theo_end = pd.Timestamp(theo_end_cal) if theo_end_cal is not None and pd.notna(theo_end_cal) else None
        expected_next = rb.date.to_period("M") + 1
        if nxt is not None and nxt.date in index:
            real_end, completed = nxt.date, True
            row["sin_saltos"] = nxt.date.to_period("M") == expected_next
            if not row["sin_saltos"]:
                notes.append(f"Después del rebalanceo del {rb.date.date()} el siguiente fue el {nxt.date.date()}: "
                             f"falta el de {expected_next}.")
        else:
            completed = False
            row["sin_saltos"] = theo_end is None or theo_end >= last_date
            if not row["sin_saltos"] and nxt is None:
                notes.append(f"No hay rebalanceo registrado para {expected_next}. "
                             "Si ya lo hiciste, revisa la carpeta registros/ejecuciones.")
            # mientras está abierto se comparan ventanas iguales: hasta hoy o hasta el próximo rebalanceo teórico
            real_end = min(theo_end, last_date) if theo_end is not None else last_date
        theo_end = min(theo_end, last_date) if theo_end is not None else last_date
        if real_end <= rb.date or theo_end <= theo_start:
            row["estado"] = "pendiente: todavía no hay cierres posteriores a la ejecución"
            rows.append(row)
            continue

        # ---- cartera real
        universe = sorted(set(rb.shares_before.index) | set(filled.index) | set(rb.target.index))
        lacking = [t for t in universe if t not in adj.columns]
        if lacking:
            row["estado"] = f"sin precios para {', '.join(lacking)} (¿cambiaste el universo en config.toml?)"
            rows.append(row)
            continue
        before = rb.shares_before.reindex(universe, fill_value=0.0)
        after = before + filled.reindex(universe, fill_value=0.0)
        px = raw.loc[rb.date, universe]
        r_real = adj.loc[real_end, universe] / adj.loc[rb.date, universe] - 1.0
        r_theo = adj.loc[theo_end, universe] / adj.loc[theo_start, universe] - 1.0

        cash_before = rb.equity_before - float(rb.value_before.sum())
        v0 = float((before * px).sum()) + cash_before
        if len(f):
            f = f.assign(monto=f["acciones"] * f["precio"],
                         costo_precio=f["signo"] * f["acciones"] * (f["precio"] - f["ticker"].map(px)))
        traded = float(f["monto"].sum()) if len(f) else 0.0
        price_cost = float(f["costo_precio"].sum()) if len(f) else 0.0
        commission = float(f["comision_usd"].fillna(0.0).sum()) if len(f) else 0.0
        cash_after = cash_before - (float((f["signo"] * f["monto"]).sum()) if len(f) else 0.0) - commission
        v1 = float((after * px * (1.0 + r_real)).sum()) + cash_after
        real = v1 / v0 - 1.0

        w_real = after * px / v0
        w = rb.target.reindex(universe, fill_value=0.0)
        comm_sup, slip_sup = _assumed_costs(f if len(f) else pd.DataFrame(columns=["ticker", "lado", "monto"]), cfg)
        theo = float((w * r_theo).sum()) - (comm_sup + slip_sup) / v0

        comp = {
            "caja_redondeo": float(((w_real - w) * r_real).sum()),
            "fecha": float((w * (r_real - r_theo)).sum()),
            "precio": -(price_cost - slip_sup) / v0,
            "comisiones": -(commission - comm_sup) / v0,
        }
        held = set(after[after.abs() > 1e-9].index)
        row.update({
            "hasta": real_end.date(),
            "completo": completed,
            "real": real,
            "teorico": theo,
            "diferencia": real - theo,
            **comp,
            "monto_operado": traded,
            "comision_real": commission,
            "comision_supuesta": comm_sup,
            "costo_precio": price_cost,
            "deslizamiento_supuesto": slip_sup,
            "precio_bps": price_cost / traded * 1e4 if traded else np.nan,
            "cartera_igual_senal": held == set(rb.target[rb.target > 0].index),
            "patrimonio_inicio": v0,
        })
        if completed:
            ib_now, ib_next = rb.equity_before, nxt.equity_before
            row["segun_ib"] = ib_next / ib_now - 1.0 if ib_now > 0 else np.nan
        row["estado"] = "completo" if completed else f"en curso (hasta el cierre del {real_end.date()})"
        rows.append(row)

    periods = pd.DataFrame(rows)
    return _evaluate(periods, cfg, notes)


# --------------------------------------------------------------------------- criterios

def _evaluate(periods: pd.DataFrame, cfg: Config, notes: list[str]) -> TrackingResult:
    if periods.empty:
        return TrackingResult(periods, [], {}, False, notes)

    done = periods[periods["completo"] == True]  # noqa: E712
    computed = periods[periods["real"].notna()] if "real" in periods else periods.iloc[0:0]
    totals: dict = {"meses_completos": len(done), "rebalanceos": len(periods)}

    if len(done):
        totals["real_acumulado"] = float((1 + done["real"]).prod() - 1)
        totals["teorico_acumulado"] = float((1 + done["teorico"]).prod() - 1)
        for k in ("caja_redondeo", "fecha", "precio", "comisiones"):
            totals[k] = float(done[k].sum())
        gaps = (done["real"] - done["segun_ib"]).abs()
        big = done.loc[gaps > IB_GAP_WARNING, "mes"].tolist()
        if big:
            notes.append(
                "En " + ", ".join(big) + " la reconstrucción difiere más de 1,5 pp del patrimonio que informa IB. "
                "Revisa que estén todos los llenados y que no hayas movido ni reiniciado la cuenta paper."
            )
    if len(computed):
        traded = float(computed["monto_operado"].sum())
        totals["comision_real"] = float(computed["comision_real"].sum())
        totals["comision_supuesta"] = float(computed["comision_supuesta"].sum())
        totals["precio_bps"] = float(computed["costo_precio"].sum()) / traded * 1e4 if traded else 0.0
        totals["deslizamiento_supuesto_bps"] = cfg.costs.slippage_bps

    fills_ok = bool(periods["llenado_completo"].all() and periods["comisiones_completas"].all())
    portfolio_ok = bool(computed["cartera_igual_senal"].all()) if len(computed) else False
    if "sin_saltos" in periods:
        no_gaps = periods["sin_saltos"].map(lambda v: True if v is None or v is np.nan or pd.isna(v) else bool(v))
    else:
        no_gaps = pd.Series(True, index=periods.index)
    no_gaps = no_gaps.astype(bool)
    date_issues = ~periods["fecha_correcta"].astype(bool) | ~periods["senal_correcta"].astype(bool) | ~no_gaps
    dates_ok = not bool(date_issues.any())
    comm_ok = len(computed) > 0 and totals["comision_real"] <= totals["comision_supuesta"] * COMMISSION_TOLERANCE
    price_ok = len(computed) > 0 and totals["precio_bps"] <= cfg.costs.slippage_bps + PRICE_TOLERANCE_BPS

    def bad_months(mask) -> str:
        return ", ".join(periods.loc[mask, "mes"].tolist())

    checks = [
        {"ok": len(done) >= MIN_COMPLETED_MONTHS,
         "texto": f"Al menos {MIN_COMPLETED_MONTHS} meses completos ({len(done)} hasta ahora)",
         "accion": "Sigue rebalanceando una vez al mes."},
        {"ok": fills_ok and portfolio_ok,
         "texto": "Todas las órdenes llenadas, con comisión informada, y cartera igual a la señal",
         "accion": ("Revisa las órdenes en TWS y corre 'python -m etfml llenados' el mismo día de la ejecución. "
                    + (f"Meses con problemas: {bad_months(~periods['llenado_completo'] | ~periods['comisiones_completas'])}."
                       if not fills_ok else ""))},
        {"ok": dates_ok,
         "texto": "Rebalanceo todos los meses, con la señal del mes y el día que asume el backtest (primer día hábil)",
         "accion": ("Rebalancea todos los meses, el primer día hábil y con --actualizar. "
                    + (f"Meses con problemas: {bad_months(date_issues)}." if not dates_ok else ""))},
        {"ok": comm_ok,
         "texto": (f"Comisiones reales hasta {COMMISSION_TOLERANCE:.1f} veces las supuestas"
                   + (f" (US$ {totals['comision_real']:,.2f} reales vs US$ {totals['comision_supuesta']:,.2f} supuestas)"
                      if len(computed) else "")),
         "accion": "Sube comision_bps o comision_minima_usd en config.toml y vuelve a correr el backtest."},
        {"ok": price_ok,
         "texto": (f"Precio de llenado vs cierre hasta {PRICE_TOLERANCE_BPS:.0f} bps sobre el deslizamiento supuesto"
                   + (f" ({totals['precio_bps']:.1f} bps reales vs {cfg.costs.slippage_bps:.1f} supuestos)"
                      if len(computed) else "")),
         "accion": ("Ejecuta más cerca del cierre, o sube slippage_bps en config.toml y vuelve a correr el backtest. "
                    "Con órdenes a mercado este número incluye lo que se movió el precio hasta el cierre.")},
    ]
    passed = all(c["ok"] for c in checks)
    return TrackingResult(periods, checks, totals, passed, notes)


# --------------------------------------------------------------------------- salida

def _pct(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.2f} %"


def _pp(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.2f}"


def _usd(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.2f}"


def format_report(result: TrackingResult) -> str:
    p = result.periods
    lines: list[str] = []
    if p.empty:
        return ("Todavía no hay ejecuciones registradas.\n"
                "Corre 'python -m etfml rebalancear --ejecutar' el primer día hábil del mes.")
    first = p.iloc[0]
    lines.append(f"Seguimiento de la cuenta paper · modelo {p.iloc[-1]['modelo']} · "
                 f"{len(p)} rebalanceo(s) desde {first['ejecutado']}")

    has = "real" in p.columns
    ret = pd.DataFrame({
        "Ejecutado": p["ejecutado"].astype(str),
        "Hasta": p["hasta"].map(lambda v: "" if pd.isna(v) else str(v)) if has else "",
        "Real": p["real"].map(_pct) if has else "",
        "Teórico": p["teorico"].map(_pct) if has else "",
        "Diferencia (pp)": p["diferencia"].map(_pp) if has else "",
        "Según IB": p["segun_ib"].map(_pct) if "segun_ib" in p else "",
        "Estado": p["estado"],
    })
    ret.index = p["mes"]
    lines += ["", "Retorno de cada mes (real reconstruido con tus llenados vs teórico del backtest):",
              ret.to_string()]

    if has:
        comp = pd.DataFrame({
            "Caja y redondeo": p["caja_redondeo"].map(_pp),
            "Fecha": p["fecha"].map(_pp),
            "Precio": p["precio"].map(_pp),
            "Comisiones": p["comisiones"].map(_pp),
        })
        comp.index = p["mes"]
        lines += ["", "De dónde sale la diferencia (puntos porcentuales; suman la diferencia):", comp.to_string()]

        costs = pd.DataFrame({
            "Operado US$": p["monto_operado"].map(_usd),
            "Comisión real": p["comision_real"].map(_usd),
            "Comisión supuesta": p["comision_supuesta"].map(_usd),
            "Precio vs cierre (bps)": p["precio_bps"].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}"),
        })
        costs.index = p["mes"]
        lines += ["", "Costos de cada rebalanceo:", costs.to_string()]

    t = result.totals
    if "real_acumulado" in t:
        lines += ["", f"Acumulado de los meses completos: real {_pct(t['real_acumulado'])} · "
                      f"teórico {_pct(t['teorico_acumulado'])}"]

    lines += ["", "Para pasar a la siguiente fase:"]
    for c in result.checks:
        lines.append(f"  [{'OK' if c['ok'] else 'NO'}] {c['texto']}")
    pending = [c for c in result.checks if not c["ok"]]
    if pending:
        lines += ["", "Qué hacer:"]
        lines += [f"  - {c['accion'].strip()}" for c in pending]
    else:
        lines += ["", "Veredicto: el paper reproduce los supuestos del backtest. Puedes pasar a la siguiente fase."]
    if result.notes:
        lines += ["", "Avisos:"] + [f"  - {n}" for n in result.notes]
    return "\n".join(lines)


def save_tracking_csv(cfg: Config, result: TrackingResult) -> str:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.logs_dir / "seguimiento.csv"
    result.periods.to_csv(path, index=False, encoding="utf-8")
    return str(path)
