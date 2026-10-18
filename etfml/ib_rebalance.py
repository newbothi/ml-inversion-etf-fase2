"""Rebalanceo en Interactive Brokers (pensado para la cuenta paper).

Seguridad:
  - Por defecto se conecta en modo solo lectura y solo muestra el plan (dry-run).
  - Para enviar órdenes hay que pasar --ejecutar y escribir SI en la consola.
  - Se niega a operar en cuentas que no sean paper (DU...) salvo que permitir_cuenta_real = true.
  - Solo toca los ETFs del universo; cualquier otra posición de la cuenta se deja intacta.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Config

PAPER_PREFIXES = ("DU", "DF")
SELL, BUY, HOLD = "VENDER", "COMPRAR", "MANTENER"


@dataclass
class AccountSnapshot:
    account_id: str
    is_paper: bool
    net_liquidation: float
    currency: str
    positions: pd.Series        # acciones por ticker (solo stocks/ETFs)
    market_values: pd.Series    # valor de mercado por ticker


def plan_orders(target_weights: pd.Series, prices: pd.Series, positions: pd.Series, net_liquidation: float,
                outside_value: float, cash_buffer: float, min_trade_usd: float,
                universe: list[str]) -> pd.DataFrame:
    """Calcula las órdenes para llegar a los pesos objetivo con acciones enteras.

    - El monto invertible excluye posiciones fuera del universo y deja un colchón en caja.
    - Los ajustes chicos a posiciones que se mantienen se omiten (banda de rebalanceo).
    - Las salidas completas siempre se ejecutan, aunque sean chicas.
    - Las ventas van primero para liberar caja.
    """
    investable = max(0.0, (net_liquidation - outside_value) * (1.0 - cash_buffer))
    held = [t for t in universe if positions.get(t, 0) != 0]
    tickers = sorted(set(target_weights.index) | set(held))
    rows = []
    for t in tickers:
        px = float(prices.get(t, float("nan")))
        if not math.isfinite(px) or px <= 0:
            raise ValueError(f"No tengo un precio válido para {t}")
        w = float(target_weights.get(t, 0.0))
        cur = int(round(float(positions.get(t, 0))))
        tgt = int(math.floor(w * investable / px)) if w > 0 else 0
        delta = tgt - cur
        note = ""
        if delta != 0 and tgt > 0 and cur > 0 and abs(delta) * px < min_trade_usd:
            note = "ajuste menor a la banda"
            delta = 0
        action = BUY if delta > 0 else SELL if delta < 0 else HOLD
        rows.append({
            "ticker": t, "precio": px, "peso_objetivo": w, "acciones_actuales": cur,
            "acciones_objetivo": tgt if not note else cur, "orden": action, "cantidad": abs(delta),
            "monto_usd": abs(delta) * px, "nota": note,
        })
    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan
    order = {SELL: 0, BUY: 1, HOLD: 2}
    return plan.sort_values(["orden", "ticker"], key=lambda s: s.map(order) if s.name == "orden" else s) \
        .reset_index(drop=True)


# --------------------------------------------------------------------------- conexión

def connect(cfg: Config, readonly: bool):
    from ib_async import IB

    ib = IB()
    try:
        ib.connect(cfg.ib.host, cfg.ib.port, clientId=cfg.ib.client_id, timeout=20, readonly=readonly)
    except Exception as exc:  # noqa: BLE001 - mensaje claro para el usuario
        raise RuntimeError(
            f"No pude conectarme a TWS/IB Gateway en {cfg.ib.host}:{cfg.ib.port}. Revisa que esté abierto, "
            "con sesión iniciada en la cuenta paper, y que la API esté habilitada en ese puerto. "
            f"Detalle: {exc}"
        ) from exc
    return ib


def snapshot_account(ib) -> AccountSnapshot:
    accounts = ib.managedAccounts()
    if not accounts:
        raise RuntimeError("IB no informó ninguna cuenta.")
    if len(accounts) > 1:
        print(f"Aviso: hay varias cuentas ({', '.join(accounts)}); se usará {accounts[0]}.")
    account = accounts[0]
    summary = ib.accountSummary(account)
    nl = [v for v in summary if v.tag == "NetLiquidation"]
    if not nl:
        raise RuntimeError("No pude leer el patrimonio neto (NetLiquidation) de la cuenta.")
    currency = nl[0].currency
    if currency != "USD":
        raise RuntimeError(
            f"La moneda base de la cuenta es {currency}. Este script asume USD; cambia la moneda base "
            "de la cuenta paper o adapta la conversión."
        )
    shares, values = {}, {}
    for item in ib.portfolio(account):
        c = item.contract
        if c.secType != "STK":
            continue
        shares[c.symbol] = shares.get(c.symbol, 0) + item.position
        values[c.symbol] = values.get(c.symbol, 0.0) + item.marketValue
    return AccountSnapshot(
        account_id=account,
        is_paper=account.upper().startswith(PAPER_PREFIXES),
        net_liquidation=float(nl[0].value),
        currency=currency,
        positions=pd.Series(shares, dtype=float),
        market_values=pd.Series(values, dtype=float),
    )


def check_account_allowed(snap: AccountSnapshot, cfg: Config) -> None:
    if not snap.is_paper and not cfg.ib.allow_live:
        raise RuntimeError(
            f"La cuenta conectada ({snap.account_id}) no parece ser paper. Por seguridad no opero en cuentas "
            "reales salvo que pongas permitir_cuenta_real = true en config.toml."
        )


def execute_orders(ib, plan: pd.DataFrame, order_type: str, wait_seconds: int = 90) -> pd.DataFrame:
    from ib_async import MarketOrder, Order, Stock

    to_send = plan[plan["orden"].isin([SELL, BUY])]
    contracts = {t: Stock(t, "SMART", "USD") for t in to_send["ticker"]}
    if contracts:
        # qualifyContracts completa cada contrato en el mismo objeto (le asigna conId). Se revisan los originales
        # porque, según la versión de ib_async, la lista que devuelve omite los que fallan o trae None en su lugar.
        ib.qualifyContracts(*contracts.values())
        bad = [t for t, c in contracts.items() if not getattr(c, "conId", 0)]
        if bad:
            raise RuntimeError(f"IB no reconoció estos tickers: {', '.join(bad)}")

    def make_order(action: str, qty: int):
        side = "SELL" if action == SELL else "BUY"
        if order_type == "MOC":
            return Order(action=side, totalQuantity=qty, orderType="MOC", tif="DAY")
        return MarketOrder(side, qty, tif="DAY")

    def wait_done(placed) -> None:
        waited = 0
        while waited < wait_seconds and not all(t.isDone() for _, t in placed):
            ib.sleep(1)
            waited += 1

    trades = []
    for phase in (SELL, BUY):
        batch = to_send[to_send["orden"] == phase]
        placed = [(row.ticker, ib.placeOrder(contracts[row.ticker], make_order(phase, int(row.cantidad))))
                  for row in batch.itertuples()]
        trades += placed
        # con órdenes a mercado se espera a que se llenen (las ventas, antes de comprar)
        if placed and order_type == "MKT":
            wait_done(placed)
    ib.sleep(3)  # da tiempo a que lleguen los reportes de comisión

    out = []
    for ticker, t in trades:
        st = t.orderStatus
        out.append({"ticker": ticker, "orden": t.order.action, "cantidad": t.order.totalQuantity,
                    "estado": st.status, "llenadas": st.filled, "precio_promedio": st.avgFillPrice,
                    "order_id": t.order.orderId, "perm_id": t.order.permId})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- registros para el seguimiento

FILL_COLUMNS = ["exec_id", "fecha_hora_utc", "fecha_ny", "cuenta", "ticker", "lado", "acciones", "precio",
                "comision_usd", "perm_id", "order_id"]
SIDE_LABELS = {"BOT": "COMPRA", "SLD": "VENTA"}


def fills_to_frame(fills, account: str) -> pd.DataFrame:
    """Convierte los Fill de ib_async en filas. Comisión NaN si IB todavía no la informó."""
    rows = []
    for f in fills:
        ex, c = f.execution, f.contract
        if getattr(c, "secType", "STK") != "STK" or (account and ex.acctNumber != account):
            continue
        ts = pd.Timestamp(ex.time)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        report = f.commissionReport
        has_report = bool(getattr(report, "execId", "")) and abs(float(report.commission)) < 1e9
        rows.append({
            "exec_id": ex.execId,
            "fecha_hora_utc": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "fecha_ny": ts.tz_convert("America/New_York").strftime("%Y-%m-%d"),
            "cuenta": ex.acctNumber,
            "ticker": c.symbol,
            "lado": SIDE_LABELS.get(ex.side, ex.side),
            "acciones": float(ex.shares),
            "precio": float(ex.price),
            "comision_usd": float(report.commission) if has_report else float("nan"),
            "perm_id": int(ex.permId),
            "order_id": int(ex.orderId),
        })
    return pd.DataFrame(rows, columns=FILL_COLUMNS)


def fetch_fills(ib, account: str) -> pd.DataFrame:
    """Ejecuciones que IB informa para hoy (la API solo entrega las del día en curso)."""
    return fills_to_frame(ib.reqExecutions(), account)


def load_fills(cfg: Config) -> pd.DataFrame:
    path = cfg.logs_dir / "llenados.csv"
    if not path.exists():
        return pd.DataFrame(columns=FILL_COLUMNS)
    return pd.read_csv(path, dtype={"exec_id": str, "cuenta": str, "ticker": str, "fecha_ny": str})


def save_fills(cfg: Config, new: pd.DataFrame) -> int:
    """Agrega llenados a registros/llenados.csv sin duplicar. Devuelve cuántos son nuevos."""
    if new.empty:
        return 0
    old = load_fills(cfg)
    known = set(old["exec_id"])
    added = int((~new["exec_id"].isin(known)).sum())
    # si un llenado ya existía, se queda la versión nueva (puede traer la comisión que antes faltaba)
    kept = old[~old["exec_id"].isin(set(new["exec_id"]))]
    merged = pd.concat([kept, new], ignore_index=True) if not kept.empty else new.copy()
    merged = merged.sort_values(["fecha_hora_utc", "exec_id"]).reset_index(drop=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cfg.logs_dir / "llenados.csv", index=False, encoding="utf-8")
    return added


def build_execution_record(snap: AccountSnapshot, sig, plan: pd.DataFrame, order_type: str,
                           universe: list[str], outside_value: float, cash_buffer: float) -> dict:
    now_ny = pd.Timestamp.now(tz="America/New_York")
    before = {t: {"acciones": float(snap.positions.get(t, 0.0)), "valor": float(snap.market_values.get(t, 0.0))}
              for t in universe if snap.positions.get(t, 0) != 0}
    return {
        "version": 1,
        "fecha_hora_ny": now_ny.strftime("%Y-%m-%d %H:%M:%S"),
        "fecha_ny": now_ny.strftime("%Y-%m-%d"),
        "cuenta": snap.account_id,
        "es_paper": snap.is_paper,
        "modelo": sig.model,
        "fecha_senal": str(sig.signal_date.date()),
        "tipo_orden": order_type,
        "patrimonio_antes": float(snap.net_liquidation),
        "fuera_universo": float(outside_value),
        "colchon_caja": float(cash_buffer),
        "posiciones_antes": before,
        "pesos_objetivo": {t: float(w) for t, w in sig.weights.items()},
        "plan": [
            {"ticker": r.ticker, "orden": r.orden, "cantidad": int(r.cantidad), "precio_plan": float(r.precio),
             "acciones_actuales": int(r.acciones_actuales), "acciones_objetivo": int(r.acciones_objetivo)}
            for r in plan.itertuples()
        ],
        "ordenes": [],
    }


def save_execution_record(cfg: Config, record: dict) -> Path:
    folder = cfg.logs_dir / "ejecuciones"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = record["fecha_hora_ny"].replace("-", "").replace(":", "").replace(" ", "_")
    path = folder / f"{stamp}_{record['cuenta']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_execution_records(cfg: Config) -> list[dict]:
    folder = cfg.logs_dir / "ejecuciones"
    if not folder.exists():
        return []
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
    return sorted(records, key=lambda r: r["fecha_hora_ny"])


def log_orders(cfg: Config, plan: pd.DataFrame, status: pd.DataFrame | None, account: str, executed: bool) -> str:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = cfg.logs_dir / f"ordenes_{stamp}{'' if executed else '_simulacion'}.csv"
    merged = plan.copy()
    if status is not None and not status.empty:
        merged = merged.merge(status[["ticker", "estado", "llenadas", "precio_promedio"]], on="ticker", how="left")
    merged.insert(0, "cuenta", account)
    merged.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return str(path)
