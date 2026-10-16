"""Fase 3, experimento 1: filtro de tendencia (acciones o refugio).

En vez de elegir 3 ETFs entre 14, cada fin de mes se decide una sola cosa: ¿estar en el
activo de riesgo (SPY) o en un refugio? Las variantes están fijas antes de ver resultados:

  señal "media10"   SPY sobre su media de los últimos 10 cierres mensuales (regla clásica de Faber, 2007)
  señal "mom12"     retorno de 12 meses de SPY mayor que el de SHY (momentum absoluto, Antonacci, 2014)
  refugio           IEF (bonos 7-10 años) o SHY (bonos 1-3 años, casi caja)

Mismo calendario y costos que el backtest de la fase 1: señal con el cierre del último día
hábil del mes, compra al cierre del día hábil siguiente.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import simulate
from .config import Config
from .data import MarketData
from .features import rebalance_calendar
from .metrics import strategy_metrics, subperiod_sharpes
from .stats import DeflatedSharpe, deflated_sharpe

EXPERIMENT = "tendencia"
BENCHMARK = "comprar_mantener"
SIGNALS = {
    "media10": "media de 10 meses",
    "mom12": "momentum de 12 meses",
}
MIN_HISTORY_MONTHS = 12


@dataclass
class Variant:
    key: str
    signal: str
    safe_asset: str

    @property
    def label(self) -> str:
        return f"{SIGNALS[self.signal].capitalize()} → {self.safe_asset}"


@dataclass
class TrendResult:
    variants: list[Variant]
    simulations: dict[str, pd.DataFrame]       # variante o benchmark -> fila por mes
    risk_on: pd.DataFrame                       # fecha de señal x señal: True = en acciones
    metrics: pd.DataFrame
    subperiods: pd.DataFrame
    risk_asset: str
    data_until: pd.Timestamp
    deflated: dict[str, DeflatedSharpe] | None = None
    verdicts: dict[str, dict] | None = None


def variants_for(cfg: Config) -> list[Variant]:
    return [Variant(f"{sig}_{safe.lower()}", sig, safe) for sig in SIGNALS for safe in cfg.trend.safe_assets]


def monthly_closes(close: pd.DataFrame, today: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cierres de fin de mes, retornos de mantener cada mes (ejecución al día hábil siguiente) y calendario."""
    if today is not None:
        close = close[close.index < pd.Timestamp(today).normalize()]
    cal = rebalance_calendar(close.index, today)
    sd = pd.DatetimeIndex(cal["signal_date"])
    month_end = close.reindex(sd)
    buy = close.reindex(pd.DatetimeIndex(cal["exec_date"])).to_numpy()
    sell = close.reindex(pd.DatetimeIndex(cal["realized_at"])).to_numpy()
    fwd = pd.DataFrame(sell / buy - 1, index=sd, columns=close.columns)
    return month_end, fwd, cal


def trend_signals(month_end: pd.DataFrame, risk: str, cash_proxy: str) -> pd.DataFrame:
    """True = estar en el activo de riesgo. NaN mientras no hay historia suficiente."""
    px = month_end[risk]
    ma10 = px.rolling(10, min_periods=10).mean()
    media10 = (px > ma10).where(ma10.notna())
    ret12 = px / px.shift(12) - 1
    cash12 = month_end[cash_proxy] / month_end[cash_proxy].shift(12) - 1
    mom12 = (ret12 > cash12).where(ret12.notna() & cash12.notna())
    return pd.DataFrame({"media10": media10, "mom12": mom12})


def run_trend(cfg: Config, market: MarketData, today: pd.Timestamp | None) -> TrendResult:
    """Simula las variantes y el benchmark. El veredicto se calcula después con evaluate()."""
    risk = cfg.trend.risk_asset
    safes = list(cfg.trend.safe_assets)
    cash_proxy = safes[-1]  # el refugio más corto es la referencia de "caja" para el momentum absoluto
    needed = [risk] + safes
    lacking = [t for t in needed if t not in market.close.columns]
    if lacking:
        raise ValueError(f"Faltan precios de {', '.join(lacking)}. Corre el comando con --actualizar.")

    month_end, fwd, _ = monthly_closes(market.close[needed], today)
    signals = trend_signals(month_end, risk, cash_proxy)

    # período común: todas las señales definidas y el retorno del mes ya conocido para todos los activos
    usable = signals.notna().all(axis=1) & fwd[needed].notna().all(axis=1)
    dates = signals.index[usable]
    if len(dates) < 36:
        raise ValueError("Muy pocos meses con datos para evaluar el filtro de tendencia (mínimo 36).")
    fwd = fwd.loc[dates, needed]
    on = signals.loc[dates].astype(bool)

    variants = variants_for(cfg)
    sims: dict[str, pd.DataFrame] = {}
    for v in variants:
        weights = {d: pd.Series({risk if on.at[d, v.signal] else v.safe_asset: 1.0}) for d in dates}
        sims[v.key] = simulate(weights, fwd, cfg.costs)
    sims[BENCHMARK] = simulate({d: pd.Series({risk: 1.0}) for d in dates}, fwd, cfg.costs)

    metrics = pd.DataFrame({k: strategy_metrics(s, cfg.risk_free_annual) for k, s in sims.items()}).T
    subs = subperiod_sharpes(sims, cfg.risk_free_annual)
    return TrendResult(variants=variants, simulations=sims, risk_on=on, metrics=metrics, subperiods=subs,
                       risk_asset=risk, data_until=market.close.index.max())


def evaluate(res: TrendResult, trial_sharpes: list[float]) -> TrendResult:
    """Sharpe deflactado y veredicto, con el Sharpe mensual de todas las pruebas registradas."""
    sims, metrics, subs, on = res.simulations, res.metrics, res.subperiods, res.risk_on
    deflated = {v.key: deflated_sharpe(sims[v.key]["net"], sims[BENCHMARK]["net"], trial_sharpes)
                for v in res.variants}
    verdicts = {}
    for v in res.variants:
        better = bool(metrics.loc[v.key, "sharpe"] > metrics.loc[BENCHMARK, "sharpe"])
        wins = int((subs.loc[v.key] > subs.loc[BENCHMARK]).sum())
        consistent = wins >= 2
        evidence = deflated[v.key].passed
        if better and consistent and evidence:
            level = "pasa"
        elif better and consistent:
            level = "prometedor"
        else:
            level = "no_pasa"
        verdicts[v.key] = {"level": level, "better": better, "consistent": consistent, "sub_wins": wins,
                           "evidence": evidence,
                           "switches": int((on[v.signal] != on[v.signal].shift()).iloc[1:].sum()),
                           "pct_risk_on": float(on[v.signal].mean())}

    res.deflated, res.verdicts = deflated, verdicts
    return res


def trend_params(cfg: Config, v: Variant) -> dict:
    from .registry import base_params

    return {**base_params(cfg), "senal": v.signal, "riesgo": cfg.trend.risk_asset, "refugio": v.safe_asset,
            "caja_referencia": cfg.trend.safe_assets[-1], "inicio": cfg.universe.start}


LEVEL_TEXT = {
    "pasa": "Pasa: mejora a comprar y mantener, de forma consistente y con evidencia estadística",
    "prometedor": "Prometedor: mejora y es consistente, pero no hay evidencia estadística suficiente",
    "no_pasa": "No pasa: no mejora a comprar y mantener de forma consistente",
}


def format_console(res: TrendResult, n_trials: int) -> str:
    m = res.metrics
    order = [v.key for v in res.variants] + [BENCHMARK]
    labels = {v.key: v.label for v in res.variants}
    labels[BENCHMARK] = f"Comprar y mantener {res.risk_asset}"
    first = res.simulations[BENCHMARK].index[0].to_period("M") + 1
    last = res.simulations[BENCHMARK].index[-1].to_period("M") + 1
    table = pd.DataFrame({
        "Retorno anual": m.loc[order, "cagr"].map(lambda v: f"{v * 100:6.1f} %"),
        "Volatilidad": m.loc[order, "volatility"].map(lambda v: f"{v * 100:6.1f} %"),
        "Sharpe": m.loc[order, "sharpe"].map(lambda v: f"{v:5.2f}"),
        "Máx. caída": m.loc[order, "max_drawdown"].map(lambda v: f"{v * 100:6.1f} %"),
        "Costo anual": m.loc[order, "cost_annual"].map(lambda v: f"{v * 100:5.2f} %"),
    })
    table.index = [labels[k] for k in order]
    lines = [f"Filtro de tendencia: {first} a {last} ({len(res.simulations[BENCHMARK])} meses), después de costos.",
             "", table.to_string(), "", f"Sharpe deflactado ({n_trials} pruebas en el registro):"]
    for v in res.variants:
        d = res.deflated[v.key]
        lines.append(f"  {v.label:<32} {d.probability * 100:5.1f} %  (umbral: Sharpe {d.annualized(d.threshold):.2f} "
                     f"= {res.risk_asset} {d.annualized(d.benchmark_sharpe):.2f} + suerte {d.annualized(d.luck_premium):.2f})")
    lines += ["", "Veredicto:"]
    for v in res.variants:
        vd = res.verdicts[v.key]
        lines.append(f"  {v.label:<32} {LEVEL_TEXT[vd['level']].split(':')[0]}")
    return "\n".join(lines)
