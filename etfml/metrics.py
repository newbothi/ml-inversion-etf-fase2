"""Métricas en dos capas: calidad de las predicciones y resultado de la estrategia."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .backtest import BASELINES, BacktestResult
from .config import STRATEGY_MODELS
from .models import IS_PROBABILISTIC

PERIODS_PER_YEAR = 12
N_SUBPERIODS = 3


# --------------------------------------------------------------------------- estrategia

def equity_curve(net: pd.Series) -> pd.Series:
    return (1.0 + net).cumprod()


def drawdown(net: pd.Series) -> pd.Series:
    eq = equity_curve(net)
    peak = eq.cummax().clip(lower=1.0)  # incluye el punto de partida (1,0)
    return eq / peak - 1.0


def strategy_metrics(sim: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    r = sim["net"]
    n = len(r)
    years = n / PERIODS_PER_YEAR
    eq = equity_curve(r)
    rf_m = (1.0 + rf_annual) ** (1.0 / PERIODS_PER_YEAR) - 1.0
    ex = r - rf_m
    sd = r.std(ddof=1)
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean()) * np.sqrt(PERIODS_PER_YEAR)
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    max_dd = drawdown(r).min()
    return {
        "cagr": cagr,
        "volatility": sd * np.sqrt(PERIODS_PER_YEAR),
        "sharpe": ex.mean() / sd * np.sqrt(PERIODS_PER_YEAR) if sd > 0 else np.nan,
        "sortino": ex.mean() * PERIODS_PER_YEAR / downside if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "turnover_annual": sim["turnover"].mean() * PERIODS_PER_YEAR,
        "cost_annual": sim["cost"].mean() * PERIODS_PER_YEAR,
        "pct_positive_months": (r > 0).mean(),
        "growth_of_1": eq.iloc[-1],
        "months": n,
    }


def sharpe(r: pd.Series, rf_annual: float = 0.0) -> float:
    rf_m = (1.0 + rf_annual) ** (1.0 / PERIODS_PER_YEAR) - 1.0
    sd = r.std(ddof=1)
    return float((r - rf_m).mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else float("nan")


def subperiod_sharpes(sims: dict[str, pd.DataFrame], rf_annual: float = 0.0,
                      n: int = N_SUBPERIODS) -> pd.DataFrame:
    dates = next(iter(sims.values())).index
    chunks = np.array_split(np.arange(len(dates)), n)
    rows = {}
    for chunk in chunks:
        d = dates[chunk]
        label = f"{d[0].to_period('M') + 1} a {d[-1].to_period('M') + 1}"
        rows[label] = {k: sharpe(v.loc[d, "net"], rf_annual) for k, v in sims.items()}
    return pd.DataFrame(rows)  # estrategias x subperíodos


def annual_returns(sims: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Retorno por año calendario (el mes se asigna al período en que se mantuvo la posición)."""
    out = {}
    for name, sim in sims.items():
        year = (sim.index.to_period("M") + 1).year
        out[name] = (1.0 + sim["net"]).groupby(year).prod() - 1.0
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- predicciones

def information_coefficients(scores: pd.Series, excess: pd.Series) -> pd.Series:
    """Correlación de Spearman, mes a mes, entre el puntaje y el retorno en exceso realizado."""
    df = pd.DataFrame({"s": scores, "e": excess}).dropna()
    g = df.groupby(level="date")
    a = g["s"].rank()
    b = g["e"].rank()
    am = a - a.groupby(level="date").transform("mean")
    bm = b - b.groupby(level="date").transform("mean")
    num = (am * bm).groupby(level="date").sum()
    den = np.sqrt((am ** 2).groupby(level="date").sum() * (bm ** 2).groupby(level="date").sum())
    return (num / den.replace(0.0, np.nan)).rename("ic")


def newey_west_tstat(x: pd.Series, lags: int | None = None) -> float:
    """t-stat de la media corregido por autocorrelación (Newey-West, kernel de Bartlett).

    Los IC de meses seguidos se parecen (features que se solapan, modelo fijo por 12 meses);
    el t-stat simple los trata como independientes y exagera la significancia.
    """
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    d = v - v.mean()
    s = d @ d / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (d[lag:] @ d[:-lag]) / n
    return float(v.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def model_metrics(pred: pd.DataFrame, name: str, top_k: int) -> dict:
    ic = information_coefficients(pred[name], pred["excess"]).dropna()
    ic_mean, ic_sd = ic.mean(), ic.std(ddof=1)
    beat = pred["beat"].astype(int)
    out = {
        "ic_mean": ic_mean,
        "ic_tstat": newey_west_tstat(ic),
        "icir_annual": ic_mean / ic_sd * np.sqrt(PERIODS_PER_YEAR) if ic_sd > 0 else np.nan,
        "ic_pct_positive": (ic > 0).mean(),
        "auc": roc_auc_score(beat, pred[name]) if beat.nunique() == 2 else np.nan,
        "log_loss": np.nan,
        "hit_rate_top_k": np.nan,
    }
    if IS_PROBABILISTIC.get(name, False):
        out["log_loss"] = log_loss(beat, np.clip(pred[name], 1e-6, 1 - 1e-6))
    picks = (
        pred[[name, "beat"]]
        .sort_values(name, ascending=False, kind="mergesort")
        .groupby(level="date")
        .head(top_k)
    )
    out["hit_rate_top_k"] = picks["beat"].mean()
    return out


def base_log_loss(pred: pd.DataFrame) -> float:
    """Log loss de predecir siempre la tasa base: un modelo útil debería quedar por debajo."""
    p = pred["beat"].mean()
    return float(log_loss(pred["beat"].astype(int), np.full(len(pred), p)))


# --------------------------------------------------------------------------- resumen

def summarize(result: BacktestResult, rf_annual: float, top_k: int) -> dict:
    sims = result.simulations
    pred = result.walkforward.predictions
    strat = pd.DataFrame({k: strategy_metrics(v, rf_annual) for k, v in sims.items()}).T
    models = pd.DataFrame({k: model_metrics(pred, k, top_k) for k in STRATEGY_MODELS}).T
    sub = subperiod_sharpes(sims, rf_annual)
    ics = pd.DataFrame({k: information_coefficients(pred[k], pred["excess"]) for k in STRATEGY_MODELS})

    verdicts = {}
    best_baseline_sharpe = strat.loc[list(BASELINES), "sharpe"].max()
    for k in STRATEGY_MODELS:
        beats_full = bool(strat.loc[k, "sharpe"] > best_baseline_sharpe)
        sub_wins = int((sub.loc[k] > sub.loc[list(BASELINES)].max()).sum())
        ic_t = models.loc[k, "ic_tstat"]
        checks = {
            "sharpe_total": beats_full,
            "subperiodos": sub_wins >= 2,
            "ic_significativo": bool(ic_t >= 2.0),
        }
        verdicts[k] = {"passed": all(checks.values()), "checks": checks, "sub_wins": sub_wins, "ic_t": ic_t}

    return {
        "strategy": strat,
        "models": models,
        "subperiods": sub,
        "annual": annual_returns(sims),
        "ics": ics,
        "verdicts": verdicts,
        "base_log_loss": base_log_loss(pred),
        "best_baseline_sharpe": best_baseline_sharpe,
    }
