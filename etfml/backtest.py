"""Walk-forward, construcción de cartera y simulación con costos."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MODEL_NAMES, STRATEGY_MODELS, Config, CostConfig
from .features import Panel
from .models import ensemble_scores, feature_weights, fit_model, predict_scores

BASELINES: tuple[str, ...] = ("buy_hold", "equal_weight", "momentum")
STRATEGY_LABELS = {
    "ridge": "Ridge (lineal)",
    "logistica": "Logística",
    "lgbm": "LightGBM",
    "ensamble": "Ensamble",
    "buy_hold": "Comprar y mantener",
    "equal_weight": "Igual peso (todos)",
    "momentum": "Momentum 12-1",
}
CASH = "_CAJA"


@dataclass
class Fold:
    train_first: pd.Timestamp
    train_last: pd.Timestamp
    test_first: pd.Timestamp
    test_last: pd.Timestamp
    train_months: int
    train_rows: int


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame                      # (date, ticker): puntajes + targets
    folds: list[Fold]
    weights: dict[str, pd.DataFrame]               # modelo -> (fold x feature)


@dataclass
class BacktestResult:
    walkforward: WalkForwardResult
    simulations: dict[str, pd.DataFrame]           # estrategia -> fila por mes
    benchmark: str
    source: str
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- walk-forward

def walk_forward(panel: Panel, cfg: Config) -> WalkForwardResult:
    labeled = panel.labeled
    feature_cols = panel.feature_cols
    date_level = labeled.index.get_level_values("date")
    dates = date_level.unique().sort_values()
    wf = cfg.walkforward

    preds, folds = [], []
    weights: dict[str, list[pd.Series]] = {n: [] for n in MODEL_NAMES}
    j = 0
    while j < len(dates):
        test_first = dates[j]
        # Purga: solo filas cuyo retorno ya se conocía el día en que se predice.
        train = labeled[labeled["realized_at"] <= test_first]
        train_dates = train.index.get_level_values("date").unique()
        if len(train_dates) < wf.min_train_months:
            j += 1
            continue
        test_dates = dates[j: j + wf.test_months]
        test = labeled[date_level.isin(test_dates)]

        out = test[["fwd_ret", "excess", "beat", "raw_mom_12_1", "raw_vol_60d"]].copy()
        for name in MODEL_NAMES:
            model = fit_model(name, cfg.models, train, feature_cols)
            out[name] = predict_scores(name, model, test[feature_cols])
            weights[name].append(feature_weights(name, model, feature_cols).rename(test_first))
        preds.append(out)
        folds.append(Fold(
            train_first=train_dates.min(), train_last=train_dates.max(),
            test_first=test_first, test_last=test_dates.max(),
            train_months=len(train_dates), train_rows=len(train),
        ))
        j += wf.test_months

    if not preds:
        raise ValueError(
            f"No hay suficientes meses para entrenar con {wf.min_train_months} meses. "
            "Baja min_entrenamiento_meses o usa una fecha de inicio más antigua."
        )
    predictions = pd.concat(preds).sort_index()
    predictions["ensamble"] = ensemble_scores(predictions)
    return WalkForwardResult(
        predictions=predictions,
        folds=folds,
        weights={n: pd.DataFrame(w) for n, w in weights.items()},
    )


# --------------------------------------------------------------------------- cartera

def select_weights(scores: pd.Series, top_k: int, weighting: str,
                   vol: pd.Series | None = None) -> dict[pd.Timestamp, pd.Series]:
    """Para cada mes: los top_k puntajes, en partes iguales o inverso a la volatilidad."""
    out: dict[pd.Timestamp, pd.Series] = {}
    for date, s in scores.dropna().groupby(level="date"):
        s = s.droplevel("date")
        top = s.sort_values(ascending=False, kind="mergesort").head(top_k)
        if weighting == "inv_vol":
            if vol is None:
                raise ValueError("inv_vol necesita volatilidades")
            inv = 1.0 / vol.xs(date, level="date").reindex(top.index)
            w = inv / inv.sum()
        else:
            w = pd.Series(1.0 / len(top), index=top.index)
        out[date] = w
    return out


def constant_weights(dates: pd.Index, tickers_by_date: dict, mode: str, benchmark: str) -> dict:
    out = {}
    for d in dates:
        if mode == "buy_hold":
            out[d] = pd.Series({benchmark: 1.0})
        else:
            tickers = tickers_by_date[d]
            out[d] = pd.Series(1.0 / len(tickers), index=tickers)
    return out


def simulate(weights: dict[pd.Timestamp, pd.Series], fwd: pd.DataFrame, costs: CostConfig) -> pd.DataFrame:
    """Simula la cartera mes a mes con comisión mínima por orden, slippage y banda de rebalanceo.

    fwd: retorno de mantener cada ETF desde exec_date hasta realized_at (fechas x tickers).
    """
    equity = costs.initial_capital_usd
    held = pd.Series(dtype=float)  # pesos al final del mes anterior, ya movidos por los precios
    rows = []
    for date in sorted(weights):
        target = weights[date]
        tickers = target.index.union(held.index.drop(CASH, errors="ignore"))
        cur = held.reindex(tickers, fill_value=0.0)
        new = target.reindex(tickers, fill_value=0.0).astype(float)

        if costs.rebalance_band_usd > 0:
            # posiciones que siguen y cuyo ajuste es chico: se dejan como están
            small = ((new - cur).abs() * equity < costs.rebalance_band_usd) & (new > 0) & (cur > 0)
            new[small] = cur[small]
            over = new.sum() - 1.0
            if over > 1e-12:
                adjustable = (~small) & (new > 0)
                if adjustable.any():
                    total_adj = new[adjustable].sum()
                    new[adjustable] *= max(total_adj - over, 0.0) / total_adj
                else:
                    new /= new.sum()

        delta = new - cur
        trades = delta.abs()
        trades = trades[trades > 1e-12]
        trade_usd = trades * equity
        commission = float(np.maximum(trade_usd * costs.commission_bps / 1e4, costs.min_commission_usd).sum()) \
            if len(trade_usd) else 0.0
        slippage = float((trade_usd * costs.slippage_bps / 1e4).sum())
        cost = (commission + slippage) / equity

        invested = new[new > 0]
        r = fwd.loc[date, invested.index]
        if r.isna().any():
            raise ValueError(f"Falta el retorno de {list(r[r.isna()].index)} en {date.date()}")
        cash_w = max(0.0, 1.0 - invested.sum())
        gross = float((invested * r).sum())
        net = (1.0 - cost) * (1.0 + gross) - 1.0
        equity *= 1.0 + net

        grown = invested * (1.0 + r)
        total = grown.sum() + cash_w
        held = grown / total
        if cash_w > 0:
            held[CASH] = cash_w / total

        rows.append({
            "date": date, "gross": gross, "cost": cost, "net": net,
            "turnover": 0.5 * float(delta.abs().sum()), "n_trades": int(len(trades)),
            "cash": cash_w, "equity": equity, "holdings": " ".join(sorted(invested.index)),
        })
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- orquestación

def run_backtest(panel: Panel, cfg: Config, source: str) -> BacktestResult:
    wf = walk_forward(panel, cfg)
    pred = wf.predictions
    fwd = pred["fwd_ret"].unstack("ticker")
    vol = pred["raw_vol_60d"]
    s = cfg.strategy

    sims: dict[str, pd.DataFrame] = {}
    for name in STRATEGY_MODELS:
        sims[name] = simulate(select_weights(pred[name], s.top_k, s.weighting, vol), fwd, cfg.costs)

    dates = fwd.index
    tickers_by_date = {d: g.index.get_level_values("ticker").tolist() for d, g in pred.groupby(level="date")}
    benchmark = cfg.universe.benchmark
    missing_bm = [d for d in dates if benchmark not in tickers_by_date[d]]
    if missing_bm:
        raise ValueError(f"El benchmark {benchmark} no tiene datos en {len(missing_bm)} meses del backtest.")
    sims["buy_hold"] = simulate(constant_weights(dates, tickers_by_date, "buy_hold", benchmark), fwd, cfg.costs)
    sims["equal_weight"] = simulate(constant_weights(dates, tickers_by_date, "equal", benchmark), fwd, cfg.costs)
    sims["momentum"] = simulate(select_weights(pred["raw_mom_12_1"], s.top_k, s.weighting, vol), fwd, cfg.costs)
    return BacktestResult(walkforward=wf, simulations=sims, benchmark=benchmark, source=source)
