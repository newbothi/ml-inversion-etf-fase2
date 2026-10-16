"""Señal del mes: entrena con toda la historia conocida y puntúa el último fin de mes."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .backtest import select_weights
from .config import MODEL_NAMES, Config
from .data import MarketData
from .features import Panel
from .models import ensemble_scores, fit_model, predict_scores

STALE_DAYS = 40


@dataclass
class LiveSignal:
    signal_date: pd.Timestamp
    model: str
    scores: pd.DataFrame       # ticker x (ridge, logistica, lgbm, ensamble)
    weights: pd.Series         # pesos objetivo (suman 1)
    prices: pd.Series          # último cierre usado para calcular acciones
    train_months: int
    warnings: list[str]


def compute_live_signal(panel: Panel, market: MarketData, cfg: Config, model: str,
                        today: pd.Timestamp) -> LiveSignal:
    last = panel.last_date
    live = panel.data.xs(last, level="date", drop_level=False)
    # Entrena solo con retornos ya conocidos a la fecha de la señal (igual que en el backtest).
    train = panel.labeled[panel.labeled["realized_at"] <= last]
    train_months = train.index.get_level_values("date").nunique()
    warnings = []
    if train_months < cfg.walkforward.min_train_months:
        warnings.append(f"Solo hay {train_months} meses para entrenar (mínimo configurado: "
                        f"{cfg.walkforward.min_train_months}).")
    if (pd.Timestamp(today).normalize() - last).days > STALE_DAYS:
        warnings.append(f"La última señal es del {last.date()}: faltan datos recientes. "
                        "Corre el comando con --actualizar.")

    scores = pd.DataFrame(index=live.index)
    for name in MODEL_NAMES:
        fitted = fit_model(name, cfg.models, train, panel.feature_cols)
        scores[name] = predict_scores(name, fitted, live[panel.feature_cols])
    scores["ensamble"] = ensemble_scores(scores)

    weights = select_weights(scores[model], cfg.strategy.top_k, cfg.strategy.weighting, live["raw_vol_60d"])[last]
    close = market.close[market.close.index < pd.Timestamp(today).normalize()]
    prices = close.ffill().iloc[-1]
    return LiveSignal(
        signal_date=last,
        model=model,
        scores=scores.droplevel("date").sort_values(model, ascending=False),
        weights=weights,
        prices=prices,
        train_months=train_months,
        warnings=warnings,
    )


def log_signal(cfg: Config, sig: LiveSignal) -> None:
    """Agrega la señal a registros/senales.csv para comparar después lo ejecutado contra el backtest."""
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.logs_dir / "senales.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["generada", "fecha_senal", "modelo", "ticker", "puntaje", "peso_objetivo", "precio"])
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        for ticker, row in sig.scores.iterrows():
            w.writerow([stamp, sig.signal_date.date(), sig.model, ticker, f"{row[sig.model]:.6f}",
                        f"{sig.weights.get(ticker, 0.0):.4f}", f"{sig.prices.get(ticker, float('nan')):.4f}"])
