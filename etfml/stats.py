"""Sharpe probabilístico y deflactado (Bailey y López de Prado, 2012 y 2014).

La idea: si pruebas muchas variantes, la mejor se verá bien aunque ninguna sirva. El Sharpe
deflactado responde "¿qué probabilidad hay de que el Sharpe verdadero supere un umbral?",
donde el umbral sube con la cantidad de pruebas que hiciste.

Aquí el umbral es: Sharpe del benchmark + cuánto le "ganaría" por pura suerte la mejor de N
variantes sin ninguna habilidad. Tratar el Sharpe del benchmark como conocido hace la prueba
más exigente, no menos.

Todo se calcula con Sharpe por período (mensual). Para mostrar, se anualiza por sqrt(12).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329
DSR_THRESHOLD = 0.95


def sharpe_per_period(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def expected_max_standard_normal(n_trials: int) -> float:
    """Valor esperado aproximado del máximo de N normales estándar independientes."""
    if n_trials < 2:
        return 0.0
    n = float(n_trials)
    return float((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e)))


def expected_max_sharpe(trial_sharpes: list[float] | np.ndarray) -> float:
    """Cuánto Sharpe (por período) se espera que tenga la mejor de N pruebas solo por suerte."""
    srs = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    if len(srs) < 2:
        return 0.0
    return float(np.sqrt(srs.var(ddof=1)) * expected_max_standard_normal(len(srs)))


def probabilistic_sharpe(returns: pd.Series, sr_star: float) -> float:
    """Probabilidad de que el Sharpe verdadero (por período) supere sr_star, corrigiendo por asimetría y colas."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    t = len(r)
    if t < 3:
        return float("nan")
    sr = sharpe_per_period(pd.Series(r))
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    denom = max(denom, 1e-12)
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(t - 1) / np.sqrt(denom)))


@dataclass
class DeflatedSharpe:
    probability: float        # Sharpe deflactado: probabilidad de superar el umbral
    sharpe: float             # Sharpe de la estrategia (por período)
    benchmark_sharpe: float   # Sharpe del benchmark en el mismo período (por período)
    luck_premium: float       # lo que la mejor de N pruebas le ganaría por suerte (por período)
    threshold: float          # benchmark + suerte (por período)
    n_trials: int

    @property
    def passed(self) -> bool:
        return bool(self.probability >= DSR_THRESHOLD)

    def annualized(self, value: float, periods_per_year: int = 12) -> float:
        return float(value * np.sqrt(periods_per_year))


def deflated_sharpe(returns: pd.Series, benchmark_returns: pd.Series,
                    trial_sharpes: list[float] | np.ndarray) -> DeflatedSharpe:
    sr = sharpe_per_period(returns)
    bench = sharpe_per_period(benchmark_returns)
    luck = expected_max_sharpe(trial_sharpes)
    threshold = bench + luck
    return DeflatedSharpe(
        probability=probabilistic_sharpe(returns, threshold),
        sharpe=sr,
        benchmark_sharpe=bench,
        luck_premium=luck,
        threshold=threshold,
        n_trials=int(np.isfinite(np.asarray(trial_sharpes, dtype=float)).sum()),
    )
