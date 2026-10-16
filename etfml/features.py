"""Calendario de rebalanceo, features mensuales y objetivo (target).

Regla de oro: la fila de la fecha t solo usa precios hasta el cierre de t.

Calendario (sin mirar el futuro):
  signal_date     último día hábil del mes: se calculan las features con el cierre de ese día
  exec_date       día hábil siguiente: se compra al cierre de ese día
  realized_at     exec_date del mes siguiente: se vende ahí y recién entonces se conoce el retorno
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData

FEATURE_COLS: list[str] = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1",
    "vol_20d", "vol_60d", "dist_ma200", "drawdown_1y", "volume_trend",
]
MACRO_COLS: list[str] = ["vix_level", "vix_change_1m", "term_spread"]

FEATURE_LABELS: dict[str, str] = {
    "ret_1m": "Retorno 1 mes",
    "ret_3m": "Retorno 3 meses",
    "ret_6m": "Retorno 6 meses",
    "ret_12m": "Retorno 12 meses",
    "mom_12_1": "Momentum 12-1 meses",
    "vol_20d": "Volatilidad 20 días",
    "vol_60d": "Volatilidad 60 días",
    "dist_ma200": "Distancia a media móvil 200 días",
    "drawdown_1y": "Caída desde máximo de 1 año",
    "volume_trend": "Tendencia del volumen (20 vs 120 días)",
    "vix_level": "VIX",
    "vix_change_1m": "Cambio del VIX en 1 mes",
    "term_spread": "Pendiente de tasas (10a - 3m)",
}


@dataclass
class Panel:
    data: pd.DataFrame        # índice (date, ticker)
    feature_cols: list[str]
    calendar: pd.DataFrame    # signal_date, exec_date, realized_at

    @property
    def labeled(self) -> pd.DataFrame:
        """Filas con retorno futuro conocido (las que sirven para entrenar y evaluar)."""
        return self.data[self.data["fwd_ret"].notna()]

    @property
    def last_date(self) -> pd.Timestamp:
        return self.data.index.get_level_values("date").max()


def rebalance_calendar(index: pd.DatetimeIndex, today: pd.Timestamp | None = None) -> pd.DataFrame:
    index = pd.DatetimeIndex(index).sort_values()
    month_ends = pd.Series(index, index=index).groupby(index.to_period("M")).max()
    signal_dates = pd.DatetimeIndex(month_ends.to_numpy())
    if today is not None:  # el mes en curso todavía no termina
        current = pd.Timestamp(today).to_period("M")
        signal_dates = signal_dates[signal_dates.to_period("M") < current]
    pos = index.get_indexer(signal_dates)
    exec_dates = [index[p + 1] if p + 1 < len(index) else pd.NaT for p in pos]
    cal = pd.DataFrame({"signal_date": signal_dates, "exec_date": pd.DatetimeIndex(exec_dates)})
    cal["realized_at"] = cal["exec_date"].shift(-1)
    return cal


def _to_long(df: pd.DataFrame) -> pd.Series:
    idx = pd.MultiIndex.from_product([df.index, df.columns], names=["date", "ticker"])
    return pd.Series(df.to_numpy(dtype=float).reshape(-1), index=idx)


def _per_date(values: pd.Series, dates: pd.Index) -> np.ndarray:
    """Repite un valor por fecha en cada fila del panel largo."""
    return values.reindex(dates).to_numpy()


def build_panel(market: MarketData, min_assets: int = 5, today: pd.Timestamp | None = None) -> Panel:
    close = market.close.copy()
    volume = market.volume.reindex_like(close)
    if today is not None:  # la barra de hoy puede ser parcial (mercado abierto)
        today = pd.Timestamp(today).normalize()
        close = close[close.index < today]
        volume = volume.loc[close.index]

    cal = rebalance_calendar(close.index, today)
    if len(cal) < 14:
        raise ValueError("Muy pocos meses de datos para calcular las features (se necesitan al menos 14).")
    sd = pd.DatetimeIndex(cal["signal_date"])

    log_ret = np.log(close).diff()
    vol20 = log_ret.rolling(20, min_periods=18).std() * np.sqrt(252)
    vol60 = log_ret.rolling(60, min_periods=55).std() * np.sqrt(252)
    ma200 = close.rolling(200, min_periods=180).mean()
    max252 = close.rolling(252, min_periods=230).max()
    vpos = volume.where(volume > 0)
    vol_trend = np.log(vpos.rolling(20, min_periods=15).mean() / vpos.rolling(120, min_periods=90).mean())

    m = close.loc[sd]  # cierre de cada fin de mes
    wide = {
        "ret_1m": m / m.shift(1) - 1,
        "ret_3m": m / m.shift(3) - 1,
        "ret_6m": m / m.shift(6) - 1,
        "ret_12m": m / m.shift(12) - 1,
        "mom_12_1": m.shift(1) / m.shift(12) - 1,
        "vol_20d": vol20.loc[sd],
        "vol_60d": vol60.loc[sd],
        "dist_ma200": (close / ma200 - 1).loc[sd],
        "drawdown_1y": (close / max252 - 1).loc[sd],
        "volume_trend": vol_trend.loc[sd],
    }
    data = pd.DataFrame({k: _to_long(v) for k, v in wide.items()})
    data = data.replace([np.inf, -np.inf], np.nan)
    data["volume_trend"] = data["volume_trend"].fillna(0.0)  # sin volumen = neutro, no descarta el ETF

    # retorno de mantener desde exec_date hasta realized_at
    buy_px = close.reindex(pd.DatetimeIndex(cal["exec_date"])).to_numpy()
    sell_px = close.reindex(pd.DatetimeIndex(cal["realized_at"])).to_numpy()
    fwd = pd.DataFrame(sell_px / buy_px - 1, index=sd, columns=close.columns)
    data["fwd_ret"] = _to_long(fwd)

    dates = data.index.get_level_values("date")
    cal_by_date = cal.set_index("signal_date")
    data["exec_date"] = _per_date(cal_by_date["exec_date"], dates)
    data["realized_at"] = _per_date(cal_by_date["realized_at"], dates)

    feature_cols = list(FEATURE_COLS)
    if market.macro is not None:
        macro = market.macro.reindex(close.index).ffill(limit=5).loc[sd]
        macro_feats = pd.DataFrame({
            "vix_level": macro["vix"],
            "vix_change_1m": macro["vix"] / macro["vix"].shift(1) - 1,
            "term_spread": macro["rate_10y"] - macro["rate_3m"],
        })
        for col in MACRO_COLS:
            data[col] = _per_date(macro_feats[col], dates)
        feature_cols += MACRO_COLS

    # filas utilizables: features completas; si el mes ya se realizó, también el retorno
    data = data.dropna(subset=feature_cols)
    unrealized = data["realized_at"].isna()
    data = data[unrealized | data["fwd_ret"].notna()]
    counts = data.groupby(level="date").size()
    data = data[data.index.get_level_values("date").isin(counts[counts >= min_assets].index)]
    if data.empty:
        raise ValueError("No quedó ningún mes con suficientes ETFs y features completas.")

    # columnas crudas para los baselines, antes de transformar a ranking
    data["raw_mom_12_1"] = data["mom_12_1"]
    data["raw_vol_60d"] = data["vol_60d"]

    # ranking dentro de cada mes: el modelo compara ETFs entre sí, no niveles absolutos
    grouped = data.groupby(level="date")
    data[FEATURE_COLS] = grouped[FEATURE_COLS].rank(pct=True) - 0.5

    mean_fwd = grouped["fwd_ret"].transform("mean")
    data["excess"] = data["fwd_ret"] - mean_fwd
    data["beat"] = np.where(data["fwd_ret"].notna(), (data["excess"] > 0).astype(float), np.nan)

    data = data.sort_index()
    return Panel(data=data, feature_cols=feature_cols, calendar=cal)
