"""Precios: descarga desde Yahoo Finance, caché en CSV y datos sintéticos de prueba."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

MACRO_TICKERS = {"vix": "^VIX", "rate_10y": "^TNX", "rate_3m": "^IRX"}


@dataclass
class MarketData:
    close: pd.DataFrame            # precio ajustado por dividendos y splits (fechas x tickers)
    volume: pd.DataFrame           # volumen diario (fechas x tickers)
    macro: pd.DataFrame | None     # vix, rate_10y, rate_3m (fechas x columnas) o None
    source: str                    # "yahoo" o "sintetico"
    close_raw: pd.DataFrame | None = None  # cierre sin ajustar por dividendos (para comparar con llenados reales)


# --------------------------------------------------------------------------- descarga

def _naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _yf_download(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, auto_adjust=False, progress=False, threads=True, multi_level_index=True
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            "Yahoo Finance no devolvió datos. Revisa tu conexión a internet o intenta de nuevo en unos minutos."
        )
    return raw


def _field(raw: pd.DataFrame, name: str, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw[name]
    else:  # un solo ticker sin MultiIndex
        df = raw[[name]].set_axis(tickers, axis=1)
    df = df.reindex(columns=tickers)
    df.columns.name = None
    df.index = _naive_dates(df.index)
    df.index.name = None
    return df


def clean_prices(close: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ordena, quita días vacíos y rellena huecos cortos (máx. 5 días) sin inventar historia."""
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    close = close.where(close > 0).dropna(how="all")
    close = close.ffill(limit=5)
    volume = volume.sort_index()
    volume = volume[~volume.index.duplicated(keep="last")].reindex(close.index)
    return close, volume


def align_raw(close_raw: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    raw = close_raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.where(raw > 0).reindex(index=close.index, columns=close.columns)
    return raw.ffill(limit=5)


def download_tickers(cfg: Config) -> list[str]:
    """Universo + tickers extra de los experimentos: un solo archivo de precios sirve para todo."""
    return list(cfg.universe.tickers) + list(cfg.extra_tickers)


def download_market_data(cfg: Config) -> MarketData:
    tickers = download_tickers(cfg)
    print(f"Descargando {len(tickers)} ETFs desde Yahoo Finance (desde {cfg.universe.start})...")
    raw = _yf_download(tickers, cfg.universe.start)
    close, volume = clean_prices(_field(raw, "Adj Close", tickers), _field(raw, "Volume", tickers))
    close_raw = align_raw(_field(raw, "Close", tickers), close)
    missing = [t for t in tickers if close[t].dropna().empty]
    missing_universe = [t for t in missing if t in cfg.universe.tickers]
    if missing_universe:
        raise RuntimeError(f"Sin datos para: {', '.join(missing_universe)}. Revisa esos tickers en config.toml.")
    if missing:
        print(f"Aviso: sin datos para {', '.join(missing)} (solo los usa el experimento de tendencia).")
        keep = [t for t in tickers if t not in missing]
        close, volume, close_raw = close[keep], volume[keep], close_raw[keep]

    macro = None
    if cfg.use_macro:
        print("Descargando series macro (VIX y tasas)...")
        names = list(MACRO_TICKERS.values())
        raw_m = _yf_download(names, cfg.universe.start)
        macro = _field(raw_m, "Close", names).rename(columns={v: k for k, v in MACRO_TICKERS.items()})
        macro = macro.sort_index().ffill(limit=5)
    return MarketData(close=close, volume=volume, macro=macro, source="yahoo", close_raw=close_raw)


# --------------------------------------------------------------------------- caché

def _paths(cfg: Config) -> dict[str, Path]:
    d = cfg.data_dir
    return {"close": d / "precios.csv", "raw": d / "precios_sin_ajustar.csv", "volume": d / "volumen.csv",
            "macro": d / "macro.csv"}


def save_cache(cfg: Config, market: MarketData) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    p = _paths(cfg)
    market.close.to_csv(p["close"], index_label="fecha", encoding="utf-8")
    market.volume.to_csv(p["volume"], index_label="fecha", encoding="utf-8")
    if market.close_raw is not None:
        market.close_raw.to_csv(p["raw"], index_label="fecha", encoding="utf-8")
    if market.macro is not None:
        market.macro.to_csv(p["macro"], index_label="fecha", encoding="utf-8")


def restrict(market: MarketData, tickers: list[str]) -> MarketData:
    raw = market.close_raw[tickers] if market.close_raw is not None else None
    return MarketData(market.close[tickers], market.volume.reindex(columns=tickers), market.macro, market.source,
                      close_raw=raw)


def load_cache(cfg: Config, tickers: list[str] | None = None) -> MarketData | None:
    """Devuelve los datos guardados, o None si no existen o no calzan con la config actual."""
    p = _paths(cfg)
    if not all(p[k].exists() for k in ("close", "raw", "volume")):
        return None
    close = pd.read_csv(p["close"], index_col="fecha", parse_dates=["fecha"])
    volume = pd.read_csv(p["volume"], index_col="fecha", parse_dates=["fecha"])
    tickers = list(tickers or cfg.universe.tickers)
    if not set(tickers).issubset(close.columns):
        return None
    if close.index.min() > pd.Timestamp(cfg.universe.start) + pd.Timedelta(days=10):
        return None
    macro = None
    if cfg.use_macro:
        if not p["macro"].exists():
            return None
        macro = pd.read_csv(p["macro"], index_col="fecha", parse_dates=["fecha"])
    close, volume = clean_prices(close[tickers], volume.reindex(columns=tickers))
    raw = pd.read_csv(p["raw"], index_col="fecha", parse_dates=["fecha"])
    if not set(tickers).issubset(raw.columns):
        return None
    close_raw = align_raw(raw[tickers], close)
    return MarketData(close=close, volume=volume, macro=macro, source="yahoo", close_raw=close_raw)


def load_market_data(cfg: Config, refresh: bool = False, tickers: list[str] | None = None) -> MarketData:
    """Precios de los tickers pedidos (por defecto, el universo). Descarga solo si hace falta."""
    wanted = list(tickers or cfg.universe.tickers)
    if not refresh:
        cached = load_cache(cfg, wanted)
        if cached is not None:
            last = cached.close.index.max().date()
            print(f"Usando datos guardados en {cfg.data_dir.name}/ (último día: {last}). "
                  "Usa --actualizar para descargar de nuevo.")
            return cached
    market = download_market_data(cfg)
    save_cache(cfg, market)
    print(f"Datos guardados en {cfg.data_dir.name}/ (último día: {market.close.index.max().date()}).")
    lacking = [t for t in wanted if t not in market.close.columns]
    if lacking:
        raise RuntimeError(f"Yahoo Finance no tiene datos para: {', '.join(lacking)}.")
    return restrict(market, wanted)


# --------------------------------------------------------------------------- sintéticos

def synthetic_market_data(
    n_assets: int = 12,
    start: str = "2004-01-02",
    end: str = "2025-12-31",
    signal_strength: float = 1.0,
    seed: int = 7,
) -> MarketData:
    """Precios inventados para probar el código sin internet.

    Cada activo tiene una tendencia mensual oculta que persiste en el tiempo (AR(1)),
    así que el momentum pasado sí predice el mes siguiente. Con signal_strength=0
    no hay nada que predecir: es ruido puro (sirve para detectar fugas de información).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    codes, months = pd.factorize(dates.to_period("M"))
    n_days, n_months = len(dates), len(months)

    phi, drift_sd = 0.9, 0.015 * signal_strength
    drift = np.zeros((n_months, n_assets))
    drift[0] = rng.normal(0.0, drift_sd, n_assets)
    for m in range(1, n_months):
        drift[m] = phi * drift[m - 1] + rng.normal(0.0, drift_sd * np.sqrt(1 - phi**2), n_assets)

    days_per_month = np.bincount(codes)[codes][:, None]
    market = rng.normal(0.0003, 0.010, n_days)[:, None]
    betas = rng.uniform(0.2, 1.2, n_assets)
    idio = rng.uniform(0.08, 0.25, n_assets) / np.sqrt(252)
    daily = drift[codes] / days_per_month + market * betas + rng.standard_normal((n_days, n_assets)) * idio
    prices = 100 * np.exp(np.cumsum(np.log1p(daily), axis=0))

    tickers = [f"SYN{i + 1:02d}" for i in range(n_assets)]
    close = pd.DataFrame(prices, index=dates, columns=tickers)
    # dos activos con historia más corta, como un ETF que se lanzó después
    close.loc[:"2008-06-30", tickers[-1]] = np.nan
    close.loc[:"2011-12-31", tickers[-2]] = np.nan
    volume = pd.DataFrame(rng.lognormal(15, 0.3, (n_days, n_assets)), index=dates, columns=tickers)
    volume = volume.where(close.notna())
    return MarketData(close=close, volume=volume, macro=None, source="sintetico", close_raw=close.copy())
