"""Datos externos para el simulador, con caché en datos/.

  Chile       dólar observado y UF diarios desde mindicador.cl (fuente: Banco Central de Chile)
  EE.UU.      retorno mensual del mercado de acciones desde 1926 (Kenneth French, Dartmouth)
  Japón       retorno mensual del mercado de acciones en dólares desde 1990 (Kenneth French)
  Inflación   IPC de EE.UU. mensual desde 1913 (FRED, serie CPIAUCNS)

Si una fuente no responde, el simulador sigue con lo que tenga y avisa qué quedó fuera.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import date

import pandas as pd

from .config import Config

MINDICADOR = "https://mindicador.cl/api/{indicator}/{year}"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRED_CPI = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"
HEADERS = {"User-Agent": "Mozilla/5.0 (ml-inversion-etf; uso personal)"}
CHILE_START_YEAR = 2004


class ExternalDataError(RuntimeError):
    pass


# --------------------------------------------------------------------------- HTTP

def _get(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    import requests

    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.content
            last = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:  # sin internet, timeout, etc.
            last = str(exc)
        time.sleep(1.5 * (attempt + 1))
    raise ExternalDataError(f"No pude descargar {url} ({last})")


# --------------------------------------------------------------------------- parsers (funciones puras, testeables)

def parse_mindicador(payload: dict) -> pd.Series:
    """{"serie": [{"fecha": "2025-12-31T03:00:00.000Z", "valor": 39727.96}, ...]} -> serie diaria ordenada."""
    rows = payload.get("serie") or []
    if not rows:
        return pd.Series(dtype=float)
    # la hora UTC corresponde a la medianoche de Chile: la fecha local son los primeros 10 caracteres
    idx = pd.to_datetime([r["fecha"][:10] for r in rows])
    s = pd.Series([float(r["valor"]) for r in rows], index=idx)
    return s[~s.index.duplicated()].sort_index()


def parse_french_monthly(text: str) -> pd.DataFrame:
    """Bloque mensual de un archivo de factores de French. Devuelve Mkt-RF, RF, ... en decimales (no %)."""
    lines = text.splitlines()
    header_at = next((i for i, ln in enumerate(lines) if "Mkt-RF" in ln and ln.lstrip().startswith(",")), None)
    if header_at is None:
        raise ExternalDataError("El archivo de French no tiene el formato esperado (falta la fila con Mkt-RF)")
    columns = [c.strip() for c in lines[header_at].split(",")[1:]]
    rows, index = [], []
    for ln in lines[header_at + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts or not re.fullmatch(r"\d{6}", parts[0]):
            break  # termina el bloque mensual (después viene el anual)
        index.append(pd.Period(f"{parts[0][:4]}-{parts[0][4:]}", freq="M"))
        rows.append([float(x) for x in parts[1:1 + len(columns)]])
    if not rows:
        raise ExternalDataError("El archivo de French no trae datos mensuales")
    df = pd.DataFrame(rows, index=pd.PeriodIndex(index, freq="M"), columns=columns)
    return df.where(df > -99.0) / 100.0  # -99,99 = sin dato


def parse_fred_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col, value_col = df.columns[0], df.columns[1]
    values = pd.to_numeric(df[value_col], errors="coerce")
    idx = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M")
    return pd.Series(values.to_numpy(), index=idx).dropna()


# Cuánto puede alejarse un dato de la mediana de sus 15 vecinos sin ser un error de la fuente.
# La UF sube como mucho ~0,1 % al día; el dólar ha llegado a moverse ~5 % en un día.
PLAUSIBLE_JUMP = {"dolar": 0.25, "uf": 0.02}


def clean_chile_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Descarta datos imposibles. Ej.: mindicador trae la UF del 29 y 30-12-2014 en ~$608 (es ~$24.600)."""
    out = df.copy()
    dropped = []
    for col, tol in PLAUSIBLE_JUMP.items():
        if col not in out.columns:
            continue
        s = out[col].dropna()
        if len(s) < 5:
            continue
        ref = s.rolling(15, center=True, min_periods=3).median()
        bad = s.index[(s / ref - 1).abs() > tol]
        if len(bad):
            out.loc[bad, col] = float("nan")
            dropped += [f"{col} {d:%Y-%m-%d}" for d in bad]
    return out, dropped


def _unzip_csv(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(name).decode("latin-1")


# --------------------------------------------------------------------------- descargas

def download_chile_indicators(today: date | None = None) -> pd.DataFrame:
    import json

    end_year = (today or date.today()).year
    out = {}
    for indicator, col in (("dolar", "dolar"), ("uf", "uf")):
        parts = []
        for year in range(CHILE_START_YEAR, end_year + 1):
            parts.append(parse_mindicador(json.loads(_get(MINDICADOR.format(indicator=indicator, year=year)))))
            time.sleep(0.2)
        out[col] = pd.concat(parts).sort_index()
    df = pd.DataFrame(out)
    df.index.name = "fecha"
    return df


def download_long_history() -> pd.DataFrame:
    us = parse_french_monthly(_unzip_csv(_get(FRENCH.format(name="F-F_Research_Data_Factors"))))
    jp = parse_french_monthly(_unzip_csv(_get(FRENCH.format(name="Japan_3_Factors"))))
    cpi = parse_fred_csv(_get(FRED_CPI).decode("utf-8"))
    df = pd.DataFrame({
        "eeuu_mercado": us["Mkt-RF"] + us["RF"],
        "eeuu_rf": us["RF"],
        "japon_mercado": jp["Mkt-RF"] + jp["RF"],
        "japon_rf": jp["RF"],
    })
    df = df.join(cpi.rename("ipc_eeuu"), how="outer")
    df.index.name = "mes"
    return df


# --------------------------------------------------------------------------- caché

def _chile_path(cfg: Config):
    return cfg.data_dir / "indicadores_chile.csv"


def _long_path(cfg: Config):
    return cfg.data_dir / "historia_larga.csv"


def load_chile_indicators(cfg: Config, refresh: bool = False) -> pd.DataFrame | None:
    path = _chile_path(cfg)

    def read_cache() -> pd.DataFrame:
        return clean_chile_indicators(pd.read_csv(path, index_col="fecha", parse_dates=["fecha"]))[0]

    if path.exists() and not refresh:
        return read_cache()
    print("Descargando dólar observado y UF desde mindicador.cl...")
    try:
        df = download_chile_indicators()
    except ExternalDataError as exc:
        print(f"Aviso: {exc}. Los resultados quedan en dólares, sin ajustar por inflación.")
        return read_cache() if path.exists() else None
    df, dropped = clean_chile_indicators(df)
    if dropped:
        print(f"Aviso: descarté {len(dropped)} dato(s) de mindicador que no calzan con los días vecinos: "
              f"{', '.join(dropped[:6])}{' ...' if len(dropped) > 6 else ''}")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")
    return df


def load_long_history(cfg: Config, refresh: bool = False) -> pd.DataFrame | None:
    path = _long_path(cfg)
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col="mes")
        df.index = pd.PeriodIndex(df.index, freq="M")
        return df
    print("Descargando historia larga (EE.UU. desde 1926, Japón desde 1990, inflación de EE.UU.)...")
    try:
        df = download_long_history()
    except (ExternalDataError, StopIteration, zipfile.BadZipFile) as exc:
        print(f"Aviso: no pude bajar la historia larga ({exc}). Se omite la comparación de escenarios.")
        if path.exists():
            old = pd.read_csv(path, index_col="mes")
            old.index = pd.PeriodIndex(old.index, freq="M")
            return old
        return None
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")
    return df


def month_end(daily: pd.DataFrame) -> pd.DataFrame:
    """Último valor disponible de cada mes, indexado por mes."""
    return daily.groupby(daily.index.to_period("M")).last()


def synthetic_external(seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Indicadores de Chile e historia larga inventados, para probar sin internet."""
    import numpy as np

    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2004-01-01", "2026-01-15")
    dolar = 600 * np.exp(np.cumsum(rng.normal(0.00005, 0.006, len(days))))
    uf = 17000 * (1.03 ** (np.arange(len(days)) / 252))
    chile = pd.DataFrame({"dolar": dolar, "uf": uf}, index=days)
    chile.index.name = "fecha"

    months = pd.period_range("1926-07", "2025-12", freq="M")
    n = len(months)
    long_df = pd.DataFrame({
        "eeuu_mercado": rng.normal(0.009, 0.05, n),
        "eeuu_rf": np.full(n, 0.003),
        "ipc_eeuu": 10 * (1.0025 ** np.arange(n)),
    }, index=months)
    jp = months >= pd.Period("1990-07", freq="M")
    long_df["japon_mercado"] = np.where(jp, rng.normal(0.002, 0.06, n), np.nan)
    long_df["japon_rf"] = np.where(jp, 0.002, np.nan)
    long_df.index.name = "mes"
    return chile, long_df
