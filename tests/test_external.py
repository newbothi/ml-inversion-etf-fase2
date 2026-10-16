"""Descarga de datos externos: formatos reales de cada fuente, caché y respaldo cuando una fuente falla."""
import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import etfml.external as ext
from etfml.config import load_config

MINDICADOR_2025 = {
    "version": "1.7.0", "autor": "mindicador.cl", "codigo": "uf", "nombre": "Unidad de fomento (UF)",
    "unidad_medida": "Pesos",
    "serie": [
        {"fecha": "2025-12-31T03:00:00.000Z", "valor": 39727.96},
        {"fecha": "2025-07-01T04:00:00.000Z", "valor": 39200.10},   # horario de invierno: 04:00Z
        {"fecha": "2025-01-01T03:00:00.000Z", "valor": 38419.17},
    ],
}

FRENCH_US = """This file was created by CMPT_ME_BEME_RETS using the 202509 CRSP database.
The 1-month TBill return is from Ibbotson and Associates, Inc.

,Mkt-RF,SMB,HML,RF
192607,    2.96,   -2.56,   -2.43,    0.22
192608,    2.64,   -1.17,    3.82,    0.25
192609,  -99.99,    0.10,    0.20,    0.23

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
  1927,   29.47,   -2.46,   -3.75,    3.12

Copyright 2025 Kenneth R. French
"""

FRENCH_JP = """This file was created using the 202509 Bloomberg database.
Missing data are indicated by -99.99.

,Mkt-RF,SMB,HML,RF
199007,  -12.94,    1.56,    0.73,    0.68
199008,   -8.10,    0.50,    0.20,    0.66
"""

FRED_NEW = "observation_date,CPIAUCNS\n1926-06-01,17.7\n1926-07-01,17.5\n1926-08-01,.\n1926-09-01,17.5\n"
FRED_OLD = "DATE,CPIAUCNS\n1990-07-01,130.4\n1990-08-01,131.6\n"


def _zip(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_mindicador_dates_are_chilean_days_and_sorted():
    s = ext.parse_mindicador(MINDICADOR_2025)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-07-01", "2025-12-31"]
    assert s.iloc[-1] == 39727.96


def test_french_monthly_block_only_in_decimals():
    df = ext.parse_french_monthly(FRENCH_US)
    assert list(df.index.astype(str)) == ["1926-07", "1926-08", "1926-09"]   # el bloque anual queda fuera
    assert df.loc[pd.Period("1926-07", "M"), "Mkt-RF"] == pytest.approx(0.0296)
    assert np.isnan(df.loc[pd.Period("1926-09", "M"), "Mkt-RF"])             # -99,99 = sin dato
    with pytest.raises(ext.ExternalDataError):
        ext.parse_french_monthly("archivo sin el formato esperado")


def test_fred_accepts_both_header_styles_and_missing_values():
    new = ext.parse_fred_csv(FRED_NEW)
    assert list(new.index.astype(str)) == ["1926-06", "1926-07", "1926-09"]  # "." se descarta
    old = ext.parse_fred_csv(FRED_OLD)
    assert old.iloc[-1] == 131.6


def test_impossible_values_from_the_source_are_dropped():
    days = pd.date_range("2014-12-15", "2015-01-15")
    uf = pd.Series(24500 + 4.0 * np.arange(len(days)), index=days)
    uf.loc["2014-12-29"], uf.loc["2014-12-30"] = 608.15, 607.38              # error real de mindicador
    dolar = pd.Series(600.0 * 1.004 ** np.arange(len(days)), index=days)     # sube fuerte, pero es real
    clean, dropped = ext.clean_chile_indicators(pd.DataFrame({"dolar": dolar, "uf": uf}))
    assert dropped == ["uf 2014-12-29", "uf 2014-12-30"]
    assert clean["uf"].isna().sum() == 2 and clean["dolar"].notna().all()
    assert ext.month_end(clean).loc[pd.Period("2014-12", "M"), "uf"] == pytest.approx(24500 + 4.0 * 16)


def test_download_long_history_combines_the_three_sources(monkeypatch):
    responses = {
        ext.FRENCH.format(name="F-F_Research_Data_Factors"): _zip("F-F_Research_Data_Factors.CSV", FRENCH_US),
        ext.FRENCH.format(name="Japan_3_Factors"): _zip("Japan_3_Factors.csv", FRENCH_JP),
        ext.FRED_CPI: FRED_NEW.encode(),
    }
    monkeypatch.setattr(ext, "_get", lambda url, **kw: responses[url])
    df = ext.download_long_history()
    jul = pd.Period("1926-07", "M")
    assert df.loc[jul, "eeuu_mercado"] == pytest.approx(0.0296 + 0.0022)       # mercado = Mkt-RF + RF
    assert df.loc[jul, "ipc_eeuu"] == 17.5
    assert df.loc[pd.Period("1990-07", "M"), "japon_mercado"] == pytest.approx(-0.1294 + 0.0068)


def test_download_chile_indicators_reads_every_year(monkeypatch):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        year = int(url.rsplit("/", 1)[1])
        payload = {"serie": [{"fecha": f"{year}-06-30T04:00:00.000Z", "valor": 800.0 + year % 100}]}
        return json.dumps(payload).encode()

    monkeypatch.setattr(ext, "_get", fake_get)
    monkeypatch.setattr(ext.time, "sleep", lambda s: None)
    df = ext.download_chile_indicators(today=pd.Timestamp("2006-03-01").date())
    assert len(calls) == 2 * 3                                                  # dólar y UF, 2004 a 2006
    assert list(df.columns) == ["dolar", "uf"] and len(df) == 3


def test_failed_download_falls_back_without_crashing(tmp_path, monkeypatch, capsys):
    cfg = replace(load_config(), data_dir=tmp_path)

    def boom(*a, **k):
        raise ext.ExternalDataError("sin internet")

    monkeypatch.setattr(ext, "download_chile_indicators", boom)
    monkeypatch.setattr(ext, "download_long_history", boom)
    assert ext.load_chile_indicators(cfg) is None
    assert ext.load_long_history(cfg) is None
    out = capsys.readouterr().out
    assert "quedan en dólares" in out and "Se omite la comparación" in out


def test_cache_is_reused(tmp_path, monkeypatch):
    cfg = replace(load_config(), data_dir=tmp_path)
    chile, long_df = ext.synthetic_external()
    monkeypatch.setattr(ext, "download_chile_indicators", lambda: chile)
    monkeypatch.setattr(ext, "download_long_history", lambda: long_df)
    ext.load_chile_indicators(cfg)
    ext.load_long_history(cfg)

    def boom(*a, **k):
        raise AssertionError("no debería descargar de nuevo")

    monkeypatch.setattr(ext, "download_chile_indicators", boom)
    monkeypatch.setattr(ext, "download_long_history", boom)
    again = ext.load_chile_indicators(cfg)
    assert again["uf"].iloc[-1] == pytest.approx(chile["uf"].iloc[-1])
    long_again = ext.load_long_history(cfg)
    assert isinstance(long_again.index, pd.PeriodIndex)
    assert long_again["eeuu_mercado"].iloc[0] == pytest.approx(long_df["eeuu_mercado"].iloc[0])
