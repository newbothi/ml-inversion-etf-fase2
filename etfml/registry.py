"""Registro de experimentos: cada estrategia distinta que evalúas con datos reales queda anotada.

Sirve para saber cuántas variantes probaste en total, que es lo que usa el Sharpe deflactado.
Una prueba = una configuración distinta. Volver a correr la misma configuración (por ejemplo,
con datos más nuevos) actualiza su fila en vez de sumar otra.

El archivo vive en experimentos/registro.csv y sí se sube a Git: es la evidencia de lo que probaste.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from .config import Config

COLUMNS = ["id", "experimento", "variante", "descripcion", "desde", "hasta", "meses", "retorno_anual",
           "volatilidad", "sharpe_anual", "sharpe_mensual", "max_caida", "costo_anual", "datos_hasta",
           "registrado", "parametros"]


def trial_id(experiment: str, variant: str, params: dict) -> str:
    payload = json.dumps({"experimento": experiment, "variante": variant, **params}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def base_params(cfg: Config) -> dict:
    """Parte de la configuración que cambia el resultado de cualquier estrategia."""
    return {"costos": asdict(cfg.costs), "rf": cfg.risk_free_annual}


def ml_params(cfg: Config) -> dict:
    return {
        **base_params(cfg),
        "universo": list(cfg.universe.tickers),
        "inicio": cfg.universe.start,
        "macro": cfg.use_macro,
        "estrategia": asdict(cfg.strategy),
        "walkforward": asdict(cfg.walkforward),
        "modelos": asdict(cfg.models),
    }


def registry_path(cfg: Config):
    return cfg.experiments_dir / "registro.csv"


def load_registry(cfg: Config) -> pd.DataFrame:
    path = registry_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"id": str, "experimento": str, "variante": str})


def make_row(experiment: str, variant: str, description: str, params: dict, metrics: dict,
             sim: pd.DataFrame, data_until: pd.Timestamp) -> dict:
    first, last = sim.index[0].to_period("M") + 1, sim.index[-1].to_period("M") + 1
    r = sim["net"]
    sd = r.std(ddof=1)
    return {
        "id": trial_id(experiment, variant, params),
        "experimento": experiment,
        "variante": variant,
        "descripcion": description,
        "desde": str(first),
        "hasta": str(last),
        "meses": int(len(sim)),
        "retorno_anual": float(metrics["cagr"]),
        "volatilidad": float(metrics["volatility"]),
        "sharpe_anual": float(metrics["sharpe"]),
        "sharpe_mensual": float(r.mean() / sd) if sd > 0 else float("nan"),
        "max_caida": float(metrics["max_drawdown"]),
        "costo_anual": float(metrics["cost_annual"]),
        "datos_hasta": str(pd.Timestamp(data_until).date()),
        "registrado": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "parametros": json.dumps(params, sort_keys=True, ensure_ascii=False, default=str),
    }


def record_trials(cfg: Config, rows: list[dict]) -> pd.DataFrame:
    """Agrega o actualiza pruebas. Devuelve el registro completo."""
    old = load_registry(cfg)
    new = pd.DataFrame(rows, columns=COLUMNS)
    kept = old[~old["id"].isin(set(new["id"]))]
    merged = pd.concat([kept, new], ignore_index=True) if not kept.empty else new
    cfg.experiments_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(registry_path(cfg), index=False, encoding="utf-8")
    return merged
