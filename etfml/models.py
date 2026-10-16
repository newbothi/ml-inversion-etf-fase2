"""Los tres modelos y el ensamble.

Todos devuelven un puntaje por ETF y mes: más alto = mejor candidato para el mes siguiente.
  ridge      regresión lineal con L2; predice el retorno en exceso sobre el promedio del grupo
  logistica  regresión logística con L2; predice la probabilidad de superar al promedio
  lgbm       LightGBM (árboles con boosting) para la misma probabilidad, muy restringido
  ensamble   promedio de los rankings de los tres
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_NAMES, ModelConfig

TARGET_COL = {"ridge": "excess", "logistica": "beat", "lgbm": "beat"}
IS_PROBABILISTIC = {"ridge": False, "logistica": True, "lgbm": True, "ensamble": False}

MODEL_LABELS = {
    "ridge": "Ridge (lineal)",
    "logistica": "Logística",
    "lgbm": "LightGBM",
    "ensamble": "Ensamble",
}


def make_model(name: str, cfg: ModelConfig):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=cfg.ridge_alpha))
    if name == "logistica":
        return make_pipeline(StandardScaler(), LogisticRegression(C=cfg.logistic_c, max_iter=2000))
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        params = {"subsample_freq": 1, **cfg.lgbm_params}
        return LGBMClassifier(**params, importance_type="gain", random_state=42, n_jobs=1, verbose=-1)
    raise ValueError(f"Modelo desconocido: {name}")


def fit_model(name: str, cfg: ModelConfig, train: pd.DataFrame, feature_cols: list[str]):
    y = train[TARGET_COL[name]]
    if TARGET_COL[name] == "beat":
        y = y.astype(int)
        if y.nunique() < 2:
            raise ValueError("El set de entrenamiento tiene una sola clase; faltan datos.")
    model = make_model(name, cfg)
    model.fit(train[feature_cols], y)
    return model


def predict_scores(name: str, model, X: pd.DataFrame) -> np.ndarray:
    if IS_PROBABILISTIC[name]:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def feature_weights(name: str, model, feature_cols: list[str]) -> pd.Series:
    """Coeficientes (modelos lineales, sobre features estandarizadas) o importancia por ganancia (LightGBM)."""
    if name in ("ridge", "logistica"):
        est = model[-1]
        coef = np.ravel(est.coef_)
        return pd.Series(coef, index=feature_cols)
    gain = pd.Series(model.feature_importances_, index=feature_cols, dtype=float)
    total = gain.sum()
    return gain / total if total > 0 else gain


def ensemble_scores(scores: pd.DataFrame) -> pd.Series:
    """Promedia el ranking (percentil dentro de cada mes) de cada modelo."""
    ranks = scores[list(MODEL_NAMES)].groupby(level="date").rank(pct=True)
    return ranks.mean(axis=1)
