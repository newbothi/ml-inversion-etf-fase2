"""Carga y valida config.toml.

Convención del proyecto: identificadores de código en inglés; todo lo que ve el
usuario (config, comandos, mensajes, reporte) en español.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"

MODEL_NAMES: tuple[str, ...] = ("ridge", "logistica", "lgbm")
STRATEGY_MODELS: tuple[str, ...] = MODEL_NAMES + ("ensamble",)
WEIGHTINGS = ("igual", "inv_vol")
ORDER_TYPES = ("MKT", "MOC")


class ConfigError(ValueError):
    """Error de configuración con mensaje para el usuario."""


@dataclass(frozen=True)
class UniverseConfig:
    tickers: tuple[str, ...]
    benchmark: str
    start: str


@dataclass(frozen=True)
class StrategyConfig:
    top_k: int
    weighting: str
    min_assets: int


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_months: int
    test_months: int


@dataclass(frozen=True)
class CostConfig:
    initial_capital_usd: float
    commission_bps: float
    min_commission_usd: float
    slippage_bps: float
    rebalance_band_usd: float


@dataclass(frozen=True)
class ModelConfig:
    ridge_alpha: float
    logistic_c: float
    lgbm_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrendConfig:
    risk_asset: str
    safe_assets: tuple[str, ...]

    @property
    def tickers(self) -> tuple[str, ...]:
        return (self.risk_asset,) + self.safe_assets


@dataclass(frozen=True)
class SavingsConfig:
    monthly_clp: float
    usd_clp: float
    horizons_years: tuple[int, ...]
    portfolio: str
    fx_spread: float
    dividend_drag: float
    btg_order_fee: float
    btg_min_order: float
    ib_per_share: float
    ib_min_fee: float
    ib_max_pct: float
    ib_swift_fee: float
    ib_ref_price: float
    ib_transfer_months: tuple[int, ...]
    waiting_rate: float = 0.0
    btg_buy_months: tuple[int, ...] = (1, 2, 3)

    @property
    def monthly_usd(self) -> float:
        return self.monthly_clp / self.usd_clp


DEFAULT_SAVINGS = SavingsConfig(100000, 981, (3, 5, 10, 15), "global", 0.01, 0.0025, 1.0, 20, 0.005, 1.0, 0.01,
                                25, 130, (1, 3, 6, 12))
PORTFOLIOS = ("global", "eeuu", "conservadora")


@dataclass(frozen=True)
class IBConfig:
    host: str
    port: int
    client_id: int
    model: str
    cash_buffer: float
    order_type: str
    allow_live: bool


@dataclass(frozen=True)
class Config:
    universe: UniverseConfig
    use_macro: bool
    strategy: StrategyConfig
    walkforward: WalkForwardConfig
    costs: CostConfig
    risk_free_annual: float
    models: ModelConfig
    ib: IBConfig
    trend: TrendConfig = TrendConfig("SPY", ("IEF", "SHY"))
    savings: SavingsConfig = DEFAULT_SAVINGS
    data_dir: Path = PROJECT_ROOT / "datos"
    reports_dir: Path = PROJECT_ROOT / "reportes"
    logs_dir: Path = PROJECT_ROOT / "registros"
    experiments_dir: Path = PROJECT_ROOT / "experimentos"

    @property
    def extra_tickers(self) -> tuple[str, ...]:
        """Tickers que se descargan además del universo (los usa el experimento de tendencia)."""
        return tuple(t for t in dict.fromkeys(self.trend.tickers) if t not in self.universe.tickers)

    def with_universe(self, tickers: list[str], benchmark: str) -> "Config":
        """Copia de la config con otro universo (se usa para los datos sintéticos)."""
        return replace(self, universe=replace(self.universe, tickers=tuple(tickers), benchmark=benchmark))


def _get(section: dict, key: str, section_name: str):
    if key not in section:
        raise ConfigError(f"Falta '{key}' en la sección [{section_name}] de config.toml")
    return section[key]


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"No encontré el archivo de configuración: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    def section(name: str) -> dict:
        if name not in raw:
            raise ConfigError(f"Falta la sección [{name}] en config.toml")
        return raw[name]

    u = section("universo")
    tickers = tuple(str(t).strip().upper() for t in _get(u, "tickers", "universo"))
    benchmark = str(_get(u, "benchmark", "universo")).strip().upper()
    universe = UniverseConfig(tickers=tickers, benchmark=benchmark, start=str(_get(u, "inicio", "universo")))

    e = section("estrategia")
    strategy = StrategyConfig(
        top_k=int(_get(e, "top_k", "estrategia")),
        weighting=str(_get(e, "ponderacion", "estrategia")),
        min_assets=int(_get(e, "min_activos", "estrategia")),
    )

    w = section("walkforward")
    walkforward = WalkForwardConfig(
        min_train_months=int(_get(w, "min_entrenamiento_meses", "walkforward")),
        test_months=int(_get(w, "meses_prueba", "walkforward")),
    )

    c = section("costos")
    costs = CostConfig(
        initial_capital_usd=float(_get(c, "capital_inicial_usd", "costos")),
        commission_bps=float(_get(c, "comision_bps", "costos")),
        min_commission_usd=float(_get(c, "comision_minima_usd", "costos")),
        slippage_bps=float(_get(c, "slippage_bps", "costos")),
        rebalance_band_usd=float(_get(c, "banda_rebalanceo_usd", "costos")),
    )

    m = section("modelos")
    lgbm = dict(m.get("lgbm", {}))
    models = ModelConfig(
        ridge_alpha=float(_get(m, "ridge_alpha", "modelos")),
        logistic_c=float(_get(m, "logistica_C", "modelos")),
        lgbm_params=lgbm,
    )

    i = section("ib")
    ib = IBConfig(
        host=str(_get(i, "host", "ib")),
        port=int(_get(i, "puerto", "ib")),
        client_id=int(_get(i, "client_id", "ib")),
        model=str(_get(i, "modelo", "ib")),
        cash_buffer=float(_get(i, "colchon_caja", "ib")),
        order_type=str(_get(i, "tipo_orden", "ib")).upper(),
        allow_live=bool(_get(i, "permitir_cuenta_real", "ib")),
    )

    t = raw.get("experimentos", {}).get("tendencia", {})
    trend = TrendConfig(
        risk_asset=str(t.get("activo_riesgo", "SPY")).strip().upper(),
        safe_assets=tuple(str(x).strip().upper() for x in t.get("refugios", ["IEF", "SHY"])),
    )

    sv = raw.get("simulador", {})
    btg, ibs = sv.get("btg", {}), sv.get("ib", {})
    d = DEFAULT_SAVINGS
    savings = SavingsConfig(
        monthly_clp=float(sv.get("aporte_mensual_clp", d.monthly_clp)),
        usd_clp=float(sv.get("tipo_cambio_clp", d.usd_clp)),
        horizons_years=tuple(int(x) for x in sv.get("plazos_anios", d.horizons_years)),
        portfolio=str(sv.get("cartera", d.portfolio)),
        fx_spread=float(sv.get("spread_cambio", d.fx_spread)),
        dividend_drag=float(sv.get("retencion_dividendos", d.dividend_drag)),
        btg_order_fee=float(btg.get("comision_orden_usd", d.btg_order_fee)),
        btg_min_order=float(btg.get("orden_minima_usd", d.btg_min_order)),
        ib_per_share=float(ibs.get("comision_por_accion_usd", d.ib_per_share)),
        ib_min_fee=float(ibs.get("comision_minima_usd", d.ib_min_fee)),
        ib_max_pct=float(ibs.get("comision_maxima_pct", d.ib_max_pct)),
        ib_swift_fee=float(ibs.get("swift_usd", d.ib_swift_fee)),
        ib_ref_price=float(ibs.get("precio_referencia_usd", d.ib_ref_price)),
        ib_transfer_months=tuple(int(x) for x in ibs.get("meses_entre_transferencias", d.ib_transfer_months)),
        waiting_rate=float(sv.get("interes_espera_anual", d.waiting_rate)),
        btg_buy_months=tuple(int(x) for x in btg.get("meses_entre_compras", d.btg_buy_months)),
    )

    cfg = Config(
        universe=universe,
        use_macro=bool(raw.get("features", {}).get("macro", False)),
        strategy=strategy,
        walkforward=walkforward,
        costs=costs,
        risk_free_annual=float(raw.get("metricas", {}).get("tasa_libre_riesgo_anual", 0.0)),
        models=models,
        ib=ib,
        trend=trend,
        savings=savings,
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    u, s = cfg.universe, cfg.strategy
    if len(set(u.tickers)) != len(u.tickers):
        raise ConfigError("Hay tickers repetidos en [universo].tickers")
    if u.benchmark not in u.tickers:
        raise ConfigError(f"El benchmark {u.benchmark} debe estar también en la lista de tickers")
    if not 1 <= s.top_k < len(u.tickers):
        raise ConfigError("top_k debe ser al menos 1 y menor que la cantidad de tickers")
    if s.min_assets <= s.top_k:
        raise ConfigError("min_activos debe ser mayor que top_k (si no, el top 3 sería 'todos')")
    if s.weighting not in WEIGHTINGS:
        raise ConfigError(f"ponderacion debe ser uno de: {', '.join(WEIGHTINGS)}")
    if cfg.walkforward.min_train_months < 24 or cfg.walkforward.test_months < 1:
        raise ConfigError("Usa al menos 24 meses de entrenamiento y 1 mes de prueba")
    if cfg.ib.model not in STRATEGY_MODELS:
        raise ConfigError(f"[ib].modelo debe ser uno de: {', '.join(STRATEGY_MODELS)}")
    if cfg.ib.order_type not in ORDER_TYPES:
        raise ConfigError(f"[ib].tipo_orden debe ser uno de: {', '.join(ORDER_TYPES)}")
    if not 0 <= cfg.ib.cash_buffer < 0.5:
        raise ConfigError("[ib].colchon_caja debe estar entre 0 y 0.5")
    if cfg.costs.initial_capital_usd <= 0:
        raise ConfigError("capital_inicial_usd debe ser positivo")
    sv = cfg.savings
    if sv.portfolio not in PORTFOLIOS:
        raise ConfigError(f"[simulador].cartera debe ser uno de: {', '.join(PORTFOLIOS)}")
    if sv.monthly_clp <= 0 or sv.usd_clp <= 0:
        raise ConfigError("[simulador]: el aporte y el tipo de cambio deben ser positivos")
    if not sv.horizons_years or min(sv.horizons_years) < 1 or not sv.ib_transfer_months \
            or min(sv.ib_transfer_months) < 1 or not sv.btg_buy_months or min(sv.btg_buy_months) < 1 \
            or 1 not in sv.btg_buy_months:
        raise ConfigError("[simulador]: los plazos y los meses entre compras o transferencias deben ser al menos 1 "
                          "(y meses_entre_compras de BTG debe incluir 1)")
    if not cfg.trend.safe_assets or cfg.trend.risk_asset in cfg.trend.safe_assets:
        raise ConfigError("[experimentos.tendencia]: los refugios deben ser distintos del activo de riesgo")
