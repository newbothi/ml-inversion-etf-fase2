"""Simulador de aportes periódicos con datos históricos.

Pregunta: si hubieras aportado lo mismo todos los meses durante N años, partiendo en cualquier mes de
la historia disponible, ¿con cuánto habrías terminado? ¿Cuánto se lleva cada corredora? ¿Cuánto hay que
aportar para llegar a una meta?

Tres tipos de escenario, todos en dinero de hoy (ajustado por inflación):
  principal       ETFs desde 2004, en pesos de hoy: el dólar observado convierte a pesos y la UF descuenta
                  la inflación chilena. Si no hay indicadores de Chile, queda en dólares sin ajustar.
  historia larga  acciones de EE.UU. desde 1926 y de Japón desde 1990, en dólares de hoy (IPC de EE.UU.).
                  Sirven para ver qué pasa en épocas peores que 2004–2026.

Cada "ventana" es un punto de partida distinto. Ojo: las ventanas se superponen, así que 150 ventanas de
10 años en 22 años de datos son, en el fondo, unos 2 períodos independientes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SavingsConfig

MIN_WINDOWS = 12


@dataclass(frozen=True)
class Portfolio:
    key: str
    label: str
    history: dict[str, float]       # ETFs con historia desde 2004 que aproximan la cartera
    orders: dict[str, float]        # lo que se compraría hoy (cada uno es una orden)
    long_weights: dict[str, float]  # para la historia larga: "mercado" (acciones) y "rf" (letras del Tesoro)


PORTFOLIOS: dict[str, Portfolio] = {
    "global": Portfolio("global", "Global (como VT)", {"SPY": 0.6, "EFA": 0.3, "EEM": 0.1}, {"VT": 1.0},
                        {"mercado": 1.0}),
    "eeuu": Portfolio("eeuu", "EE.UU. (como SPY o VOO)", {"SPY": 1.0}, {"VOO": 1.0}, {"mercado": 1.0}),
    "conservadora": Portfolio("conservadora", "70 % global y 30 % bonos",
                              {"SPY": 0.42, "EFA": 0.21, "EEM": 0.07, "IEF": 0.30}, {"VT": 0.7, "IEF": 0.3},
                              {"mercado": 0.7, "rf": 0.3}),
}


@dataclass(frozen=True)
class Scheme:
    key: str
    label: str
    broker: str          # "btg" o "ib"
    every: int           # meses entre compras (BTG) o transferencias (IB)


FREQUENCY = {1: "mensual", 2: "cada 2 meses", 3: "trimestral", 6: "semestral", 12: "anual"}


def frequency_label(k: int) -> str:
    return FREQUENCY.get(int(k), f"cada {k} meses")


def schemes_for(cfg: SavingsConfig) -> list[Scheme]:
    out = [Scheme(f"btg_{k}", f"BTG, compra {frequency_label(k)}", "btg", k) for k in sorted(cfg.btg_buy_months)]
    out += [Scheme(f"ib_{k}", f"IB, transferencia {frequency_label(k)}", "ib", k) for k in sorted(cfg.ib_transfer_months)]
    return out


@dataclass
class Scenario:
    key: str
    label: str
    unit: str                # "CLP" o "USD"
    real: bool               # True = dinero de hoy (ajustado por inflación)
    returns: pd.Series       # retornos mensuales de la cartera en esa unidad, índice mensual (Period)
    unit_to_usd: float       # para calcular comisiones en dólares: dólares por unidad

    @property
    def unit_label(self) -> str:
        if self.unit == "CLP":
            return "pesos de hoy" if self.real else "pesos"
        return "dólares de hoy" if self.real else "dólares"

    @property
    def years_of_data(self) -> float:
        return len(self.returns) / 12


# --------------------------------------------------------------------------- costos

def _ib_commission_vec(order: np.ndarray, cfg: SavingsConfig) -> np.ndarray:
    fee = np.minimum(np.maximum(cfg.ib_min_fee, cfg.ib_per_share * order / cfg.ib_ref_price), cfg.ib_max_pct * order)
    return np.where(order > 0, fee, 0.0)


def ib_commission(order_usd: float, cfg: SavingsConfig) -> float:
    """IBKR Pro fija: US$0,005 por acción, mínimo US$1, máximo 1 % del valor de la orden."""
    return float(_ib_commission_vec(np.array([float(order_usd)]), cfg)[0])


def _cost_usd(amount_usd: np.ndarray, scheme: Scheme, portfolio: Portfolio, cfg: SavingsConfig) -> dict:
    fx = amount_usd * cfg.fx_spread
    after_fx = amount_usd - fx
    if scheme.broker == "btg":
        commission = np.full_like(amount_usd, cfg.btg_order_fee * len(portfolio.orders))
        transfer = np.zeros_like(amount_usd)
    else:
        transfer = np.full_like(amount_usd, cfg.ib_swift_fee)
        net = np.maximum(after_fx - transfer, 0.0)
        commission = sum(_ib_commission_vec(net * w, cfg) for w in portfolio.orders.values())
    return {"fx": fx, "transfer": transfer, "commission": commission, "total": fx + transfer + commission}


def investment_cost(amount_usd: float, scheme: Scheme, portfolio: Portfolio, cfg: SavingsConfig) -> dict:
    """Costo de convertir y comprar `amount_usd` dólares (versión para un solo monto)."""
    out = _cost_usd(np.array([float(amount_usd)]), scheme, portfolio, cfg)
    return {k: float(v[0]) for k, v in out.items()}


def _can_invest(amount_usd: np.ndarray, scheme: Scheme, portfolio: Portfolio, cfg: SavingsConfig) -> np.ndarray:
    if scheme.broker == "btg":
        return amount_usd * (1 - cfg.fx_spread) * min(portfolio.orders.values()) >= cfg.btg_min_order
    return np.ones_like(amount_usd, dtype=bool)


# --------------------------------------------------------------------------- simulación (todas las ventanas a la vez)

def simulate_matrix(rets: np.ndarray, contribution: float, scheme: Scheme, portfolio: Portfolio,
                    cfg: SavingsConfig, unit_to_usd: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rets: (ventanas x meses). Aporta al inicio de cada mes y mide al final.

    Devuelve el valor final, los costos acumulados y la trayectoria del valor, todo en la unidad del escenario.
    """
    rets = np.atleast_2d(rets)
    n_windows, months = rets.shape
    drag = 1 - cfg.dividend_drag / 12
    wait = (1 + cfg.waiting_rate) ** (1 / 12)
    invested = np.zeros(n_windows)
    pending = np.zeros(n_windows)
    costs = np.zeros(n_windows)
    values = np.empty((n_windows, months))
    for j in range(months):
        pending += contribution
        if (j + 1) % scheme.every == 0:
            usd = pending * unit_to_usd
            ok = _can_invest(usd, scheme, portfolio, cfg)
            cost = _cost_usd(usd, scheme, portfolio, cfg)["total"] / unit_to_usd
            invested = np.where(ok, invested + pending - cost, invested)
            costs = np.where(ok, costs + cost, costs)
            pending = np.where(ok, 0.0, pending)
        invested = invested * (1 + rets[:, j]) * drag
        pending = pending * wait
        values[:, j] = invested + pending
    return values[:, -1].copy(), costs, values


def simulate_window(rets: np.ndarray, contribution: float, scheme: Scheme, portfolio: Portfolio,
                    cfg: SavingsConfig, unit_to_usd: float = 1.0):
    """Una sola ventana (útil para pruebas). Devuelve valor final, costos, trayectoria y lo aportado."""
    final, costs, values = simulate_matrix(np.asarray(rets, dtype=float)[None, :], contribution, scheme, portfolio,
                                           cfg, unit_to_usd)
    contributed = contribution * np.arange(1, len(rets) + 1)
    return float(final[0]), float(costs[0]), values[0], contributed


def money_weighted_return(contribution: float, months: int, final):
    """Tasa anual equivalente: la que convierte aportes iguales al inicio de cada mes en el valor final."""
    final_arr = np.atleast_1d(np.asarray(final, dtype=float))
    lo = np.full_like(final_arr, -0.99)
    hi = np.full_like(final_arr, 1.0)
    for _ in range(100):
        mid = (lo + hi) / 2
        safe = np.where(np.abs(mid) < 1e-12, 1e-12, mid)
        fv = np.where(np.abs(mid) < 1e-12, contribution * months,
                      contribution * ((1 + safe) ** months - 1) / safe * (1 + safe))
        above = fv > final_arr
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    out = (1 + (lo + hi) / 2) ** 12 - 1
    out = np.where(final_arr <= 0, -1.0, out)
    return float(out[0]) if np.ndim(final) == 0 else out


@dataclass
class WindowSet:
    starts: list
    years: int
    contribution: float
    final: np.ndarray
    costs: np.ndarray
    paths: np.ndarray
    contributed_path: np.ndarray
    irr: np.ndarray | None = None

    @property
    def contributed(self) -> float:
        return float(self.contributed_path[-1])

    @property
    def worst_moment(self) -> np.ndarray:
        return (self.paths / self.contributed_path).min(axis=1) - 1

    def __len__(self) -> int:
        return len(self.final)


def run_windows(scenario: Scenario, years: int, scheme: Scheme, portfolio: Portfolio, cfg: SavingsConfig,
                contribution: float, with_irr: bool = True) -> WindowSet:
    months = years * 12
    arr = scenario.returns.to_numpy(dtype=float)
    if len(arr) < months:
        raise ValueError(f"{scenario.label}: no hay {years} años de datos")
    mat = np.lib.stride_tricks.sliding_window_view(arr, months)
    final, costs, paths = simulate_matrix(mat, contribution, scheme, portfolio, cfg, scenario.unit_to_usd)
    ws = WindowSet(starts=list(scenario.returns.index[: len(final)]), years=years, contribution=contribution,
                   final=final, costs=costs, paths=paths, contributed_path=contribution * np.arange(1, months + 1))
    if with_irr:
        ws.irr = money_weighted_return(contribution, months, final)
    return ws


def summarize(ws: WindowSet) -> dict:
    f = ws.final
    return {
        "ventanas": len(ws),
        "aportado": ws.contributed,
        "peor": float(f.min()),
        "p10": float(np.percentile(f, 10)),
        "mediana": float(np.median(f)),
        "p90": float(np.percentile(f, 90)),
        "mejor": float(f.max()),
        "pct_perdida": float((f < ws.contributed).mean()),
        "rentabilidad_mediana": float(np.median(ws.irr)) if ws.irr is not None else float("nan"),
        "peor_momento": float(ws.worst_moment.min()),
        "costos_mediana": float(np.median(ws.costs)),
    }


# --------------------------------------------------------------------------- escenarios

def etf_returns(close: pd.DataFrame, portfolio: Portfolio, today: pd.Timestamp | None) -> pd.Series:
    """Retorno mensual nominal en dólares de la cartera de ETFs (rebalanceada cada mes), solo meses completos."""
    tickers = list(portfolio.history)
    lacking = [t for t in tickers if t not in close.columns]
    if lacking:
        raise ValueError(f"Faltan precios de {', '.join(lacking)}. Corre el comando con --actualizar.")
    px = close[tickers]
    if today is not None:
        px = px[px.index < pd.Timestamp(today).to_period("M").to_timestamp()]
    month_end = px.groupby(px.index.to_period("M")).last()
    rets = month_end.pct_change().dropna(how="any")
    return (rets[tickers] * pd.Series(portfolio.history)).sum(axis=1).rename(portfolio.key)


monthly_returns = etf_returns  # nombre anterior


def pesos_scenario(usd_rets: pd.Series, chile_month_end: pd.DataFrame, usd_clp_today: float,
                   label: str) -> Scenario | None:
    """Retornos en pesos de hoy: (1 + r en US$) × variación del dólar ÷ variación de la UF."""
    fx_chg = chile_month_end["dolar"] / chile_month_end["dolar"].shift(1)
    uf_chg = chile_month_end["uf"] / chile_month_end["uf"].shift(1)
    rets = ((1 + usd_rets) * fx_chg.reindex(usd_rets.index) / uf_chg.reindex(usd_rets.index) - 1).dropna()
    if len(rets) < 36:
        return None
    return Scenario("principal", label, "CLP", True, rets, 1 / usd_clp_today)


def usd_scenario(usd_rets: pd.Series, label: str) -> Scenario:
    return Scenario("principal", label, "USD", False, usd_rets, 1.0)


def to_real_usd(rets: pd.Series, cpi: pd.Series) -> pd.Series:
    infl = cpi / cpi.shift(1)
    return ((1 + rets) / infl.reindex(rets.index) - 1).dropna()


def long_scenarios(long_df: pd.DataFrame, portfolio: Portfolio, etf_usd_rets: pd.Series) -> list[Scenario]:
    """Escenarios en dólares de hoy: ETFs desde 2004, EE.UU. desde 1926 y Japón desde 1990."""
    cpi = long_df["ipc_eeuu"].dropna()
    w = portfolio.long_weights
    out = []
    etf_real = to_real_usd(etf_usd_rets, cpi)
    if len(etf_real) >= 36:
        out.append(Scenario("etf", f"ETFs {etf_real.index[0].year}–{etf_real.index[-1].year}", "USD", True,
                            etf_real, 1.0))
    for key, name in (("eeuu", "EE.UU."), ("japon", "Japón")):
        nominal = (w.get("mercado", 0) * long_df[f"{key}_mercado"] + w.get("rf", 0) * long_df[f"{key}_rf"]).dropna()
        real = to_real_usd(nominal, cpi)
        if len(real) >= 36:
            what = "acciones" if w.get("rf", 0) == 0 else "70 % acciones y 30 % letras del Tesoro"
            out.append(Scenario(key, f"{name} {real.index[0].year}–{real.index[-1].year}, {what}", "USD", True,
                                real, 1.0))
    return out


# --------------------------------------------------------------------------- comparaciones

def _contribution(scn: Scenario, clp: float, usd_clp: float) -> float:
    return clp if scn.unit == "CLP" else clp / usd_clp


def _from_units(scn: Scenario, value: float, usd_clp: float) -> float:
    """Convierte un monto del escenario a pesos (de hoy, si el escenario es real)."""
    return value if scn.unit == "CLP" else value * usd_clp


def amount_comparison(scn: Scenario, years: int, portfolio: Portfolio, cfg: SavingsConfig,
                      amounts_clp: list[float]) -> pd.DataFrame:
    """Para varios montos: la mejor frecuencia en BTG y la mejor en IB (por valor final típico)."""
    rows = []
    schemes = schemes_for(cfg)
    for clp in amounts_clp:
        c = _contribution(scn, clp, cfg.usd_clp)
        res = {s.key: summarize(run_windows(scn, years, s, portfolio, cfg, c, with_irr=False)) for s in schemes}
        row = {"aporte_clp": clp, "aporte_usd": clp / cfg.usd_clp, "aportado": res["btg_1"]["aportado"]}
        for broker in ("btg", "ib"):
            keys = [k for k in res if k.startswith(broker + "_")]
            best = max(keys, key=lambda k: res[k]["mediana"])
            row.update({f"{broker}_mejor": int(best.split("_")[1]), f"{broker}_costos": res[best]["costos_mediana"],
                        f"{broker}_mediana": res[best]["mediana"]})
        rows.append(row)
    return pd.DataFrame(rows)


def find_breakeven(scn: Scenario, years: int, portfolio: Portfolio, cfg: SavingsConfig) -> float | None:
    """Aporte mensual (CLP) desde el que la mejor opción de IB deja más plata que la mejor de BTG."""
    def ib_minus_btg(clp: float) -> float:
        t = amount_comparison(scn, years, portfolio, cfg, [clp]).iloc[0]
        return t["ib_mediana"] - t["btg_mediana"]

    lo, hi = 20_000.0, 5_000_000.0
    if ib_minus_btg(hi) <= 0:
        return None
    if ib_minus_btg(lo) > 0:
        return lo
    for _ in range(18):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ib_minus_btg(mid) > 0 else (mid, hi)
    return round(hi, -3)


def scenario_comparison(scenarios: list[Scenario], horizons: list[int], portfolio: Portfolio,
                        cfg: SavingsConfig) -> pd.DataFrame:
    """Para cada escenario y plazo: resultados como múltiplo de lo aportado (dinero de hoy)."""
    btg = schemes_for(cfg)[0]
    rows = []
    for scn in scenarios:
        c = _contribution(scn, cfg.monthly_clp, cfg.usd_clp)
        for y in horizons:
            if len(scn.returns) - y * 12 + 1 < MIN_WINDOWS:
                continue
            s = summarize(run_windows(scn, y, btg, portfolio, cfg, c))
            rows.append({
                "escenario": scn.label, "clave": scn.key, "anios": y, "ventanas": s["ventanas"],
                "independientes": int(scn.years_of_data // y),
                "pct_perdida": s["pct_perdida"], "peor_x": s["peor"] / s["aportado"],
                "p10_x": s["p10"] / s["aportado"], "mediana_x": s["mediana"] / s["aportado"],
                "rentabilidad_real": s["rentabilidad_mediana"], "peor_momento": s["peor_momento"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- meta

def required_contribution(scn: Scenario, years: int, target: float, level: float, portfolio: Portfolio,
                          cfg: SavingsConfig) -> float:
    """Aporte mensual mínimo (en la unidad del escenario) para llegar a `target` en `level` de los casos.

    level = 0,5: en la mitad de los casos · 0,9: en 9 de cada 10 · 1,0: incluso en el peor caso histórico.
    """
    btg = schemes_for(cfg)[0]
    months = years * 12
    mat = np.lib.stride_tricks.sliding_window_view(scn.returns.to_numpy(dtype=float), months)

    def reached(c: float) -> float:
        final, _, _ = simulate_matrix(mat, c, btg, portfolio, cfg, scn.unit_to_usd)
        return float(final.min() if level >= 1 else np.quantile(final, 1 - level))

    lo, hi = 0.0, max(target / months, 1.0)
    while reached(hi) < target:
        hi *= 2
        if hi > target * 10:
            return float("nan")
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if reached(mid) >= target else (mid, hi)
    return hi


def goal_table(scenarios: list[Scenario], years: int, target_clp: float, confidence: float,
               portfolio: Portfolio, cfg: SavingsConfig) -> pd.DataFrame:
    """Cuánto aportar al mes (en pesos de hoy) para la meta, en cada escenario."""
    btg = schemes_for(cfg)[0]
    rows = []
    for scn in scenarios:
        if len(scn.returns) - years * 12 + 1 < MIN_WINDOWS:
            continue
        target = target_clp if scn.unit == "CLP" else target_clp / cfg.usd_clp
        row = {"escenario": scn.label, "clave": scn.key, "unidad": scn.unit_label}
        for name, level in (("tipico", 0.5), ("confianza", confidence), ("peor", 1.0)):
            row[name] = _from_units(scn, required_contribution(scn, years, target, level, portfolio, cfg), cfg.usd_clp)
        ws = run_windows(scn, years, btg, portfolio, cfg, _contribution(scn, cfg.monthly_clp, cfg.usd_clp),
                         with_irr=False)
        row["prob_con_aporte_actual"] = float((ws.final >= target).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- orquestación

@dataclass
class SavingsResult:
    portfolio: Portfolio
    cfg: SavingsConfig
    main: Scenario
    main_years: int
    by_horizon: pd.DataFrame
    main_windows: WindowSet
    costs_table: pd.DataFrame
    amounts_table: pd.DataFrame
    breakeven_clp: float | None
    comparison: pd.DataFrame | None = None
    goal: pd.DataFrame | None = None
    goal_target: float | None = None
    confidence: float = 0.9
    notes: list[str] = field(default_factory=list)


def run_savings(close: pd.DataFrame, cfg: SavingsConfig, today: pd.Timestamp | None,
                chile: pd.DataFrame | None = None, long_history: pd.DataFrame | None = None,
                main_years: int | None = None, goal: float | None = None, confidence: float = 0.9,
                with_breakeven: bool = True) -> SavingsResult:
    """chile: dólar y UF de fin de mes (índice mensual). long_history: retornos de French e IPC de EE.UU."""
    portfolio = PORTFOLIOS[cfg.portfolio]
    usd = etf_returns(close, portfolio, today)
    notes: list[str] = []

    main = None
    if chile is not None and not chile.empty:
        main = pesos_scenario(usd, chile, cfg.usd_clp, f"{portfolio.label}, en pesos de hoy")
    if main is None:
        main = usd_scenario(usd, f"{portfolio.label}, en dólares")
        notes.append("Sin dólar observado ni UF: los resultados quedan en dólares y sin descontar inflación.")

    horizons = [y for y in cfg.horizons_years if len(main.returns) - y * 12 + 1 >= MIN_WINDOWS]
    if not horizons:
        raise ValueError(f"Hay {main.years_of_data:.0f} años de datos: no alcanza para los plazos pedidos.")
    main_years = main_years or (10 if 10 in horizons else max(horizons))
    if main_years not in horizons:
        raise ValueError(f"El plazo de {main_years} años no cabe en los {main.years_of_data:.0f} años de datos.")

    contribution = _contribution(main, cfg.monthly_clp, cfg.usd_clp)
    schemes = schemes_for(cfg)
    by_h = {y: summarize(run_windows(main, y, schemes[0], portfolio, cfg, contribution)) for y in horizons}
    main_ws = run_windows(main, main_years, schemes[0], portfolio, cfg, contribution)

    rows = {}
    for sch in schemes:
        s = summarize(run_windows(main, main_years, sch, portfolio, cfg, contribution, with_irr=False))
        rows[sch.key] = {"esquema": sch.label, "costos": s["costos_mediana"],
                         "costos_pct": s["costos_mediana"] / s["aportado"], "mediana": s["mediana"],
                         "aportado": s["aportado"]}
    costs_table = pd.DataFrame(rows).T
    costs_table["vs_btg"] = costs_table["mediana"] - costs_table.loc["btg_1", "mediana"]

    amounts = sorted({30_000, 50_000, 100_000, 200_000, 500_000, cfg.monthly_clp})
    amounts_table = amount_comparison(main, main_years, portfolio, cfg, amounts)
    breakeven = find_breakeven(main, main_years, portfolio, cfg) if with_breakeven else None

    comparison, goal_df = None, None
    scenarios = [main]
    if long_history is not None and not long_history.empty:
        extra = long_scenarios(long_history, portfolio, usd)
        comparison = scenario_comparison(extra, [5, 10, 15, 20], portfolio, cfg)
        scenarios += [s for s in extra if s.key != "etf"]
    else:
        notes.append("Sin historia larga: se omiten los escenarios de EE.UU. desde 1926 y Japón desde 1990.")
    if goal:
        goal_df = goal_table(scenarios, main_years, goal, confidence, portfolio, cfg)

    return SavingsResult(portfolio=portfolio, cfg=cfg, main=main, main_years=main_years,
                         by_horizon=pd.DataFrame(by_h).T, main_windows=main_ws, costs_table=costs_table,
                         amounts_table=amounts_table, breakeven_clp=breakeven, comparison=comparison,
                         goal=goal_df, goal_target=goal, confidence=confidence, notes=notes)


# --------------------------------------------------------------------------- consola

def _clp(x: float) -> str:
    return "$" + f"{x:,.0f}".replace(",", ".")


def _usd(x: float) -> str:
    return "US$ " + f"{x:,.0f}".replace(",", ".")


def money(scn: Scenario, x: float) -> str:
    return _clp(x) if scn.unit == "CLP" else _usd(x)


def pct_label(p: float) -> str:
    return f"{p * 100:.0f} %"


def format_console(res: SavingsResult) -> str:
    cfg, m = res.cfg, res.main
    first, last = m.returns.index[0], m.returns.index[-1]
    lines = [
        f"Simulador de aportes: {_clp(cfg.monthly_clp)} al mes (dólar a {_clp(cfg.usd_clp)}) · {res.portfolio.label}",
        f"Resultados en {m.unit_label}. Historia: {first} a {last} ({len(m.returns)} meses); cada fila resume "
        "todos los meses posibles de partida.",
        "",
    ]
    h = res.by_horizon
    table = pd.DataFrame({
        "Aportado": h["aportado"].map(lambda v: money(m, v)),
        "Peor caso": h["peor"].map(lambda v: money(m, v)),
        "Malo (10 %)": h["p10"].map(lambda v: money(m, v)),
        "Típico": h["mediana"].map(lambda v: money(m, v)),
        "Bueno (90 %)": h["p90"].map(lambda v: money(m, v)),
        "Terminó perdiendo": h["pct_perdida"].map(pct_label),
        "Rentab. típica": h["rentabilidad_mediana"].map(lambda v: f"{v * 100:.1f} %/año"),
        "Peor momento": h["peor_momento"].map(pct_label),
    })
    table.index = [f"{int(y)} años" for y in h.index]
    lines += ["Con BTG, compra mensual:", table.to_string(), ""]

    if res.comparison is not None and not res.comparison.empty:
        c = res.comparison
        ct = pd.DataFrame({
            "Escenario": c["escenario"], "Plazo": c["anios"].map(lambda v: f"{v} años"),
            "Terminó perdiendo": c["pct_perdida"].map(pct_label),
            "Peor caso": c["peor_x"].map(lambda v: f"{v:.2f}x"),
            "Típico": c["mediana_x"].map(lambda v: f"{v:.2f}x"),
            "Rentab. real típica": c["rentabilidad_real"].map(lambda v: f"{v * 100:.1f} %/año"),
            "Peor momento": c["peor_momento"].map(pct_label),
            "Períodos indep.": c["independientes"],
        })
        lines += ["¿Y si el futuro se parece a otra época? (dólares de hoy; 1,00x = recuperas lo aportado)",
                  ct.to_string(index=False), ""]

    if res.goal is not None and not res.goal.empty:
        g = res.goal
        gt = pd.DataFrame({
            "Escenario": g["escenario"],
            "Típico (50 %)": g["tipico"].map(_clp),
            f"Con {pct_label(res.confidence)}": g["confianza"].map(_clp),
            "Peor caso": g["peor"].map(_clp),
            "Con tu aporte llegas": g["prob_con_aporte_actual"].map(lambda v: f"en {pct_label(v)} de los casos"),
        })
        lines += [f"Meta: {_clp(res.goal_target)} de hoy en {res.main_years} años. Aporte mensual necesario "
                  "(pesos de hoy):", gt.to_string(index=False), ""]

    c = res.costs_table
    ct = pd.DataFrame({
        "Costos totales": c["costos"].map(lambda v: money(m, v)),
        "% de lo aportado": c["costos_pct"].map(lambda v: f"{v * 100:.1f} %"),
        "Valor típico al final": c["mediana"].map(lambda v: money(m, v)),
        "Diferencia vs BTG mensual": c["vs_btg"].map(lambda v: ("+" if v >= 0 else "−") + money(m, abs(v))),
    })
    ct.index = c["esquema"]
    lines += [f"Costos a {res.main_years} años con este aporte:", ct.to_string(), ""]

    a = res.amounts_table
    at = pd.DataFrame({
        "Mejor en BTG": [f"{frequency_label(k)} ({money(m, v)})" for k, v in zip(a["btg_mejor"], a["btg_costos"], strict=True)],
        "Mejor en IB": [f"{frequency_label(k)} ({money(m, v)})" for k, v in zip(a["ib_mejor"], a["ib_costos"], strict=True)],
        "Conviene": [("BTG" if b >= i else "IB") for b, i in zip(a["btg_mediana"], a["ib_mediana"], strict=True)],
    })
    at.index = a["aporte_clp"].map(_clp)
    lines += [f"¿Dónde y cada cuánto comprar según el aporte? (a {res.main_years} años; entre paréntesis, costos totales)",
              at.to_string()]
    lines.append(f"IB empieza a convenir desde unos {_clp(res.breakeven_clp)} al mes." if res.breakeven_clp else
                 "Con estos supuestos, IB no le gana a BTG con ningún aporte de hasta $5.000.000 al mes.")
    if res.notes:
        lines += [""] + [f"Aviso: {n}" for n in res.notes]
    return "\n".join(lines)
