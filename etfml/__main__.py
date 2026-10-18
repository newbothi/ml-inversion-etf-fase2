"""Línea de comandos.

  python -m etfml descargar
  python -m etfml backtest [--sintetico] [--actualizar]
  python -m etfml senal [--modelo ensamble] [--actualizar]
  python -m etfml rebalancear [--modelo ensamble] [--ejecutar] [--actualizar]
  python -m etfml llenados
  python -m etfml seguimiento [--actualizar]
  python -m etfml tendencia [--sintetico] [--actualizar]
  python -m etfml experimentos
  python -m etfml simulador [--aporte 100000] [--anios 10] [--cartera global] [--meta 20000000] [--dolar 981]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime

import pandas as pd

from .config import STRATEGY_MODELS, ConfigError, load_config

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _today() -> pd.Timestamp:
    # fecha del mercado de EE.UU., no de Chile
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def _pct(x: float) -> str:
    return f"{x * 100:6.1f} %"


def cmd_descargar(args, cfg) -> int:
    from .data import load_market_data

    load_market_data(cfg, refresh=True)
    return 0


def cmd_backtest(args, cfg) -> int:
    from .backtest import STRATEGY_LABELS, run_backtest
    from .data import load_market_data, synthetic_market_data
    from .features import build_panel
    from .metrics import summarize
    from .report import write_outputs

    if args.sintetico:
        market = synthetic_market_data()
        cfg = cfg.with_universe(list(market.close.columns), market.close.columns[0])
        today = pd.Timestamp("2026-01-15")
        print("Usando datos sintéticos (inventados) para probar el código.")
    else:
        market = load_market_data(cfg, refresh=args.actualizar)
        today = _today()

    panel = build_panel(market, min_assets=cfg.strategy.min_assets, today=today)
    print(f"Panel: {panel.labeled.index.get_level_values('date').nunique()} meses con retorno conocido, "
          f"{len(panel.feature_cols)} features. Entrenando walk-forward...")
    result = run_backtest(panel, cfg, market.source)
    summary = summarize(result, cfg.risk_free_annual, cfg.strategy.top_k)

    s = summary["strategy"]
    table = pd.DataFrame({
        "Retorno anual": s["cagr"].map(_pct),
        "Volatilidad": s["volatility"].map(_pct),
        "Sharpe": s["sharpe"].map(lambda v: f"{v:5.2f}"),
        "Máx. caída": s["max_drawdown"].map(_pct),
        "Costo anual": s["cost_annual"].map(_pct),
    })
    table.index = [STRATEGY_LABELS[k] for k in table.index]
    print("\nResultado fuera de muestra, después de costos:")
    print(table.to_string())

    m = summary["models"]
    print("\nCalidad de las predicciones:")
    print(pd.DataFrame({
        "IC medio": m["ic_mean"].map(lambda v: f"{v:6.3f}"),
        "t-stat IC (NW)": m["ic_tstat"].map(lambda v: f"{v:5.1f}"),
        "AUC": m["auc"].map(lambda v: f"{v:5.3f}"),
    }, index=m.index).rename(index=STRATEGY_LABELS).to_string())

    print("\nVeredicto (los 3 criterios a la vez):")
    for k, v in summary["verdicts"].items():
        print(f"  {STRATEGY_LABELS[k]:<16} {'PASA' if v['passed'] else 'no pasa'}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = cfg.reports_dir / f"{stamp}{'_sintetico' if args.sintetico else ''}"
    report = write_outputs(result, summary, cfg, out_dir, focus=cfg.ib.model)
    print(f"\nReporte: {report}")
    if not args.sintetico:  # los datos inventados no cuentan como pruebas
        registry = _register_ml(cfg, result, summary, market)
        print(f"Registro de experimentos: {len(registry)} prueba(s) en {cfg.experiments_dir.name}/registro.csv")
    return 0


def _register_ml(cfg, result, summary, market):
    from .backtest import STRATEGY_LABELS
    from .registry import make_row, ml_params, record_trials

    rows = []
    for k in ("ridge", "logistica", "lgbm", "ensamble", "momentum"):
        params = ml_params(cfg)
        if k == "momentum":  # no depende de los modelos
            params = {key: v for key, v in params.items() if key not in ("modelos", "macro")}
        rows.append(make_row("rotacion_ml", k, STRATEGY_LABELS[k], params, summary["strategy"].loc[k].to_dict(),
                             result.simulations[k], market.close.index.max()))
    return record_trials(cfg, rows)


def cmd_tendencia(args, cfg) -> int:
    from dataclasses import replace

    from .config import TrendConfig
    from .data import load_market_data, synthetic_market_data
    from .registry import make_row, record_trials
    from .trend import evaluate, format_console, run_trend, trend_params
    from .trend_report import write_trend_outputs

    if args.sintetico:
        market = synthetic_market_data()
        cfg = replace(cfg.with_universe(list(market.close.columns), "SYN01"),
                      trend=TrendConfig("SYN01", ("SYN02", "SYN03")))
        today = pd.Timestamp("2026-01-15")
        print("Usando datos sintéticos (inventados) para probar el código. No se registra nada.")
    else:
        market = load_market_data(cfg, refresh=args.actualizar, tickers=list(cfg.trend.tickers))
        today = _today()

    res = run_trend(cfg, market, today)
    if args.sintetico:
        trial_sharpes = [s["net"].mean() / s["net"].std() for k, s in res.simulations.items() if k != "comprar_mantener"]
        months = res.simulations["comprar_mantener"].index.to_period("M") + 1
        registry = pd.DataFrame([{"experimento": "tendencia", "descripcion": v.label,
                                  "desde": str(months[0]), "hasta": str(months[-1]),
                                  "sharpe_anual": res.metrics.loc[v.key, "sharpe"],
                                  "max_caida": res.metrics.loc[v.key, "max_drawdown"], "datos_hasta": "sintético"}
                                 for v in res.variants])
    else:
        rows = [make_row("tendencia", v.key, v.label, trend_params(cfg, v), res.metrics.loc[v.key].to_dict(),
                         res.simulations[v.key], res.data_until) for v in res.variants]
        registry = record_trials(cfg, rows)
        trial_sharpes = registry["sharpe_mensual"].astype(float).tolist()
    evaluate(res, trial_sharpes)

    print()
    print(format_console(res, len(trial_sharpes)))
    if not args.sintetico and "rotacion_ml" not in set(registry["experimento"]):
        print("\nAviso: el registro no tiene las pruebas de la fase 1. Corre 'python -m etfml backtest' una vez "
              "para que cuenten en el Sharpe deflactado, y después vuelve a correr este comando.")
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = cfg.reports_dir / f"{stamp}_tendencia{'_sintetico' if args.sintetico else ''}"
    report = write_trend_outputs(res, registry, cfg, out_dir, synthetic=args.sintetico)
    print(f"\nReporte: {report}")
    return 0


def cmd_simulador(args, cfg) -> int:
    from dataclasses import replace

    from .config import ConfigError
    from .data import load_market_data, synthetic_market_data
    from .external import load_chile_indicators, load_long_history, month_end, synthetic_external
    from .savings import PORTFOLIOS, format_console, run_savings
    from .savings_report import write_savings_outputs

    sv = cfg.savings
    if args.aporte is not None:
        sv = replace(sv, monthly_clp=args.aporte)
    if args.cartera is not None:
        sv = replace(sv, portfolio=args.cartera)
    if args.anios is not None and args.anios not in sv.horizons_years:
        sv = replace(sv, horizons_years=tuple(sorted(set(sv.horizons_years) | {args.anios})))
    if sv.monthly_clp <= 0:
        raise ConfigError("El aporte debe ser positivo")
    if args.meta is not None and args.meta <= 0:
        raise ConfigError("La meta debe ser positiva")
    if not 0.5 <= args.confianza < 1:
        raise ConfigError("--confianza debe estar entre 0.5 y 0.99")
    portfolio = PORTFOLIOS[sv.portfolio]

    if args.sintetico:
        raw = synthetic_market_data()
        close = raw.close.rename(columns={"SYN01": "SPY", "SYN02": "EFA", "SYN03": "EEM", "SYN04": "IEF"})
        chile_daily, long_history = synthetic_external()
        today = pd.Timestamp("2026-01-15")
        print("Usando datos sintéticos (inventados) para probar el código.")
    else:
        close = load_market_data(cfg, refresh=args.actualizar, tickers=list(portfolio.history)).close
        chile_daily = load_chile_indicators(cfg, refresh=args.actualizar)
        long_history = load_long_history(cfg, refresh=args.actualizar)
        today = _today()

    chile = None
    if chile_daily is not None and not chile_daily.empty:
        chile_daily = chile_daily[chile_daily.index <= today]
        chile = month_end(chile_daily)
        if args.dolar is None:  # el dólar observado más reciente reemplaza al de config.toml
            sv = replace(sv, usd_clp=float(chile_daily["dolar"].dropna().iloc[-1]))
            print(f"Dólar observado al {chile_daily['dolar'].dropna().index[-1].date()}: ${sv.usd_clp:,.2f}")
    if args.dolar is not None:
        sv = replace(sv, usd_clp=args.dolar)
    if sv.usd_clp <= 0:
        raise ConfigError("El dólar debe ser positivo")

    res = run_savings(close, sv, today, chile=chile, long_history=long_history, main_years=args.anios,
                      goal=args.meta, confidence=args.confianza)
    print()
    print(format_console(res))
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = cfg.reports_dir / f"{stamp}_simulador{'_sintetico' if args.sintetico else ''}"
    report = write_savings_outputs(res, out_dir, synthetic=args.sintetico)
    print(f"\nReporte: {report}")
    return 0


def cmd_experimentos(args, cfg) -> int:
    from .registry import load_registry

    reg = load_registry(cfg)
    if reg.empty:
        print("El registro está vacío. Se llena solo al correr 'backtest' y 'tendencia' con datos reales.")
        return 0
    view = pd.DataFrame({
        "Experimento": reg["experimento"],
        "Variante": reg["descripcion"],
        "Período": reg["desde"].astype(str) + " a " + reg["hasta"].astype(str),
        "Sharpe": reg["sharpe_anual"].map(lambda v: f"{v:5.2f}"),
        "Máx. caída": reg["max_caida"].map(lambda v: f"{v * 100:6.1f} %"),
        "Datos hasta": reg["datos_hasta"],
    })
    print(f"{len(reg)} prueba(s) registradas en {cfg.experiments_dir.name}/registro.csv:\n")
    print(view.to_string(index=False))
    return 0


def _signal(args, cfg):
    from .data import load_market_data
    from .features import build_panel
    from .live_signal import compute_live_signal, log_signal

    model = args.modelo or cfg.ib.model
    today = _today()
    market = load_market_data(cfg, refresh=args.actualizar)
    panel = build_panel(market, min_assets=cfg.strategy.min_assets, today=today)
    sig = compute_live_signal(panel, market, cfg, model, today)

    print(f"\nSeñal con datos al cierre del {sig.signal_date.date()} · modelo: {model} · "
          f"entrenado con {sig.train_months} meses")
    for w in sig.warnings:
        print(f"Aviso: {w}")
    view = sig.scores.copy()
    view["peso_objetivo"] = sig.weights.reindex(view.index).fillna(0.0).map(lambda v: f"{v:.0%}" if v else "")
    print(view.round(4).to_string())
    log_signal(cfg, sig)
    return sig


def cmd_senal(args, cfg) -> int:
    _signal(args, cfg)
    print(f"\nSeñal guardada en {cfg.logs_dir.name}/senales.csv")
    return 0


def cmd_rebalancear(args, cfg) -> int:
    from .ib_rebalance import (
        build_execution_record,
        check_account_allowed,
        connect,
        execute_orders,
        fetch_fills,
        log_orders,
        plan_orders,
        save_execution_record,
        save_fills,
        snapshot_account,
    )

    sig = _signal(args, cfg)
    universe = list(cfg.universe.tickers)
    ib = connect(cfg, readonly=not args.ejecutar)
    try:
        snap = snapshot_account(ib)
        print(f"\nCuenta {snap.account_id} ({'paper' if snap.is_paper else 'REAL'}) · "
              f"patrimonio neto US$ {snap.net_liquidation:,.2f}")
        check_account_allowed(snap, cfg)
        outside = float(snap.market_values[[t for t in snap.market_values.index if t not in universe]].sum())
        if outside:
            print(f"Posiciones fuera del universo (no se tocan): US$ {outside:,.2f}")
        plan = plan_orders(sig.weights, sig.prices, snap.positions, snap.net_liquidation, outside,
                           cfg.ib.cash_buffer, cfg.costs.rebalance_band_usd, universe)
        print("\nPlan de órdenes (precios del último cierre):")
        print(plan.to_string(index=False, formatters={"precio": "{:,.2f}".format, "monto_usd": "{:,.2f}".format,
                                                      "peso_objetivo": "{:.0%}".format}))
        pending = plan[plan["orden"] != "MANTENER"]
        if pending.empty:
            print("\nLa cartera ya está alineada con la señal. No hay órdenes que enviar.")
            log_orders(cfg, plan, None, snap.account_id, executed=False)
            if args.ejecutar:  # el mes cuenta como rebalanceado para el seguimiento
                record = build_execution_record(snap, sig, plan, cfg.ib.order_type, universe, outside,
                                                cfg.ib.cash_buffer)
                print(f"Mes registrado para el seguimiento: {save_execution_record(cfg, record)}")
            return 0
        if not args.ejecutar:
            path = log_orders(cfg, plan, None, snap.account_id, executed=False)
            print(f"\nSimulación: no se envió nada. Plan guardado en {path}")
            print("Para enviar las órdenes: python -m etfml rebalancear --ejecutar")
            return 0
        answer = input(f"\nSe enviarán {len(pending)} órdenes {cfg.ib.order_type} a {snap.account_id}. "
                       "Escribe SI para confirmar: ").strip()
        if answer != "SI":
            print("Cancelado. No se envió ninguna orden.")
            return 1
        record = build_execution_record(snap, sig, plan, cfg.ib.order_type, universe, outside, cfg.ib.cash_buffer)
        save_execution_record(cfg, record)  # se guarda antes de enviar, por si algo se corta
        status = execute_orders(ib, plan, cfg.ib.order_type)
        record["ordenes"] = json.loads(status.to_json(orient="records"))
        record_path = save_execution_record(cfg, record)
        print("\nEstado de las órdenes:")
        print(status.drop(columns=["order_id", "perm_id"]).to_string(index=False))
        added = save_fills(cfg, fetch_fills(ib, snap.account_id))
        path = log_orders(cfg, plan, status, snap.account_id, executed=True)
        print(f"\nRegistro guardado en {path}")
        print(f"Foto de la cuenta para el seguimiento: {record_path}")
        print(f"Llenados registrados: {added}")
        if cfg.ib.order_type == "MOC" or not (status["estado"] == "Filled").all():
            print("\nIMPORTANTE: hay órdenes que todavía no se llenan. Después del cierre (16:00 de Nueva York) "
                  "y el mismo día, corre: python -m etfml llenados")
        return 0
    finally:
        ib.disconnect()


def cmd_llenados(args, cfg) -> int:
    from .ib_rebalance import connect, fetch_fills, save_fills

    ib = connect(cfg, readonly=True)
    try:
        accounts = ib.managedAccounts()
        if not accounts:
            raise RuntimeError("IB no informó ninguna cuenta.")
        fills = fetch_fills(ib, accounts[0])
    finally:
        ib.disconnect()
    if fills.empty:
        print("IB no informó ejecuciones hoy. La API solo entrega las del día en curso: corre este comando "
              "el mismo día en que ejecutaste las órdenes.")
        return 0
    added = save_fills(cfg, fills)
    print(fills.drop(columns=["exec_id", "perm_id", "order_id"]).to_string(index=False))
    print(f"\n{added} llenado(s) nuevo(s) guardado(s) en {cfg.logs_dir.name}/llenados.csv")
    if fills["comision_usd"].isna().any():
        print("Aviso: IB todavía no informa la comisión de algunos llenados. Vuelve a correr el comando en unos minutos.")
    return 0


def cmd_seguimiento(args, cfg) -> int:
    from .data import load_market_data
    from .ib_rebalance import load_execution_records, load_fills
    from .tracking import format_report, save_tracking_csv, track

    records = load_execution_records(cfg)
    if not records:
        print("Todavía no hay ejecuciones registradas. Corre 'python -m etfml rebalancear --ejecutar' "
              "el primer día hábil del mes.")
        return 0
    market = load_market_data(cfg, refresh=args.actualizar)
    result = track(records, load_fills(cfg), market, cfg, _today())
    print()
    print(format_report(result))
    path = save_tracking_csv(cfg, result)
    print(f"\nDetalle guardado en {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m etfml", description="Machine learning para rotación de ETFs")
    parser.add_argument("--config", help="ruta a otro config.toml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("descargar", help="descarga precios desde Yahoo Finance y los guarda en datos/")

    p = sub.add_parser("backtest", help="walk-forward de los 3 modelos + ensamble contra los baselines")
    p.add_argument("--sintetico", action="store_true", help="usa datos inventados (no necesita internet)")
    p.add_argument("--actualizar", action="store_true", help="descarga los precios de nuevo")

    for name, helptext in (("senal", "calcula la cartera objetivo de este mes"),
                           ("rebalancear", "compara la señal con tu cuenta IB y arma las órdenes")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--modelo", choices=STRATEGY_MODELS, help="por defecto, el de [ib].modelo")
        p.add_argument("--actualizar", action="store_true", help="descarga los precios de nuevo")
        if name == "rebalancear":
            p.add_argument("--ejecutar", action="store_true", help="envía las órdenes (pide confirmación)")

    p = sub.add_parser("tendencia", help="fase 3: filtro de tendencia (acciones o refugio) contra comprar y mantener")
    p.add_argument("--sintetico", action="store_true", help="usa datos inventados (no registra nada)")
    p.add_argument("--actualizar", action="store_true", help="descarga los precios de nuevo")
    sub.add_parser("experimentos", help="muestra todas las pruebas registradas")

    p = sub.add_parser("simulador", help="plan de aportes mensuales: cuánto, por cuánto tiempo y BTG o IB")
    p.add_argument("--aporte", type=float, help="aporte mensual en pesos (por defecto, el de config.toml)")
    p.add_argument("--anios", type=int, help="plazo principal en años (por defecto, 10)")
    p.add_argument("--cartera", choices=["global", "eeuu", "conservadora"], help="por defecto, la de config.toml")
    p.add_argument("--dolar", type=float, help="pesos por dólar (por defecto, el dólar observado más reciente)")
    p.add_argument("--meta", type=float, help="monto a juntar, en pesos de hoy: calcula cuánto aportar al mes")
    p.add_argument("--confianza", type=float, default=0.9, help="seguridad para la meta (por defecto 0.9)")
    p.add_argument("--sintetico", action="store_true", help="usa datos inventados")
    p.add_argument("--actualizar", action="store_true", help="descarga los precios de nuevo")

    sub.add_parser("llenados", help="guarda las ejecuciones y comisiones de hoy desde IB")
    p = sub.add_parser("seguimiento", help="compara la cuenta paper con lo que asume el backtest")
    p.add_argument("--actualizar", action="store_true", help="descarga los precios de nuevo")

    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        handler = {"descargar": cmd_descargar, "backtest": cmd_backtest, "senal": cmd_senal,
                   "rebalancear": cmd_rebalancear, "llenados": cmd_llenados,
                   "seguimiento": cmd_seguimiento, "tendencia": cmd_tendencia,
                   "experimentos": cmd_experimentos, "simulador": cmd_simulador}[args.cmd]
        return handler(args, cfg)
    except (ConfigError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrumpido.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
