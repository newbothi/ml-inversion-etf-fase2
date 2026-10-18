"""Reporte HTML del experimento de tendencia (mismo estilo y motor de gráficos que el de la fase 1)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Config
from .metrics import annual_returns, drawdown, equity_curve
from .report import CSS, ICON_NO, ICON_OK, JS, _chart, _esc, _num, _pct, _table, _x_labels
from .stats import DSR_THRESHOLD
from .trend import BENCHMARK, LEVEL_TEXT, TrendResult

ICON_MAYBE = ('<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5" fill="none" '
              'stroke="currentColor" stroke-width="2"/><path d="M5.5 8h5" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round"/></svg>')
STATUS = {"pasa": ("good", ICON_OK), "prometedor": ("warn", ICON_MAYBE), "no_pasa": ("bad", ICON_NO)}

BENCHMARK_SLOT = 5  # mismo color que "Comprar y mantener" en el reporte de la fase 1


def _slots(res: TrendResult) -> dict[str, int]:
    slots = {v.key: i + 1 for i, v in enumerate(res.variants)}
    slots[BENCHMARK] = BENCHMARK_SLOT
    return slots


def _labels(res: TrendResult) -> tuple[dict[str, str], dict[str, str]]:
    long = {v.key: v.label for v in res.variants}
    short = {v.key: f"{'Media' if v.signal == 'media10' else 'Mom.'} → {v.safe_asset}" for v in res.variants}
    long[BENCHMARK] = f"Comprar y mantener {res.risk_asset}"
    short[BENCHMARK] = res.risk_asset
    return long, short


def _key(slot: int) -> str:
    return f'<span class="key" style="background:var(--s{slot})"></span>'


def _specs(res: TrendResult) -> dict:
    order = [v.key for v in res.variants] + [BENCHMARK]
    slots = _slots(res)
    long, short = _labels(res)
    idx = res.simulations[BENCHMARK].index

    def series(fn):
        return [{"name": long[k], "short": short[k], "slot": slots[k],
                 "values": [round(float(x), 6) for x in fn(res.simulations[k]["net"])]} for k in order]

    x = _x_labels(idx)
    return {
        "growth": {"x": x, "series": series(lambda r: [1.0] + list(equity_curve(r).to_numpy())),
                   "fmt": "mult", "log": True, "ref": 1.0},
        "drawdown": {"x": x, "series": series(lambda r: [0.0] + list(drawdown(r).to_numpy())),
                     "fmt": "pct", "log": False, "ref": 0.0, "max0": True},
    }


def _verdict_cards(res: TrendResult) -> str:
    slots = _slots(res)
    cards = []
    for v in res.variants:
        vd, d = res.verdicts[v.key], res.deflated[v.key]
        status_cls, icon = STATUS[vd["level"]]
        title = LEVEL_TEXT[vd["level"]].split(":")[0]
        status = f'<span class="status {status_cls}">{icon}{_esc(title)}</span>'
        checks = [
            (vd["better"], f"Sharpe mayor que comprar y mantener {res.risk_asset}"),
            (vd["consistent"], f"Le gana en al menos 2 de 3 subperíodos ({vd['sub_wins']} de 3)"),
            (vd["evidence"], f"Sharpe deflactado de al menos {_pct(DSR_THRESHOLD, 0)} ({_pct(d.probability, 1)})"),
        ]
        items = "".join(
            f'<li><span class="mini {"good" if ok else "bad"}">{ICON_OK if ok else ICON_NO}</span>'
            f"<span>{_esc(text)}</span></li>" for ok, text in checks)
        info = (f'<p class="card-note">En acciones el {_pct(vd["pct_risk_on"], 0)} de los meses · '
                f'{vd["switches"]} cambios de posición</p>')
        cards.append(f'<div class="card"><div class="card-head">{_key(slots[v.key])}<strong>{_esc(v.label)}</strong>'
                     f'</div>{status}<ul class="checks">{items}</ul>{info}</div>')
    return '<div class="cards">' + "".join(cards) + "</div>"


def _results_table(res: TrendResult) -> str:
    m = res.metrics
    slots = _slots(res)
    long, _ = _labels(res)
    order = [v.key for v in res.variants] + [BENCHMARK]
    headers = ["Estrategia", "Retorno/año", "Volatilidad", "Sharpe", "Sortino", "Máx. caída", "Calmar",
               "Costo/año", "Meses +", "US$ 1 →"]
    rows = [[_key(slots[k]) + _esc(long[k]), _pct(m.loc[k, "cagr"]), _pct(m.loc[k, "volatility"]),
             _num(m.loc[k, "sharpe"]), _num(m.loc[k, "sortino"]), _pct(m.loc[k, "max_drawdown"]),
             _num(m.loc[k, "calmar"]), _pct(m.loc[k, "cost_annual"], 2), _pct(m.loc[k, "pct_positive_months"], 0),
             _num(m.loc[k, "growth_of_1"]) + "x"] for k in order]
    classes = [""] * len(res.variants) + ["sep"]
    return _table(headers, rows, row_classes=classes)


def _subperiod_table(res: TrendResult) -> str:
    sub = res.subperiods
    slots = _slots(res)
    long, _ = _labels(res)
    order = [v.key for v in res.variants] + [BENCHMARK]
    rows = [[_key(slots[k]) + _esc(long[k])] + [_num(x) for x in sub.loc[k]] for k in order]
    return _table(["Estrategia"] + [str(c) for c in sub.columns], rows,
                  row_classes=[""] * len(res.variants) + ["sep"])


def _annual_table(res: TrendResult) -> str:
    a = annual_returns(res.simulations)
    long, short = _labels(res)
    order = [v.key for v in res.variants] + [BENCHMARK]
    rows = [[str(y)] + [_pct(a.loc[y, k]) for k in order] for y in a.index]
    return _table(["Año"] + [short[k] for k in order], rows)


def _dsr_table(res: TrendResult) -> str:
    slots = _slots(res)
    rows = []
    for v in res.variants:
        d = res.deflated[v.key]
        rows.append([_key(slots[v.key]) + _esc(v.label), _num(d.annualized(d.sharpe)),
                     _num(d.annualized(d.benchmark_sharpe)), _num(d.annualized(d.luck_premium)),
                     _num(d.annualized(d.threshold)), _pct(d.probability, 1)])
    return _table(["Variante", "Sharpe", f"Sharpe {res.risk_asset}", "+ suerte", "= umbral",
                   "Probabilidad de superarlo"], rows)


def _registry_table(registry: pd.DataFrame) -> str:
    names = {"rotacion_ml": "Rotación ML (fase 1)", "tendencia": "Filtro de tendencia"}
    rows = [[_esc(names.get(r.experimento, r.experimento)), _esc(r.descripcion), f"{r.desde} a {r.hasta}",
             _num(r.sharpe_anual), _pct(r.max_caida), _esc(str(r.datos_hasta))] for r in registry.itertuples()]
    return _table(["Experimento", "Variante", "Período", "Sharpe", "Máx. caída", "Datos hasta"], rows,
                  text_cols={1, 2, 5})


EXTRA_CSS = ".card-note{color:var(--muted);font-size:12px;margin:10px 0 0}"


def build_trend_html(res: TrendResult, registry: pd.DataFrame, cfg: Config, synthetic: bool = False) -> str:
    idx = res.simulations[BENCHMARK].index
    first, last = idx[0].to_period("M") + 1, idx[-1].to_period("M") + 1
    n = len(registry)
    js = JS.replace("__SPECS__", json.dumps(_specs(res), ensure_ascii=False))
    safe = " o ".join(cfg.trend.safe_assets)
    c = cfg.costs
    ml_missing = "rotacion_ml" not in set(registry["experimento"])
    if synthetic:
        warn = ('<div class="banner"><strong>Datos sintéticos.</strong> Este reporte solo sirve para comprobar que el '
                'código funciona. Los precios son inventados y nada quedó en el registro.</div>')
    elif ml_missing:
        warn = ('<div class="banner">El registro no tiene las pruebas de la fase 1. Corre '
                '<code>python -m etfml backtest</code> para que cuenten en el Sharpe deflactado.</div>')
    else:
        warn = ""

    body = f"""
<main>
  <h1>Fase 3 · Filtro de tendencia</h1>
  <p class="lede">Cada fin de mes se decide una sola cosa: estar en {res.risk_asset} o refugiarse en {safe}.
  Dos señales fijadas antes de ver resultados × {len(cfg.trend.safe_assets)} refugios = {len(res.variants)} variantes,
  todas comparadas con comprar y mantener {res.risk_asset}, después de costos.</p>
  <p class="meta">Período: {first} a {last} ({len(idx)} meses) · datos hasta {res.data_until.date()} ·
  {n} pruebas en el registro de experimentos</p>
  {warn}

  <h2>Veredicto</h2>
  <p class="note">"Prometedor" significa que mejora y es consistente, pero con esta cantidad de años y de pruebas
  no se puede descartar que sea suerte. "Pasa" exige además evidencia estadística.</p>
  {_verdict_cards(res)}

  <h2>Resultado de cada variante</h2>
  {_results_table(res)}

  <h2>Crecimiento de 1 dólar</h2>
  <p class="note">Escala logarítmica: la misma distancia vertical es el mismo porcentaje de ganancia.</p>
  {_chart("growth", "Crecimiento de 1 dólar por variante")}

  <h2>Caídas desde el máximo</h2>
  <p class="note">Lo que más promete un filtro de tendencia es esquivar las caídas grandes. Aquí se ve si lo logra
  y cuánto le cuesta en los años buenos.</p>
  {_chart("drawdown", "Caídas desde el máximo")}

  <h2>Consistencia por subperíodo (Sharpe neto)</h2>
  {_subperiod_table(res)}

  <h2>Sharpe deflactado</h2>
  <p class="note">Si pruebas muchas variantes, la mejor se ve bien aunque ninguna sirva. El umbral es el Sharpe de
  {res.risk_asset} más lo que la mejor de {n} pruebas le ganaría solo por suerte (calculado con la dispersión de los
  Sharpe de todas las pruebas registradas). La última columna es la probabilidad de que el Sharpe verdadero supere ese
  umbral, corrigiendo por asimetría y colas gruesas. Se exige {_pct(DSR_THRESHOLD, 0)}.</p>
  {_dsr_table(res)}

  <h2>Retornos por año</h2>
  {_annual_table(res)}

  <h2>Registro de experimentos ({n} pruebas)</h2>
  <p class="note">Cada estrategia distinta evaluada con datos reales. Volver a correr la misma configuración actualiza
  su fila; cambiar algo cuenta como una prueba nueva.</p>
  {_registry_table(registry)}

  <h2>Supuestos y advertencias</h2>
  <ul class="plain">
    <li>Señal con el cierre del último día hábil del mes; cambio de posición al cierre del día hábil siguiente.</li>
    <li>La media de 10 meses usa 10 cierres mensuales (incluido el actual). El momentum compara el retorno de 12 meses
      de {res.risk_asset} con el de {cfg.trend.safe_assets[-1]}.</li>
    <li>Costos: comisión de {_num(c.commission_bps, 1)} bps con mínimo de US$ {_num(c.min_commission_usd)} por orden y
      slippage de {_num(c.slippage_bps, 1)} bps por lado, sobre US$ {_num(c.initial_capital_usd, 0)}.</li>
    <li>Las reglas vienen de estudios publicados en 2007 (media de 10 meses) y 2014 (momentum absoluto): los años
      anteriores a esas fechas eran parte de lo que esos autores vieron.</li>
    <li>No incluye impuestos: cada cambio de posición puede generar ganancias de capital que tributan.</li>
    <li>Un buen resultado histórico no garantiza resultados futuros. Esto no es una recomendación de inversión.</li>
  </ul>
</main>"""
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Filtro de tendencia</title><style>{CSS}{EXTRA_CSS}</style></head><body>{body}"
            f"<script>{js}</script></body></html>")


def write_trend_outputs(res: TrendResult, registry: pd.DataFrame, cfg: Config, out_dir: Path,
                        synthetic: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "reporte.html"
    report.write_text(build_trend_html(res, registry, cfg, synthetic), encoding="utf-8")
    pd.DataFrame({k: s["net"] for k, s in res.simulations.items()}).to_csv(
        out_dir / "retornos_mensuales.csv", index_label="fecha_senal", encoding="utf-8")
    res.metrics.to_csv(out_dir / "metricas.csv", index_label="estrategia", encoding="utf-8")
    res.risk_on.astype(int).to_csv(out_dir / "senales.csv", index_label="fecha_senal", encoding="utf-8")
    return report
