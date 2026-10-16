"""Reporte HTML autocontenido (sin internet) + CSVs con los resultados."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BASELINES, STRATEGY_LABELS, BacktestResult
from .config import MODEL_NAMES, STRATEGY_MODELS, Config
from .features import FEATURE_LABELS
from .metrics import drawdown, equity_curve

# Colores: el color sigue a la entidad en todos los gráficos (paleta validada para daltonismo).
SLOTS = {"ridge": 1, "logistica": 2, "lgbm": 3, "ensamble": 4, "buy_hold": 5, "equal_weight": 6, "momentum": 7}
SHORT_LABELS = {"ridge": "Ridge", "logistica": "Logística", "lgbm": "LightGBM", "ensamble": "Ensamble",
                "buy_hold": "Mantener", "equal_weight": "Igual peso", "momentum": "Momentum"}


# --------------------------------------------------------------------------- formato

def _num(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def _pct(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return _num(x * 100, decimals) + " %"


def _esc(s: str) -> str:
    return html.escape(str(s))


def _table(headers: list[str], rows: list[list[str]], first_col_left: bool = True,
           row_classes: list[str] | None = None, text_cols: set[int] | None = None) -> str:
    text_cols = set(text_cols or set()) | ({0} if first_col_left else set())
    # los encabezados pueden partirse en dos líneas, pero nunca entre un número y su "%"
    th = "".join(f'<th class="{"txt" if j in text_cols else "num"}">{_esc(h).replace(" %", "&nbsp;%")}</th>'
                 for j, h in enumerate(headers))
    body = []
    for i, r in enumerate(rows):
        cls = f' class="{row_classes[i]}"' if row_classes and row_classes[i] else ""
        cells = "".join(
            f'<td class="{"txt" if j in text_cols else "num"}">{c}</td>' for j, c in enumerate(r)
        )
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<div class="table-wrap"><table class="data"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _key(name: str) -> str:
    return f'<span class="key" style="background:var(--s{SLOTS[name]})"></span>'


ICON_OK = ('<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3 3 7-7" fill="none" '
           'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ICON_NO = ('<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" '
           'stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>')


# --------------------------------------------------------------------------- datos de gráficos

def _x_labels(index: pd.DatetimeIndex) -> list[str]:
    # punto inicial (capital 1,0 al comprar) + cada mes mantenido (el mes siguiente a la señal)
    held = index.to_period("M") + 1
    return [str(index[0].to_period("M"))] + [str(p) for p in held]


def _series(sims: dict, names: list[str], fn) -> list[dict]:
    out = []
    for n in names:
        vals = fn(sims[n]["net"])
        out.append({"name": STRATEGY_LABELS[n], "short": SHORT_LABELS[n], "slot": SLOTS[n],
                    "values": [round(float(v), 6) for v in vals]})
    return out


def _chart_specs(result: BacktestResult, summary: dict, focus: str) -> dict:
    sims = result.simulations
    idx = next(iter(sims.values())).index
    x = _x_labels(idx)

    def growth(r):
        return [1.0] + list(equity_curve(r).to_numpy())

    def dd(r):
        return [0.0] + list(drawdown(r).to_numpy())


    # el orden de las series en el 2.º gráfico está elegido para que los pares vecinos
    # sigan siendo distinguibles con daltonismo (validado con la paleta)
    baseline_order = ["buy_hold", "equal_weight", "momentum"] if focus == "ridge" else \
        ["momentum", "buy_hold", "equal_weight"]
    focus_set = [focus] + baseline_order

    ics = summary["ics"]
    ic_x = [f"{d:%Y-%m}" for d in ics.index]
    ic_series = [{"name": STRATEGY_LABELS[n], "short": SHORT_LABELS[n], "slot": SLOTS[n],
                  "values": [round(float(v), 6) for v in ics[n].fillna(0.0).cumsum()]} for n in STRATEGY_MODELS]
    return {
        "models": {"x": x, "series": _series(sims, list(STRATEGY_MODELS), growth), "fmt": "mult", "log": True,
                   "ref": 1.0},
        "focus": {"x": x, "series": _series(sims, focus_set, growth), "fmt": "mult", "log": True, "ref": 1.0},
        "drawdown": {"x": x, "series": _series(sims, focus_set, dd), "fmt": "pct", "log": False, "ref": 0.0,
                     "max0": True},
        "ic": {"x": ic_x, "series": ic_series, "fmt": "num", "log": False, "ref": 0.0},
    }


# --------------------------------------------------------------------------- secciones

def _verdict_section(summary: dict, focus: str) -> str:
    names = {
        "sharpe_total": "Sharpe neto mayor que el mejor baseline en todo el período",
        "subperiodos": "Le gana al mejor baseline en al menos 2 de 3 subperíodos",
        "ic_significativo": "IC con t-stat (Newey-West) de al menos 2",
    }
    cards = []
    for k in STRATEGY_MODELS:
        v = summary["verdicts"][k]
        status = ('<span class="status good">' + ICON_OK + "Pasa los 3 criterios</span>") if v["passed"] else \
            ('<span class="status bad">' + ICON_NO + "No pasa</span>")
        items = []
        for ck, label in names.items():
            ok = v["checks"][ck]
            extra = ""
            if ck == "subperiodos":
                extra = f" ({v['sub_wins']} de 3)"
            if ck == "ic_significativo":
                extra = f" (t = {_num(v['ic_t'], 1)})"
            icon = f'<span class="mini {"good" if ok else "bad"}">{ICON_OK if ok else ICON_NO}</span>'
            items.append(f"<li>{icon}<span>{_esc(label)}{extra}</span></li>")
        tag = '<span class="tag">usado en IB</span>' if k == focus else ""
        cards.append(
            f'<div class="card"><div class="card-head">{_key(k)}<strong>{_esc(STRATEGY_LABELS[k])}</strong>{tag}</div>'
            f"{status}<ul class=\"checks\">{''.join(items)}</ul></div>"
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def _strategy_table(summary: dict, focus: str) -> str:
    s = summary["strategy"]
    headers = ["Estrategia", "Retorno/año", "Volatilidad", "Sharpe", "Sortino", "Máx. caída", "Calmar",
               "Rotación/año", "Costo/año", "Meses +", "US$ 1 →"]
    rows, classes = [], []
    for k in list(STRATEGY_MODELS) + list(BASELINES):
        r = s.loc[k]
        rows.append([
            _key(k) + _esc(STRATEGY_LABELS[k]), _pct(r.cagr), _pct(r.volatility), _num(r.sharpe), _num(r.sortino),
            _pct(r.max_drawdown), _num(r.calmar), _pct(r.turnover_annual, 0), _pct(r.cost_annual, 2),
            _pct(r.pct_positive_months, 0), _num(r.growth_of_1) + "x",
        ])
        classes.append("focus" if k == focus else ("sep" if k == BASELINES[0] else ""))
    return _table(headers, rows, row_classes=classes)


def _model_table(summary: dict) -> str:
    m = summary["models"]
    headers = ["Modelo", "IC medio", "t-stat IC (NW)", "ICIR anual", "Meses con IC > 0", "AUC", "Log loss",
               "Acierto top-k"]
    rows = []
    for k in STRATEGY_MODELS:
        r = m.loc[k]
        rows.append([_key(k) + _esc(STRATEGY_LABELS[k]), _num(r.ic_mean, 3), _num(r.ic_tstat, 1),
                     _num(r.icir_annual), _pct(r.ic_pct_positive, 0), _num(r.auc, 3), _num(r.log_loss, 4),
                     _pct(r.hit_rate_top_k, 1)])
    return _table(headers, rows)


def _subperiod_table(summary: dict) -> str:
    sub = summary["subperiods"]
    headers = ["Estrategia"] + [str(c) for c in sub.columns]
    rows = [[_key(k) + _esc(STRATEGY_LABELS[k])] + [_num(v) for v in sub.loc[k]]
            for k in list(STRATEGY_MODELS) + list(BASELINES)]
    classes = ["sep" if k == BASELINES[0] else "" for k in list(STRATEGY_MODELS) + list(BASELINES)]
    return _table(headers, rows, row_classes=classes)


def _annual_table(summary: dict) -> str:
    a = summary["annual"]
    order = list(STRATEGY_MODELS) + list(BASELINES)
    headers = ["Año"] + [STRATEGY_LABELS[k] for k in order]
    rows = [[str(year)] + [_pct(a.loc[year, k]) for k in order] for year in a.index]
    return _table(headers, rows)


def _weights_table(result: BacktestResult) -> str:
    w = {k: v.mean() for k, v in result.walkforward.weights.items()}
    df = pd.DataFrame(w)
    df = df.reindex(df["ridge"].abs().sort_values(ascending=False).index)
    headers = ["Variable", "Ridge (‰ de retorno mensual)", "Logística (coef.)", "LightGBM (% ganancia)"]
    rows = [[_esc(FEATURE_LABELS.get(f, f)), _num(r.ridge * 1000, 2) + " ‰", _num(r.logistica, 3),
             _pct(r.lgbm, 1)] for f, r in df.iterrows()]
    return _table(headers, rows)


# --------------------------------------------------------------------------- página

CSS = """
:root{
  color-scheme:light;
  --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--wash:rgba(11,11,11,.04);
  --good:#006300;--bad:#d03b3b;--warn-bg:#fdf3dc;--warn-ink:#5c4300;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
    --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--wash:rgba(255,255,255,.05);
    --good:#0ca30c;--bad:#e66767;--warn-bg:#3a2f12;--warn-ink:#f3d58a;
    --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--wash:rgba(255,255,255,.05);
  --good:#0ca30c;--bad:#e66767;--warn-bg:#3a2f12;--warn-ink:#f3d58a;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;line-height:1.25;margin:0 0 6px;font-weight:650}
h2{font-size:18px;margin:40px 0 4px;font-weight:650}
.lede{color:var(--ink-2);margin:0 0 4px}
.meta{color:var(--muted);font-size:13px;margin:0}
.note{color:var(--ink-2);font-size:14px;margin:0 0 14px;max-width:760px}
.banner{background:var(--warn-bg);color:var(--warn-ink);border-radius:10px;padding:12px 14px;margin:18px 0 0;font-size:14px}
.panel{background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring);padding:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:12px}
.card{background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring);padding:14px}
.card-head{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.tag{margin-left:auto;font-size:12px;color:var(--ink-2);background:var(--wash);border-radius:6px;padding:1px 6px}
.status{display:inline-flex;align-items:center;gap:6px;font-weight:600;font-size:14px}
.status.good{color:var(--good)}.status.bad{color:var(--bad)}.status.warn{color:var(--warn-ink)}
.icon{width:16px;height:16px;flex:none}
.checks{list-style:none;padding:0;margin:10px 0 0;font-size:13px;color:var(--ink-2)}
.checks li{display:flex;gap:6px;align-items:flex-start;margin:4px 0}
.mini{display:inline-flex;margin-top:2px}.mini.good{color:var(--good)}.mini.bad{color:var(--bad)}
.mini .icon{width:14px;height:14px}
.key{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:8px;vertical-align:middle;flex:none}
.table-wrap{overflow-x:auto;margin-top:12px;background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring)}
table.data{border-collapse:collapse;width:100%;font-size:13px}
table.data th{text-align:right;font-weight:600;color:var(--ink-2);padding:9px 12px;border-bottom:1px solid var(--grid);vertical-align:bottom;min-width:7ch}
table.data th.txt{text-align:left}
table.data td{padding:7px 12px;border-bottom:1px solid var(--grid);white-space:nowrap}
table.data tr:last-child td{border-bottom:0}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.txt{text-align:left}
tr.focus td{background:var(--wash);font-weight:600}
tr.sep td{border-top:2px solid var(--axis)}
.chart{margin-top:12px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink-2);margin:0 0 6px}
.legend span{display:inline-flex;align-items:center}
.plot{position:relative}
.plot svg{display:block;width:100%;height:auto;overflow:visible}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.endlabel{fill:var(--ink-2);font-size:12px}
.tip{position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);border-radius:8px;
  box-shadow:0 0 0 1px var(--ring),0 4px 14px rgba(0,0,0,.12);padding:8px 10px;font-size:12px;display:none;min-width:150px;z-index:2}
.tip .t-date{color:var(--ink-2);margin-bottom:4px}
.tip .t-row{display:flex;align-items:center;gap:6px;justify-content:space-between;font-variant-numeric:tabular-nums}
.tip .t-row span:first-child{display:inline-flex;align-items:center}
ul.plain{color:var(--ink-2);font-size:14px;padding-left:20px;max-width:820px}
ul.plain li{margin:4px 0}
code{font-family:ui-monospace,Consolas,monospace;font-size:13px;background:var(--wash);padding:1px 4px;border-radius:4px}
"""

JS = r"""
const SPECS = __SPECS__;
const LOCALE = 'es-CL';
const FMT = {
  usd: v => 'US$ ' + Math.round(v).toLocaleString(LOCALE),
  clp: v => '$' + Math.round(v).toLocaleString(LOCALE),
  mult: v => v.toLocaleString(LOCALE,{minimumFractionDigits:2,maximumFractionDigits:2}) + 'x',
  pct: v => (v*100).toLocaleString(LOCALE,{minimumFractionDigits:1,maximumFractionDigits:1}) + ' %',
  num: v => v.toLocaleString(LOCALE,{minimumFractionDigits:2,maximumFractionDigits:2}),
};
const TICK = {
  usd: v => 'US$ ' + (Math.abs(v) >= 1000 ? (v / 1000).toLocaleString(LOCALE,{maximumFractionDigits:1}) + ' mil' : Math.round(v)),
  clp: v => '$' + (Math.abs(v) >= 1e6 ? (v / 1e6).toLocaleString(LOCALE,{maximumFractionDigits:1}) + ' M'
    : Math.abs(v) >= 1000 ? (v / 1000).toLocaleString(LOCALE,{maximumFractionDigits:0}) + ' mil' : Math.round(v)),
  mult: v => v.toLocaleString(LOCALE,{maximumFractionDigits:2}) + 'x',
  pct: v => Math.round(v*100) + ' %',
  num: v => v.toLocaleString(LOCALE,{maximumFractionDigits:1}),
};
function linTicks(min, max, count){
  const span = (max - min) || Math.abs(max) || 1;
  const raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), e = raw / mag;
  const step = mag * (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1);
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step, t = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) t.push(+v.toFixed(10));
  return {lo, hi, ticks: t};
}
function logTicks(min, max){
  const pick = ms => { const c = []; for (let e = -3; e <= 4; e++) for (const m of ms) c.push(m * Math.pow(10, e));
    return c.filter(v => v >= min * 0.999 && v <= max * 1.001); };
  let t = pick([1, 2, 5]);
  if (t.length < 3) t = pick([1, 1.5, 2, 3, 4, 5, 6, 8]);
  while (t.length > 7) t = t.filter((_, i) => i % 2 === 0);
  return t;
}
function monthLabel(s){
  const [y, m] = s.split('-').map(Number);
  return new Date(y, m - 1, 1).toLocaleDateString(LOCALE, {month: 'short', year: 'numeric'});
}
function render(el, spec){
  const W = Math.max(320, el.clientWidth), H = W < 560 ? 230 : 280;
  const all = spec.series.flatMap(s => s.values).filter(v => Number.isFinite(v));
  const endVals = spec.series.map(s => s.values[s.values.length - 1]);
  // etiquetas directas al final solo con espacio y si no chocan (si chocan: leyenda + tooltip)
  const labelsFit = W >= 560 && spec.series.length <= 4;
  const m = {t: 10, r: labelsFit ? 124 : 12, b: 26, l: 50};
  const ph = H - m.t - m.b, n = spec.x.length;
  let ymin = Math.min(...all, spec.ref), ymax = Math.max(...all, spec.ref);
  if (spec.max0) ymax = 0;
  let yScale, ticks;
  if (spec.log){
    const lmin = Math.log(ymin) - 0.04, lmax = Math.log(ymax) + 0.04;
    yScale = v => m.t + ph - (Math.log(v) - lmin) / (lmax - lmin) * ph;
    ticks = logTicks(Math.exp(lmin), Math.exp(lmax));
  } else {
    const r = linTicks(ymin, ymax, 5);
    if (spec.max0) r.hi = 0;
    yScale = v => m.t + ph - (v - r.lo) / (r.hi - r.lo) * ph;
    ticks = r.ticks.filter(v => v <= r.hi + 1e-12);
  }
  let showLabels = labelsFit;
  if (showLabels){
    const ys = endVals.map(v => yScale(v)).sort((a, b) => a - b);
    for (let i = 1; i < ys.length; i++) if (ys[i] - ys[i - 1] < 15) showLabels = false;
  }
  const r2 = showLabels ? m.r : 12;
  const pw2 = W - m.l - r2;
  const X = i => m.l + (n === 1 ? 0 : i * pw2 / (n - 1));
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${spec.title}">`;
  for (const t of ticks){
    const y = yScale(t);
    const isRef = Math.abs(t - spec.ref) < 1e-9;
    svg += `<line x1="${m.l}" x2="${m.l + pw2}" y1="${y}" y2="${y}" stroke="var(${isRef ? '--axis' : '--grid'})" stroke-width="1"/>`;
    svg += `<text class="tick" x="${m.l - 8}" y="${y + 4}" text-anchor="end">${TICK[spec.fmt](t)}</text>`;
  }
  if (!ticks.some(t => Math.abs(t - spec.ref) < 1e-9)){
    const y = yScale(spec.ref);
    svg += `<line x1="${m.l}" x2="${m.l + pw2}" y1="${y}" y2="${y}" stroke="var(--axis)" stroke-width="1"/>`;
  }
  // eje x: fechas "AAAA-MM" (marca cada enero) o meses transcurridos (spec.xkind = "months", marca cada año)
  const years = [];
  if (spec.xkind === 'months') spec.x.forEach((s, i) => { if ((i + 1) % 12 === 0) years.push(i); });
  else spec.x.forEach((s, i) => { if (s.endsWith('-01')) years.push(i); });
  const every = Math.max(1, Math.ceil(years.length / (W < 560 ? 4 : 8)));
  years.filter((_, k) => k % every === 0).forEach(i => {
    const label = spec.xkind === 'months' ? `${(i + 1) / 12} ${i === 11 ? 'año' : 'años'}` : spec.x[i].slice(0, 4);
    svg += `<text class="tick" x="${X(i)}" y="${H - 6}" text-anchor="middle">${label}</text>`;
  });
  for (const s of spec.series){
    const d = s.values.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${yScale(v).toFixed(1)}`).join('');
    svg += `<path d="${d}" fill="none" stroke="var(--s${s.slot})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  }
  spec.series.forEach(s => {
    const v = s.values[n - 1], y = yScale(v);
    svg += `<circle cx="${X(n - 1)}" cy="${y}" r="4" fill="var(--s${s.slot})" stroke="var(--surface)" stroke-width="2"/>`;
    if (showLabels) svg += `<text class="endlabel" x="${X(n - 1) + 9}" y="${y + 4}">${s.short} ${FMT[spec.fmt](v)}</text>`;
  });
  svg += `<line class="xhair" x1="0" x2="0" y1="${m.t}" y2="${m.t + ph}" stroke="var(--axis)" stroke-width="1" visibility="hidden"/>`;
  svg += `<g class="hdots" visibility="hidden">` + spec.series.map(s =>
    `<circle r="4" fill="var(--s${s.slot})" stroke="var(--surface)" stroke-width="2"/>`).join('') + `</g>`;
  svg += `<rect class="hit" x="${m.l}" y="${m.t}" width="${pw2}" height="${ph}" fill="transparent"/>`;
  svg += `</svg>`;
  const legend = spec.series.map(s => `<span><span class="key" style="background:var(--s${s.slot})"></span>${s.name}</span>`).join('');
  el.innerHTML = `<div class="legend">${legend}</div><div class="plot">${svg}<div class="tip"></div></div>`;

  const plot = el.querySelector('.plot'), svgEl = plot.querySelector('svg'), tip = plot.querySelector('.tip');
  const xh = svgEl.querySelector('.xhair'), dots = svgEl.querySelectorAll('.hdots circle'), g = svgEl.querySelector('.hdots');
  const hit = svgEl.querySelector('.hit');
  const move = ev => {
    const rect = svgEl.getBoundingClientRect();
    const px = (ev.clientX - rect.left) * (W / rect.width);
    const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / pw2 * (n - 1))));
    const x = X(i);
    xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.setAttribute('visibility', 'visible');
    g.setAttribute('visibility', 'visible');
    spec.series.forEach((s, k) => { dots[k].setAttribute('cx', x); dots[k].setAttribute('cy', yScale(s.values[i])); });
    const rows = spec.series.map(s => ({s, v: s.values[i]})).sort((a, b) => b.v - a.v)
      .map(({s, v}) => `<div class="t-row"><span><span class="key" style="background:var(--s${s.slot})"></span>${s.name}</span><span>${FMT[spec.fmt](v)}</span></div>`).join('');
    tip.innerHTML = `<div class="t-date">${spec.xkind === 'months' ? spec.x[i] : monthLabel(spec.x[i])}</div>${rows}`;
    tip.style.display = 'block';
    const scale = rect.width / W, tw = tip.offsetWidth;
    let left = x * scale + 14;
    if (left + tw > rect.width) left = x * scale - tw - 14;
    tip.style.left = Math.max(0, left) + 'px';
    tip.style.top = (m.t * scale + 6) + 'px';
  };
  const leave = () => { tip.style.display = 'none'; xh.setAttribute('visibility', 'hidden'); g.setAttribute('visibility', 'hidden'); };
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerdown', move);
  hit.addEventListener('pointerleave', leave);
}
document.querySelectorAll('[data-chart]').forEach(el => {
  const spec = SPECS[el.dataset.chart];
  spec.title = el.dataset.title || '';
  let w = 0;
  const draw = () => { if (el.clientWidth !== w) { w = el.clientWidth; render(el, spec); } };
  draw();
  if ('ResizeObserver' in window) new ResizeObserver(draw).observe(el);
});
"""


def _chart(key: str, title: str) -> str:
    return f'<div class="panel chart" data-chart="{key}" data-title="{_esc(title)}"></div>'


def build_html(result: BacktestResult, summary: dict, cfg: Config, focus: str) -> str:
    sims = result.simulations
    idx = next(iter(sims.values())).index
    folds = result.walkforward.folds
    first_hold, last_hold = idx[0].to_period("M") + 1, idx[-1].to_period("M") + 1
    c = cfg.costs
    universe = ", ".join(sorted({t for t in result.walkforward.predictions.index.get_level_values("ticker")}))
    synthetic = result.source == "sintetico"

    banner = ""
    if synthetic:
        banner = ('<div class="banner"><strong>Datos sintéticos.</strong> Este reporte solo sirve para comprobar que '
                  'el código funciona. Los precios son inventados: estos números no dicen nada del mercado real.</div>')

    specs = _chart_specs(result, summary, focus)
    js = JS.replace("__SPECS__", json.dumps(specs, ensure_ascii=False))
    focus_label = STRATEGY_LABELS[focus]

    body = f"""
<main>
  <h1>Backtest walk-forward: rotación mensual de ETFs</h1>
  <p class="lede">Cada mes se compran los {cfg.strategy.top_k} ETFs con mejor puntaje
  ({'partes iguales' if cfg.strategy.weighting == 'igual' else 'peso inverso a la volatilidad'}).
  Todo lo que se muestra es fuera de muestra y después de costos.</p>
  <p class="meta">Período evaluado: {first_hold} a {last_hold} ({len(idx)} meses) ·
  {len(folds)} reentrenamientos · primer entrenamiento con {folds[0].train_months} meses ·
  fuente: {'sintética' if synthetic else 'Yahoo Finance'}</p>
  <p class="meta">Universo: {_esc(universe)}</p>
  {banner}

  <h2>Veredicto</h2>
  <p class="note">Un modelo solo merece dinero real si le gana a los baselines simples fuera de muestra y de forma
  consistente. Los tres criterios tienen que cumplirse a la vez.</p>
  {_verdict_section(summary, focus)}

  <h2>Resultado de cada estrategia</h2>
  <p class="note">Sharpe y Sortino con tasa libre de riesgo de {_pct(cfg.risk_free_annual)}. La rotación anual es la
  fracción de la cartera que se compra y vende en un año.</p>
  {_strategy_table(summary, focus)}

  <h2>Crecimiento de 1 dólar: los cuatro modelos</h2>
  <p class="note">Escala logarítmica: la misma distancia vertical es el mismo porcentaje de ganancia.</p>
  {_chart("models", "Crecimiento de 1 dólar por modelo")}

  <h2>{_esc(focus_label)} contra los baselines</h2>
  <p class="note">El modelo configurado en <code>[ib].modelo</code> frente a comprar y mantener
  {_esc(result.benchmark)}, igual peso en todos los ETFs y momentum 12-1 sin machine learning.</p>
  {_chart("focus", "Crecimiento de 1 dólar contra baselines")}

  <h2>Caídas desde el máximo</h2>
  <p class="note">Cuánto llegó a perder cada estrategia desde su punto más alto. Esto es lo que hay que aguantar.</p>
  {_chart("drawdown", "Caídas desde el máximo")}

  <h2>Calidad de las predicciones</h2>
  <p class="note">IC: correlación de Spearman entre el puntaje y el retorno del mes siguiente, calculada mes a mes.
  Sobre 0,05 de forma estable ya es útil. AUC 0,5 = azar. Log loss de referencia (predecir siempre la tasa base):
  {_num(summary["base_log_loss"], 4)}; un modelo útil queda por debajo.</p>
  {_model_table(summary)}
  {_chart("ic", "IC acumulado por modelo")}

  <h2>Consistencia por subperíodo (Sharpe neto)</h2>
  <p class="note">Si un modelo gana solo en un tramo, probablemente fue suerte o un régimen de mercado puntual.</p>
  {_subperiod_table(summary)}

  <h2>Retornos por año</h2>
  {_annual_table(summary)}

  <h2>Qué variables usa cada modelo</h2>
  <p class="note">Promedio entre reentrenamientos. Coeficientes lineales sobre variables estandarizadas
  (positivo = más de esa variable, mejor puntaje); en LightGBM, el porcentaje de la ganancia total.</p>
  {_weights_table(result)}

  <h2>Supuestos y advertencias</h2>
  <ul class="plain">
    <li>Las features usan el cierre del último día hábil del mes; la compra se simula al cierre del día hábil siguiente.</li>
    <li>Costos: comisión de {_num(c.commission_bps, 1)} bps con mínimo de US$ {_num(c.min_commission_usd)} por orden,
      slippage de {_num(c.slippage_bps, 1)} bps por lado, banda de rebalanceo de US$ {_num(c.rebalance_band_usd, 0)}
      y capital inicial de US$ {_num(c.initial_capital_usd, 0)}.</li>
    <li>No incluye impuestos, retención sobre dividendos, costo de cambiar pesos a dólares ni transferencias SWIFT.</li>
    <li>Los precios están ajustados por dividendos: se asume que se reinvierten completos.</li>
    <li>Se comparan cuatro modelos y tres baselines. Mientras más variantes pruebes, más probable es que alguna
      se vea bien por suerte: si ajustas parámetros mirando este reporte, exige más que estos criterios.</li>
    <li>El universo se eligió hoy con ETFs que existen hoy, lo que introduce un sesgo de supervivencia leve.</li>
    <li>Un buen resultado histórico no garantiza resultados futuros. Esto no es una recomendación de inversión.</li>
  </ul>
</main>"""
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Backtest ETFs</title><style>{CSS}</style></head><body>{body}"
            f"<script>{js}</script></body></html>")


def write_outputs(result: BacktestResult, summary: dict, cfg: Config, out_dir: Path, focus: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "reporte.html"
    report.write_text(build_html(result, summary, cfg, focus), encoding="utf-8")

    sims = result.simulations
    pd.DataFrame({k: v["net"] for k, v in sims.items()}).to_csv(
        out_dir / "retornos_mensuales.csv", index_label="fecha_senal", encoding="utf-8")
    summary["strategy"].to_csv(out_dir / "metricas_estrategias.csv", index_label="estrategia", encoding="utf-8")
    summary["models"].to_csv(out_dir / "metricas_modelos.csv", index_label="modelo", encoding="utf-8")
    pred = result.walkforward.predictions[list(STRATEGY_MODELS) + ["fwd_ret", "excess", "beat"]]
    pred.to_csv(out_dir / "predicciones.csv", encoding="utf-8")
    holdings = pd.DataFrame({k: sims[k]["holdings"] for k in list(MODEL_NAMES) + ["ensamble", "momentum"]})
    holdings.to_csv(out_dir / "carteras.csv", index_label="fecha_senal", encoding="utf-8")
    return report
