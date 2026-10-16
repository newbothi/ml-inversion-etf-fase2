"""Reporte HTML del simulador de aportes (mismo estilo y motor de gráficos que los otros reportes)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report import CSS, JS, _chart, _esc, _num, _pct, _table
from .savings import SavingsResult, frequency_label


def _clp(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else "$" + _num(x, 0)


def _usd(x: float) -> str:
    return "US$ " + _num(x, 0)


def _money(res: SavingsResult, x: float) -> str:
    return _clp(x) if res.main.unit == "CLP" else _usd(x)


def _specs(res: SavingsResult) -> dict:
    ws = res.main_windows
    months = ws.paths.shape[1]
    fmt = "clp" if res.main.unit == "CLP" else "usd"
    x = [f"Mes {j + 1} (año {j // 12 + 1})" for j in range(months)]

    def pct(q):
        return [round(float(v), 2) for v in np.percentile(ws.paths, q, axis=0)]

    fan = {"x": x, "xkind": "months", "fmt": fmt, "log": False, "ref": 0.0, "series": [
        {"name": "Típico (mediana)", "short": "Típico", "slot": 1, "values": pct(50)},
        {"name": "Malo (10 % de los casos)", "short": "Malo", "slot": 2, "values": pct(10)},
        {"name": "Bueno (90 % de los casos)", "short": "Bueno", "slot": 3, "values": pct(90)},
        {"name": "Lo aportado", "short": "Aportado", "slot": 4,
         "values": [round(float(v), 2) for v in ws.contributed_path]},
    ]}
    start = {"x": [str(s) for s in ws.starts], "fmt": "mult", "log": False, "ref": 1.0, "series": [
        {"name": f"Valor final / lo aportado ({res.main_years} años)", "short": "Final", "slot": 1,
         "values": [round(float(v), 4) for v in ws.final / ws.contributed]},
    ]}
    return {"fan": fan, "start": start}


def _worst_window_text(res: SavingsResult) -> str:
    ws = res.main_windows
    i = int(np.argmin(ws.final / ws.contributed))
    return f"partió en {ws.starts[i]} y terminó en {ws.starts[i] + res.main_years * 12 - 1}"


def _fx_note(res: SavingsResult) -> str:
    if res.main.unit != "CLP":
        return ""
    return (" En pesos también pesa cómo está el dólar al final: si el peso se fortalece justo cuando terminas, tus"
            " dólares valen menos pesos, y al revés.")


def _summary_cards(res: SavingsResult) -> str:
    s = res.by_horizon.loc[res.main_years]
    cards = [
        ("Aportas en total", _money(res, s["aportado"]), f"{_clp(res.cfg.monthly_clp)} al mes durante {res.main_years} años"),
        ("Resultado típico", _money(res, s["mediana"]), "la mitad de los casos terminó con más que esto"),
        ("Peor caso de la historia", _money(res, s["peor"]), _worst_window_text(res)),
        ("Terminó perdiendo", _pct(s["pct_perdida"], 0), f"de {int(s['ventanas'])} puntos de partida posibles"),
    ]
    html = "".join(f'<div class="card"><p class="stat-label">{_esc(a)}</p><p class="stat-value">{_esc(b)}</p>'
                   f'<p class="card-note">{_esc(c)}</p></div>' for a, b, c in cards)
    return f'<div class="cards">{html}</div>'


def _horizon_table(res: SavingsResult) -> str:
    h = res.by_horizon
    rows = [[f"{int(y)} años", _money(res, r["aportado"]), _money(res, r["peor"]), _money(res, r["p10"]),
             _money(res, r["mediana"]), _money(res, r["p90"]), _pct(r["pct_perdida"], 0),
             _pct(r["rentabilidad_mediana"], 1), _pct(r["peor_momento"], 0)] for y, r in h.iterrows()]
    rent = "Rentabilidad real típica/año" if res.main.real else "Rentabilidad típica/año"
    return _table(["Plazo", "Aportado", "Peor caso", "Malo (10 %)", "Típico", "Bueno (90 %)", "Terminó perdiendo",
                   rent, "Peor momento"], rows,
                  row_classes=["focus" if int(y) == res.main_years else "" for y in h.index])


def _comparison_section(res: SavingsResult) -> str:
    c = res.comparison
    if c is None or c.empty:
        return ""
    rows, classes, previous = [], [], None
    for r in c.itertuples():
        rows.append([_esc(r.escenario), f"{r.anios} años", str(r.independientes), _pct(r.pct_perdida, 0),
                     _num(r.peor_x) + "x", _num(r.p10_x) + "x", _num(r.mediana_x) + "x",
                     _pct(r.rentabilidad_real, 1), _pct(r.peor_momento, 0)])
        classes.append("sep" if previous is not None and r.escenario != previous else "")
        previous = r.escenario
    table = _table(["Escenario", "Plazo", "Períodos independientes", "Terminó perdiendo", "Peor caso", "Malo (10 %)",
                    "Típico", "Rentabilidad real típica/año", "Peor momento"], rows, row_classes=classes,
                   text_cols={0})
    return f"""
  <h2>¿Y si el futuro se parece a otra época?</h2>
  <p class="note">2004–2026 es un solo período, y bastante bueno. Esta tabla repite la simulación con otras historias,
  todas en dólares de hoy (descontando la inflación de EE.UU.) y como múltiplo de lo aportado: 1,00x es recuperar
  exactamente lo que pusiste. "Períodos independientes" es cuántos plazos completos sin superponerse caben en cada
  historia: con pocos, los porcentajes son poco confiables. Japón desde 1990 es el ejemplo clásico de un mercado que
  tardó décadas en recuperarse.</p>
  {table}"""


def _goal_section(res: SavingsResult) -> str:
    g = res.goal
    if g is None or g.empty:
        return ""
    conf = _pct(res.confidence, 0)
    rows = [[_esc(r.escenario), _clp(r.tipico), _clp(r.confianza), _clp(r.peor),
             _pct(r.prob_con_aporte_actual, 0)] for r in g.itertuples()]
    table = _table(["Si el futuro se parece a…", "Típico (50 %)", f"Con {conf} de seguridad", "Incluso en el peor caso",
                    f"Con {_clp(res.cfg.monthly_clp)} al mes llegas en"], rows, text_cols={0},
                   row_classes=["focus"] + [""] * (len(rows) - 1))
    return f"""
  <h2>Tu meta: {_clp(res.goal_target)} de hoy en {res.main_years} años</h2>
  <p class="note">Cuánto tendrías que aportar al mes, en pesos de hoy, para llegar a la meta según cada historia.
  "Con {conf} de seguridad" significa que en 9 de cada 10 puntos de partida se llegó. Planificar con el caso típico
  es arriesgado: la mitad de las veces no alcanza.</p>
  {table}"""


def _costs_table(res: SavingsResult) -> str:
    c = res.costs_table
    rows = [[_esc(r["esquema"]), _money(res, r["costos"]), _pct(r["costos_pct"], 1), _money(res, r["mediana"]),
             ("+" if r["vs_btg"] >= 0 else "−") + _money(res, abs(r["vs_btg"]))] for _, r in c.iterrows()]
    return _table(["Cómo inviertes", "Costos totales", "% de lo aportado", "Valor típico al final",
                   "Diferencia vs BTG mensual"], rows, row_classes=["focus"] + [""] * (len(rows) - 1))


def _amounts_table(res: SavingsResult) -> str:
    a = res.amounts_table
    rows = []
    for r in a.itertuples():
        best = "BTG" if r.btg_mediana >= r.ib_mediana else "IB"
        rows.append([_clp(r.aporte_clp), _usd(r.aporte_usd),
                     f"{frequency_label(r.btg_mejor)} · {_money(res, r.btg_costos)}",
                     f"{frequency_label(r.ib_mejor)} · {_money(res, r.ib_costos)}",
                     ("+" if r.btg_mediana >= r.ib_mediana else "−") + _money(res, abs(r.btg_mediana - r.ib_mediana)),
                     best])
    return _table(["Aporte mensual", "En dólares", "Mejor en BTG · costos", "Mejor en IB · costos",
                   "BTG deja de más", "Conviene"], rows, text_cols={2, 3, 5})


EXTRA_CSS = """
.stat-label{color:var(--ink-2);font-size:13px;margin:0}
.stat-value{font-size:26px;font-weight:650;margin:4px 0 0}
.card-note{color:var(--muted);font-size:12px;margin:8px 0 0}
"""


def build_savings_html(res: SavingsResult, synthetic: bool = False) -> str:
    cfg, m = res.cfg, res.main
    js = JS.replace("__SPECS__", json.dumps(_specs(res), ensure_ascii=False))
    p = res.portfolio
    proxy = " + ".join(f"{_pct(w, 0)} {t}" for t, w in p.history.items())
    buy = " y ".join(f"{t} ({_pct(w, 0)})" if len(p.orders) > 1 else t for t, w in p.orders.items())
    banners = []
    if synthetic:
        banners.append("<strong>Datos sintéticos.</strong> Este reporte solo sirve para ver el formato.")
    banners += [_esc(n) for n in res.notes]
    banner = "".join(f'<div class="banner">{b}</div>' for b in banners)
    unit_text = ("en pesos de hoy: cada aporte se convierte con el dólar observado de ese mes y todo se ajusta por "
                 "inflación con la UF" if m.unit == "CLP" else "en dólares, sin ajustar por inflación")
    breakeven = (f"Con estos supuestos, IB empieza a convenir desde unos {_clp(res.breakeven_clp)} al mes."
                 if res.breakeven_clp else
                 "Con estos supuestos, IB no le gana a BTG con ningún aporte mensual de hasta $5.000.000.")
    real_word = "real " if m.real else ""

    body = f"""
<main>
  <h1>Plan de aportes mensuales</h1>
  <p class="lede">¿Qué habría pasado si hubieras invertido {_clp(cfg.monthly_clp)} de hoy todos los meses en una cartera
  {_esc(p.label)}, partiendo en cualquier mes desde {m.returns.index[0]}? Resultados {unit_text}, después de costos e
  impuesto sobre dividendos.</p>
  <p class="meta">Historia: {m.returns.index[0]} a {m.returns.index[-1]} ({len(m.returns)} meses) · cartera histórica:
  {_esc(proxy)} · lo que comprarías hoy: {_esc(buy)} · dólar de hoy: {_clp(cfg.usd_clp)}</p>
  {banner}

  <h2>A {res.main_years} años, comprando todos los meses en BTG</h2>
  {_summary_cards(res)}
  {_goal_section(res)}

  <h2>Cómo evoluciona la plata mes a mes</h2>
  <p class="note">Las tres líneas de colores resumen todos los puntos de partida posibles: el caso típico, el malo
  (solo 1 de cada 10 fue peor) y el bueno (solo 1 de cada 10 fue mejor). Cuando una línea queda bajo "lo aportado",
  ibas perdiendo en ese momento.</p>
  {_chart("fan", "Valor de la cartera mes a mes")}

  <h2>Resultado según cuándo empezaste</h2>
  <p class="note">Cada punto es alguien que partió ese mes y aportó {res.main_years} años. Sobre 1,0x terminó ganando.
  Los peores resultados no son de quienes partieron antes de una crisis, sino de quienes <strong>terminaron</strong> en
  una caída: con aportes mensuales, lo que más pesa es cómo está el mercado al final, cuando ya juntaste más plata. Por eso
  es habitual, en los últimos años antes de necesitar la plata, ir pasándola de a poco a algo más estable.{_fx_note(res)}</p>
  {_chart("start", "Valor final dividido por lo aportado, según el mes de partida")}

  <h2>¿Cuánto tiempo hace falta?</h2>
  <p class="note">"Peor momento" es la peor pérdida que alguien vio en el camino, comparando lo que tenía con lo que
  había aportado hasta ese día. Es lo que hay que aguantar sin vender. La rentabilidad {real_word}típica es la tasa
  anual que convierte tus aportes en el resultado final.</p>
  {_horizon_table(res)}
  {_comparison_section(res)}

  <h2>BTG o Interactive Brokers con este aporte ({res.main_years} años)</h2>
  <p class="note">En IB cada envío es una transferencia SWIFT con comisión fija, así que hay que juntar la plata y
  enviarla cada varios meses, y mientras espera en el banco no está invertida. En BTG cada compra cuesta US$ 1,
  así que con montos chicos también puede convenir juntar algunos meses.</p>
  {_costs_table(res)}

  <h2>¿Y con otros montos? ¿Cada cuánto comprar?</h2>
  <p class="note">{breakeven} Con aportes chicos, la comisión fija de cada compra pesa más: juntar 2 o 3 meses y
  comprar de una vez puede dejar un poco más de plata. Con aportes grandes da lo mismo.</p>
  {_amounts_table(res)}

  <h2>Supuestos</h2>
  <ul class="plain">
    <li>Pesos de hoy: el aporte es el mismo en UF todos los meses (o sea, sube con la inflación) y cada resultado se
      lleva a pesos de hoy con la UF. El dólar observado y la UF vienen del Banco Central (vía mindicador.cl).</li>
    <li>Historia larga: acciones de EE.UU. desde 1926 y de Japón desde 1990 (base de datos de Kenneth French, en dólares),
      descontando la inflación de EE.UU. (FRED). Allí la cartera "conservadora" usa letras del Tesoro en vez de bonos.</li>
    <li>Cambiar pesos a dólares cuesta {_pct(cfg.fx_spread, 1)} (supuesto: compara el dólar que te dan con el observado).</li>
    <li>BTG: US$ {_num(cfg.btg_order_fee)} por orden y orden mínima de US$ {_num(cfg.btg_min_order, 0)} ("desde US$1"
      según su sitio, así que órdenes grandes podrían costar más). IB: US$ {_num(cfg.ib_per_share, 3)} por acción,
      mínimo US$ {_num(cfg.ib_min_fee)} y máximo {_pct(cfg.ib_max_pct, 0)} de la orden, más US$ {_num(cfg.ib_swift_fee, 0)}
      por transferencia SWIFT. La plata que espera la transferencia gana {_pct(cfg.waiting_rate, 1)} al año.</li>
    <li>EE.UU. retiene 15 % de los dividendos (convenio Chile–EE.UU. vigente desde 2024): unos
      {_pct(cfg.dividend_drag, 2)} al año menos de rentabilidad.</li>
    <li>No incluye el impuesto a la renta en Chile por ganancias y dividendos. Consulta con un contador.</li>
    <li>Los resultados pasados no garantizan los futuros. Esto no es una recomendación de inversión.</li>
  </ul>
</main>"""
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Plan de aportes</title><style>{CSS}{EXTRA_CSS}</style></head><body>{body}"
            f"<script>{js}</script></body></html>")


def write_savings_outputs(res: SavingsResult, out_dir: Path, synthetic: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "reporte.html"
    report.write_text(build_savings_html(res, synthetic), encoding="utf-8")
    res.by_horizon.to_csv(out_dir / "resultados_por_plazo.csv", index_label="anios", encoding="utf-8")
    res.costs_table.to_csv(out_dir / "costos_btg_vs_ib.csv", index_label="esquema_id", encoding="utf-8")
    res.amounts_table.to_csv(out_dir / "comparacion_por_monto.csv", index=False, encoding="utf-8")
    if res.comparison is not None:
        res.comparison.to_csv(out_dir / "escenarios.csv", index=False, encoding="utf-8")
    if res.goal is not None:
        res.goal.to_csv(out_dir / "meta.csv", index=False, encoding="utf-8")
    ws = res.main_windows
    pd.DataFrame({"inicio": [str(s) for s in ws.starts], "aportado": ws.contributed, "final": ws.final,
                  "costos": ws.costs, "rentabilidad_anual": ws.irr, "peor_momento": ws.worst_moment}) \
        .to_csv(out_dir / "ventanas.csv", index=False, encoding="utf-8")
    return report
