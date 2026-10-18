# ml-inversion-etf

[![Tests](https://github.com/newbothi/ml-inversion-etf-fase2/actions/workflows/tests.yml/badge.svg)](https://github.com/newbothi/ml-inversion-etf-fase2/actions/workflows/tests.yml)

Rotación mensual de ETFs con machine learning, evaluada con walk-forward y conectada a la cuenta **paper** de Interactive Brokers.

Cada fin de mes, tres modelos (Ridge, regresión logística y LightGBM) y su ensamble puntúan 14 ETFs líquidos según qué tan probable es que le ganen al promedio del grupo el mes siguiente. La estrategia compra los 3 mejores. Todo se compara contra baselines sin machine learning: comprar y mantener SPY, igual peso en todos y momentum 12-1.

Con datos reales, **ningún modelo le ganó a los baselines** (ver [Resultados](#resultados)). La fase 3 cambió la pregunta y probó un filtro de tendencia, con un registro de cada experimento y el Sharpe deflactado para no confundir suerte con habilidad: cortó la caída máxima casi a la mitad, pero tampoco demostró ser mejor que comprar y mantener SPY.

> Esto es un proyecto de aprendizaje, no una recomendación de inversión. Un buen backtest no garantiza resultados futuros.

## Resultados

Datos de Yahoo Finance hasta el 15 de octubre de 2026. Todo fuera de muestra y después de costos, partiendo con US$ 10.000.

### Rotación de ETFs con machine learning (2010–2026)

| Estrategia | Retorno anual | Volatilidad | Sharpe | Máx. caída | Costo anual |
|---|---:|---:|---:|---:|---:|
| Ridge | 8,3 % | 14,1 % | 0,64 | −20,3 % | 1,1 % |
| Logística | 7,8 % | 15,9 % | 0,55 | −23,3 % | 0,9 % |
| LightGBM | 6,6 % | 15,2 % | 0,50 | −25,2 % | 1,1 % |
| Ensamble | 8,4 % | 14,8 % | 0,62 | −19,4 % | 1,0 % |
| **Comprar y mantener SPY** | **14,3 %** | 15,3 % | **0,95** | −23,8 % | 0,0 % |
| Igual peso (14 ETFs) | 10,6 % | 12,0 % | 0,91 | −19,6 % | 0,1 % |
| Momentum 12-1 | 11,9 % | 14,7 % | 0,84 | −21,9 % | 0,5 % |

| Modelo | IC medio | t-stat del IC (Newey-West) | AUC |
|---|---:|---:|---:|
| Ridge | −0,005 | −0,2 | 0,494 |
| Logística | 0,019 | 0,7 | 0,501 |
| LightGBM | −0,004 | −0,2 | 0,498 |
| Ensamble | 0,005 | 0,2 | 0,497 |

**Veredicto: ningún modelo pasa los criterios.**

- **Los modelos no predicen.** El IC (correlación entre el puntaje y el retorno del mes siguiente) es prácticamente 0 y el AUC es 0,50: como lanzar una moneda. Ningún t-stat se acerca a 2.
- **Su ~8 % anual viene de estar invertidos, no de elegir bien.** Quedan unos 6 puntos por año bajo SPY y pagan cerca de 1 % al año en costos por rotar cada mes (con US$ 10.000, la comisión mínima de US$ 1 por orden pesa).
- **Ni el momentum simple le ganó a SPY.** En 2010–2026 las acciones de EE.UU. le ganaron a casi todo lo demás del universo (bonos, oro, emergentes, Europa), así que rotar hacia esos activos restaba.
- **No es un error del código.** Los tests muestran que, con datos sintéticos que sí tienen señal, los modelos la encuentran (IC de 0,14), y que con ruido puro no inventan nada. Con estas 10 variables de precio, a un mes y en estos 14 ETFs, no hubo señal que encontrar.

Decisión: estos modelos no van con dinero real. La fase 3 prueba una pregunta más simple.

### Filtro de tendencia (fase 3, 2005–2026)

Cada fin de mes: ¿en SPY o en un refugio? Febrero de 2005 a septiembre de 2026 (260 meses), después de costos.

| Variante | Retorno anual | Volatilidad | Sharpe | Máx. caída | Costo anual | Sharpe deflactado |
|---|---:|---:|---:|---:|---:|---:|
| Media de 10 meses → IEF | 8,9 % | 11,5 % | 0,80 | −29,4 % | 0,17 % | 34,5 % |
| Media de 10 meses → SHY | 8,3 % | 10,9 % | 0,79 | −27,5 % | 0,17 % | 32,7 % |
| Momentum de 12 meses → IEF | 8,5 % | 12,9 % | 0,70 | −28,4 % | 0,10 % | 20,6 % |
| Momentum de 12 meses → SHY | 8,5 % | 12,4 % | 0,73 | −26,8 % | 0,10 % | 24,5 % |
| **Comprar y mantener SPY** | **11,0 %** | 16,5 % | 0,72 | **−52,9 %** | 0,00 % | — |

| Sharpe por subperíodo | 2005-02 a 2012-04 | 2012-05 a 2019-07 | 2019-08 a 2026-09 |
|---|---:|---:|---:|
| Media de 10 meses → IEF | **0,84** | 0,70 | 0,87 |
| Media de 10 meses → SHY | **0,81** | 0,67 | 0,90 |
| Momentum de 12 meses → IEF | **0,77** | 0,70 | 0,66 |
| Momentum de 12 meses → SHY | **0,78** | 0,70 | 0,72 |
| Comprar y mantener SPY | 0,32 | **1,04** | **0,95** |

**Veredicto: ninguna variante pasa.** El umbral del Sharpe deflactado fue 0,90 (SPY 0,72 + 0,18 por haber probado 9 estrategias en total).

- **Cortó la caída máxima casi a la mitad:** entre −27 % y −29 %, contra −53 % de SPY. En la crisis de 2008 (octubre de 2007 a marzo de 2009), la media de 10 meses con IEF ganó 9 % mientras SPY perdió 46 %.
- **Pero todo su mérito viene de esa crisis.** Le gana a SPY solo en el primer subperíodo. Desde 2012 queda atrás en Sharpe, y en todo el período rinde 2 a 3 puntos menos por año.
- **Falla en caídas rápidas o cuando los bonos también caen.** En 2022 la media de 10 meses entró y salió de SPY varias veces en mal momento (adentro en abril y diciembre, que fueron malos; afuera en los repuntes) y los bonos también cayeron: perdió 25 % a 29 % contra 19 % de SPY. Su peor caída es de ese episodio. El momentum de 12 meses tomó completa la caída de marzo de 2020 y se perdió el rebote.
- **Mejor Sharpe que SPY, pero no demostrado.** La media de 10 meses tiene Sharpe 0,80 contra 0,72, pero la probabilidad de que su Sharpe verdadero supere el umbral es 33 % a 35 %, lejos del 95 % exigido.

Lectura: un filtro de tendencia no es una mejora demostrada. Funciona como un seguro contra crisis largas tipo 2008, que se paga con menos retorno en los años normales y que puede fallar en caídas rápidas (2020) o cuando acciones y bonos caen juntos (2022).

### Cómo leer las métricas

| Métrica | Qué mide | Cómo leerla |
|---|---|---|
| Retorno anual | Crecimiento compuesto por año | Poca diferencia anual pesa mucho en el tiempo: en 21,7 años, el 11 % de SPY convirtió 1 dólar en 9,5 y el 8,9 % del filtro, en 6,3 |
| Volatilidad | Cuánto se mueven los retornos (desviación estándar anualizada) | Más alta = más sustos. SPY 16,5 % vs filtro 11 %: el filtro es más tranquilo |
| Sharpe | Retorno por unidad de volatilidad | Sirve para comparar estrategias **en el mismo período**. Sobre 1 es excelente por décadas; 0,5 a 0,8 es lo normal de un índice |
| Máx. caída | La peor pérdida desde un máximo hasta el punto más bajo | Lo que habrías tenido que aguantar sin vender. −53 % = ver la mitad de tu plata desaparecer |
| Costo anual | Comisiones y deslizamiento, como porcentaje del capital por año | Se resta directo del retorno |
| Sharpe por subperíodo | El Sharpe en tres tramos iguales | Si una estrategia gana en un solo tramo, su ventaja depende de un episodio |
| IC | Correlación entre el puntaje del modelo y lo que pasó el mes siguiente | 0 = no predice; 0,05 estable ya es útil; negativo = predice al revés |
| t-stat del IC | Si el IC medio es distinguible de 0 | Sobre 2 es evidencia; entre −2 y 2 puede ser azar |
| AUC | Qué tan bien separa ganadores de perdedores | 0,5 = azar; 0,55 ya sería interesante en finanzas |
| Sharpe deflactado | Probabilidad de que el Sharpe verdadero supere el umbral (benchmark + suerte por la cantidad de pruebas) | Se exige 95 %. Bajo 50 % = ni siquiera es más probable que sí a que no |

## Requisitos

- Windows 11 (también funciona en Linux o macOS)
- Python 3.11 o superior (probado con 3.13)
- Para la fase 2: TWS o IB Gateway con una cuenta paper de Interactive Brokers

## Instalación

En PowerShell, dentro de la carpeta del proyecto (la que tiene `config.toml`):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell no deja activar el entorno, corre una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Fase 1: comprobar que todo funciona (sin internet)

```powershell
python -m pytest
python -m etfml backtest --sintetico
```

Los 92 tests verifican, entre otras cosas, que **cambiar precios futuros no cambia ninguna feature ni señal del pasado** (sin fuga de información), que los modelos encuentran una señal plantada en datos sintéticos, que **no inventan señal en ruido puro**, que el seguimiento de la fase 2 reconstruye bien una cuenta simulada día a día y que el Sharpe deflactado calza con su fórmula y que el simulador de aportes reproduce exacto los casos que se pueden calcular a mano (incluida la conversión a pesos de hoy y la calculadora de metas). El backtest sintético genera un reporte en `reportes\` con números inventados, solo para ver el formato.

### Control de calidad

En cada push, GitHub Actions ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) revisa el código con Ruff y corre todos los tests con Python 3.12 y 3.13, pandas 2.2 y 3.0, en Linux y en Windows. Falla si:

- algún test no pasa,
- la cobertura baja del 90 % (hoy es 94 %),
- pandas o numpy avisan que algo del código dejará de funcionar en su próxima versión (esos avisos se tratan como errores),
- Ruff encuentra errores comunes, imports desordenados o sintaxis anticuada.

Para hacer lo mismo en tu PC:

```powershell
pip install -r requirements-dev.txt
python -m pytest -m "not slow"        # pasada rápida mientras programas (~15 s, omite 9 tests lentos)
python -m pytest --cov                # todos los tests, con el informe de cobertura
ruff check .                          # linter (ruff check --fix . corrige lo automático)
```

---

## Fase 2: validar con datos reales y la cuenta paper

La fase 2 responde dos preguntas distintas:

1. **¿La estrategia vale la pena?** La responde el backtest con datos reales (paso 2.1).
2. **¿Ejecutarla en la vida real se parece a lo que supone el backtest?** La responden 2 o 3 meses de cuenta paper (pasos 2.2 a 2.5).

Ojo: 2 o 3 meses de paper **no** dicen si la estrategia gana plata (son muy pocos meses, puro ruido). Lo que sí dicen es si las órdenes, los precios de llenado, las comisiones y las fechas calzan con los supuestos del backtest. Si calzan, puedes confiar en el backtest; si no, hay que corregir los supuestos y volver a correrlo.

### 2.1 Backtest con datos reales (hoy, no necesita cuenta)

```powershell
python -m etfml backtest
```

La primera vez descarga los precios desde Yahoo Finance y los guarda en `datos\`. Las siguientes usa la copia guardada; agrega `--actualizar` para bajar datos nuevos. Abre el `reporte.html` que aparece al final (doble clic: funciona sin internet).

Un modelo **pasa** solo si cumple los tres criterios a la vez:

1. Su Sharpe neto (después de costos) supera al mejor baseline en todo el período.
2. Le gana al mejor baseline en al menos 2 de 3 subperíodos.
3. El IC (correlación entre su puntaje y el retorno real del mes siguiente) es significativo: t-stat de Newey-West de al menos 2.

Qué hacer según el resultado:

- **Ningún modelo pasa:** no pongas dinero real. Puedes igual hacer los pasos 2.2 a 2.5 para practicar la ejecución, pero lo importante pasa a ser la fase 3 (mejorar el modelo con rigor). Si el momentum simple le gana a todo, esa puede ser tu estrategia: el ML no es obligatorio.
- **Algún modelo pasa:** pon ese modelo en `config.toml` → `[ib].modelo` y sigue con 2.2.

Si cambias parámetros mirando el reporte y lo vuelves a correr muchas veces, terminarás encontrando algo que "funciona" por suerte. Cambia pocas cosas y anota cada prueba.

### 2.2 Abrir la cuenta en IB y activar la paper (mientras corre lo demás)

1. Abre una cuenta individual en interactivebrokers.com con Chile como residencia y el plan **IBKR Pro** (el que IB exige para la API). Te pedirán identificación, comprobante de domicilio y el formulario **W-8BEN** (declaración de que no eres residente tributario en EE.UU.).
2. Cuando la aprueben, **fondéala** con una transferencia SWIFT en dólares desde tu banco. IB exige que la cuenta real esté abierta y con fondos para usar la paper por API. No hay mínimo para cuentas cash; lo que pesa es la comisión del SWIFT que cobra tu banco.
3. En el Portal del Cliente crea la **cuenta paper**. Su número empieza con `DU`.
4. Si el portal te deja reiniciar el saldo de la paper, ponlo parecido al capital que usarías de verdad (por ejemplo, US$ 10.000). Con el saldo de fábrica (muy grande) las comisiones mínimas no pesan y la prueba se parece menos a tu caso real.
5. Usa la cuenta paper **solo** para esta prueba: no compres otras cosas en ella, no muevas fondos y no reinicies el saldo a mitad de camino.

Configura TWS (o IB Gateway):

- Inicia sesión **en la cuenta paper**.
- Configuración global → API → Settings: activa "Enable ActiveX and Socket Clients", deja el puerto `7497` (en IB Gateway paper es `4002`; cámbialo en `config.toml` → `[ib].puerto`) y `127.0.0.1` como IP de confianza.
- Desmarca "Read-Only API" (si no, no se pueden enviar órdenes).
- La moneda base de la cuenta debe ser USD.

Los precios para calcular cuántas acciones comprar salen del último cierre de Yahoo Finance, así que no necesitas pagar suscripciones de datos de mercado en IB.

### 2.3 Rutina mensual (primer día hábil de cada mes, cerca del cierre)

El backtest supone que compras **al cierre del primer día hábil del mes**. Para parecerte a eso, ejecuta ese día entre las **15:00 y las 15:45 de Nueva York**. En Chile es 1 o 2 horas más según la época del año (revisa la diferencia horaria el día que lo hagas).

Con TWS abierto en la cuenta paper:

```powershell
python -m etfml senal --actualizar        # qué ETFs tocan este mes (también baja los precios nuevos)
python -m etfml rebalancear               # simulación: muestra el plan, no envía nada
python -m etfml rebalancear --ejecutar    # envía las órdenes después de que escribas SI
```

`rebalancear --ejecutar` hace todo lo que el seguimiento necesita:

- Antes de operar guarda una foto de la cuenta (patrimonio, posiciones y pesos objetivo) en `registros\ejecuciones\`.
- Envía primero las ventas, espera a que se llenen y después envía las compras.
- Al terminar guarda cada llenado (precio y comisión) en `registros\llenados.csv`.

Si al final dice que hay órdenes sin llenar, corre esto **el mismo día**, después de las 16:00 de Nueva York (la API de IB solo entrega las ejecuciones del día en curso):

```powershell
python -m etfml llenados
```

Puedes correrlo las veces que quieras: no duplica nada y completa las comisiones que IB haya informado tarde.

Sobre el tipo de orden: queda en `tipo_orden = "MKT"` (a mercado). `"MOC"` (al cierre) calza mejor con el backtest, pero envía compras y ventas a la vez, y en una cuenta cash IB puede rechazar las compras porque la plata de las ventas todavía no está disponible. Si usas MOC, envíalas antes de las 15:45 de Nueva York y corre `llenados` después del cierre.

Protecciones del rebalanceo:

- Sin `--ejecutar` se conecta en modo solo lectura y no puede enviar órdenes.
- Con `--ejecutar` muestra el plan y pide que escribas `SI`.
- Se niega a operar en cuentas que no sean paper, salvo que pongas `permitir_cuenta_real = true`.
- Solo toca los ETFs del universo. Cualquier otra posición de la cuenta queda intacta y su valor no se usa para comprar.
- Los ajustes menores a `banda_rebalanceo_usd` se omiten.

### 2.4 Seguimiento (desde el segundo mes)

```powershell
python -m etfml seguimiento --actualizar
```

No se conecta a IB: trabaja con lo que quedó en `registros\` y con los precios de Yahoo. Para cada rebalanceo reconstruye el mes de tu cartera con los llenados reales y lo compara con el **teórico**: la misma señal simulada con los supuestos del backtest (compra al cierre del primer día hábil y costos de `config.toml`). Un mes queda "completo" cuando ya hiciste el rebalanceo del mes siguiente; mientras tanto aparece "en curso".

Qué muestra:

| Sección | Qué significa |
|---|---|
| Real | Retorno del mes de tu cartera paper, reconstruido con tus llenados y comisiones reales |
| Teórico | Lo que el backtest habría obtenido con la misma señal ese mes |
| Según IB | Retorno según el patrimonio que informa IB, como control. Si difiere mucho del real, falta registrar algún llenado o se movió la cuenta |
| Caja y redondeo | Efecto del 2 % que queda en caja y de comprar acciones enteras. Siempre es chico |
| Fecha | Efecto de no ejecutar el día que supone el backtest. Es 0 si ejecutaste el primer día hábil |
| Precio | Precio de llenado contra el cierre de ese día, frente al deslizamiento supuesto (`slippage_bps`) |
| Comisiones | Comisiones reales frente a las supuestas (`comision_bps` y `comision_minima_usd`) |

Las cuatro últimas suman exactamente la diferencia entre real y teórico, así que siempre sabes de dónde sale. El detalle completo queda en `registros\seguimiento.csv`.

### 2.5 Cuándo pasar a la siguiente fase

El comando termina con estos criterios y te dice qué hacer con cada uno que falle:

| Criterio | Si falla |
|---|---|
| Al menos 2 meses completos | Sigue rebalanceando una vez al mes |
| Todas las órdenes llenadas, con comisión, y cartera igual a la señal | Revisa las órdenes en TWS y corre `llenados` el mismo día |
| Rebalanceo todos los meses, con la señal del mes y el primer día hábil | Ejecuta a tiempo y con `--actualizar` |
| Comisiones reales hasta 1,5 veces las supuestas | Sube los costos en `config.toml` y **vuelve a correr el backtest** |
| Precio de llenado hasta 20 bps sobre el deslizamiento supuesto | Ejecuta más cerca del cierre o sube `slippage_bps` y **vuelve a correr el backtest** |

Si todo pasa, el backtest describe bien cómo se ejecuta la estrategia en la práctica. Si un modelo además pasó el backtest real (2.1), recién ahí tiene sentido pensar en las siguientes fases.

Si tienes que corregir costos, el veredicto del backtest puede cambiar: con costos más altos, una estrategia que rota mucho puede dejar de pasar.

---

## Fase 3: cambiar la pregunta (filtro de tendencia)

La rotación con ML no encontró señal: elegir 3 entre activos tan distintos (acciones, bonos, oro) mes a mes es muy difícil. La fase 3 hace una pregunta más simple: **cada fin de mes, ¿estar en acciones (SPY) o en un refugio?**

### Las variantes (fijadas antes de ver resultados)

| Señal | Regla | Origen |
|---|---|---|
| Media de 10 meses | En SPY si su cierre está sobre el promedio de los últimos 10 cierres mensuales | Faber, 2007 |
| Momentum de 12 meses | En SPY si su retorno de 12 meses supera al de SHY (casi caja) | Antonacci, 2014 |

Cada señal se prueba con dos refugios: **IEF** (bonos del Tesoro de 7 a 10 años) y **SHY** (bonos de 1 a 3 años, casi caja). Son 4 variantes, todas contra comprar y mantener SPY, con el mismo calendario y los mismos costos que la fase 1. Están fijas en el código a propósito: si las ajustas después de ver el resultado, ya no sabes si encontraste algo o si te acomodaste a la historia.

### Cómo correrlo

```powershell
python -m etfml backtest       # una vez: deja las 5 estrategias de la fase 1 en el registro (usa los datos guardados)
python -m etfml tendencia      # las 4 variantes, el veredicto y el reporte (la primera vez descarga SHY)
python -m etfml experimentos   # lista todas las pruebas registradas
```

Para ver el formato sin internet: `python -m etfml tendencia --sintetico` (no registra nada).

### Cómo se decide

Cada variante recibe uno de tres veredictos:

| Veredicto | Qué exige |
|---|---|
| **Pasa** | Sharpe mayor que comprar y mantener SPY, le gana en al menos 2 de 3 subperíodos y Sharpe deflactado de al menos 95 % |
| **Prometedor** | Las dos primeras, pero sin evidencia estadística suficiente |
| **No pasa** | No mejora a SPY de forma consistente |

**Sharpe deflactado.** Si pruebas muchas variantes, la mejor se ve bien aunque ninguna sirva. El Sharpe deflactado (Bailey y López de Prado, 2014) calcula la probabilidad de que el Sharpe verdadero de una variante supere un umbral. Aquí el umbral es el Sharpe de SPY **más** lo que la mejor de N pruebas le ganaría solo por suerte, donde N es el total de pruebas del registro (las 5 de la fase 1 más las 4 de esta). También corrige por asimetría y colas gruesas de los retornos.

**Registro de experimentos.** `experimentos\registro.csv` anota cada estrategia distinta evaluada con datos reales. Volver a correr la misma configuración (por ejemplo, con datos más nuevos) actualiza su fila; cambiar cualquier cosa (una señal, un costo) cuenta como una prueba nueva y sube la exigencia para todas. Ese archivo sí se sube a Git: es la evidencia de cuántas cosas probaste.

### Qué esperar, honestamente

Lo que más promete un filtro de tendencia es esquivar las caídas grandes (2008, 2020, 2022), a cambio de quedarse atrás en algunos años buenos y de pagar costos cada vez que cambia de posición. Con unos 20 años de datos es difícil demostrar estadísticamente una mejora de Sharpe chica, así que "Prometedor" o "No pasa" son resultados razonables, no errores. Mira también el gráfico de caídas del reporte: puede que una variante no gane en Sharpe pero sí reduzca mucho las pérdidas máximas, y eso puede importarte.

---

## Simulador de aportes mensuales

Después de que ninguna estrategia activa le ganara al índice, la pregunta práctica es otra: **si aporto lo mismo todos los meses a un ETF diversificado, ¿cuánto puedo esperar, por cuánto tiempo tengo que aguantar y dónde me conviene comprar?**

```powershell
python -m etfml simulador --actualizar                      # la primera vez: baja dólar, UF e historia larga
python -m etfml simulador --aporte 50000 --anios 10         # $50.000 al mes, plazo principal de 10 años
python -m etfml simulador --meta 20000000 --anios 10        # ¿cuánto aportar para juntar $20 millones de hoy?
python -m etfml simulador --cartera conservadora            # "global", "eeuu" o "conservadora"
```

Para cada plazo simula **todos los meses de partida posibles** desde 2004: quien empezó justo antes de 2008, quien empezó en el fondo de la crisis, y todos los demás. Cada mes convierte pesos a dólares, compra (en BTG) o junta la plata y la envía por SWIFT (en IB), con las comisiones de cada una y la retención de impuesto de EE.UU. sobre los dividendos.

**Todo en pesos de hoy.** El aporte es el mismo en UF todos los meses (sube con la inflación), cada mes se convierte con el dólar observado de ese mes y el resultado se lleva a pesos de hoy con la UF. Así, "$12.000.000 aportados" y "$19.000.000 al final" se comparan directo con lo que cuestan las cosas hoy. El dólar observado y la UF se bajan de mindicador.cl (datos del Banco Central); si no hay conexión, los resultados quedan en dólares sin ajustar.

**Otras épocas.** 2004–2026 es un solo período y fue bueno. El simulador repite todo con las acciones de EE.UU. desde 1926 (incluye la Gran Depresión y los años 70) y las de Japón desde 1990 (un mercado que tardó más de 30 años en recuperar su máximo), en dólares de hoy. Datos de la biblioteca de Kenneth French (Dartmouth) y la inflación de EE.UU. de FRED. La tabla muestra cuántos períodos **independientes** caben en cada historia: las ventanas se superponen, así que 150 ventanas de 10 años en 22 años de datos son en el fondo solo 2 períodos distintos.

**Calculadora de metas.** Con `--meta` calcula cuánto aportar al mes, en pesos de hoy, para llegar al monto en el plazo: en el caso típico (50 %), con 90 % de seguridad (`--confianza` para cambiarlo) e incluso en el peor caso histórico, para cada época. También dice en qué porcentaje de los casos llegarías con tu aporte actual.

| Cartera | Qué compras hoy | Historia que la aproxima |
|---|---|---|
| `global` | VT (todo el mundo, 0,06 % al año) | 60 % SPY + 30 % EFA + 10 % EEM |
| `eeuu` | VOO o SPY | SPY |
| `conservadora` | 70 % VT + 30 % IEF | lo mismo con 30 % IEF |

### Resultado con $100.000 al mes en la cartera global (datos al 15 de octubre de 2026, dólar a $987)

Comprando todos los meses en BTG, después de costos, **en pesos de hoy**:

| Plazo | Aportado | Peor caso | Malo (10 %) | Típico | Bueno (90 %) | Terminó perdiendo | Rentab. real típica | Peor momento |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 años | $3.600.000 | $2.082.983 | $3.221.232 | $4.008.026 | $4.705.291 | 21 % | 7,1 % | −42 % |
| 5 años | $6.000.000 | $3.517.879 | $5.286.085 | $7.578.863 | $8.661.431 | 20 % | 9,3 % | −42 % |
| 10 años | $12.000.000 | $15.175.110 | $16.874.538 | $19.628.052 | $22.385.847 | 0 % | 9,5 % | −42 % |
| 15 años | $18.000.000 | $29.373.205 | $32.534.596 | $39.440.375 | $44.344.260 | 0 % | 9,8 % | −42 % |

Si el futuro se parece a otra época (dólares de hoy, como múltiplo de lo aportado):

| Historia | Perdió a 10 años | Peor caso a 10 años | Típico a 10 años | Perdió a 20 años | Períodos indep. a 10 años |
|---|---:|---:|---:|---:|---:|
| ETFs 2004–2026 | 0 % | 1,08x | 1,40x | 0 % | 2 |
| EE.UU. 1926–2026 | 12 % | 0,53x | 1,50x | 4 % | 10 |
| Japón 1990–2026 | 39 % | 0,54x | 1,07x | 20 % | 3 |

- **Plazo:** a 3 y 5 años, 1 de cada 5 puntos de partida terminó con menos poder de compra del que puso. A 10 años o más, ninguno en 2004–2026, pero eso son solo 2 períodos independientes: con la historia de EE.UU. desde 1926, un plan de 10 años terminó perdiendo en 12 % de los casos y uno de 20 años en 4 %. En el camino, quien partió en abril de 2006 llegó a ver, en febrero de 2009, un −42 % sobre lo aportado.
- **El dólar ayudó, y hoy parte caro:** de 2004 a 2026 la cartera rindió 6,4 % real al año en dólares de hoy y 7,4 % en pesos de hoy. La diferencia es que el dólar subió más que la inflación chilena. En las ventanas de 10 años el efecto es mayor (9,5 % típico en pesos contra 6,5 % en dólares), porque muchas partieron con el dólar barato, entre 2008 y 2013. Hoy el dólar real está más caro que en el 96 % de los meses desde 2004; solo en 2022 estuvo más alto. Si volviera a su promedio real de estos 22 años, cada dólar valdría unos 21 % menos pesos. **Para planificar es más prudente usar las cifras en dólares de hoy (6 a 8 % real al año) que el 9,5 %.**
- **En pesos, el dólar al final pesa tanto como el mercado:** el peor resultado a 10 años partió en febrero de 2004 y terminó en enero de 2014 (1,27x). No hubo crisis al final: el dólar terminó más barato que al partir ($547 contra $594), mientras los precios en Chile subieron 39 %.
- **Meta de $20.000.000 de hoy en 10 años:** con $100.000 al mes se llegó en 43 % de los casos de 2004–2026 y en 30 % con la historia de EE.UU. desde 1926. Para llegar con 90 % de seguridad: unos $118.000 al mes según 2004–2026, o $171.000 según EE.UU. desde 1926 (que incluye la Gran Depresión y los años 70).
- **BTG o IB:** con estos supuestos, BTG deja más plata que IB con cualquier aporte de hasta $5.000.000 al mes. En IB cada transferencia SWIFT cuesta unos US$ 25: transfiriendo todos los meses se va 26 % de lo aportado, y aun transfiriendo una vez al año, 3,1 % (contra 1,5 % en BTG), porque la plata espera sin invertirse.
- **Cada cuánto comprar:** con $100.000 al mes, comprar cada 2 meses en BTG deja 0,5 % más que comprar todos los meses (unos $97.000 en 10 años), porque la comisión de US$ 1 pesa 1 % en una compra de US$ 101. Juntar 3 meses ahorra más comisión, pero la plata espera más sin invertirse y deja menos. Sobre $500.000 al mes da lo mismo (0,1 % o menos).
- **Global o EE.UU.:** EE.UU. rindió más en este período (12 % contra 9 % anual en dólares), pero elegirlo por eso es el mismo error que elegir el mejor backtest: 2004–2026 fue una época excepcional para EE.UU. que no tiene por qué repetirse. Japón desde 1990 muestra qué pasa cuando un solo país se estanca.

Supuestos que conviene revisar en `config.toml`: el costo de cambiar pesos a dólares (1 %; compáralo con lo que te cobra BTG) y la comisión de BTG ("desde US$ 1" por operación). El dólar se toma solo del último dólar observado; `--dolar` lo reemplaza.

---

## Qué falta considerar antes de usar dinero real

- Impuestos en Chile (declaración en el SII) y la retención de 15 % sobre dividendos de EE.UU. (convenio Chile–EE.UU. vigente desde 2024). Consúltalo con un contador.
- El costo real de convertir pesos a dólares en tu corredora y de las transferencias SWIFT.
- Las comisiones vigentes de BTG e IB.
- El impuesto a la herencia de EE.UU.: para no residentes aplica sobre activos de EE.UU. (como VT, VOO o SPY) sobre US$ 60.000. Si llegas a ese monto, los ETFs equivalentes domiciliados en Irlanda (UCITS, disponibles en IB) lo evitan.

## Comandos

| Comando | Qué hace | Necesita |
|---|---|---|
| `python -m etfml descargar` | Descarga precios y los guarda en `datos\` | Internet |
| `python -m etfml backtest [--sintetico] [--actualizar]` | Walk-forward de los 3 modelos y el ensamble contra los baselines, con reporte HTML | Internet la primera vez |
| `python -m etfml senal [--actualizar]` | Cartera objetivo de este mes | Internet la primera vez |
| `python -m etfml rebalancear [--ejecutar]` | Plan de órdenes para la cuenta paper; con `--ejecutar`, las envía | TWS o IB Gateway |
| `python -m etfml llenados` | Guarda las ejecuciones y comisiones de hoy | TWS o IB Gateway |
| `python -m etfml seguimiento [--actualizar]` | Compara la cuenta paper con lo que supone el backtest | Lo guardado en `registros\` |
| `python -m etfml tendencia [--sintetico] [--actualizar]` | Fase 3: filtro de tendencia contra comprar y mantener, con reporte HTML | Internet la primera vez |
| `python -m etfml experimentos` | Lista todas las pruebas registradas | Nada |
| `python -m etfml simulador [--aporte] [--anios] [--cartera] [--meta] [--confianza] [--dolar]` | Aportes mensuales en pesos de hoy: resultados por plazo, otras épocas, meta y BTG contra IB | Internet la primera vez |

## Estructura

```
ml-inversion-etf/
├── config.toml            todos los parámetros
├── requirements.txt
├── requirements-dev.txt   + pytest-cov y ruff (control de calidad)
├── pyproject.toml         configuración de pytest, cobertura y ruff
├── .github/workflows/
│   └── tests.yml          CI: ruff y tests en 5 combinaciones de Python, pandas y sistema
├── etfml/
│   ├── __main__.py        comandos
│   ├── config.py          carga y valida config.toml
│   ├── data.py            Yahoo Finance (precios ajustados y sin ajustar), caché en CSV y datos sintéticos
│   ├── features.py        calendario, features mensuales y target (sin mirar el futuro)
│   ├── models.py          Ridge, logística, LightGBM y ensamble
│   ├── backtest.py        walk-forward con purga, cartera y simulación con costos
│   ├── metrics.py         IC, AUC, log loss, Sharpe, Sortino, caídas, subperíodos
│   ├── report.py          reporte HTML autocontenido + CSVs
│   ├── live_signal.py     señal del mes con toda la historia conocida
│   ├── ib_rebalance.py    conexión a IB (ib_async), plan, ejecución, fotos de la cuenta y llenados
│   ├── tracking.py        seguimiento de la cuenta paper contra el backtest
│   ├── trend.py           fase 3: filtro de tendencia
│   ├── trend_report.py    reporte HTML del filtro de tendencia
│   ├── stats.py           Sharpe probabilístico y deflactado
│   ├── registry.py        registro de experimentos
│   ├── savings.py         simulador de aportes, escenarios y calculadora de metas
│   ├── savings_report.py  reporte HTML del simulador
│   └── external.py        dólar y UF (mindicador.cl), historia larga (Kenneth French) e IPC de EE.UU. (FRED)
├── tests/                 92 tests (9 marcados como lentos)
├── experimentos/
│   └── registro.csv       cada estrategia evaluada con datos reales (se sube a Git)
├── datos/                 precios, dólar y UF, historia larga (se crea solo)
├── reportes/              un reporte por corrida del backtest (se crea solo)
└── registros/             se crea solo
    ├── senales.csv        cada señal calculada
    ├── ordenes_*.csv      cada plan y cada envío de órdenes
    ├── ejecuciones/       foto de la cuenta antes de cada rebalanceo ejecutado
    ├── llenados.csv       precio y comisión de cada llenado
    └── seguimiento.csv    resultado del último seguimiento
```

Convención: identificadores de código en inglés; configuración, comandos, mensajes y reporte en español.

## Cómo está armado (y por qué)

- **Objetivo relativo, no absoluto.** Se predice si un ETF le gana al promedio del grupo, no si sube. Así el modelo no depende de adivinar la dirección del mercado.
- **Features como ranking dentro de cada mes.** Un retorno de 3 % significa cosas distintas en 2008 y en 2017; su posición frente a los demás ETFs es más estable.
- **Calendario sin trampas.** Las features usan el cierre del último día hábil del mes; la compra se simula al cierre del día hábil siguiente; el retorno de ese mes recién se conoce al mes siguiente, y solo entonces puede entrar al entrenamiento (purga).
- **Reentrenamiento anual con ventana creciente** en el backtest, hiperparámetros fijos (no se optimizan mirando el resultado). La señal en vivo entrena con toda la historia conocida.
- **Costos realistas para cuentas chicas:** comisión mínima por orden, slippage y banda de rebalanceo.
- **t-stat de Newey-West** para el IC: los meses seguidos se parecen y el t-stat simple exagera la significancia.
- **Sharpe deflactado y registro de experimentos.** Cada estrategia probada queda anotada, y la exigencia sube con la cantidad de pruebas: es la defensa contra encontrar algo bueno por suerte.
- **Seguimiento con precios sin ajustar.** Los llenados reales se comparan con el cierre real del día, no con el precio ajustado por dividendos (que en un ETF de bonos puede diferir varios puntos). Los retornos sí usan el ajustado, para contar los dividendos.
