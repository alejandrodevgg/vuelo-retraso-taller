# HOWTO — informe de drift con Evidently (sección s9, opcional)

Evidently **no** vive en ninguna imagen del stack: es una herramienta de
análisis, no un servicio. Se ejecuta como contenedor desechable.

Con vuelos el drift **no hay que inyectarlo: es temporal y real**. La tasa de
retraso >15 min pasa de 6.81 % (2023) a 7.27 % (2024) y 10.26 % (2025) en la
misma muestra (anclas literales en `docs/NUMEROS-VUELOS.md`). Por eso el
`drift_report.py` compara **años del mismo CSV**: referencia 2023 (lo que vio
el modelo al entrenar el fit slice) contra actual 2025 (el corte de test).

## 1) Por defecto: crudo 2023 vs crudo 2025 (un solo comando)

```bash
cd pipeline
docker run --rm -v "$PWD:/work" -w /work python:3.11-slim sh -c '
  pip install --no-cache-dir evidently==0.7.23 "pandas==2.2.3" >/dev/null 2>&1 &&
  python estados-finalizados/evidently/drift_report.py
'
# abre estados-finalizados/evidently/report_drift.html en el navegador
```

> **Por qué el `pandas==2.2.3` en el pip install**: evidently 0.7.x todavía
> importa `infer_dtype_from_object` de pandas, eliminado en pandas 3. Sin el
> pin, pip te baja el pandas último y el script muere con `ImportError` antes
> de empezar. Es el mismo pandas del `requirements.txt` del pipeline: paridad.

> La primera vez baja dependencias (~2 min). Si tu Wi-Fi del aula es malo,
> esto es lo último que debes dejar para el final de la sesión.

> **Ruido esperado**: durante el cálculo salen varios
> `RuntimeWarning: invalid value encountered in divide` de numpy (el preset
> correlaciona columnas con varianza casi nula, como la etiqueta binaria). Son
> warning, no errores: el informe se guarda igual (exit 0, verificado).

## 2) Referencia = lo que realmente vio el modelo (el volumen limpio)

Si ya corriste el job `datos` (s3), la referencia honesta es el CSV limpio que
vive en el **volumen** `mlops-pipeline_pipeline-data` (montado en
`/app/data/work` de los contenedores), recortado al año de referencia:

```bash
docker run --rm \
  -v "$PWD:/work" -w /work \
  -v mlops-pipeline_pipeline-data:/data:ro \
  python:3.11-slim sh -c '
    pip install --no-cache-dir evidently==0.7.23 "pandas==2.2.3" >/dev/null 2>&1 &&
    python estados-finalizados/evidently/drift_report.py \
      --reference /data/cleaned_flights.csv --reference-year 2023 \
      --current data/raw/flights.csv --current-year 2025
  '
```

(El nombre exacto del volumen es `docker volume ls | grep pipeline`.)

> **Diferencia con la casa**: allí, correr el job `datos` con el crudo con skew
> **contaminaba** la referencia del volumen (escribía antes de que pandera
> vetara). Con vuelos eso ya no pasa: la banda de volumen mata el crudo
> equivocado **a la entrada**, antes de escribir nada (verificado 2026-11-02:
> tras el `run -e RAW_FILE=data/raw/house_data.csv datos` que muere en la
> banda, el `cleaned_flights.csv` del volumen seguía intacto para el punto 2).

## 3) Qué mirar en el HTML

- La tarjeta **Data Drift Summary**: cuántas columnas han cambiado de
  distribución (KPI grande arriba a la izquierda).
- Salida OBSERVADA (2026-11-02, puntos 1 y 2 dan la misma lista):
  `columnas con drift: ['adep', 'ades', 'delay_min', 'dof', 'icao_operator', 'source_month', 'source_year']`
- Lectura honesta de esa lista — tres grupos:
  - **Artefacto del corte**: `source_year`, `source_month` y `dof` disparan
    *por construcción* (has partido el CSV por año; su distribución cambia
    por decreto). Enseñanza: antes de alarmarse, mira si el drift es real o
    es el recorte que se delata a sí mismo.
  - **Drift de verdad**: `delay_min` (la severidad sube: media 2.40 en el
    total, con 2025 más gordo) y `adep`/`ades`/`icao_operator` (cambia la
    **composición** de aeropuertos y operadores entre años: mix shift, no
    solo deriva numérica).
  - **Lo que NO dispara**: las features horarias (`dep_hour`/`dep_weekday`)
    están estables entre años — el mundo no se movió por ahí. Y fíjate en la
    etiqueta: `delayed_15min` pasó de 6.81 % a 10.26 % y **el test de drift no
    la lista**: un shift de 3.45 puntos en una binaria desbalanceada no pasa el
    umbral del detector. Lección: el detector de drift no es un dashboard de
    métricas — ni lo ve todo, ni lo ve igual.
- La diferencia de actitud con pandera: el contract **vetó** el job (s3);
  Evidently **mide** cuánto se fue el mundo del entrenamiento (s9). Los dos
  son necesarios y no se solapan.

## 4) El bucle que se cierra (CT)

Drift detectado → re-entrena (`docker compose run --rm entrena python -m
src.models.train_model --register --alias Production` — ojo: `compose run`
**sustituye** el command del servicio, hay que escribir el comando completo)
→ nueva versión en el registry → mueve el alias:

```bash
# version N = la que acabe de crear el reentrenamiento (mira la UI o:
# mlflow CLI: docker compose run --rm entrena mlflow models list)
docker compose run --rm entrena python -c "
import mlflow, os
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_registered_model_alias('vuelo-retraso', 'Production', 2)
print('Production -> version 2')
"
```

Mejor desde la UI (Registry → vuelo-retraso → versiones → alias Production a la
nueva) — y la API **no se reinicia**: resuelve `models:/vuelo-retraso@Production`
(con `@`: así, en MLflow 3) en cada arranque, y el
`docker compose up -d --force-recreate api` de mañana
recoge la versión nueva sin tocar una línea de código. Eso es exactamente lo
que el `.pkl` local del libro no puede dar.

## Versión probada

- evidently==0.7.23 + pandas==2.2.3 sobre python:3.11-slim (linux/arm64), generado el
  2026-11-02 con el `flights.csv` de este repo (regenerado con
  `scripts/make_flights_sample.py`). Salida observada (punto 1):
  `59709 filas ref vs 72424 actuales` y
  `columnas con drift: ['adep', 'ades', 'delay_min', 'dof', 'icao_operator', 'source_month', 'source_year']`
  (report_drift.html: 4.305.323 bytes). El punto 2 (referencia del volumen) se
  verificó la misma tarde: mismas filas, misma lista.
