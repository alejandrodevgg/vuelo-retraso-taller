# streamlit_app/ — front del taller (dual-mode)

Una pantalla: eliges un vuelo, el front pregunta al modelo y te enseña la
**probabilidad** (no solo la etiqueta), la curva de riesgo de las 24 horas y la
identidad del modelo servido. Nunca cae a datos mock: si el backend falla, el
front lo testiga con su comando de reproducción.

## Los dos modos (misma UI, una env)

| | `MODE=api` (default) | `MODE=direct` |
|---|---|---|
| Para qué | Taller en local: compose levanta MLflow + API; el front llama a la API por nombre de servicio | App pública en Streamlit Community Cloud (o portátil sin API): el **modelo viaja empaquetado** en la imagen del front |
| Env | `API_URL=http://api:8000` (dentro de compose; `http://localhost:18000` si expones el puerto y cambias el default) | `MODEL_DIR=/ruta/al/artefacto` (descarga local de `models:/vuelo-retraso@Production`, ver `despliegue/README.md`) |
| Dependencias activas | `streamlit`, `requests`, `plotly` | + `mlflow-skinny`, `scikit-learn`, `xgboost` (pyfunc carga el Pipeline real) |
| Salud | `/health` + `/metrics` de la API en la franja «Salud del modelo» sobre el panel de control | sin API que observar: la identidad del artefacto viaja al pie de página |

El modo direct **no levanta ni llama a ninguna API**: `mlflow.pyfunc.load_model(MODEL_DIR)`
y a correr. Requiere `src/` en el `PYTHONPATH` (el pickle referencia clases
propias) — el `Dockerfile.streamlit` ya lo copia.

## Correrlo en el taller (modo api)

Desde la raíz del `pipeline/`, con el stack arriba (`up -d --build api`):

```bash
docker compose -f estados-finalizados/compose-v0.6.yaml --project-directory . up -d --build front
# → http://localhost:8501
```

Sin docker (uv, apuntando al API expuesto):

```bash
API_URL=http://localhost:18000 uv run --project . \
    python -m streamlit run streamlit_app/app.py   # pins en streamlit_app/requirements.txt
```

## Qué se proyecta en clase

- El **gauge** y la píldora: la probabilidad manda; el umbral 0,50 y la
  precisión medida (0,2087 en test 2025) explicados en su nota.
- La **curva 24 h** (`/batch-predict`): el alumno ve el modelo como superficie
  de decisión, no como un formulario.
- La **franja de salud del modelo** (solo modo api): versión del registry +
  predicciones servidas desde `/metrics` — «esto es un servicio, no un
  archivo».
- El **pie de página**: una línea `st.caption` con la evidencia del servicio
  (`Modo: direct (artefacto empaquetado) · Ruta: ./artefacto` en modo direct;
  la URL del API en modo api). Desde 2026-11-05 no hay barra lateral: controles
  arriba, resultado grande centrado, curva 24 h a fondo de página.

## Honestidad de los datos de la UI

- Cifras del hero y de la nota: copiadas de `docs/NUMEROS-VUELOS.md`.
- Top-20 aeropuertos y top-10 operadores: extraídos del **artefacto registrado**
  (`GroupCategories.known_`), no de la documentación. Si reentrenas con otra
  muestra, actualiza `app.py`.
- Aeropuerto fuera del top-20: **aviso** (se colapsa a `OTH`), no error. Campo
  mal formado: **bloqueo visual** antes de llamar (el 422 del API es la
  demostración de la frontera, no el plan).
