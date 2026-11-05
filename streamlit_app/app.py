# streamlit_app/app.py — front-end del taller: P(retraso de llegada > 15 min)
# =============================================================================
# Dos modos, la MISMA pantalla (doc en streamlit_app/README.md):
#   MODE=api     (default) — habla con la API FastAPI vía API_URL: /predict,
#                            /batch-predict, /health y /metrics. Es el modo del
#                            taller en compose (http://api:8000 por nombre de
#                            servicio, nunca localhost).
#   MODE=direct            — carga el artefacto con mlflow.pyfunc.load_model
#                            sobre MODEL_DIR (modelo empaquetado con su
#                            prepor; NO levanta ni llama a ninguna API). Pensado
#                            para la app pública, donde la API del portátil
#                            del docente no es reachable: ver ../../../spaces/.
#
# Layout 2026-11-05: SIN barra lateral — panel de control arriba, resultado
# grande centrado debajo, curva 24 h después y la evidencia de servicio
# (modo/ruta o URL del API) en el pie de página.
#
# Herencia del original (backbone del libro): el front es UN TESTIGO, no un
# mentiroso — si el modelo no carga o la API falla, el error se muestra con su
# comando de reproducción; NUNCA se cae a datos mock.
#
# Verdad de la UI: las cifras citadas salen SOLO de docs/NUMEROS-VUELOS.md y
# docs/SMOKE-FLIGHTS.md. Los top-20 de aeropuertos y top-10 de operadores están
# EXTRAÍDOS del artefacto registrado (GroupCategories.known_ del preprocessor de
# vuelo-retraso@Production, 2026-11-02): si reentrenas con otra muestra, actualiza
# aquí — la UI avisa de los colapsos a OTH, pero quien decide qué conoce el
# modelo es src/features/engineer.py.
import json
import os
import re
import socket
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ------------------------------------------------------------ paleta Inmune --
# Tokens de assets/brand/paleta-inmune.css (raíz del repo). Ningún hex fuera de
# estas constantes.
TINTA = "#0e2a33"      # --inm-tinta   (fondo — ya puesto en config.toml)
ESPUMA = "#eafffb"     # --inm-espuma  (texto claro sobre tinta)
MENTA = "#9fd8dc"      # --inm-menta   (texto secundario sobre tinta)
TEAL = "#0898ab"       # --inm-teal    (acentos, series normales)
AVISO = "#f4b400"      # --inm-aviso   (umbral, hora seleccionada)
ALERTA = "#d64545"     # --inm-fig-alerta (clase RETRASO)
DEGRADADO = ("linear-gradient(118deg,#004149 0%,#006688 38%,"
             "#0898ab 72%,#32d8be 100%)")  # --inm-degradado, literal

# ------------------------------------------------------------------ dominio --
# Aeropuertos/operadores CONOCIDOS por el modelo (los que sobreviven al
# colapso top-K; el resto del mundo es OTH y los nulos de operador son UNK).
TOP_ADEP = ["EHAM", "LFPG", "EDDF", "EGLL", "LEBL", "EDDM", "LEMD", "EKCH",
            "EIDW", "LOWW", "EGKK", "LSZH", "LIRF", "LEPA", "LPPT", "EDDB",
            "EGSS", "ESSA", "EGCC", "EBBR"]
TOP_ADES = ["EHAM", "EGLL", "EDDF", "LEBL", "LEMD", "EDDM", "LFPG", "LSZH",
            "EIDW", "EKCH", "LOWW", "EGKK", "LIRF", "LEPA", "EDDB", "EGSS",
            "ENGM", "ESSA", "LTFM", "LSGG"]
TOP_OPERATORS = ["RYR", "DLH", "EZY", "BAW", "AFR", "AUA", "KLC", "THY",
                 "VLG", "WZZ"]
CIUDADES = {"EHAM": "Ámsterdam", "LFPG": "París CDG", "EDDF": "Fráncfort",
            "EGLL": "Londres Heathrow", "LEBL": "Barcelona El Prat",
            "EDDM": "Múnich", "LEMD": "Madrid Barajas", "EKCH": "Copenhague",
            "EIDW": "Dublín", "LOWW": "Viena", "EGKK": "Londres Gatwick",
            "LSZH": "Zúrich", "LIRF": "Roma Fiumicino", "LEPA": "Palma",
            "LPPT": "Lisboa", "EDDB": "Berlín", "EGSS": "Londres Stansted",
            "ESSA": "Estocolmo", "EGCC": "Mánchester", "EBBR": "Bruselas",
            "ENGM": "Oslo", "LTFM": "Estambul", "LSGG": "Ginebra"}
OPERADOR_NOMBRE = {"RYR": "Ryanair", "DLH": "Lufthansa", "EZY": "easyJet",
                   "BAW": "British Airways", "AFR": "Air France",
                   "AUA": "Austrian", "KLC": "KLM", "THY": "Turkish",
                   "VLG": "Vueling", "WZZ": "Wizz Air"}
OTRO = "✏️ Otro…"                       # centinela de los selectores libres
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado",
        "Domingo"]                      # dep_weekday: Mon=0 (contrato del API)

# Anclas citadas en la UI — copiadas LITERALMENTE de docs/NUMEROS-VUELOS.md y
# docs/SMOKE-FLIGHTS.md. Regenerar allí, no retocar aquí.
ANCLA_FILAS = "198.753"
ANCLA_TASA_TOTAL = "8,22%"
ANCLA_LEBL = ("5,65%", "n.º 5 en volumen · 4.707 vuelos")
ANCLA_LEMD = ("4,88%", "n.º 7 en volumen · 4.217 vuelos")
PRECISION_TEST = 0.2087                 # XGBoost, test 2025 (metrics.json)
RECALL_TEST = 0.6191

ICAO_RE = re.compile(r"^[A-Z]{4}$")     # espejo de src/api/schemas.py
OPERADOR_RE = re.compile(r"^[A-Z0-9]{2,3}$")
UMBRAL = 0.5                            # class_threshold del contrato

st.set_page_config(page_title="Retrasos aéreos · MLOps", layout="wide")

MODE = os.getenv("MODE", "api").strip().lower()
API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
MODEL_DIR = os.getenv("MODEL_DIR", "")

st.markdown("""
<style>
.hero { background: %s; border-radius: 14px; padding: 22px 28px 20px; }
.hero-k { letter-spacing: .2em; font-size: 11px; text-transform: uppercase;
          color: rgba(234,255,251,.75); }
.hero-t { font-size: 30px; font-weight: 800; color: %s; margin-top: 4px; }
.hero-s { font-size: 15px; color: rgba(234,255,251,.85); max-width: 760px;
          margin-top: 6px; }
/* Sin sidebar (2026-11-05): el resultado manda, así que se proyecta grande
   y centrado en la zona principal. */
.prob-big { font-size: 64px; font-weight: 800; line-height: 1.1;
            text-align: center; }
.pill { display: inline-block; vertical-align: middle; margin-left: 14px;
        padding: 5px 18px; border-radius: 999px; font-weight: 700;
        font-size: 18px; color: %s; }
.nota { border-left: 4px solid %s; padding: 8px 14px; font-size: 14px;
        background: rgba(244,180,0,.08); border-radius: 0 8px 8px 0; }
.hora-regla { display: flex; justify-content: space-between;
              font-size: 11px; color: %s; padding: 0 2px; }
/* El botoncito gris "Deploy" de Streamlit no: esta app ya se despliega por su
   cuenta (spaces/) y en clase sobra. */
div[data-testid="stAppDeployButton"] { display: none !important; }
</style>
""" % (DEGRADADO, ESPUMA, TINTA, AVISO, MENTA), unsafe_allow_html=True)


# ------------------------------------------------------- backend (2 modos) --
def _frozen(rows: list[dict]) -> str:
    """Clave cacheable para st.cache_data (lista de dicts -> JSON canónico)."""
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


@st.cache_data(ttl=600, show_spinner=False)
def _api_predict_one(row_json: str) -> dict:
    r = requests.post(f"{API_URL}/predict", json=json.loads(row_json), timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def _api_predict_many(rows_json: str) -> list[dict]:
    r = requests.post(f"{API_URL}/batch-predict", json=json.loads(rows_json),
                      timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _api_health() -> dict:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _api_traffic() -> dict:
    """Resumen de /metrics (Prometheus en texto): lo que el pie necesita."""
    text = requests.get(f"{API_URL}/metrics", timeout=5).text
    served, errors = 0, 0
    for line in text.splitlines():
        if not line.startswith("http_requests_total{"):
            continue
        # Ojo: prometheus-fastapi-instrumentator etiqueta status="2xx" (clase),
        # nunca el código numérico — quien busque \d aquí cuenta siempre 0.
        m = re.search(r'handler="(/[^"]*)",method="\w+",status="([^"]+)"', line)
        if not m or m.group(1) not in ("/predict", "/batch-predict"):
            continue
        n = int(float(line.rsplit(" ", 1)[1]))
        served += n
        errors += n if not m.group(2).startswith("2") else 0
    total = sum(int(float(l.rsplit(" ", 1)[1]))
                for l in text.splitlines() if l.startswith("http_requests_total{"))
    return {"served": served, "errors": errors, "total": total}


@st.cache_resource(show_spinner="Cargando el artefacto pyfunc…")
def _direct_model():
    """MODE=direct: el modelo viaja DENTRO del contenedor del Space."""
    import mlflow.pyfunc  # import perezoso: en MODE=api no se toca ni su librería
    return mlflow.pyfunc.load_model(MODEL_DIR)


@st.cache_data(ttl=600, show_spinner=False)
def _direct_predict_many(rows_json: str) -> list[dict]:
    rows = json.loads(rows_json)
    frame = pd.DataFrame(rows)  # los None viajan como nulos: el prepor los
    # colapsa a UNK/OTH igual que en la API (mismo artefacto, misma frontera)
    probas = _direct_model().predict(frame)
    now = datetime.now(timezone.utc).isoformat()
    return [{
        "probability_delay": round(float(p), 4),
        "predicted_class": int(float(p) >= UMBRAL),
        "class_threshold": UMBRAL,
        "model_version": os.getenv("MODEL_DIR", "artefacto local"),
        "model_source": f"direct: mlflow.pyfunc sobre {MODEL_DIR}",
        "prediction_time": now,
    } for p in probas]


def predict_one(row: dict) -> dict:
    if MODE == "direct":
        return _direct_predict_many(_frozen([row]))[0]
    return _api_predict_one(_frozen(row))


def predict_many(rows: list[dict]) -> list[dict]:
    if MODE == "direct":
        return _direct_predict_many(_frozen(rows))
    return _api_predict_many(_frozen(rows))


# -------------------------------------------------------------------- héroe --
st.markdown(f"""
<div class="hero">
  <div class="hero-k">Taller MLOps · datos → MLflow → registry → modelo servido</div>
  <div class="hero-t">Retrasos aéreos · Eurocontrol 2023–2025</div>
  <div class="hero-s">¿Llegará este vuelo con más de 15 minutos de retraso?
  Un clasificador entrenado con {ANCLA_FILAS} vuelos reales responde — y este
  front enseña a leer su probabilidad, no solo su etiqueta.</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Retrasos en la muestra", ANCLA_TASA_TOTAL,
          help=f"{ANCLA_FILAS} vuelos 2023–2025 (Eurocontrol OPDI vía HF, "
               "muestra docente Apache-2.0). Fuente: docs/NUMEROS-VUELOS.md.")
m2.metric("LEBL · El Prat", ANCLA_LEBL[0], delta=ANCLA_LEBL[1], delta_color="off")
m3.metric("LEMD · Barajas", ANCLA_LEMD[0], delta=ANCLA_LEMD[1], delta_color="off")

# ------------------------------------------------- salud (solo en modo api) --
# Antes vivía en la barra lateral; sin sidebar, la identidad del servicio es
# una franja compacta sobre el panel de control (s6 y s9 del taller).
if MODE != "direct":
    st.markdown("#### ❤️‍🩹 Salud del modelo")
    h1, h2, h3, h4 = st.columns([3.2, 1.9, 1.9, 1.4])
    try:
        health = _api_health()
        ok = health.get("status") == "healthy"
        h1.markdown(f"**{'🟢' if ok else '🔴'} {health.get('model_source', '?')}**")
        h1.caption(health.get("backend", ""))
        h2.metric("Versión en serving", health.get("model_version", "?"))
        traffic = _api_traffic()
        h3.metric("Predicciones servidas", traffic["served"])
        if h4.button("Actualizar panel"):
            _api_health.clear()
            _api_traffic.clear()
            st.rerun()
        st.caption(
            f"{traffic['total']} llamadas al API en total · "
            f"{traffic['errors']} con error. Este panel Lee **el registry y el "
            "/metrics de Prometheus**: el modelo no es un archivo, es un "
            "servicio observado (s6 y s9 del taller)."
        )
    except requests.RequestException as e:
        st.error(f"API no responde en `{API_URL}`")
        st.code(f"curl -s {API_URL}/health   # ¿compose up? ¿modelo registrado?",
                language=None)

st.divider()

# --------------------------------------------------------- panel de control --
# Deep-link del taller: ?adep=..&ades=..&hour=..&weekday=..&op=..&demo=1
# (demo=1 prediction a la carga — es el enlace que se proyecta en la slide
#  y el que usa el screenshot headless del smoke; sin parámetros, el form
#  abre en la fila del smoke documentada: LEMD→EGLL, Sáb 11:00, BAW).
qp = st.query_params


def _prefill(widget_key: str, free_key: str, value: str | None, known: list[str]):
    if value is None:
        return
    st.session_state[widget_key] = value if value.upper() in known else OTRO
    if st.session_state[widget_key] == OTRO:
        st.session_state[free_key] = value.upper()


_prefill("sel_Origen", "free_Origen", qp.get("adep"), TOP_ADEP)
_prefill("sel_Destino", "free_Destino", qp.get("ades"), TOP_ADES)
if qp.get("hour", "").isdigit():
    st.session_state["k_hour"] = min(23, int(qp["hour"]))
if qp.get("weekday", "").isdigit():
    st.session_state["k_weekday"] = min(6, int(qp["weekday"]))
if qp.get("op") == "∅":
    st.session_state["k_op"] = "∅ sin atribuir (UNK)"
elif qp.get("op"):
    op_in = qp["op"].upper()
    st.session_state["k_op"] = op_in if op_in in TOP_OPERATORS else OTRO
    if st.session_state["k_op"] == OTRO:
        st.session_state["free_op"] = op_in


def _pick_aeropuerto(label: str, top: list[str], default: str):
    """Selectbox sobre lo que el modelo conoce + puerta a código libre."""
    sel = st.selectbox(
        label, options=top + [OTRO],
        index=top.index(default) if default in top else 0,
        key=f"sel_{label}",
        format_func=lambda c: f"{c} · {CIUDADES[c]}" if c in CIUDADES else c,
    )
    if sel == OTRO:
        return st.text_input(f"Código ICAO ({label})", max_chars=4,
                             key=f"free_{label}").strip().upper()
    return sel


st.markdown("#### El vuelo")
# Dos filas: aeropuertos/día/operador arriba; hora + avisos + botón abajo.
f1 = st.columns([2.5, 2.5, 1.7, 2.8])
with f1[0]:
    adep = _pick_aeropuerto("Origen", TOP_ADEP, "LEMD")
with f1[1]:
    ades = _pick_aeropuerto("Destino", TOP_ADES, "EGLL")
with f1[2]:
    dep_weekday = st.selectbox("Día de salida", options=list(range(7)), index=5,
                               format_func=lambda d: f"{DIAS[d]} ({d})",
                               key="k_weekday")
with f1[3]:
    sel_op = st.selectbox(
        "Operador aéreo (opcional)", options=["∅ sin atribuir (UNK)"] + TOP_OPERATORS + [OTRO],
        index=1 + TOP_OPERATORS.index("BAW"), key="k_op",
        format_func=lambda o: o if o.startswith("∅") or o == OTRO
        else f"{o} · {OPERADOR_NOMBRE[o]}")
    icao_operator = None
    if sel_op == OTRO:
        icao_operator = st.text_input("IATA/ICAO del operador (2-3)", max_chars=3,
                                      key="free_op").strip().upper()
    elif not sel_op.startswith("∅"):
        icao_operator = sel_op

f2h, f2m, f2b = st.columns([6.4, 3.6, 2.0], vertical_alignment="center")
with f2h:
    dep_hour = st.slider("Hora de salida (UTC)", 0, 23, 11,
                         format="%02d:00", key="k_hour")
    st.markdown('<div class="hora-regla"><span>00:00</span><span>06:00</span>'
                '<span>12:00</span><span>18:00</span><span>23:00</span></div>',
                unsafe_allow_html=True)

# Validación VISUAL antes de llamar: el 422 del API es la frontera, no el
# plan. El aviso OTH es transparencia: el modelo colapsa (no rechaza).
errores, avisos = [], []
for nombre, codigo, top in (("origen", adep, TOP_ADEP), ("destino", ades, TOP_ADES)):
    if not codigo:
        errores.append(f"Falta el aeropuerto de {nombre}.")
    elif not ICAO_RE.match(codigo):
        errores.append(f"`{codigo}` no es un código ICAO de 4 letras "
                       "mayúsculas (el API respondería **422**).")
    elif codigo not in top:
        avisos.append(f"`{codigo}` no está entre los 20 aeropuertos que el "
                      "modelo conoce → lo codifica como **`OTH`** "
                      "(funciona; solo pierde especificidad en ese campo).")
if icao_operator and not OPERADOR_RE.match(icao_operator):
    errores.append(f"`{icao_operator}` no es un operador de 2-3 caracteres "
                   "`[A-Z0-9]` (el API respondería **422**).")
elif icao_operator and icao_operator not in TOP_OPERATORS:
    avisos.append(f"Operador `{icao_operator}` fuera del top-10 → también "
                  "se colapsa a **`OTH`**.")

with f2m:
    for a in avisos:
        st.warning(a, icon="ℹ️")
    for e in errores:
        st.error(e, icon="🚫")

with f2b:
    volar = st.button("Predecir retraso ✈️", use_container_width=True,
                      disabled=bool(errores) or not adep or not ades)

flight = {"dep_hour": dep_hour, "dep_weekday": dep_weekday,
          "adep": adep, "ades": ades, "icao_operator": icao_operator,
          "typecode": None}  # typecode se queda en UNK: menos UI, mismo modelo
if qp.get("demo") == "1" and "flight" not in st.session_state:
    volar = not errores and bool(adep) and bool(ades)  # enlace de la slide
if volar:
    st.session_state["flight"] = flight
    try:
        st.session_state["result"] = predict_one(flight)
    except requests.RequestException as e:
        st.session_state.pop("result", None)
        st.error(f"La API falló: {e}")
        st.code(f"curl -s {API_URL}/health   # ¿está el API up? ¿modelo registrado?",
                language=None)

# ---------------------------------------------------- resultado (centrado) ---
res = st.session_state.get("result")
vuelo = st.session_state.get("flight")
_, res_col, __ = st.columns([1, 3.6, 1])
with res_col:
    if res is None:
        st.markdown("#### Pronóstico")
        st.info("Configura el vuelo en el panel de control y pulsa **Predecir "
                "retraso**. Luego mira la curva: es la misma llamada hecha 24 "
                "veces, una por hora del día.")
    else:
        p = res["probability_delay"]
        es_retraso = res["predicted_class"] == 1
        color = ALERTA if es_retraso else TEAL
        st.markdown(f"""
        <div class="prob-big" style="color:{color};">{p:.1%}
          <span class="pill" style="background:{color};">
            {'RETRASO &gt; 15 min' if es_retraso else 'A TIEMPO (según el modelo)'}
          </span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            f"P(llegada con > 15 min de retraso) para "
            f"**{vuelo['adep']} → {vuelo['ades']}** · "
            f"{DIAS[vuelo['dep_weekday']]} {vuelo['dep_hour']:02d}:00 UTC · "
            f"modelo `{res['model_version']}` desde `{res['model_source']}` · "
            f"{res['prediction_time']}"
        )

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=round(p * 100, 1),
            number={"suffix": " %", "font": {"size": 30, "color": ESPUMA}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100],
                         "tickfont": {"color": MENTA, "size": 11}},
                "bar": {"color": color, "thickness": 0.28},
                "bordercolor": MENTA, "borderwidth": 1,
                "steps": [{"range": [0, 50], "color": "rgba(8,152,171,.14)"},
                          {"range": [50, 100], "color": "rgba(214,69,69,.16)"}],
                "threshold": {"value": 50, "thickness": 0.85,
                              "line": {"color": AVISO, "width": 3}},
            }))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", height=170,
                            margin=dict(l=10, r=10, t=5, b=5))
        st.plotly_chart(gauge, use_container_width=True, theme=None,
                        config={"staticPlot": True})

        st.markdown(f"""
        <div class="nota">El umbral de decisión es <b>0,50</b>: con prob ≥ 50 %
        el modelo grita «retraso». En el test 2025 su <b>precisión medida es
        {PRECISION_TEST}</b>: de cada 10 vuelos que marca, <b>~2 aciertan</b>
        (a cambio de cazar ~{RECALL_TEST:.0%} de los retrasos reales).
        Por eso manda la probabilidad, no la píldora.</div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------- curva 24 h (a fondo) --
# La pieza de producto: el batch-predict de las 24 horas con el resto
# fijo. Una sola llamada /batch-predict (probado en SMOKE-FLIGHTS §5).
if res is not None and vuelo:
    with st.expander("📈 Riesgo a lo largo del día (mismo vuelo, 24 horas)",
                     expanded=True):
        try:
            curva = predict_many([{**vuelo, "dep_hour": h}
                                  for h in range(24)])
            probs = [c["probability_delay"] for c in curva]
            bars = go.Bar(
                x=[f"{h:02d}" for h in range(24)], y=probs,
                marker_color=[AVISO if h == vuelo["dep_hour"] else TEAL
                              for h in range(24)],
                marker_line_width=0,
            )
            dia = go.Figure(bars)
            dia.add_hline(y=UMBRAL, line_color=ALERTA, line_width=1.5,
                          line_dash="dash",
                          annotation_text="umbral 0,50",
                          annotation_font_color=ALERTA)
            dia.update_layout(
                height=230, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title=None, tickfont=dict(color=MENTA, size=10),
                           dtick=3),
                yaxis=dict(title="P(retraso)", tickformat=".0%",
                           rangemode="tozero",
                           gridcolor="rgba(159,216,220,.15)",
                           tickfont=dict(color=MENTA)),
            )
            st.plotly_chart(dia, use_container_width=True, theme=None,
                            config={"displayModeBar": False})
            pico = max(range(24), key=lambda i: probs[i])
            fuente = ("`/batch-predict`, 24 filas por una llamada"
                      if MODE != "direct" else
                      "24 horas en un solo `predict` del artefacto")
            st.caption(f"Pico del día a las {pico:02d}:00 "
                       f"({probs[pico]:.1%}) · tu hora, en ámbar. "
                       f"Fuente: {fuente}.")
        except requests.RequestException as e:
            st.error(f"`/batch-predict` falló: {e}")

# ------------------------------------------------- pie de página (evidencia) --
# La identidad del servicio ya no ocupa una barra lateral: es una línea
# discreta al final de la página (mismas cadenas de siempre).
mode_line = (f"Modo: `direct` (artefacto empaquetado) · "
             f"Ruta: `{MODEL_DIR or 'MODEL_DIR sin definir'}`"
             if MODE == "direct" else f"API: {API_URL}")
st.caption(f"Taller MLOps · pipeline local · hostname: {socket.gethostname()} · "
           f"{mode_line}")
