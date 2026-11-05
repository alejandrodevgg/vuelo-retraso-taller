# inference_registry.py — Sección s6 · modo REGISTRY (el delta sobre el libro)
# ============================================================================
# Copiar a src/api/inference_registry.py en la sección 6 y levantar el api con
# INFERENCE_MODULE=inference_registry (viene así por defecto en v0.5/v0.6):
#
#   cp estados-finalizados/inference_registry.py src/api/
#   docker compose up -d --build api   # el --build importa: src/api va DENTRO de la imagen
#   # (compose reutiliza la imagen rota ya construida si no lo pones — probado en vivo)
#
# Diferencia conceptual con el starter (inference.py, el patrón del libro):
# aquí NO hay un .pkl en disco. Se pide por URI al tracking server:
# models:/vuelo-retraso@Production. El alias lo resuelve el SERVIDOR en cada
# carga; promover la versión 2 mañana no toca este código ni esta imagen.
# Además, el artefacto logged es un Pipeline completo (group-collapse + one-hot
# + clasificador) con SIGNATURA: pyfunc valida el input y aplica el prepor uno
# mismo. Ojo: el pipeline usa clases PROPIAS (src.features.engineer) — el pickle
# las importa al deserializar, por eso Dockerfile.api copia TODO src/ y exporta
# PYTHONPATH=/app (gotcha de la conversión a clasificación, probado 2026-11-02).
import logging
import os
from datetime import datetime

import mlflow.pyfunc
import pandas as pd

from schemas import DelayPredictionResponse, FlightPredictionRequest

logger = logging.getLogger("inference-registry")

MODEL_NAME = os.getenv("MODEL_NAME", "vuelo-retraso")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")
CLASS_THRESHOLD = 0.5

_model = None
_meta = {}


def load() -> dict:
    """Pull the Production alias from the registry at container start."""
    global _model, _meta
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    # MLflow 3: los ALIAS se resuelven con "@" (models:/vuelo-retraso@Production).
    # La forma con "/" seguida de nombre (models:/vuelo-retraso/Production) se
    # interpretó SIEMPRE como STAGE del registry 2.x — los stages ya no existen
    # en 3.x y eso da "No versions of model ... stage 'Production' found".
    # Gotcha probado.
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        _model = mlflow.pyfunc.load_model(uri)
    except Exception as e:
        raise RuntimeError(
            f"No pude cargar {uri} desde {tracking_uri}. "
            "¿Registraste y promoviste el modelo? (s5: --register --alias Production "
            "o create model version + alias desde la UI)."
        ) from e
    mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    _meta = {
        "status": "healthy",
        "backend": "mlflow.pyfunc (registry: models:/<name>@<alias>)",
        "model_version": str(mv.version),
        "model_source": uri,
    }
    logger.info(f"Loaded {uri} -> version {mv.version}")
    return _meta


def _to_frame(requests: list[FlightPredictionRequest]) -> pd.DataFrame:
    """Raw flight rows only — el prepor (top-K + UNK + OTH + one-hot) viaja
    dentro del artefacto y lo aplica pyfunc solo, guiado por la SIGNATURA."""
    return pd.DataFrame([r.model_dump() for r in requests])


def _response_from_proba(proba: float, now: str) -> DelayPredictionResponse:
    return DelayPredictionResponse(
        probability_delay=round(proba, 4),
        predicted_class=int(proba >= CLASS_THRESHOLD),
        class_threshold=CLASS_THRESHOLD,
        model_version=_meta.get("model_version", "?"),
        model_source=_meta.get("model_source", "?"),
        prediction_time=now,
    )


def predict_delay(request: FlightPredictionRequest) -> DelayPredictionResponse:
    # El pipeline logged termina en ProbaPredictor: predict() devuelve la
    # PROBABILIDAD (pyfunc no expone predict_proba). Devuelve ndarray: indexar
    # posicional. (patrón probado 2026-11-02)
    proba = float(_model.predict(_to_frame([request]))[0])
    return _response_from_proba(proba, datetime.now().isoformat())


def batch_predict(requests: list[FlightPredictionRequest]) -> list[DelayPredictionResponse]:
    if not requests:
        return []
    now = datetime.now().isoformat()
    return [_response_from_proba(float(p), now) for p in _model.predict(_to_frame(requests)).tolist()]
