# src/api/inference.py  — STARTER MODE (Google L0, the book's way)
#
# Loads a .pkl from the local disk with joblib. It WORKS... and it is BLIND:
# there is no registry behind it. Nobody can answer "what version is serving
# right now?" from here — the filename is the only metadata. Section 6 of the
# workshop replaces this file's logic with estados-finalizados/inference_registry.py.
#
# The pkl is a full sklearn Pipeline ending in ProbaPredictor: .predict() IS
# P(delay > 15 min). Class = probability >= 0.5 (threshold documented in the
# response). Unpickling the pipeline imports src.features.engineer — that is
# why Dockerfile.api ships the whole src/ tree and sets PYTHONPATH=/app.
import logging
import os
from datetime import datetime

import joblib
import pandas as pd

from schemas import DelayPredictionResponse, FlightPredictionRequest

logger = logging.getLogger("inference-pkl")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/trained/flight_delay_model.pkl")
CLASS_THRESHOLD = 0.5

_model = None


def load() -> dict:
    """Load the model at startup; return health metadata."""
    global _model
    try:
        _model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not load local pkl at {MODEL_PATH}: {e}. "
            "Run the `entrena` job first (it drops the model in the shared volume)."
        ) from e
    logger.info(f"Loaded local model (L0 mode): {MODEL_PATH}")
    return {
        "status": "healthy",
        "backend": "joblib-pkl (L0: local file, no registry)",
        "model_version": "unknown (that is the problem)",
        "model_source": MODEL_PATH,
    }


def _to_frame(requests: list[FlightPredictionRequest]) -> pd.DataFrame:
    """One request -> one raw flight row. NO feature math here: the collapsed
    groups (top-K + UNK + OTH) and the one-hot live INSIDE the logged pipeline,
    so serving and training can never diverge."""
    return pd.DataFrame([r.model_dump() for r in requests])


def predict_delay(request: FlightPredictionRequest) -> DelayPredictionResponse:
    proba = float(_model.predict(_to_frame([request]))[0])
    return DelayPredictionResponse(
        probability_delay=round(proba, 4),
        predicted_class=int(proba >= CLASS_THRESHOLD),
        class_threshold=CLASS_THRESHOLD,
        model_version="unknown (joblib-pkl mode)",
        model_source=MODEL_PATH,
        prediction_time=datetime.now().isoformat(),
    )


def batch_predict(requests: list[FlightPredictionRequest]) -> list[DelayPredictionResponse]:
    if not requests:
        return []
    now = datetime.now().isoformat()
    out = []
    for req, proba in zip(requests, _model.predict(_to_frame(requests))):
        p = float(proba)
        out.append(DelayPredictionResponse(
            probability_delay=round(p, 4),
            predicted_class=int(p >= CLASS_THRESHOLD),
            class_threshold=CLASS_THRESHOLD,
            model_version="unknown (joblib-pkl mode)",
            model_source=MODEL_PATH,
            prediction_time=now,
        ))
    return out
