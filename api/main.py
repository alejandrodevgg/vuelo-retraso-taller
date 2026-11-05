# src/api/main.py — FastAPI inference service (flight-delay classifier).
#
# Which backend answers is decided by ONE env var, not by a rebuild:
#   INFERENCE_MODULE=inference           -> starter/L0 (joblib pkl)
#   INFERENCE_MODULE=inference_registry  -> registry mode (pyfunc, models:/<name>@Production)
# CORS is wide open (as in the book) so the Streamlit browser-side calls never
# fight the API; inside compose, Streamlit talks server-to-server anyway.
import importlib
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from schemas import DelayPredictionResponse, FlightPredictionRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

INFERENCE_MODULE = os.getenv("INFERENCE_MODULE", "inference")
try:
    impl = importlib.import_module(INFERENCE_MODULE)
except ModuleNotFoundError as e:
    raise RuntimeError(
        f"INFERENCE_MODULE={INFERENCE_MODULE!r} not found in src/api/. "
        "For registry mode: cp estados-finalizados/inference_registry.py src/api/ "
        "(section 6 of the workshop)."
    ) from e

app = FastAPI(
    title="Flight Delay Prediction API",
    description="MLOps workshop pipeline (modernized from the School of Devops / Packt backbone). Classification: P(arrival delay > 15 min).",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus /metrics (scraped in section 9; harmless before that)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Load once at import time = at container start. If the model cannot load,
# the container dies loudly: fail fast > fail silent.
HEALTH = impl.load()


@app.get("/health", response_model=dict)
async def health_check():
    return HEALTH


@app.post("/predict", response_model=DelayPredictionResponse)
async def predict(request: FlightPredictionRequest):
    return impl.predict_delay(request)


@app.post("/batch-predict", response_model=list[DelayPredictionResponse])
async def batch_predict_endpoint(requests: list[FlightPredictionRequest]):
    return impl.batch_predict(requests)
