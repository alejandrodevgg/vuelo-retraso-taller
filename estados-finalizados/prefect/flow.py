# flow.py — STRETCH OPCIONAL (fuera del arco v0.1→v0.7 del taller)
# =================================================================
# Un único flow de Prefect 3 que envuelve datos -> entrena como dos tasks.
# Mismo código de pipeline que los jobs de compose: el orquestador no cambia
# TU pipeline, solo decide CUÁNDO y CON QUÉ RÉGIMEN DE FALLOS se ejecuta.
#
# Este directorio es lectura/demo si sobra tiempo en s9-s10. El equivalente
# "de producción" que sí veis entero en el taller es el CI de s8 (GitHub
# Actions); Prefect sería elCT local disparado por drift (el loop de Evidently).
#
# ✅ Smoke ephemeral real: 2026-11-02 (ver docs/SMOKE-RETRAIN.md §0 del taller
#    mlops-prefect). fixes aplicados tras pagarlo ejecutando:
#    - `log_level` era kwarg de @flow en Prefect 2.x: en 3.6 da TypeError
#      directo. En 3.x el nivel se ajusta por settings, no por flow.
#    - Los paths «house» estaban obsoletos: el núcleo actual produce
#      data/work/featured_flights.csv (src/run_pipeline.py lo verifica).
import subprocess
import sys

from prefect import flow, task


@task(retries=1, retry_delay_seconds=10)
def datos() -> str:
    subprocess.run([sys.executable, "-m", "src.run_pipeline"], check=True)
    return "data/work/featured_flights.csv"


@task(retries=0)
def entrena(featured: str) -> str:
    subprocess.run(
        [
            sys.executable, "-m", "src.models.train_model",
            "--config", "configs/model_config.yaml",
            "--data", featured,
            "--models-dir", "models",
            "--register",
        ],
        check=True,
    )
    return "registered"


@flow(name="flight-delay-pipeline")
def pipeline():
    featured = datos()
    entrena(featured)


if __name__ == "__main__":
    pipeline()
