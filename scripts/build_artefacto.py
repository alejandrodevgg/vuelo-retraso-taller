# scripts/build_artefacto.py
# Reconstruye `artefacto/` — el modelo pyfunc que carga la app de Streamlit en
# MODE=direct (`mlflow.pyfunc.load_model(<repo>/artefacto)`).
#
# Por qué existe: el artefacto original vivía en el volumen Docker del servidor
# MLflow del taller y se perdió con él. El recuperable es
# `models/trained/flight_delay_model.pkl`: ES el Pipeline completo (prepor +
# delay_model + ProbaPredictor), con clases propias (`src.features.engineer`,
# `src.models.proba`) importables desde la raíz del repo.
#
# Cómo funciona: loguea el pkl con `mlflow.sklearn.log_model` contra un
# file store LOCAL EFÍMERO (un mlruns en un directorio temporal que se elimina
# al terminar) y copia el directorio del modelo registrado a `artefacto/` en la
# raíz del repo. No toca ni necesita ningún tracking server.
#
# Determinista y re-ejecutable: limpia `artefacto/` antes de copiar y fija los
# `pip_requirements` del artefacto con los pins del front, en lugar de capturar
# los del entorno de build (que bajo `uv run` serían los del env temporal).
#
# Uso (desde la raíz del repo, env con mlflow + deps del pipeline):
#   python scripts/build_artefacto.py
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # para que joblib resuelva src.* del pkl

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402

PKL_PATH = REPO_ROOT / "models" / "trained" / "flight_delay_model.pkl"
ARTEFACTO_DIR = REPO_ROOT / "artefacto"

# Pins con los que el front (streamlit_app/requirements.txt) carga el
# artefacto: quedan escritos en su requirements.txt para trazabilidad.
PIP_REQUIREMENTS = [
    "mlflow-skinny==3.17.2",
    "scikit-learn==1.5.2",
    "xgboost==1.7.6",
    "pandas==2.2.3",
    "numpy==1.26.4",
    "pandera==0.35.0",
    "joblib==1.5.2",
    "PyYAML==6.0.2",
]


def _logged_model_dir(model) -> Path:
    """Localiza en disco el directorio del modelo recién logueado."""
    # Vía principal (mlflow 3.x): LoggedModel.artifact_location apunta al
    # directorio de artefactos del modelo en el file store.
    lm = mlflow.MlflowClient().get_logged_model(model.model_id)
    if lm.artifact_location and lm.artifact_location.startswith("file:"):
        cand = Path(lm.artifact_location[len("file:"):])
        if (cand / "MLmodel").is_file():
            return cand
    # Red de seguridad: buscar el MLmodel bajo el mlruns del tmp (la forma de
    # la ruta cambia entre versiones de mlflow; no confiar en una sola).
    exp = mlflow.get_experiment_by_name(_EXPERIMENT)
    if exp is not None:
        base = Path(temp_dir) / "mlruns" / exp.experiment_id
        for hits in (sorted(base.rglob("artefacto/MLmodel")),
                     sorted(base.rglob("MLmodel"))):
            for hit in reversed(hits):  # el mas reciente si hay varios runs
                if hit.is_file():
                    return hit.parent
    sys.exit("ERROR: no se localizo el directorio del modelo logueado")


_EXPERIMENT = "artefacto-rebuild"

if not PKL_PATH.is_file():
    sys.exit(f"ERROR: falta {PKL_PATH}")

pipeline = joblib.load(PKL_PATH)  # prepor + delay_model + ProbaPredictor

temp_dir = tempfile.mkdtemp(prefix="artefacto-build-")
try:
    # mlflow 3.x pone el file store en modo mantenimiento y exige opt-in
    # expreso; aqui es justo lo que queremos: un mlruns efimero sin servidor.
    import os

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri((Path(temp_dir) / "mlruns").as_uri())
    mlflow.set_experiment(_EXPERIMENT)
    with mlflow.start_run():
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="artefacto",
            # Serializacion joblib/pickle: el pkl original usa clases propias
            # (src.features.engineer.GroupCategories, src.models.proba.
            # ProbaPredictor) y el front las importa de src/ al deserializar;
            # el formato skops de mlflow 3.x las rechazaria como no confiables.
            serialization_format="pickle",
            pip_requirements=PIP_REQUIREMENTS,
        )
    src_dir = _logged_model_dir(logged)

    # Copia limpia y reproducible: artefacto/ siempre arranca de cero.
    if ARTEFACTO_DIR.exists():
        shutil.rmtree(ARTEFACTO_DIR)
    shutil.copytree(src_dir, ARTEFACTO_DIR)
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

files = sorted(p.relative_to(ARTEFACTO_DIR).as_posix()
               for p in ARTEFACTO_DIR.rglob("*") if p.is_file())
print(f"artefacto/ generado en {ARTEFACTO_DIR} ({len(files)} ficheros):")
for f in files:
    print(f"  {f}")
