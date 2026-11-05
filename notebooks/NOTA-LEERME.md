# notebooks/ — REFERENCIA DE LECTURA (no se ejecutan en el taller)

Copiados **tal cual** del backbone `Docker-for-AI-ML` (Packt "Docker for
AI/ML", upstream School of Devops MLOps Bootcamp):

| Notebook | Fase que muestra | Qué hacer con él |
|---|---|---|
| `00_data_engineering.ipynb` | Limpieza (mediana/moda + IQR) | Leer al lado de `src/data/run_processing.py` |
| `01_exploratory_data_analysis.ipynb` | EDA (matplotlib/seaborn) | Opcional: ábrelo en tu entorno local si quieres explorar |
| `02_feature_engineering.ipynb` | Features + preprocessor | Leer al lado de `src/features/engineer.py` |
| `03_experimentation.ipynb` | GridSearch de 4 modelos + nested runs | **Lectura del patrón parent/hijos** que `src/models/train_model.py` conserva recortado |

**Por qué no se ejecutan:** estos notebooks están pinneados al mundo MLflow
2.3.1 de 2023 (`mlflow.start_run(nested=True)`, `transition_model_version_stage`
en el script del repo…). El taller trabaja contra MLflow **3.17.2**, donde el
registro se goberna con **aliases**, no con stages. La lógica buena del libro
(viaje del CSV, comparación de modelos, árbol de runs) vive ahora en `src/`,
ejecutable y dockerizada; los notebooks quedan como mapa del "cómo lo contaba
el curso".

Nada de esto es un fallo de estos materiales: es una fuente citada.
