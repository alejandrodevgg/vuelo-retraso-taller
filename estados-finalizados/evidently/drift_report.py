# drift_report.py — Sección s9 (OPCIONAL) · Evidently: referencia vs actual
# ==========================================================================
# Compara el dataset de REFERENCIA (lo que vio el modelo al entrenar) contra el
# dataset ACTUAL y genera un HTML con el resumen de data drift. Con vuelos la
# deriva NO hay que inyectarla: es temporal y real — la tasa de retraso >15 min
# pasa de 6.81% (2023) a 10.26% (2025) en la MISMA muestra (docs/NUMEROS-VUELOS.md).
# Por eso los defaults son el crudo de vuelos filtrado por source_year:
# referencia = 2023, actual = 2025.
#
# Ejecución (una sola vez, contenedor desechable; evidently no vive en ninguna
# imagen del stack — se instala en el momento, ~1-2 min de pip):
#
#   docker run --rm -v "$PWD:/work" -w /work python:3.11-slim sh -c '
#     pip install --no-cache-dir evidently==0.7.23 "pandas==2.2.3" >/dev/null &&
#     python estados-finalizados/evidently/drift_report.py
#   '
#
# (Si tu pipeline ya generó data/work/cleaned_flights.csv, puedes usarlo
#  como referencia montando el volumen: ver HOWTO.md en esta carpeta.)
#
# El cierre del loop: drift alto -> retraining -> nueva version en el registry
# -> el alias Production apunta a la nueva -> la API la sirve sin redeploy.
# Eso es CT (continuous training), la exclusiva de ML sobre CI/CD.
import argparse

import pandas as pd

# evidently 0.7.x reorganizó la libreria: la API clasica (Report + presets)
# vivo movida a evidently.legacy.*; "from evidently.report import ..." ya NO
# existe en 0.7.23 (ModuleNotFoundError probado en el smoke). El pin de
# pandas==2.2.3 del HOWTO es igualmente obligatorio: evidently 0.7.23 importa
# una funcion de pandas eliminada en pandas 3.
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

# Identificadores de vuelo, fuera del informe por dos motivos: cardinalidad
# casi única (flt_id) o derivada de los anteriores (route) — no son features
# (el contract los valida como DATOS, FEATURE_COLUMNS los veta) y a Evidently
# solo le aportaría columnas degeneradas.
DROP_DEFAULT = "flt_id,route"


def load_slice(path: str, year: int, drop: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    if year > 0:
        if "source_year" not in df.columns:
            raise SystemExit(f"{path}: falta la columna source_year (¿es el crudo de vuelos?)")
        df = df[df["source_year"] == year]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/raw/flights.csv")
    parser.add_argument("--current", default="data/raw/flights.csv")
    parser.add_argument("--reference-year", type=int, default=2023,
                        help="0 = no filtrar por año (p.ej. un CSV ya recortado)")
    parser.add_argument("--current-year", type=int, default=2025,
                        help="0 = no filtrar por año")
    parser.add_argument("--drop", default=DROP_DEFAULT,
                        help="columnas a excluir del informe, separadas por comas")
    parser.add_argument("--out", default="estados-finalizados/evidently/report_drift.html")
    args = parser.parse_args()

    drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    reference = load_slice(args.reference, args.reference_year, drop)
    current = load_slice(args.current, args.current_year, drop)
    # Ojo: el contract de pandera VETA los datos enfermos (s3); Evidently NO:
    # su trabajo es CUANTIFICAR el desplazamiento, no frenarlo. Compara
    # distribuciones columna a columna (el salto 2023->2025 aparece como drift
    # en sus gráficos, sin vetar el informe).
    print(f"[evidently] referencia {args.reference}"
          + (f" (source_year={args.reference_year})" if args.reference_year > 0 else "")
          + f" -> {len(reference)} filas")
    print(f"[evidently] actual    {args.current}"
          + (f" (source_year={args.current_year})" if args.current_year > 0 else "")
          + f" -> {len(current)} filas")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(args.out)
    print(f"[evidently] informe de drift guardado en {args.out}")

    # Resumen por consola: columnas cuyo test de drift dispara (p<=0.05).
    for metric in report.as_dict()["metrics"]:
        cols = (metric.get("result") or {}).get("drift_by_columns") or {}
        if cols:
            drifted = sorted(c for c, d in cols.items() if d.get("drift_detected"))
            print(f"[evidently] {len(reference)} filas ref vs {len(current)} actuales")
            print(f"[evidently] columnas con drift: {drifted or 'ninguna'}")


if __name__ == "__main__":
    main()
