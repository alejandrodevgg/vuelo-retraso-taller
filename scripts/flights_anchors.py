#!/usr/bin/env python3
"""Print the numeric anchors of the flights sample CSV to stdout.

Reads the CSV produced by scripts/make_flights_sample.py and prints, literally,
the only numbers that teaching material is allowed to quote: row counts (total
and per year), delayed>15min rates (total, per year, per month), delay_min
mean/median/p95, top departure airports and operators with their rates, route
cardinality and per-column missingness. Nothing is computed on the remote
source and nothing is invented: run it and copy its output.

Run from pipeline/ with an ephemeral env:

    cd talleres/mlops-pipeline/materiales-alumno/pipeline
    uv run --quiet --with pandas --with numpy \
        python scripts/flights_anchors.py

Real execution of record: 2026-11-02 (output archived in
talleres/mlops-pipeline/docs/NUMEROS-VUELOS.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        default=str(PIPELINE_ROOT / "data" / "raw" / "flights.csv"),
        help="Sample CSV produced by make_flights_sample.py.",
    )
    return parser.parse_args()


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    print(f"=== ANCLAS {args.csv} ===")
    print(f"filas totales: {len(df):,}")

    print("\n-- filas por ano --")
    for year, n in df["source_year"].value_counts().sort_index().items():
        print(f"{year}: {n:,}")

    print("\n-- tasa delayed_15min --")
    print(f"total: {pct(df['delayed_15min'].mean())}")
    for year, g in df.groupby("source_year"):
        print(f"{year}: {pct(g['delayed_15min'].mean())}")

    print("\n-- delay_min --")
    print(
        f"media: {df['delay_min'].mean():.2f} | "
        f"mediana: {df['delay_min'].median():.2f} | "
        f"p95: {df['delay_min'].quantile(0.95):.2f}"
    )

    print("\n-- top-10 aeropuertos de salida (volumen, tasa retraso) --")
    top = (
        df.groupby("adep")
        .agg(n=("delayed_15min", "size"), rate=("delayed_15min", "mean"))
        .sort_values("n", ascending=False)
    )
    for airport, row in top.head(10).iterrows():
        print(f"{airport}: {int(row['n']):,} vuelos, {pct(row['rate'])}")
    for special in ("LEBL", "LEMD"):
        if special in top.index:
            row = top.loc[special]
            rank = list(top.index).index(special) + 1
            print(
                f"{special} (puesto {rank}): {int(row['n']):,} vuelos, "
                f"{pct(row['rate'])}"
            )
        else:
            print(f"{special}: sin vuelos de salida en la muestra")

    print("\n-- tasa delayed_15min por mes (source_month YYYYMM) --")
    monthly = df.groupby("source_month")["delayed_15min"].mean()
    cells = " ".join(f"{m}={pct(v)}" for m, v in monthly.sort_index().items())
    print(cells)

    print("\n-- top-5 aerolineas ICAO (volumen, tasa retraso) --")
    ops = (
        df.groupby("icao_operator")
        .agg(n=("delayed_15min", "size"), rate=("delayed_15min", "mean"))
        .sort_values("n", ascending=False)
    )
    for op, row in ops.head(5).iterrows():
        print(f"{op}: {int(row['n']):,} vuelos, {pct(row['rate'])}")

    print("\n-- rutas --")
    print(f"routes distintas: {df['route'].nunique():,}")

    print("\n-- % filas faltantes por columna conservada --")
    missing = df.isna().mean() * 100
    for col, v in missing.items():
        print(f"{col}: {v:.4f}%")

    print("\n-- control de tipos (nulas no numericas en delay_min: "
          f"{int(df['delay_min'].isna().sum())}) --")
    assert np.isclose(df["delayed_15min"].isin([0, 1]).mean(), 1.0)


if __name__ == "__main__":
    main()
