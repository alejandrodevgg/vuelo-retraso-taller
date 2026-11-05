#!/usr/bin/env python3
"""Deterministic sampler for the Eurocontrol OPDI flight-delays parquet.

Streams the remote Hugging Face parquet (row-group range reads via fsspec +
pyarrow; the 787 MB file is NEVER downloaded whole) and writes a stratified
natural random sample to CSV for the MLOps classification practice.

- Raw source: https://huggingface.co/datasets/345rf4gt56t4r3e3/flight-delays-europe-2023-2025/resolve/main/merged_flights_with_delay_2023_2025.parquet
  Eurocontrol OPDI-derived dataset, Apache-2.0 (see its HF README: the authors
  warn it is "not an accurate research source"; we use it for teaching only).
- Sampling is a plain uniform Bernoulli draw per row (SEED=42): no class
  rebalancing and no per-year quota, so the natural per-year delayed>15min
  rates (documented drift ~6.97% / 7.43% / 10.37% for 2023/2024/2025 on the
  full 15.3M rows) are preserved as-is in the sample.
- Default --frac is 0.013, sized to land ~150-250k rows on the 15,300,770-row
  source. (The originally drafted frac ~0.04 would yield ~612k rows, well
  above the target band; 0.013 was chosen instead. Override with --frac.)
- Validity filters before sampling: adep/ades non-null and non-empty,
  delay_min and delayed_15min non-null, dof (a timestamp[ns] column, despite
  docs calling it a YYYY-MM-DD string) within 2023-2025.
- Rerun-safe: overwrites the output CSV. Deterministic given the same
  --seed/--frac against the same source parquet (single shared RNG stream in
  fixed row-group/batch order; a different source file reorders draws).

Run from pipeline/ with an ephemeral env (no project venv touched):

    cd talleres/mlops-pipeline/materiales-alumno/pipeline
    uv run --quiet --with pyarrow --with pandas --with fsspec --with aiohttp \
        python scripts/make_flights_sample.py

Real execution of record: 2026-11-02 (anchors documented in
talleres/mlops-pipeline/docs/NUMEROS-VUELOS.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

URL = (
    "https://huggingface.co/datasets/345rf4gt56t4r3e3/"
    "flight-delays-europe-2023-2025/resolve/main/"
    "merged_flights_with_delay_2023_2025.parquet"
)

PIPELINE_ROOT = Path(__file__).resolve().parents[1]

# Source columns we stream (column pruning keeps network I/O small).
KEEP = [
    "dof",
    "source_year",
    "source_month",
    "first_seen",
    "adep",
    "ades",
    "route",
    "flt_id",
    "typecode",
    "icao_aircraft_class",
    "icao_operator",
    "duration_min",
    "expected_duration_min",
    "delay_min",
    "delayed_15min",
]

# Output column order in the CSV.
OUT_COLUMNS = [
    "dof",
    "source_year",
    "source_month",
    "dep_hour",
    "dep_weekday",
    "adep",
    "ades",
    "route",
    "flt_id",
    "typecode",
    "icao_aircraft_class",
    "icao_operator",
    "duration_min",
    "expected_duration_min",
    "delay_min",
    "delayed_15min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=str(PIPELINE_ROOT / "data" / "raw" / "flights.csv"),
        help="Output CSV path (default: data/raw/flights.csv under pipeline/).",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.013,
        help="Bernoulli sampling fraction per valid row (default 0.013 -> ~200k rows).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    parser.add_argument("--url", default=URL, help="Source parquet URL.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=262144,
        help="Rows per streamed batch (fixed; keeps the RNG stream deterministic).",
    )
    return parser.parse_args()


def valid_mask(table: pa.Table) -> pa.Array:
    """Row validity filters, applied before the sampling draw."""
    year = pc.year(table["dof"])
    masks = [
        pc.and_(pc.is_valid(table["adep"]), pc.not_equal(table["adep"], pa.scalar(""))),
        pc.and_(pc.is_valid(table["ades"]), pc.not_equal(table["ades"], pa.scalar(""))),
        pc.is_valid(table["delay_min"]),
        pc.is_valid(table["delayed_15min"]),
        pc.and_(
            pc.is_valid(table["dof"]),
            pc.and_(pc.greater_equal(year, 2023), pc.less_equal(year, 2025)),
        ),
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = pc.and_(combined, m)
    return combined


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = PIPELINE_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    chunks: list[pd.DataFrame] = []
    scanned = kept = 0

    print(f"streaming {args.url}", file=sys.stderr, flush=True)
    with fsspec.open(args.url) as f:
        pf = pq.ParquetFile(f)
        total = pf.metadata.num_rows
        print(
            f"source: {total:,} rows, {pf.metadata.num_row_groups} row groups",
            file=sys.stderr,
            flush=True,
        )
        for i, batch in enumerate(
            pf.iter_batches(columns=KEEP, batch_size=args.batch_size)
        ):
            table = pa.Table.from_batches([batch])
            table = table.filter(valid_mask(table))
            scanned += batch.num_rows
            n = table.num_rows
            if n:
                drawn = rng.random(n) < args.frac
                table = table.filter(pa.array(drawn))
            kept += table.num_rows
            if table.num_rows:
                df = table.to_pandas()
                # Derived departure-time features (first_seen is UTC timestamp[ns]).
                # pandas dayofweek is Mon=0..Sun=6 and version-stable (Arrow's
                # day_of_week default differs from its documented one here).
                fs = pd.to_datetime(df["first_seen"])
                df["dep_hour"] = fs.dt.hour.astype("Int64")
                df["dep_weekday"] = fs.dt.dayofweek.astype("Int64")
                df["dof"] = pd.to_datetime(df["dof"]).dt.strftime("%Y-%m-%d")
                chunks.append(df[OUT_COLUMNS])
            print(
                f"batch {i}: scanned {scanned:,}/{total:,} rows, kept {kept:,}",
                file=sys.stderr,
                flush=True,
            )

    sample = pd.concat(chunks, ignore_index=True)
    sample.to_csv(out, index=False)
    by_year = sample["source_year"].value_counts().sort_index()
    print(
        f"WROTE {out} — {len(sample):,} rows ({scanned:,} scanned), "
        f"{out.stat().st_size / 1e6:.1f} MB; by year: "
        + ", ".join(f"{y}: {n:,}" for y, n in by_year.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
