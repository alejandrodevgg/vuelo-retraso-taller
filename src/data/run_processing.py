# src/data/run_processing.py
# Data engineering stage: raw flights CSV -> cleaned CSV, ready for the contract.
#
# Deliberate deltas vs the house version it was modernized from:
#   * VOLUME BAND first: ~199k rows in, ~199k rows out. A job that silently
#     processes 12 rows of a truncated download "succeeds" and trains garbage;
#     the band check makes an underfed pipeline impossible to miss. It is a
#     ±2% band around an executed anchor (docs/NUMEROS-VUELOS.md), never an
#     equality: the sample is regenerable (scripts/make_flights_sample.py) and
#     another seed legitimately shifts the row count.
#   * NO imputation of categorical nulls here. The house job median/mode-filled
#     everything on the way in; here `icao_operator` is 38.8% null BY NATURE
#     and that missingness is signal (see engineer.py). The model turns it into
#     an "UNK" category — the job must not erase it first.
#   * NO outlier clipping of the target. `delay_min` has a long tail (media
#     2.40, p95 22.62, negative tail = early flights): clipping it, or dropping
#     the delayed rows, would amputate the very drift story this dataset exists
#     to teach (6.81% -> 7.27% -> 10.26% by year). We clean shape, not truth.
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data-processor")

# Volume sanity band (±2% around the executed anchor). Import source: the
# anchors doc — regenerate both together if the sample ever changes.
from src.features.engineer import ANCHOR_ROWS, ROW_BAND_PCT

# Columns that must survive a row (a null here = corrupt row, drop and count).
REQUIRED_COLUMNS = [
    "dof", "source_year", "source_month", "dep_hour", "dep_weekday",
    "adep", "ades", "route", "delay_min", "delayed_15min",
]


def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    return df


def check_volume_band(df):
    """Fail RED if the raw input is not the ~199k-row sample we agreed on."""
    lo = int(ANCHOR_ROWS * (1 - ROW_BAND_PCT))
    hi = int(ANCHOR_ROWS * (1 + ROW_BAND_PCT))
    n = len(df)
    if not lo <= n <= hi:
        raise SystemExit(
            f"Volume sanity FAILED: {n} rows outside the band [{lo:,}, {hi:,}] "
            f"(anchor {ANCHOR_ROWS:,} ±{ROW_BAND_PCT:.0%}). The sample is truncated, "
            "stale, or from another build — regenerate it with "
            "scripts/make_flights_sample.py (see docs/NUMEROS-VUELOS.md)."
        )
    logger.info(f"Volume band OK: {n:,} rows inside [{lo:,}, {hi:,}]")


def clean_data(df):
    """Drop corrupt rows (nulls on required keys), cast dtypes. No imputation."""
    logger.info("Cleaning dataset")
    df_cleaned = df.copy()

    missing = df_cleaned[REQUIRED_COLUMNS].isnull().any(axis=1)
    if missing.any():
        logger.info(f"Dropping {int(missing.sum())} rows missing required keys")
        df_cleaned = df_cleaned[~missing]
    else:
        logger.info("No rows missing required keys (sampler guarantees this; belt + suspenders)")

    # Uniform dtypes for the contract: years/hours int, durations float.
    for col in ["source_year", "source_month", "dep_hour", "dep_weekday", "delayed_15min"]:
        df_cleaned[col] = df_cleaned[col].astype("int64")
    for col in ["duration_min", "expected_duration_min", "delay_min"]:
        df_cleaned[col] = df_cleaned[col].astype("float64")

    logger.info(f"Clean dataset shape: {df_cleaned.shape}")
    return df_cleaned


def process_data(input_file, output_file):
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(input_file)
    check_volume_band(df)
    df_cleaned = clean_data(df)

    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")
    return df_cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the raw flight-delay dataset.")
    parser.add_argument("--input", default="data/raw/flights.csv")
    parser.add_argument("--output", default="data/work/cleaned_flights.csv")
    args = parser.parse_args()
    process_data(args.input, args.output)
