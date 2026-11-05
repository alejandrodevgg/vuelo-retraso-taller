# src/run_pipeline.py
# Entrypoint of the `datos` job: raw -> cleaned -> featured (+ contract gate).
# One command, two stages, exit code 1 if the contract breaks.
#
# Point it at a WRONG schema to see the gate work (strict=True kills it — the
# house CSV has 7 columns of a different world):
#   docker compose run --rm -e RAW_FILE=data/raw/house_data.csv datos
import logging
import os

from src.data.run_processing import process_data
from src.features.engineer import run_feature_engineering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datos-job")


def main():
    raw_file = os.getenv("RAW_FILE", "data/raw/flights.csv")
    cleaned = os.getenv("CLEANED_FILE", "data/work/cleaned_flights.csv")
    featured = os.getenv("FEATURED_FILE", "data/work/featured_flights.csv")

    logger.info(f"=== datos job: {raw_file} -> {cleaned} -> {featured} ===")
    process_data(raw_file, cleaned)
    run_feature_engineering(cleaned, featured)
    logger.info("datos job OK")


if __name__ == "__main__":
    main()
