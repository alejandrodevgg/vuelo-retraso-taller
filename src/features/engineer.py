# src/features/engineer.py
# Feature engineering + DATA CONTRACT (pandera) for the FLIGHT-DELAY dataset.
#
# The contract is the section-3 device: it validates the RAW shape of the world
# (16 columns, ranges, nullability) BEFORE any feature math. Point the datos job
# at garbage and it dies red — garbage in must stop at the door, not train a
# garbage model.
#
# ─── FORBIDDEN COLUMNS (LEAKAGE) — read this before touching the feature list ───
# delay = actual_duration − expected_duration. Handing the model ANY component of
# that identity is handing it the answer:
#   * `delay_min`      — the label itself, in minutes.
#   * `delayed_15min`  — the label (0/1).
#   * `duration_min`   — actual duration: delay_min = duration_min −
#                        expected_duration_min, so this column leaks the label.
#   * `expected_duration_min` — the other half of the same identity.
#   * `unix_time`, `first_seen`/`last_seen` and raw `dof` timestamps — raw time
#     keys would let the model memorize "which days were bad" instead of learning
#     flight-level patterns (and they are not available pre-departure anyway).
# This is THE leakage lesson of the class: a model that filters on duration is a
# model that peeks at the answer key. `run_processing` keeps these columns for
# contract + bookkeeping; `FEATURE_COLUMNS` below is the only place that decides
# what the model is allowed to see.
# ────────────────────────────────────────────────────────────────────────────────
#
# Deliberate non-decisions (documented, not omissions):
#   * `icao_operator` is 38.8% NULL (verified anchor, docs/NUMEROS-VUELOS.md).
#     We do NOT impute here: the missingness IS the signal (cheap/LCC traffic
#     without ICAO attribution). It becomes an explicit "UNK" category inside
#     the model's preprocessor.
#   * `route` (24.802 distinct in 199k rows) is dropped: one-hot on it is
#     unfeasible; adep/ades separately scale (see workshop EDA).
#   * `dep_hour`/`dep_weekday` are CATEGORIES (time slots), not magnitudes: an
#     hour is not "double" another hour, so they one-hot, they do not scale.
import argparse
import logging

import pandas as pd
import pandera.pandas as pa
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("feature-engineering")

# Target label (BTS-standard: arrival delay > 15 min).
TARGET = "delayed_15min"

# What the model may see. Anything else in the CSV is contract bookkeeping,
# split metadata (`source_year`) or outright FORBIDDEN (see leakage banner).
FEATURE_COLUMNS = ["dep_hour", "dep_weekday", "adep", "ades", "icao_operator", "typecode"]

# Top-K collapse sizes (chosen with run evidence, 2026-11-02: typecode adds
# ~+0.03 test ROC-AUC, so it stays; airports top-20 / operators top-10 follow
# the anchor volume table in docs/NUMEROS-VUELOS.md).
TOP_K = {"adep": 20, "ades": 20, "icao_operator": 10, "typecode": 20}

# ---------------------------------------------------------------- contract ---
# Values describe the DATASET AS AGREED (the 16 columns emitted by
# scripts/make_flights_sample.py), not whatever the CSV happens to contain.
# Categories are deliberately permissive (no pattern on typecode/operator):
# the raw data has lowercase operators and non-ICAO airport codes (ES-0109,
# LR82, Z19O…) — collapsing them to OTH is the model's job, rejecting them
# here would be the contract lying about the world.
flights_contract = pa.DataFrameSchema(
    {
        "dof": pa.Column(
            str, pa.Check.str_matches(r"\d{4}-\d{2}-\d{2}"),
            title="departure date ISO YYYY-MM-DD",
        ),
        "source_year": pa.Column(
            int, pa.Check.isin([2023, 2024, 2025]), title="sample covers 2023-2025"
        ),
        "source_month": pa.Column(
            int, [pa.Check.ge(202301), pa.Check.le(202512)], title="YYYYMM"
        ),
        "dep_hour": pa.Column(int, [pa.Check.ge(0), pa.Check.le(23)]),
        "dep_weekday": pa.Column(int, [pa.Check.ge(0), pa.Check.le(6)], title="Mon=0"),
        "adep": pa.Column(
            str, pa.Check.str_matches(r"[A-Z0-9-]{2,7}"),
            title="departure aerodrome code (ICAO-ish: reality has non-ICAO too)",
        ),
        "ades": pa.Column(str, pa.Check.str_matches(r"[A-Z0-9-]{2,7}")),
        "route": pa.Column(str, pa.Check.str_contains("-"), title="adep-ades"),
        "flt_id": pa.Column(str, nullable=True),
        "typecode": pa.Column(str, nullable=True, title="aircraft type, 3.9% missing"),
        "icao_aircraft_class": pa.Column(str, nullable=True, title="14.8% missing"),
        "icao_operator": pa.Column(str, nullable=True, title="38.8% missing — by design"),
        # FORBIDDEN as features (leakage), contracted as data:
        "duration_min": pa.Column(float, pa.Check.ge(0), title="FORBIDDEN FEATURE: leaks the label"),
        "expected_duration_min": pa.Column(float, pa.Check.ge(0), title="FORBIDDEN FEATURE: leaks the label"),
        "delay_min": pa.Column(float, title="FORBIDDEN FEATURE: it IS the label in minutes"),
        TARGET: pa.Column(int, pa.Check.isin([0, 1]), title="arrival delay > 15 min"),
    },
    strict=True,  # unknown columns also break the contract
    coerce=True,  # int64 years etc. are cast to the declared dtypes before checks
)

# Anchor for the volume sanity band — see run_processing.py. The student may
# regenerate with another seed, so this is a band, never an equality.
ANCHOR_ROWS = 198_753
ROW_BAND_PCT = 0.02


def validate_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Raise a readable pandera SchemaErrors if the data broke the deal."""
    logger.info("Checking data contract (pandera)...")
    validated = flights_contract.validate(df, lazy=True)
    logger.info(f"Contract OK: {len(validated)} rows pass.")
    return validated


# -------------------------------------------------------------- features -----


class GroupCategories(BaseEstimator, TransformerMixin):
    """Collapse one raw column into a fixed set of groups (top-K + UNK + OTH).

    Travels INSIDE the logged model: training data and serving JSON go through
    the exact same function, so "unknown airport at 20:03 on a Friday" cannot
    be encoded two different ways. Unseen values -> `other`; nulls -> `fill_na`
    when the missingness is its own category (38.8% of operators!).
    """

    def __init__(self, known=None, fill_na=None, other="OTH"):
        self.known = known
        self.fill_na = fill_na
        self.other = other

    def fit(self, X, y=None):
        if self.known is None:  # defensive default; the workshop always passes top-K
            s = self._norm(X)
            self.known_ = sorted(s.dropna().unique().tolist())
        else:
            self.known_ = list(self.known)
        return self

    def _norm(self, X):
        s = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(X)
        s = s.astype("string").str.strip()
        # Numbers arrive as "8" or "8.0" depending on the source (CSV re-read
        # vs JSON int): canonicalize numerics to plain int strings once, here.
        if s.dropna().str.fullmatch(r"\d+(\.0)?").all():
            s = s.astype("float").astype("Int64").astype("string")
        else:
            s = s.str.upper()
        return s

    def transform(self, X, y=None):
        s = self._norm(X)
        if self.fill_na is not None:
            s = s.fillna(self.fill_na)
        s = s.where(s.isin(self.known_), self.other)
        return pd.DataFrame({"grp": s.fillna(self.other)})

    def get_feature_names_out(self, input_features=None):
        return pd.Index(self.known_ + ([self.other] if self.other not in self.known_ else []))


def build_group_maps(df: pd.DataFrame) -> dict:
    """Top-K levels per column, counted on the FIT slice only (never on test!).

    Ties are broken by code name (not by value_counts order) so the maps are
    reproducible across runs — same seed, same digits.
    """
    maps = {}
    for col, k in TOP_K.items():
        s = df[col].astype("string").str.strip().str.upper()
        vc = s.value_counts()
        vc = vc.reindex(sorted(vc.index)).sort_values(ascending=False, kind="stable")
        known = vc.head(k).index.tolist()
        fill_na = "UNK" if col in ("icao_operator", "typecode") else None
        if fill_na:  # UNK is a real category, it must survive the collapse
            known = sorted(set(known) | {"UNK"})
        maps[col] = {"known": known, "fill_na": fill_na}
    # Hours and weekdays are closed sets: every level is known by construction.
    maps["dep_hour"] = {"known": [str(h) for h in range(24)], "fill_na": None}
    maps["dep_weekday"] = {"known": [str(d) for d in range(7)], "fill_na": None}
    return maps


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Project the cleaned CSV down to the model-facing table.

    Unlike the house version there are no derived arithmetic features: hour and
    weekday are already slots, airports already keys, and every ratio we could
    compute from `duration_min`/`expected_duration_min` would be the leaked
    label in disguise (see leakage banner).
    """
    logger.info("Creating feature table (6 allowed features + label + split key)")
    out = df[FEATURE_COLUMNS + ["source_year", TARGET]].copy()
    logger.info(f"Featured shape: {out.shape}")
    return out


def create_preprocessor(group_maps: dict) -> ColumnTransformer:
    """Preprocessing pipeline packed INSIDE the trained model.

    One branch per column: GroupCategories (top-K + UNK + OTH) then one-hot.
    No scaler: after one-hot every feature is 0/1 — there is no magnitude to
    normalize (the house pipeline scaled sqft because it WAS a magnitude; this
    one honestly does not need it).
    """
    logger.info("Creating preprocessor pipeline (group collapse + one-hot)")
    transformers = [
        (
            col,
            Pipeline(
                steps=[
                    ("grp", GroupCategories(**group_maps[col])),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            [col],
        )
        for col in FEATURE_COLUMNS
    ]
    return ColumnTransformer(transformers=transformers)


def run_feature_engineering(input_file, output_file):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # 1) the contract gate (fails LOUD and RED on drifted/skewed input)
    df = validate_contract(df)

    # 2) feature projection (FORBIDDEN columns stop here, by construction)
    df_featured = create_features(df)

    df_featured.to_csv(output_file, index=False)
    logger.info(f"Saved featured (not encoded) data to {output_file}")
    return df_featured


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering for flight-delay data.")
    parser.add_argument("--input", required=True, help="Cleaned CSV file")
    parser.add_argument("--output", required=True, help="Output featured CSV")
    args = parser.parse_args()
    run_feature_engineering(args.input, args.output)
