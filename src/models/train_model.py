# src/models/train_model.py
# Experimentation stage on MLflow 3 — BINARY CLASSIFICATION of flight delays.
# Two candidates (LogisticRegression baseline + XGBoost booster), autolog,
# signature, registry hand-off. Same skeleton as the house regression, new math.
#
# TEMPORAL SPLIT (the drift story of the course):
#   fit  = 2023 + 80% of 2024          val = remaining 20% of 2024 (in-year)
#   test = ALL of 2025 (never seen, never touched)
# The base rate climbs 7.1% (fit) -> 10.3% (test 2025) on its own — the model
# was never told "2025 exists" — so val-vs-test is where calibration decay
# becomes a NUMBER, not a vibe. A random split would hide it.
#
# IMBALANCE (8:1 natural): both candidates handle it explicitly (LR
# class_weight="balanced", XGB scale_pos_weight) so precision/recall at the
# fixed 0.5 threshold are non-degenerate. Without it, both models sit at
# recall=0 predicting "all on time" and the confusion matrix is a lie by
# omission. Probed 2026-11-02.
#
# WHAT predict() MEANS: the logged pipeline ends in ProbaPredictor, so
# p.predict(df) is P(delay > 15 min) — one call, both for the local pkl (L0)
# and for pyfunc from the registry (L1). pyfunc has no predict_proba; the
# wrapper makes probability the model's native output and the API applies the
# documented 0.5 threshold for the class.
#
# MLflow 3 note: `transition_model_version_stage` is GONE in 3.x. Promotion is
# now done with ALIASES: client.set_registered_model_alias(name, "Production",
# version). That is what --register --alias Production below does.
import argparse
import json
import logging
import os
import platform

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import yaml
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:  # run as a module: `python -m src.models.train_model` from the repo root
    from src.features.engineer import (
        FEATURE_COLUMNS,
        TARGET,
        build_group_maps,
        create_preprocessor,
    )
    from src.models.proba import ProbaPredictor
except ImportError:  # run by path: `python src/models/train_model.py` (CI gist style)
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.features.engineer import (
        FEATURE_COLUMNS,
        TARGET,
        build_group_maps,
        create_preprocessor,
    )
    from src.models.proba import ProbaPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train-model")

DECISION_THRESHOLD = 0.5  # documented: probabilities are the artifact, 0.5 is the policy


def parse_args():
    parser = argparse.ArgumentParser(description="Train delay classifiers and log them to MLflow.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data", default="data/work/featured_flights.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--metrics-out", default=None,
                        help="optional JSON path with the summary metrics (CI gate reads it)")
    parser.add_argument("--mlflow-tracking-uri", default=None,
                        help="e.g. http://localhost:5555 (host) or http://mlflow:5000 (compose)")
    parser.add_argument("--register", action="store_true",
                        help="register the BEST run's model in the Model Registry")
    parser.add_argument("--alias", default=None,
                        help="registry alias for the new version, e.g. Production")
    return parser.parse_args()


def build_candidates(seed: int, scale_pos_weight: float):
    # Two models, no GridSearch bloat: a baseline and a booster. Both face the
    # 8:1 imbalance head-on — different weapons, same battlefield.
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            objective="binary:logistic", scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=seed,
        ),
    }


def temporal_split(data, cfg, seed):
    """fit = (train_years minus val_year) + (1-frac) of val_year; val = rest of
    val_year; test = test_year. The val_year carve-out must not re-include the
    whole val year in fit (a duplicated fit would silently leak val rows back
    in — caught by row arithmetic in the smoke, not by the metrics)."""
    split = cfg["split"]
    train_years, val_year, test_year = split["train_years"], split["val_year"], split["test_year"]
    frac = float(split["val_holdout_frac"])
    data = data.sort_values(["source_year", "dep_hour", "adep", "ades"], kind="stable")
    base_years = [y for y in train_years if y != val_year]
    pool = data[data.source_year == val_year]
    rng = np.random.RandomState(seed)
    is_val = rng.rand(len(pool)) < frac
    val = pool[is_val]
    fit = pd.concat([data[data.source_year.isin(base_years)], pool[~is_val]])
    test = data[data.source_year == test_year]
    if len(fit) + len(val) + len(test) != len(data):
        raise RuntimeError(
            f"Split arithmetic broken: fit({len(fit)}) + val({len(val)}) + "
            f"test({len(test)}) != total({len(data)}) — rows duplicated or lost."
        )
    logger.info(
        f"Temporal split: fit={len(fit):,} (rate {fit[TARGET].mean():.4f}) | "
        f"val={len(val):,} (rate {val[TARGET].mean():.4f}) | "
        f"test={len(test):,} (rate {test[TARGET].mean():.4f})"
    )
    return fit, val, test


def clf_metrics(y_true, proba, prefix):
    hard = (proba >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, hard, labels=[0, 1]).ravel()
    return {
        f"{prefix}_precision": float(precision_score(y_true, hard, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, hard)),
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, proba)),
        f"{prefix}_tp": float(tp),
        f"{prefix}_fp": float(fp),
        f"{prefix}_fn": float(fn),
        f"{prefix}_tn": float(tn),
        f"{prefix}_predicted_positive_rate": float(hard.mean()),
    }


def main(args):
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)
    model_cfg = cfg["model"]
    seed = int(cfg.get("random_state", 42))
    target = model_cfg["target_variable"]

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(model_cfg["experiment_name"])

    data = pd.read_csv(args.data)
    X_cols = FEATURE_COLUMNS  # the ONLY features: forbidden columns already died in engineer
    fit_df, val_df, test_df = temporal_split(data, cfg, seed)
    X_fit, y_fit = fit_df[X_cols], fit_df[target]
    X_val, y_val = val_df[X_cols], val_df[target]
    X_test, y_test = test_df[X_cols], test_df[target]
    logger.info(f"Data loaded: {data.shape}, features={X_cols}")

    scale_pos_weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
    logger.info(f"scale_pos_weight for XGBoost (from FIT only): {scale_pos_weight:.4f}")

    # The top-K group maps are computed on the FIT slice ONLY. Computing them
    # over the whole CSV would be a (mild, real) leak: 2025 airports in the
    # "top" set is knowing the future's traffic mix.
    group_maps = build_group_maps(fit_df)
    logger.info(
        "Group maps: "
        + " | ".join(f"{c}: {len(m['known'])} known" for c, m in group_maps.items())
    )

    # autolog: params, metrics, artifacts, environment AND signature.
    mlflow.sklearn.autolog()
    mlflow.xgboost.autolog()

    results = {}
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_param("split", f"fit=train_years+{1 - float(cfg['split']['val_holdout_frac']):.0%} of {cfg['split']['val_year']} | val={cfg['split']['val_year']} holdout | test={cfg['split']['test_year']}")
        mlflow.log_param("decision_threshold", DECISION_THRESHOLD)
        mlflow.log_text(json.dumps(group_maps, indent=2, default=str), "group_maps.json")
        for name, candidate in build_candidates(seed, scale_pos_weight).items():
            logger.info(f"Training {name}...")
            with mlflow.start_run(run_name=name, nested=True):
                # The preprocessor travels INSIDE the model (group collapse +
                # one-hot + estimator = one transform+predict unit).
                pipe = Pipeline([
                    ("preprocessor", create_preprocessor(group_maps)),
                    ("delay_model", ProbaPredictor(candidate)),
                ])
                pipe.fit(X_fit, y_fit)

                metrics = {}
                metrics.update(clf_metrics(y_val, pipe.predict(X_val), "val"))
                metrics.update(clf_metrics(y_test, pipe.predict(X_test), "test"))
                mlflow.log_metrics(metrics)
                run_id = mlflow.active_run().info.run_id
                results[name] = {"run_id": run_id, **metrics}
                logger.info(
                    f"{name} -> val_auc={metrics['val_roc_auc']:.4f} "
                    f"test_auc={metrics['test_roc_auc']:.4f} "
                    f"test_prec={metrics['test_precision']:.4f} "
                    f"test_rec={metrics['test_recall']:.4f}"
                )

    best_name = max(results, key=lambda k: results[k]["test_roc_auc"])
    best = results[best_name]
    logger.info(f"BEST: {best_name} (run {best['run_id']}) test_auc={best['test_roc_auc']:.4f}")

    if args.metrics_out:
        summary = {
            "best": best_name,
            "best_run_id": best["run_id"],
            "decision_threshold": DECISION_THRESHOLD,
            "models": results,
        }
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info(f"Metrics summary written to {args.metrics_out}")

    # Autolog OFF during the final refit: this one is for the local .pkl only.
    # (Gotcha observed 2026-11-01: `mlflow.autolog(disable=True)` did NOT stop
    # sklearn's patched fit in 3.17.2 and opened a stray fourth run; the
    # context manager below is the reliable switch.)
    from mlflow.utils.autologging_utils import disable_autologging

    # L0 hand-off kept ON PURPOSE for the comparison in section 6: a .pkl on
    # disk. It works — and nobody knows which one is on that disk.
    # Refit on FIT ONLY (not all data): the test year stays untouched, and the
    # pkl-vs-registry diff in s6 must isolate the SERVING path, not data drift.
    full_pipe = Pipeline([
        ("preprocessor", create_preprocessor(group_maps)),
        ("delay_model", ProbaPredictor(build_candidates(seed, scale_pos_weight)[best_name])),
    ])
    os.makedirs(f"{args.models_dir}/trained", exist_ok=True)
    pkl_path = f"{args.models_dir}/trained/flight_delay_model.pkl"
    with disable_autologging():
        full_pipe.fit(X_fit, y_fit)
        joblib.dump(full_pipe, pkl_path)
    logger.info(f"Saved local pkl (the L0 way): {pkl_path}")

    if args.register:
        client = MlflowClient()
        model_name = model_cfg["name"]
        try:
            client.create_registered_model(model_name)
            logger.info(f"Created registered model '{model_name}'")
        except mlflow.exceptions.MlflowException:
            logger.info(f"Registered model '{model_name}' already exists")
        # MLflow 3 gotcha (observed 2026-11-01 on 3.17.2): with --serve-artifacts
        # the logged model lives in a first-class LoggedModel entity (models:/m-...),
        # NOT in the run folder. Passing source="runs:/<id>/model" creates a version
        # whose artifacts cannot be resolved later ("No artifacts found to download"
        # + MLmodel missing). The working promotion recipe is the LoggedModel id:
        logged = [
            lm for lm in client.search_logged_models(experiment_ids=[mlflow.get_experiment_by_name(model_cfg["experiment_name"]).experiment_id])
            if lm.source_run_id == best["run_id"]
        ]
        if not logged:
            raise RuntimeError(f"No LoggedModel found for run {best['run_id']} — no hay nada que registrar.")
        model_id = logged[-1].model_id
        mv = client.create_model_version(
            name=model_name,
            source=f"models:/{model_id}",
            model_id=model_id,
            run_id=best["run_id"],
        )
        logger.info(f"Created version {mv.version} of '{model_name}' (from LoggedModel {model_id})")
        if args.alias:
            client.set_registered_model_alias(model_name, args.alias, mv.version)
            logger.info(f"Alias '{args.alias}' -> version {mv.version}")
        client.set_registered_model_tag(model_name, "best_algorithm", best_name)
        client.set_registered_model_tag(model_name, "task", "binary-classification-delay15")
        client.set_registered_model_tag(model_name, "split", "fit=2023+2024 | val=2024 holdout | test=2025")
        client.set_registered_model_tag(model_name, "python_version", platform.python_version())
        client.set_registered_model_tag(model_name, "scikit_learn_version", sklearn.__version__)
        client.set_registered_model_tag(model_name, "xgboost_version", xgb.__version__)


if __name__ == "__main__":
    main(parse_args())
