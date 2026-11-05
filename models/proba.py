# src/models/proba.py
# The one adapter class that TRAVELS INSIDE the logged model artifact.
#
# Why a dedicated module: anything the pickle references must be importable at
# its recorded module path. Classes defined in a script run via
# `python -m src.models.train_model` serialize as `__main__.X` — and `__main__`
# does not exist in the API container. GroupCategories lives in engineer.py for
# the same reason. Gotcha paid 2026-11-02 (the house regression never hit it:
# its pipeline was 100% stock sklearn classes).
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ProbaPredictor(BaseEstimator):
    """Adapter: the pipeline's predict() returns P(label=1), not a hard class.

    pyfunc has no predict_proba — so probability becomes the model's native
    output and the decision threshold (0.5, documented in the API response) is
    serving-side policy.

    Deliberately NOT a ClassifierMixin: mlflow then treats it as a regressor
    for autolog purposes and does not try to compute class metrics over
    probabilities. The real metrics are the ones train_model logs explicitly.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.base import clone

        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y).astype(int))
        return self

    def predict(self, X):
        return self.estimator_.predict_proba(X)[:, 1]
