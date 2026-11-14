import numpy as np
import pytest

from xsell.data import FEATURES, TARGET
from xsell.features import EXCLUDED, MODEL_FEATURES
from xsell.models import build_logistic, build_xgboost


@pytest.mark.parametrize("build", [build_logistic, build_xgboost])
def test_pipeline_trains_and_returns_probabilities(raw_df, build):
    X, y = raw_df[FEATURES], raw_df[TARGET]
    proba = build().fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_xgboost_is_reproducible(raw_df):
    X, y = raw_df[MODEL_FEATURES], raw_df[TARGET]
    first = build_xgboost(drop=EXCLUDED).fit(X, y).predict_proba(X)
    second = build_xgboost(drop=EXCLUDED).fit(X, y).predict_proba(X)
    assert np.array_equal(first, second)


def test_xgboost_learns_the_signal(raw_df):
    """El modelo debe puntuar más alto, de media, a los interesados."""
    X, y = raw_df[MODEL_FEATURES], raw_df[TARGET]
    proba = build_xgboost(drop=EXCLUDED).fit(X, y).predict_proba(X)[:, 1]
    assert proba[y == 1].mean() > proba[y == 0].mean()
