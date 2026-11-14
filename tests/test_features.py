import numpy as np
import pytest

from xsell.data import FEATURES
from xsell.features import MODEL_FEATURES, build_preprocessor


@pytest.mark.parametrize("model_type", ["linear", "tree"])
def test_preprocessor_output_is_numeric_without_nans(raw_df, model_type):
    X = raw_df[FEATURES]
    out = build_preprocessor(model_type).fit_transform(X)
    assert out.shape[0] == len(X)
    assert not np.isnan(out).any()


def test_preprocessor_handles_unseen_categories(raw_df):
    X = raw_df[FEATURES]
    preprocessor = build_preprocessor("tree").fit(X)

    new = X.head(1).copy()
    new["Region_Code"] = 999  # región que no existía al entrenar
    new["Policy_Sales_Channel"] = 999
    assert not np.isnan(preprocessor.transform(new)).any()


def test_preprocessor_drop_removes_variable(raw_df):
    X = raw_df[FEATURES]
    with_gender = build_preprocessor("tree").fit(X).get_feature_names_out()
    without_gender = build_preprocessor("tree", drop=("Gender",)).fit(X).get_feature_names_out()
    assert any(name.startswith("Gender") for name in with_gender)
    assert not any(name.startswith("Gender") for name in without_gender)


def test_preprocessor_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        build_preprocessor("otro")


def test_model_features_exclude_sensitive_and_useless_columns():
    assert "Gender" not in MODEL_FEATURES
    assert "Vintage" not in MODEL_FEATURES
    assert "id" not in MODEL_FEATURES
