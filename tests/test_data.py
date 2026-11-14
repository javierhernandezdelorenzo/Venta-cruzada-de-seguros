import pytest

from xsell.data import FEATURES, ID_COL, TARGET, load_data, split_data


def test_load_data_returns_expected_schema(csv_path):
    df = load_data(csv_path)
    assert list(df.columns) == [ID_COL, *FEATURES, TARGET]
    # Los códigos de región y canal llegan como float y se convierten a entero.
    assert df["Region_Code"].dtype.kind == "i"
    assert df["Policy_Sales_Channel"].dtype.kind == "i"


def test_load_data_fails_on_missing_column(tmp_path, raw_df):
    path = tmp_path / "bad.csv"
    raw_df.drop(columns="Age").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Faltan columnas"):
        load_data(path)


def test_load_data_fails_on_nulls(tmp_path, raw_df):
    raw_df.loc[0, "Annual_Premium"] = None
    path = tmp_path / "bad.csv"
    raw_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nulos"):
        load_data(path)


def test_load_data_fails_on_duplicated_ids(tmp_path, raw_df):
    raw_df.loc[1, "id"] = raw_df.loc[0, "id"]
    path = tmp_path / "bad.csv"
    raw_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicados"):
        load_data(path)


def test_split_is_stratified_and_reproducible(csv_path):
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) + len(X_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01

    # Misma semilla -> misma partición.
    X_train_again, *_ = split_data(df)
    assert X_train.index.equals(X_train_again.index)
