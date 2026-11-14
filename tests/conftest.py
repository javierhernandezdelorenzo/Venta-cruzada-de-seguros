"""Datos sintéticos para los tests.

Los tests no dependen del fichero de Kaggle: así se pueden ejecutar en
cualquier máquina y en integración continua.
"""

import numpy as np
import pandas as pd
import pytest


def make_dataset(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Dataset pequeño con el mismo esquema que train.csv y algo de señal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "Gender": rng.choice(["Male", "Female"], n_rows),
            "Age": rng.integers(20, 80, n_rows),
            "Driving_License": 1,
            "Region_Code": rng.choice([8.0, 28.0, 41.0, 46.0], n_rows),
            "Previously_Insured": rng.integers(0, 2, n_rows),
            "Vehicle_Age": rng.choice(["< 1 Year", "1-2 Year", "> 2 Years"], n_rows),
            "Vehicle_Damage": rng.choice(["Yes", "No"], n_rows),
            "Annual_Premium": rng.uniform(2630, 60000, n_rows).round(),
            "Policy_Sales_Channel": rng.choice([26.0, 124.0, 152.0], n_rows),
            "Vintage": rng.integers(10, 300, n_rows),
        }
    )
    # Interesados: sobre todo quienes no tienen seguro y han tenido daños.
    p = 0.02 + 0.5 * ((df["Previously_Insured"] == 0) & (df["Vehicle_Damage"] == "Yes"))
    df["Response"] = (rng.random(n_rows) < p).astype(int)
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_dataset()


@pytest.fixture
def csv_path(tmp_path, raw_df):
    path = tmp_path / "train.csv"
    raw_df.to_csv(path, index=False)
    return path
