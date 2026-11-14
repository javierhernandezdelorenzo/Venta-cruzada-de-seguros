"""Carga y validación del dataset de venta cruzada."""

from pathlib import Path

import pandas as pd

ID_COL = "id"
TARGET = "Response"

# Columnas que debe traer el fichero de entrada, además de id y target.
FEATURES = [
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]

# Códigos categóricos que el CSV trae como float (28.0, 152.0...).
CODE_COLS = ["Region_Code", "Policy_Sales_Channel"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Lee el CSV de train y comprueba que tiene el esquema esperado.

    Falla pronto (ValueError) si faltan columnas, hay nulos, ids repetidos
    o el target no es binario: preferimos parar aquí antes que entrenar
    con datos rotos.
    """
    df = pd.read_csv(path)

    expected = [ID_COL, *FEATURES, TARGET]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")

    if df[expected].isna().any().any():
        raise ValueError("El dataset contiene nulos y no se esperaba ninguno")

    if df[ID_COL].duplicated().any():
        raise ValueError("Hay ids de cliente duplicados")

    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("El target debe ser binario (0/1)")

    # Son identificadores, no magnitudes: los dejamos como enteros.
    df[CODE_COLS] = df[CODE_COLS].astype(int)

    return df[expected]
