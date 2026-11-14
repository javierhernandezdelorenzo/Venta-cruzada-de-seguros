"""Preprocesado de variables.

Las decisiones vienen del EDA (notebooks/01_eda.ipynb):

- `id` y `Vintage` no entran al modelo (identificador y variable sin señal).
- Región y canal son categóricas con muchas categorías raras: se agrupan las
  poco frecuentes en una sola.
- La edad tiene una relación no lineal con el target: el modelo lineal la
  necesita por tramos; el de árboles la aprende solo.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

BINARY_NUMERIC = ["Driving_License", "Previously_Insured"]  # ya vienen como 0/1
BINARY_TEXT = ["Gender", "Vehicle_Damage"]  # texto con dos valores
HIGH_CARDINALITY = ["Region_Code", "Policy_Sales_Channel"]
VEHICLE_AGE_ORDER = ["< 1 Year", "1-2 Year", "> 2 Years"]

# Variables que se excluyen del modelo final por decisión documentada en
# notebooks/02_modelado.ipynb: Gender es sensible y no mejora el ranking.
EXCLUDED = ("Gender",)

# Columnas que recibe el modelo, y por tanto las que pide el endpoint.
# Quedan fuera id, Vintage (sin señal) y las de EXCLUDED.
MODEL_FEATURES = [
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
]

# Una categoría de región o canal necesita al menos este peso en los datos de
# entrenamiento para tener columna propia; el resto se agrupa en "infrecuentes".
MIN_FREQUENCY = 0.01


def _one_hot_rare_grouped() -> OneHotEncoder:
    """One-hot que agrupa categorías raras y no falla con categorías nuevas."""
    return OneHotEncoder(
        min_frequency=MIN_FREQUENCY,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )


def build_preprocessor(model_type: str, drop: tuple[str, ...] = ()) -> ColumnTransformer:
    """Devuelve el preprocesado para un modelo "linear" o "tree".

    `drop` permite excluir variables (p. ej. Gender) sin tocar el resto.
    Las columnas no listadas en ningún bloque se descartan (remainder="drop").
    """
    if model_type == "linear":
        blocks = [
            # Edad en 10 tramos para capturar la forma de U invertida.
            (
                "age",
                KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile"),
                ["Age"],
            ),
            # Prima con cola muy larga: logaritmo y escalado.
            (
                "premium",
                make_pipeline(FunctionTransformer(np.log1p), StandardScaler()),
                ["Annual_Premium"],
            ),
            ("vehicle_age", OneHotEncoder(sparse_output=False), ["Vehicle_Age"]),
        ]
    elif model_type == "tree":
        blocks = [
            # Los árboles no necesitan tramos ni escalado.
            ("numeric", "passthrough", ["Age", "Annual_Premium"]),
            ("vehicle_age", OrdinalEncoder(categories=[VEHICLE_AGE_ORDER]), ["Vehicle_Age"]),
        ]
    else:
        raise ValueError(f"model_type debe ser 'linear' o 'tree', no {model_type!r}")

    # Bloques comunes a los dos tipos de modelo.
    blocks += [
        ("binary_numeric", "passthrough", BINARY_NUMERIC),
        ("binary_text", OneHotEncoder(drop="if_binary", sparse_output=False), BINARY_TEXT),
        ("high_cardinality", _one_hot_rare_grouped(), HIGH_CARDINALITY),
    ]

    transformers = []
    for name, transformer, columns in blocks:
        columns = [c for c in columns if c not in drop]
        if columns:
            transformers.append((name, transformer, columns))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
