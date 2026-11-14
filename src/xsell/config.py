"""Constantes compartidas por todo el proyecto y lectura de la configuración."""

from pathlib import Path

import yaml

# Semilla única: se pasa a todo lo que tenga aleatoriedad (split, CV, modelos).
SEED = 42

# Fracción de train.csv que se aparta como test final. No se toca hasta el final.
TEST_SIZE = 0.2

# Nº de particiones de la validación cruzada sobre el conjunto de entrenamiento.
CV_FOLDS = 5

# Alias del registro de modelos que definen el ciclo de vida (ver README).
ALIAS_CHALLENGER = "challenger"  # última versión entrenada, pendiente de validar
ALIAS_CHAMPION = "champion"  # versión validada, la que sirve el endpoint


def load_config(path: str | Path) -> dict:
    """Lee el YAML de configuración de entrenamiento."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
