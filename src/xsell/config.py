"""Constantes compartidas por todo el proyecto."""

# Semilla única: se pasa a todo lo que tenga aleatoriedad (split, CV, modelos).
SEED = 42

# Fracción de train.csv que se aparta como test final. No se toca hasta el final.
TEST_SIZE = 0.2

# Nº de particiones de la validación cruzada sobre el conjunto de entrenamiento.
CV_FOLDS = 5
