# Venta cruzada de seguros

Modelo de propensión para identificar qué clientes de seguro de salud tienen más
probabilidad de estar interesados en un seguro de vehículo, planteado para que
otro equipo pueda usarlo y mantenerlo.

> Estado: en construcción. Fase actual: análisis exploratorio.

## Estructura

```
├── data/raw/            # train.csv (no se versiona)
├── notebooks/
│   └── 01_eda.ipynb     # análisis exploratorio y decisiones de modelado
├── src/xsell/
│   └── data.py          # carga y validación del dataset
└── pyproject.toml       # dependencias
```

## Puesta en marcha

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Datos

Descarga el dataset [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
de Kaggle y copia `train.csv` en `data/raw/`. Solo se usa el fichero de train:
el de test no trae etiqueta y no se puede evaluar.

### EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Decisiones principales

### Datos (ver `notebooks/01_eda.ipynb`)

- **Target desbalanceado (12,3 %)**: se evalúa con métricas de ranking (PR-AUC,
  lift y captura por deciles), no con accuracy.
- **Baseline de negocio**: la regla "sin seguro de vehículo y con daños previos"
  captura el 97,7 % de los interesados contactando al 48 % de la cartera. El
  modelo se juzga por lo que mejora sobre esa regla.
- **Variables excluidas**: `id` (identificador) y `Vintage` (sin señal).
- **Región y canal de venta**: categóricas, agrupando los valores poco frecuentes.
