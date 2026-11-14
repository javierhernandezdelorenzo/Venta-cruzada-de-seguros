# Venta cruzada de seguros

Modelo de propensión para identificar qué clientes de seguro de salud tienen más
probabilidad de estar interesados en un seguro de vehículo, planteado para que
otro equipo pueda usarlo y mantenerlo.

> Estado: en construcción. Fase actual: modelado y evaluación.

## Estructura

```
├── data/raw/            # train.csv (no se versiona)
├── notebooks/
│   ├── 01_eda.ipynb     # análisis exploratorio y decisiones de modelado
│   └── 02_modelado.ipynb # comparación de modelos y evaluación final
├── src/xsell/
│   ├── config.py        # semilla, tamaño de test, particiones de CV
│   ├── data.py          # carga, validación y partición del dataset
│   ├── features.py      # preprocesado de variables
│   ├── models.py        # pipelines candidatos (logística y XGBoost)
│   └── evaluate.py      # métricas de ranking y tabla de lift
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

### Modelo (ver `notebooks/02_modelado.ipynb`)

- **Validación**: 20 % apartado como test final estratificado; las decisiones se
  toman con validación cruzada de 5 particiones sobre el 80 % restante.
- **Modelo elegido: XGBoost.** PR-AUC 0,368 en test, frente a 0,354 de la
  regresión logística y 0,248 de la regla de negocio.
- **`Gender` excluida**: es una variable sensible y quitarla no cambia el
  resultado.
- **Sin reequilibrar clases**: el objetivo es ordenar clientes y las
  probabilidades salen bien calibradas tal cual.
- **Mensaje para negocio**: contactando al 20 % de la cartera con mayor score se
  captura el 58 % de los interesados (41 % eligiendo al azar dentro del segmento
  de la regla). Si se va a contactar a toda la mitad "buena" de la cartera, la
  regla sola es suficiente.
- **Reproducibilidad**: una única semilla (`config.SEED`) para partición,
  validación cruzada y modelos.
