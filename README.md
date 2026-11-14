# Venta cruzada de seguros

Modelo de propensión para identificar qué clientes de seguro de salud tienen más
probabilidad de estar interesados en un seguro de vehículo, planteado para que
otro equipo pueda usarlo y mantenerlo.

> Estado: en construcción. Fase actual: entrenamiento trazable y ciclo de vida del modelo.

## Estructura

```
├── configs/
│   └── train.yaml        # configuración de entrenamiento y promoción
├── data/raw/             # train.csv (no se versiona)
├── notebooks/
│   ├── 01_eda.ipynb      # análisis exploratorio y decisiones de modelado
│   └── 02_modelado.ipynb # comparación de modelos y evaluación final
├── src/xsell/
│   ├── config.py         # semilla, constantes y lectura de la configuración
│   ├── data.py           # carga, validación y partición del dataset
│   ├── features.py       # preprocesado de variables
│   ├── models.py         # pipelines candidatos (logística y XGBoost)
│   ├── evaluate.py       # métricas de ranking y tabla de lift
│   ├── train.py          # entrenamiento + registro en MLflow
│   ├── tune.py           # búsqueda de hiperparámetros trazada
│   └── promote.py        # validación y promoción a producción
└── pyproject.toml        # dependencias con versiones fijadas
```

## Puesta en marcha

Requiere Python 3.11 o superior.

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

### Notebooks

```bash
jupyter notebook notebooks/
```

### Entrenar y promocionar un modelo

```bash
python -m xsell.train      # entrena, evalúa y registra una versión "challenger"
python -m xsell.promote    # la pasa a "champion" si supera las validaciones
```

Ambos leen `configs/train.yaml`. Para explorar experimentos y versiones:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Búsqueda de hiperparámetros (opcional, unos minutos):

```bash
python -m xsell.tune --trials 12
```

## Ciclo de vida del modelo

Se gestiona con el registro de modelos de MLflow y dos alias:

| Estado | Cómo se llega | Qué significa |
|---|---|---|
| Experimento | `xsell.train` o `xsell.tune` | Ejecución trazada: parámetros, huella de los datos, métricas y artefactos. |
| `challenger` | `xsell.train` | Última versión entrenada, pendiente de validar. |
| `champion` | `xsell.promote` | Versión validada. Es la que sirve el endpoint. |
| Sin alias | Al ser sustituida | Versión retirada. Sigue en el registro para auditoría y vuelta atrás. |

**Validaciones para promocionar** (`promote.py`):

1. PR-AUC en test por encima de un mínimo absoluto (`promotion.min_test_pr_auc`).
2. PR-AUC en test no inferior al del `champion` actual. Es comparable porque la
   partición de test es siempre la misma (semilla fija).

**Vuelta atrás:** reasignar el alias `champion` a una versión anterior desde la
interfaz de MLflow. No hay que redesplegar código.

**Qué se registra en cada entrenamiento:** hiperparámetros, semilla, variables
usadas, hash SHA-256 y número de filas del fichero de datos, métricas de
validación cruzada y de test, tabla de lift, la configuración usada y el
pipeline completo (preprocesado + modelo) con su esquema de entrada.

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
- **Hiperparámetros sin ajustar**: la búsqueda aleatoria no mejoró los valores
  de partida más allá del ruido entre particiones. El límite lo ponen los
  datos, no el modelo.

### Reproducibilidad

- Una única semilla (`config.SEED`) para partición, validación cruzada, modelo
  y búsqueda de hiperparámetros. Dos entrenamientos con los mismos datos y la
  misma configuración dan exactamente las mismas predicciones.
- Versiones de librerías fijadas en `pyproject.toml`.
- Cada versión del modelo guarda el hash de los datos y la configuración con
  que se entrenó.
