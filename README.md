# Venta cruzada de seguros

Modelo de propensión para identificar qué clientes de seguro de salud tienen más
probabilidad de estar interesados en un seguro de vehículo, planteado para que
otro equipo pueda usarlo y mantenerlo.

**Resultado en una frase:** contactando al 20 % de la cartera con mayor score se
captura el 58 % de los clientes interesados, y la mitad de la cartera se puede
descartar perdiendo solo el 1 %.

## Contenido

```
├── .github/workflows/ci.yml   # mismas validaciones, en GitHub
├── .pre-commit-config.yaml    # validación automática en local
├── configs/
│   └── train.yaml             # configuración de entrenamiento y promoción
├── data/raw/                  # train.csv (no se versiona)
├── notebooks/
│   ├── 01_eda.ipynb           # análisis exploratorio y decisiones de modelado
│   └── 02_modelado.ipynb      # comparación de modelos, evaluación y mensaje a negocio
├── src/xsell/
│   ├── config.py              # semilla, constantes y lectura de la configuración
│   ├── data.py                # carga, validación y partición del dataset
│   ├── features.py            # preprocesado de variables
│   ├── models.py              # pipelines (logística y XGBoost)
│   ├── evaluate.py            # métricas de ranking y tabla de lift
│   ├── train.py               # entrenamiento + registro en MLflow
│   ├── tune.py                # búsqueda de hiperparámetros trazada
│   ├── promote.py             # validación y promoción a producción
│   └── api.py                 # endpoint de predicción (FastAPI)
├── tests/                     # tests unitarios y de integración
└── pyproject.toml             # dependencias con versiones fijadas
```

## Cómo ejecutarlo

Requiere Python 3.11 o superior.

### 1. Entorno

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Datos

Descarga el dataset [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
de Kaggle y copia `train.csv` en `data/raw/`. Solo se usa el fichero de train:
el de test no trae etiqueta y no se puede evaluar.

### 3. Entrenar y promocionar

```bash
python -m xsell.train      # entrena, evalúa y registra una versión "challenger"
python -m xsell.promote    # la pasa a "champion" si supera las validaciones
```

Ambos leen `configs/train.yaml`. El entrenamiento tarda menos de un minuto.

### 4. Servir el modelo

```bash
uvicorn xsell.api:app --port 8000
```

Documentación interactiva en http://127.0.0.1:8000/docs. Ejemplo de petición:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customers": [{"Age": 44, "Driving_License": 1, "Region_Code": 28,
       "Previously_Insured": 0, "Vehicle_Age": "> 2 Years", "Vehicle_Damage": "Yes",
       "Annual_Premium": 40454, "Policy_Sales_Channel": 26}]}'
```

```json
{"model_version": "1", "probabilities": [0.4377]}
```

- `POST /predict`: de 1 a 1.000 clientes por petición; devuelve la probabilidad
  de interés de cada uno, en el mismo orden, y la versión del modelo usada.
- `GET /health`: estado del servicio y versión del modelo cargado.
- Las entradas inválidas (edad fuera de rango, categoría desconocida, campo
  ausente) se rechazan con un 422 que indica el campo.
- Si no hay ningún modelo `champion`, la API responde 503.

### 5. Validación automática

```bash
pre-commit run --all-files   # ejecutar todas las validaciones a mano
pre-commit install           # que se ejecuten solas en cada commit
```

Ejecuta, en este orden: `ruff check` (errores y estilo), `ruff format --check`
(formato) y `pytest` (29 tests). Los tests usan datos sintéticos, así que no
necesitan el fichero de Kaggle. Incluyen un test de integración que recorre el
flujo completo: entrenar, promocionar, arrancar la API y predecir.

GitHub Actions ejecuta lo mismo en cada push y pull request.

### Otros comandos

```bash
jupyter notebook notebooks/                            # notebooks de análisis
mlflow ui --backend-store-uri sqlite:///mlflow.db      # experimentos y versiones
python -m xsell.tune --trials 12                       # búsqueda de hiperparámetros
```

## Parte 1: el modelo

El detalle está en los dos notebooks. Resumen:

### Qué hay en los datos (`01_eda.ipynb`)

- 381.109 clientes, sin nulos ni duplicados. **12,3 % de interesados.**
- Dos variables lo explican casi todo: quien **ya tiene seguro de vehículo** no
  está interesado (0,09 %) y quien **no ha tenido daños** tampoco (0,5 %).
- **Baseline de negocio**: la regla "sin seguro y con daños previos" captura el
  97,7 % de los interesados contactando al 48 % de la cartera.
- La edad tiene una relación no lineal con el interés (máximo entre 30 y 50 años).
- Región y canal de venta tienen muchas categorías con muy pocos clientes.
- `Vintage` no tiene ninguna relación con el target.

### Decisiones

| Decisión | Motivo |
|---|---|
| Métricas de ranking (PR-AUC, lift, captura) | Target desbalanceado y uso real: decidir a quién contactar primero. |
| Comparar contra la regla, no contra el azar | Es lo que negocio puede hacer hoy sin modelo. |
| 20 % de test apartado + CV de 5 particiones | Las decisiones se toman con CV; test se usa una vez. |
| XGBoost como modelo final | Mejor que la logística de forma consistente y sin tramos manuales de edad. |
| Excluir `Gender` | Variable sensible; quitarla no cambia el resultado. |
| Excluir `Vintage` e `id` | Sin señal e identificador. |
| Agrupar regiones y canales con menos del 1 % | Evita aprender de muy pocos casos y admite categorías nuevas. |
| No reequilibrar clases | No mejora el ranking y las probabilidades salen calibradas. |
| No ajustar hiperparámetros | La búsqueda no mejoró más allá del ruido entre particiones. |

### Resultados en test

| | PR-AUC | ROC-AUC | Interesados capturados contactando al 20 % |
|---|---|---|---|
| Regla de negocio | 0,248 | 0,785 | 40,9 % (al azar dentro del segmento) |
| Regresión logística | 0,354 | 0,853 | 56,7 % |
| **XGBoost** | **0,368** | **0,857** | **57,9 %** |

### Qué le diría a negocio

1. La mitad de la cartera prácticamente no tiene interesados: no merece la pena
   contactarla.
2. Si hay presupuesto para contactar a toda la otra mitad, la regla simple basta:
   el modelo apenas la mejora (98,2 % frente a 97,7 %).
3. Si el presupuesto es menor, el modelo indica por dónde empezar: en el 10 % con
   mayor score, 4 de cada 10 clientes están interesados (3,2 veces la media).
4. El punto de corte óptimo depende del coste por contacto y del valor de una
   póliza; con esos dos datos se calcula hasta dónde es rentable llamar.

### Limitaciones

- `Response` es interés declarado, no contratación.
- No hay fechas: no se puede validar en el tiempo.
- El canal de venta describe cómo llegó el cliente, no al cliente. Si cambia la
  estrategia comercial, su relación con el interés puede cambiar.

## Parte 2: producción

### Experimentos y versiones trazables

MLflow, con SQLite local como almacén. Cada entrenamiento registra:
hiperparámetros, semilla, variables usadas, hash SHA-256 y número de filas del
fichero de datos, métricas de validación cruzada y de test, tabla de lift, la
configuración usada y el pipeline completo (preprocesado + modelo) con su
esquema de entrada.

### Ciclo de vida del modelo

Registro de modelos de MLflow con dos alias:

| Estado | Cómo se llega | Qué significa |
|---|---|---|
| Experimento | `xsell.train` o `xsell.tune` | Ejecución trazada. |
| `challenger` | `xsell.train` | Última versión entrenada, pendiente de validar. |
| `champion` | `xsell.promote` | Versión validada. Es la que sirve el endpoint. |
| Sin alias | Al ser sustituida | Versión retirada. Sigue en el registro para auditoría y vuelta atrás. |

Validaciones para promocionar (`promote.py`):

1. PR-AUC en test por encima de un mínimo absoluto (`promotion.min_test_pr_auc`).
2. PR-AUC en test no inferior al del `champion` actual. Es comparable porque la
   partición de test es siempre la misma.

Vuelta atrás: reasignar el alias `champion` a una versión anterior desde la
interfaz de MLflow y reiniciar la API. No hay que tocar código.

### Reproducibilidad

- Una única semilla (`config.SEED`) para partición, validación cruzada, modelo y
  búsqueda de hiperparámetros. Dos entrenamientos con los mismos datos y la misma
  configuración dan exactamente las mismas predicciones (hay un test que lo
  comprueba).
- Versiones de librerías fijadas en `pyproject.toml`.
- Cada versión del modelo guarda el hash de los datos y la configuración con que
  se entrenó.

### Endpoint

FastAPI. Carga el `champion` al arrancar, valida la entrada con pydantic y
devuelve la versión del modelo en cada respuesta, para poder trazar cualquier
predicción hasta su entrenamiento.

### Integración y validación automática

`pre-commit` con hooks locales (ruff + pytest), ejecutable a mano o en cada
commit, y el mismo flujo en GitHub Actions.

## Qué he priorizado y qué queda fuera

**Priorizado:** que el flujo completo (datos, entrenamiento, registro,
promoción, servicio) funcione de principio a fin, sea reproducible y esté
cubierto por tests, con el menor número de piezas posible para que sea fácil de
entender y mantener.

**Fuera, y qué haría con más tiempo:**

- **Monitorización**: registrar las peticiones y vigilar la distribución de las
  variables (sobre todo canal de venta y edad) y de los scores. Sin la respuesta
  real del cliente no se puede medir el rendimiento directamente, así que el
  drift de entrada sería la primera alarma.
- **Reentrenamiento**: definir el disparador (periódico o por drift) y renovar
  el conjunto de test con datos nuevos, para no acabar sobreajustando al actual
  a base de comparar candidatos contra él.
- **Despliegue**: contenedor Docker con el modelo incluido en la imagen, y
  MLflow en un servidor compartido en lugar de SQLite local.
- **Validación de datos más rica**: hoy se comprueban esquema, nulos y
  duplicados; faltarían rangos y distribuciones esperadas.
- **Punto de corte de negocio**: con el coste por contacto y el valor de una
  póliza, convertir el score en una decisión de contactar o no.
