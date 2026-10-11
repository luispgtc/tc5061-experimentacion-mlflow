# tc5061-experimentacion-mlflow

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Código de experimentación base con MLflow Tracking. Una regresión logística predice si una persona gana
más de 50,000 USD al año (`income`) con el dataset [UCI Adult](https://doi.org/10.24432/C5XW20).

TC5061 · Operaciones de aprendizaje automático (MLOps) · Semana 4 · Luis Pablo Guerrero Garza (A01794858)

## 1. Preparar el ambiente (sólo la primera vez)

Requiere Python 3.12. Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas con las que se obtuvieron los resultados e instala el paquete
`adult_income` en modo editable. Los datos no van en el repositorio: la primera corrida descarga
`adult.data` de UCI a `data/raw/`.

## 2. Entrenar

```bash
python -m adult_income.modeling.train
```

Sin argumentos corre con los valores por defecto. Cada ejecución crea una corrida en el experimento
`adult-income_logreg_hiperparametros` y al final imprime las métricas y el `run_id`.

Para cambiar hiperparámetros no hace falta editar el código:

```bash
python -m adult_income.modeling.train --C 0.01 --class-weight balanced
python -m adult_income.modeling.train --help
```

| Argumento | Defecto | Qué controla |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización: valores chicos regularizan más |
| `--l1-ratio` | `0.0` | Tipo de penalización: 0 = L2, 1 = L1, intermedio = elastic net (requiere `--solver saga`) |
| `--class-weight` | `none` | `balanced` pesa cada clase en proporción inversa a su frecuencia |
| `--max-iter` | `1000` | Iteraciones máximas del optimizador |
| `--solver` | `lbfgs` | `lbfgs` (sólo L2) o `saga` (L1 y elastic net) |
| `--seed` | `42` | Semilla del split, de la validación cruzada y del modelo |
| `--experiment-name` | `adult-income_logreg_hiperparametros` | Experimento de MLflow donde se agrupan las corridas |
| `--run-name` | se arma con los hiperparámetros | Nombre de la corrida |

### Corridas del experimento

Éstas son las corridas que se comparan en el reporte. Con `make` disponible, `make experiments` ejecuta las mismas.

```bash
python -m adult_income.modeling.train
python -m adult_income.modeling.train --C 0.01
python -m adult_income.modeling.train --C 100
python -m adult_income.modeling.train --class-weight balanced
python -m adult_income.modeling.train --class-weight balanced --C 0.01
python -m adult_income.modeling.train --solver saga --l1-ratio 1 --C 0.1
python -m adult_income.modeling.train --solver saga --l1-ratio 0.5
python -m adult_income.modeling.train --solver saga --max-iter 10
python -m adult_income.modeling.train --run-name repeticion_default
```

La última repite la configuración por defecto y debe dar exactamente las mismas métricas que la primera.

## 3. Ver y comparar las corridas

```bash
mlflow ui
```

Abre http://127.0.0.1:5000, entra al experimento `adult-income_logreg_hiperparametros`, selecciona las
corridas y da clic en **Compare**. La UI puede abrirse después de correr el script.

Si no se define `MLFLOW_TRACKING_URI`, MLflow 3.16 guarda las corridas en `mlflow.db` y los artefactos en
`mlruns/`, ambos en la carpeta desde la que se ejecuta. Por eso el script y `mlflow ui` se corren desde la raíz
del repositorio. Para usar otro servidor de tracking basta con definir esa variable.

## 4. Qué registra cada corrida

| Tipo | Contenido |
|---|---|
| Parámetros | Los 5 hiperparámetros, `seed`, `test_size`, `cv_folds`, tamaños de entrenamiento y prueba, y `data_sha256` (huella del archivo de datos) |
| Métricas (prueba) | `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `roc_auc` |
| Métricas (entrenamiento) | `cv_f1_mean` y `cv_f1_std` (validación cruzada estratificada de 5 pliegues) y `n_iter` (iteraciones usadas por el optimizador) |
| Artefactos | `matriz_confusion.png`, `classification_report.txt`, `metricas.json` |
| Modelo | El pipeline completo (preprocesamiento y regresión logística) con `mlflow.sklearn.log_model`, con firma y ejemplo de entrada |

La métrica principal es el **F1 de la clase `>50K`**. La clase positiva es sólo el 24 % de los datos, así que
un modelo que siempre responda `<=50K` ya obtiene 76 % de accuracy.

## 5. Reproducibilidad

- **Semilla:** `--seed` (por defecto 42) se pasa como `random_state` al split estratificado, a la validación
  cruzada (`StratifiedKFold`) y a `LogisticRegression`. Además se fijan `random.seed` y `numpy.random.seed`. Dos
  ejecuciones con los mismos argumentos producen las mismas métricas; la corrida `repeticion_default` lo comprueba.
- **Dependencias:** versiones fijadas con `==` en `requirements.txt` (Python 3.12.14).
- **Datos:** cada corrida registra el SHA-256 de `adult.data`. Si UCI cambiara el archivo, la huella lo
  delataría.
- **Sin fuga de información:** lo que aprende de los datos (imputación, escalas, categorías) está dentro del
  pipeline y se ajusta sólo con el conjunto de entrenamiento.

## 6. Estructura

Generada con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) 2.3.0 (`ccds`).

```
├── Makefile           <- make data / train / experiments / lint
├── README.md
├── pyproject.toml     <- Metadatos del paquete adult_income y configuración de ruff
├── requirements.txt   <- Dependencias con versión fija
├── data/raw/          <- adult.data (se descarga; no se versiona)
├── notebooks/
│   └── 1.0-lpgg-eda-adult-income.ipynb   <- Sólo exploración: EDA y evidencia de la limpieza
└── adult_income/      <- Código fuente
    ├── config.py      <- Rutas, columnas, semilla y nombre del experimento
    ├── dataset.py     <- Descarga, carga y limpieza (clean)
    ├── features.py    <- Pipeline de preprocesamiento
    ├── plots.py       <- Matriz de confusión
    └── modeling/
        └── train.py   <- Argumentos de línea de comandos, entrenamiento, evaluación y MLflow
```

El notebook no repite la lógica del script: importa `load_raw` y `clean` de `adult_income.dataset`.

## Fuente de datos

Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20

## Declaración de uso de inteligencia artificial

Anthropic. (2026). Claude Code (modelo `claude-opus-5-5`) [Modelo de lenguaje grande], aplicación de
escritorio, utilizado para generación de código y depuración. https://claude.com/claude-code

El trabajo se hizo con *human in the loop*: la IA proponía y ejecutaba, y el autor decidía en cada punto
de control. La IA generó la estructura con `ccds`, migró a este paquete el código del notebook baseline de la
semana 3, redactó este README, ejecutó las corridas y creó el repositorio con la sesión de GitHub del autor, a
petición de éste. El autor reunió y filtró la información del curso, decidió el alcance, la visibilidad y el
historial del repositorio, es autor del análisis de resultados del reporte (la IA sólo ordenó sus
respuestas) y revisó el código y el repositorio.
