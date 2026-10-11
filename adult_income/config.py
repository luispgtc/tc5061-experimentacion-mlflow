"""Rutas y constantes del proyecto. Ningún otro módulo escribe una ruta a mano."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Fuente: UCI Machine Learning Repository, dataset «Adult» (Becker y Kohavi, 1996).
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
RAW_FILE = RAW_DATA_DIR / "adult.data"

# Nombres de columna en el orden de adult.names; el archivo no trae encabezado.
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
TARGET = "income"

# fnlwgt es un peso muestral, no un atributo de la persona; education es redundante con
# education-num (correspondencia uno a uno, verificada en el notebook de EDA).
EXCLUDED = ["fnlwgt", "education"]

NUM_SKEWED = ["capital-gain", "capital-loss"]  # más de 90 % de ceros y cola muy larga
NUM_REST = ["age", "education-num", "hours-per-week"]
CATEGORICAL = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
EXPERIMENT_NAME = "adult-income_logreg_hiperparametros"
