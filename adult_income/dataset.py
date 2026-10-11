"""Carga y limpieza del dataset UCI Adult.

python -m adult_income.dataset      # descarga adult.data a data/raw/ si no existe
"""

import urllib.request

import numpy as np
import pandas as pd

from adult_income.config import COLUMNS, DATA_URL, EXCLUDED, RAW_FILE, TARGET


def download_raw() -> None:
    """Descarga adult.data de UCI a data/raw/ la primera vez; después usa la copia local."""
    if RAW_FILE.exists():
        return
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, RAW_FILE)


def load_raw() -> pd.DataFrame:
    """Lee adult.data tal como viene. skipinitialspace: el archivo separa con «, »."""
    download_raw()
    return pd.read_csv(RAW_FILE, header=None, names=COLUMNS, skipinitialspace=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las decisiones de limpieza que no aprenden nada de los datos.

    Lo que sí aprende de los datos (imputación, escalas, categorías) vive en el pipeline
    de features.py y se ajusta sólo con el conjunto de entrenamiento.
    """
    out = df.copy()
    out[TARGET] = (out[TARGET] == ">50K").astype(int)
    texto = out.select_dtypes(exclude="number").columns
    out[texto] = out[texto].replace("?", np.nan)  # faltantes codificados como «?»
    out = out.drop_duplicates().reset_index(drop=True)
    return out.drop(columns=EXCLUDED)


def load_dataset() -> pd.DataFrame:
    """Dataset limpio, listo para separar en entrenamiento y prueba."""
    return clean(load_raw())


if __name__ == "__main__":
    download_raw()
    print(f"Datos en {RAW_FILE}")
