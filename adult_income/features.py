"""Preprocesamiento: transformaciones que se ajustan sólo con el conjunto de entrenamiento."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from adult_income.config import CATEGORICAL, NUM_REST, NUM_SKEWED


def build_preprocessor() -> ColumnTransformer:
    """log1p + escala en las numéricas sesgadas, escala en el resto y one-hot en categóricas."""
    return ColumnTransformer(
        [
            # log1p comprime la cola (99999 -> ~11.5) y luego se estandariza.
            (
                "num_log",
                Pipeline(
                    [
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("escala", StandardScaler()),
                    ]
                ),
                NUM_SKEWED,
            ),
            ("num", StandardScaler(), NUM_REST),
            # Faltante -> categoría propia; categorías nuevas en producción se ignoran.
            (
                "cat",
                Pipeline(
                    [
                        ("imputa", SimpleImputer(strategy="constant", fill_value="Desconocido")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
