"""Entrena la regresión logística y registra la corrida en MLflow Tracking.

python -m adult_income.modeling.train
python -m adult_income.modeling.train --C 0.01 --class-weight balanced
python -m adult_income.modeling.train --help
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from adult_income.config import (
    CV_FOLDS,
    EXPERIMENT_NAME,
    NUM_REST,
    NUM_SKEWED,
    RAW_FILE,
    SEED,
    TARGET,
    TEST_SIZE,
)
from adult_income.dataset import load_dataset
from adult_income.features import build_preprocessor
from adult_income.plots import LABELS, confusion_matrix_figure


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regresión logística sobre UCI Adult (ingreso > 50K) registrada en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización: valores chicos regularizan más.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.0,
        help="Tipo de penalización: 0 = L2, 1 = L1, intermedio = elastic net. "
        "Un valor mayor que 0 requiere --solver saga.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="balanced pesa cada clase en proporción inversa a su frecuencia (>50K es ~24 %%).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Iteraciones máximas del optimizador.",
    )
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "saga"],
        default="lbfgs",
        help="Optimizador. lbfgs sólo admite L2; saga admite L1 y elastic net.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla del split, de la validación cruzada y del modelo.",
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="Experimento de MLflow donde se agrupan las corridas.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre de la corrida; si se omite se arma con los hiperparámetros.",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.l1_ratio <= 1.0:
        parser.error("--l1-ratio debe estar entre 0 y 1")
    if args.l1_ratio > 0 and args.solver != "saga":
        parser.error("--l1-ratio mayor que 0 requiere --solver saga")
    return args


def set_seed(seed: int) -> None:
    """Fija las semillas globales; scikit-learn además recibe random_state en cada paso."""
    random.seed(seed)
    np.random.seed(seed)


def split(df: pd.DataFrame, seed: int):
    """80/20 estratificado: prueba conserva la proporción de >50K."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)


def build_model(args: argparse.Namespace) -> Pipeline:
    classifier = LogisticRegression(
        C=args.C,
        l1_ratio=args.l1_ratio,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        max_iter=args.max_iter,
        solver=args.solver,
        random_state=args.seed,
    )
    return Pipeline([("prep", build_preprocessor()), ("modelo", classifier)])


def evaluate(model: Pipeline, X, y) -> dict[str, float]:
    """Métricas en prueba para la clase positiva (>50K)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def file_sha256(path: Path) -> str:
    """Huella del archivo de datos: identifica la versión exacta con la que se entrenó."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_run(args, model, metrics, cv_scores, X_train, X_test, y_test) -> str:
    """Registra parámetros, métricas, artefactos y el modelo en una corrida de MLflow."""
    mlflow.set_experiment(args.experiment_name)
    run_name = args.run_name or (
        f"C={args.C}_l1={args.l1_ratio}_cw={args.class_weight}_{args.solver}_it{args.max_iter}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("actividad", "S04_A1")
        mlflow.log_params(
            {
                "C": args.C,
                "l1_ratio": args.l1_ratio,
                "class_weight": args.class_weight,
                "max_iter": args.max_iter,
                "solver": args.solver,
                "seed": args.seed,
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "data_sha256": file_sha256(RAW_FILE),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())
        mlflow.log_metric("n_iter", int(model.named_steps["modelo"].n_iter_[0]))

        pred = model.predict(X_test)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fig = confusion_matrix_figure(y_test, pred, run_name)
            fig.savefig(tmp / "matriz_confusion.png", dpi=120)
            report = classification_report(y_test, pred, target_names=LABELS, digits=3)
            (tmp / "classification_report.txt").write_text(report, encoding="utf-8")
            (tmp / "metricas.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            for name in ("matriz_confusion.png", "classification_report.txt", "metricas.json"):
                mlflow.log_artifact(str(tmp / name))

        # Numéricas como float en el ejemplo: así la firma admite faltantes en producción.
        example = X_train.head(5).astype({c: "float64" for c in NUM_SKEWED + NUM_REST})
        # Formato skops (el seguro por defecto); OneHotEncoder usa numpy.dtype.
        mlflow.sklearn.log_model(
            model, name="model", input_example=example, skops_trusted_types=["numpy.dtype"]
        )
        return run.info.run_id


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    X_train, X_test, y_train, y_test = split(load_dataset(), args.seed)
    model = build_model(args)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    run_id = log_run(args, model, metrics, cv_scores, X_train, X_test, y_test)

    print(" ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"cv_f1={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Run ID: {run_id} | Model URI: runs:/{run_id}/model")


if __name__ == "__main__":
    main()
