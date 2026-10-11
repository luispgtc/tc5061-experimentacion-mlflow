"""Gráficas que se registran como artefactos."""

from matplotlib.figure import Figure  # sin pyplot: no abre ventanas al correr desde la terminal
from sklearn.metrics import ConfusionMatrixDisplay

LABELS = ["<=50K", ">50K"]


def confusion_matrix_figure(y_true, y_pred, title: str) -> Figure:
    fig = Figure(figsize=(5, 4.5))
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=LABELS, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig
