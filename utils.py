"""
Utilidades para el sistema OCR clásico.

Incluye funciones de visualización, guardado/carga de modelos, logging
y métricas personalizadas (como top-k accuracy).
"""

from __future__ import annotations

import os
import logging
from typing import List, Sequence, Optional

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def setup_logging(log_path: Optional[str] = None) -> None:
    """
    Configura logging básico. Si se indica log_path, guarda a archivo.
    """
    handlers = []
    handlers.append(logging.StreamHandler())
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def ensure_dir(path: str) -> None:
    """
    Crea el directorio padre del path si no existe.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_model(model, path: str) -> None:
    """
    Guarda el modelo usando joblib.
    """
    ensure_dir(path)
    joblib.dump(model, path)


def load_model(path: str):
    """
    Carga el modelo desde path con joblib.
    """
    return joblib.load(path)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str = "Matriz de confusión",
                          figsize: tuple[int, int] = (8, 8), save_path: Optional[str] = None) -> None:
    """
    Dibuja y opcionalmente guarda la matriz de confusión.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=150)
    plt.show()


def visualize_predictions(images: List[np.ndarray], y_true: Sequence[int], y_pred: Sequence[int],
                          class_names: Sequence[str], cols: int = 8,
                          save_path: Optional[str] = None, title: str = "Predicciones") -> None:
    """
    Muestra una cuadrícula de imágenes con etiqueta verdadera y predicha.
    """
    n = len(images)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap="gray")
            t = class_names[y_true[i]] if (0 <= y_true[i] < len(class_names)) else str(y_true[i])
            p = class_names[y_pred[i]] if (0 <= y_pred[i] < len(class_names)) else str(y_pred[i])
            ax.set_title(f"T:{t} | P:{p}", fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=150)
    plt.show()


def compute_topk_accuracy(y_true: Sequence[int], y_proba: np.ndarray, k: int = 2) -> float:
    """
    Calcula top-k accuracy usando probabilidades por clase.
    """
    if y_proba.ndim != 2:
        raise ValueError("y_proba debe ser (n_samples, n_classes)")
    topk = np.argsort(-y_proba, axis=1)[:, :k]
    y_true = np.array(y_true)
    correct = np.any(topk == y_true[:, None], axis=1)
    return float(correct.mean())


if __name__ == "__main__":
    # Prueba mínima de top-k accuracy
    y_true = [0, 1, 2]
    y_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.2, 0.7, 0.1],
        [0.4, 0.3, 0.3],
    ])
    print("Top-2 accuracy:", compute_topk_accuracy(y_true, y_proba, k=2))

