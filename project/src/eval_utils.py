from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve
from sklearn.calibration import CalibrationDisplay


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_plots(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    threshold: float = 0.5,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_roc.png", dpi=160)
    plt.close()

    # Precision-Recall curve
    plt.figure(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.title(f"Precision-Recall Curve (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pr.png", dpi=160)
    plt.close()

    # Confusion matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_confusion.png", dpi=160)
    plt.close()

    # Calibration curve
    plt.figure(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, strategy="quantile")
    bs = brier_score_loss(y_true, y_prob)
    plt.title(f"Calibration Curve (Brier={bs:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_calibration.png", dpi=160)
    plt.close()

    # Predicted probability distribution
    plt.figure(figsize=(6, 4.5))
    sns.histplot(y_prob[y_true == 0], bins=30, stat="density", color="#4C72B0", alpha=0.55, label="true=0")
    sns.histplot(y_prob[y_true == 1], bins=30, stat="density", color="#DD8452", alpha=0.55, label="true=1")
    plt.xlabel("Predicted P(cancel)")
    plt.ylabel("Density")
    plt.title("Predicted Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pred_hist.png", dpi=160)
    plt.close()


def save_learning_curves(
    *,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    seed: int = 42,
) -> None:
    """
    Generic learning curves (train size vs score) for report visuals.
    Uses ROC-AUC and F1 (thresholded at 0.5 via sklearn's default predict).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sizes = np.linspace(0.1, 1.0, 6)
    for scoring, title in [("roc_auc", "Learning Curve (ROC-AUC)"), ("f1", "Learning Curve (F1)")]:
        sizes, train_scores, val_scores = learning_curve(
            estimator,
            X,
            y,
            train_sizes=train_sizes,
            cv=5,
            scoring=scoring,
            n_jobs=1,
            shuffle=True,
            random_state=seed,
        )
        plt.figure(figsize=(6, 4.5))
        plt.plot(sizes, np.mean(train_scores, axis=1), marker="o", label="train")
        plt.plot(sizes, np.mean(val_scores, axis=1), marker="o", label="cv")
        plt.fill_between(
            sizes,
            np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
            np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
            alpha=0.15,
        )
        plt.title(title)
        plt.xlabel("Training samples")
        plt.ylabel(scoring)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_learning_{scoring}.png", dpi=160)
        plt.close()

def save_classification_report(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    threshold: float = 0.5,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
    (out_dir / f"{prefix}_report.txt").write_text(rep)

