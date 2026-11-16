"""Development-only threshold selection and descriptive held-out evaluation."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilistic_predictions(y_true, y_prob, threshold=0.5):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if (
        len(y_true) != len(y_prob)
        or not len(y_true)
        or not np.isfinite(y_prob).all()
        or not np.isin(y_true, [0, 1]).all()
        or (y_prob < 0).any()
        or (y_prob > 1).any()
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Expected aligned binary labels and probabilities in [0, 1]")
    predicted = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "average_precision": float(average_precision_score(y_true, y_prob)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)) if both_classes else None,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)) if tp + fn else None,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "n": len(y_true),
    }


def choose_threshold(y_development, oof_probabilities) -> float:
    """Maximize F1 using only out-of-fold development predictions."""
    precision, recall, thresholds = precision_recall_curve(y_development, oof_probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(f1)])


def bootstrap_intervals(y_true, probabilities, threshold, repeats=200, seed=42):
    """Stratified test bootstrap, conditional on fixed fits and observed prevalence."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(y_true)
    strata = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(len(group) == 0 for group in strata):
        raise ValueError("Both classes are required for stratified bootstrap")
    rows = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        rows.append(evaluate_probabilistic_predictions(labels[idx], probabilities[idx], threshold))
    keys = ["average_precision", "roc_auc", "brier", "precision", "recall", "f1", "accuracy"]
    return {key: np.quantile([row[key] for row in rows], [0.025, 0.975]).tolist() for key in keys}


def subgroup_metrics(features, labels, probabilities, threshold):
    rows = []
    groups = {
        "Sex": features["Sex"].map({0: "female", 1: "male"}),
        "Age": pd.cut(features["Age"], [0, 5, 9, 13], labels=["18-44", "45-64", "65+"]),
    }
    for variable, values in groups.items():
        for group in values.unique():
            mask = (values == group).to_numpy()
            rows.append(
                {
                    "variable": variable,
                    "group": str(group),
                    **evaluate_probabilistic_predictions(
                        np.asarray(labels)[mask], probabilities[mask], threshold
                    ),
                }
            )
    return pd.DataFrame(rows)


def get_calibration_data(y_true, y_prob, n_bins=10):
    return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
