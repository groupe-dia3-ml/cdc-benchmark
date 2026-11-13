import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

def evaluate_probabilistic_predictions(y_true, y_prob, threshold=0.5):
    """
    Evaluates probabilistic predictions using standard screening metrics:
    - ROC-AUC
    - Average Precision (PR-AUC)
    - Brier score (calibration metric)
    - Threshold-dependent metrics (Recall, Precision, Balanced Accuracy, F1)
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "Average_Precision": float(average_precision_score(y_true, y_prob)),
        "Brier_Score": float(brier_score_loss(y_true, y_prob)),
        "Threshold": threshold,
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Recall_Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics

def get_calibration_data(y_true, y_prob, n_bins=10):
    """
    Computes true vs predicted probabilities for calibration curves.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return prob_true, prob_pred
