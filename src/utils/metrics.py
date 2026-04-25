import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def compute_auc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return roc_auc_score(y_true, y_pred)


def compute_logloss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    return log_loss(y_true, y_pred)

def compute_metrics(y_true, y_pred):
    return {
        "auc": compute_auc(y_true, y_pred),
        "logloss": compute_logloss(y_true, y_pred),
    }