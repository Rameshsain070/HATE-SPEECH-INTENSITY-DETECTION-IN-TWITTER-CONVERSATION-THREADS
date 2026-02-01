import numpy as np
from scipy.stats import pearsonr


def pearson_corr(y_true, y_pred):
    """
    Compute Pearson Correlation Coefficient (PCC).

    Used to measure how well the predicted hate intensity
    trend aligns with the ground truth trend.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) < 2:
        return 0.0

    return pearsonr(y_true, y_pred)[0]


def rmse(y_true, y_pred):
    """
    Root Mean Square Error between true and predicted values.
    Lower is better.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mse(y_true, y_pred):
    """
    Mean Squared Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)