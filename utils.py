import numpy as np
import pytz


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Computes MAPE metric
    :param y_true: True values
    :param y_pred: Predicted values
    :return: MAPE
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
