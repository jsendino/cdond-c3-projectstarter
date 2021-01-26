import numpy as np
import pandas as pd
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


def value_imputation(df, cols, value=0):
    """Directly impute fixed value for missing data"""
    data = df.copy()
    data.loc[:, cols] = data.loc[:, cols].fillna(value=value)
    return data


def train_test_val_split(
    X,
    y,
    date_col="DATE_ID",
    train_periods=["2019-10-01", "2019-12-31"],
    test_periods=["2020-01-01", "2020-01-31"],
    val_periods=["2020-08-01", "2020-09-30"],
):
    """
    Split dataset using absolute time thresholds on the point_in_time column
    :param data: Dataset to split
    :param split_time_threshold: Dict with thresholds for train, validation and test and max min date
    :return:
    """
    periods_dict = dict(
        zip(["train", "test", "val"], [train_periods, test_periods, val_periods])
    )

    for split, period in periods_dict.items():
        period_dt = [pd.to_datetime(dt) for dt in period]
        periods_dict[split] = period_dt

    # split dataframes for train test and validation(post-covid)
    idx_train = X.DATE_ID.between(*periods_dict["train"])
    X_train, y_train = X[idx_train].drop("DATE_ID", axis=1), y[idx_train]

    idx_test = X.DATE_ID.between(*periods_dict["test"])
    X_test, y_test = X[idx_test].drop("DATE_ID", axis=1), y[idx_test]

    idx_validation = X.DATE_ID.between(*periods_dict["val"])
    X_validation, y_validation = (
        X[idx_validation].drop("DATE_ID", axis=1),
        y[idx_validation],
    )

    return X_train, y_train, X_test, y_test, X_validation, y_validation