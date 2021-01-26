import pytest

import pandas as pd
import numpy as np


from utils import (
    train_test_val_split,
    mean_absolute_percentage_error,
    value_imputation, 
)


@pytest.fixture
def data():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "date": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04", "2019-01-05"]),
        "a": [1, 1, 1, np.nan, 2], 
        "b": [np.nan]* 5, 
        "target": [1, 1, 1, 0, 1],
    })
    
    return df


def test_mean_absolute_percentage_error_zero_target():
    true = [1, 3, 0, 2]
    pred = [0, 2, 1, 4]

    test = mean_absolute_percentage_error(true, pred)
    
    assert test == np.inf
    

def test_value_imputation(data):
    test_df = value_imputation(data, ["a", "b"], value="a")
    
    assert (~test_df["a"].isna()).all() 
    assert (~test_df["b"].isna()).all() 
