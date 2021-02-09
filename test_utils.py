import pytest

import numpy as np


from utils import (
    mean_absolute_percentage_error,
)


def test_mean_absolute_percentage_error_zero_target():
    true = [1, 3, 0, 2]
    pred = [0, 2, 1, 4]

    test = mean_absolute_percentage_error(true, pred)
    
    assert test == np.inf
