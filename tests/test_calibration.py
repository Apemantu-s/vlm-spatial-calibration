import numpy as np
from calibration import *


def fake_data(n, sharpen, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, n)
    y = (rng.random(n) < sigmoid(z)).astype(int)
    return z * sharpen, y


def test_calibrated_model_has_low_ece():
    d, y = fake_data(50000, 1.0)
    assert ece(p_yes(d), y) < 0.02


def test_overconfident_model_has_high_ece():
    d, y = fake_data(50000, 4.0)
    assert ece(p_yes(d), y) > 0.1


def test_ece_by_hand():
    p = np.array([0.9, 0.9, 0.9, 0.1])
    y = np.array([1, 1, 0, 0])
    assert abs(ece(p, y, 15) - 0.15) < 1e-9


def test_confident_wrong():
    p = np.array([0.999, 0.999, 0.6, 0.001])
    y = np.array([1, 0, 0, 1])
    assert confident_wrong(p, y) == 0.5