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


def test_temperature_is_recovered():
    d, y = fake_data(50000, 4.0)
    T = fit_temperature(d, y)
    assert abs(T - 4.0) < 0.3
    assert ece(p_yes(d, T), y) < 0.02


def test_temperature_keeps_accuracy():
    d, y = fake_data(2000, 2.0)
    assert accuracy(p_yes(d), y) == accuracy(p_yes(d, 3.0), y)


def test_confident_wrong():
    p = np.array([0.999, 0.999, 0.6, 0.001])
    y = np.array([1, 0, 0, 1])
    assert confident_wrong(p, y) == 0.5


def test_bootstrap_sees_a_real_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    groups = ["img" + str(i // 2) for i in range(600)]
    d_bad = rng.normal(0, 1, 600)
    d_good = (y * 2 - 1) * 2 + rng.normal(0, 1, 600)
    out = bootstrap_diff(groups, y, d_bad, d_good, n_boot=200)
    assert out["accuracy_diff_ci95"][0] > 0.2
    same = bootstrap_diff(groups, y, d_bad, d_bad, n_boot=200)
    assert same["accuracy_diff_ci95"] == [0.0, 0.0]