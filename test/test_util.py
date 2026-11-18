import numpy as np
import pytest

from snow import util


def test_derivative_of_sin():
    assert util.derivative(np.sin, 1.0, 2, 0.1) == pytest.approx(np.cos(1.0), rel=1e-8)


def test_derivative_of_polynomial_is_exact():
    f = lambda x: 3 * x**3 - x
    assert util.derivative(f, 2.0, 2, 0.5) == pytest.approx(35.0, rel=1e-12)


def test_derivative_accepts_size_one_array_output():
    f = lambda x: np.array([x**2])
    assert util.derivative(f, 3.0, 2, 0.1) == pytest.approx(6.0, rel=1e-12)


def test_absorption_coeff_db_per_cm_to_per_m():
    # 1 dB/cm of power loss = ln(10)/10 per cm = 23.03 / m
    assert util.absorption_coeff(1.0) == pytest.approx(10 * np.log(10), rel=1e-12)
    assert util.absorption_coeff(0.0) == 0


def test_sech():
    x = np.linspace(-3, 3, 7)
    np.testing.assert_allclose(util.sech(x), 1 / np.cosh(x))


def test_check():
    assert util.check(1, 0, 2)
    assert util.check(0, 0, 2)
    assert not util.check(3, 0, 2)
