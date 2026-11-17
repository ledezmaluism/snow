"""Pure-numpy parts of lumerical.py (no Lumerical installation needed)."""
import numpy as np
import pytest

import lumerical


@pytest.fixture
def xy():
    return np.linspace(-1e-6, 1e-6, 41), np.linspace(-0.5e-6, 0.5e-6, 21)


def uniform(xy, vec):
    x, y = xy
    ones = np.ones((x.size, y.size), complex)
    return lumerical.field_2D(x, y, vec[0] * ones, vec[1] * ones, vec[2] * ones)


def test_cross_product_of_unit_vectors(xy):
    C = uniform(xy, (1, 0, 0)).cross(uniform(xy, (0, 1, 0)))
    np.testing.assert_allclose(C.x, 0)
    np.testing.assert_allclose(C.y, 0)
    np.testing.assert_allclose(C.z, 1)


def test_dot_conj_magsq_rescale(xy):
    E = uniform(xy, (1 + 1j, 2, 0))
    np.testing.assert_allclose(E.magsq(), 6)
    np.testing.assert_allclose(E.conj().x, 1 - 1j)
    np.testing.assert_allclose(E.rescale(2).magsq(), 24)


def test_mode_power_flux_of_uniform_field(xy):
    x, y = xy
    m = lumerical.mode()
    m.set_manually(uniform(xy, (1, 0, 0)), uniform(xy, (0, 1, 0)), neff=2.0)
    area = (x[-1] - x[0]) * (y[-1] - y[0])
    assert m.N() == pytest.approx(0.5 * area, rel=1e-12)
