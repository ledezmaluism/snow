"""Shared fixtures for the snow test suite.

Tests marked ``xfail(strict=True)`` document known bugs: they fail today and
pytest reports XPASS (an error, because strict) once a bug is fixed, so the
marker has to be removed together with the fix.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.constants import c

# Make `import snow` and `import lumerical` work without installing the package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from snow import materials, waveguides  # noqa: E402

ps = 1e-12
fs = 1e-15
um = 1e-6
nm = 1e-9


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: nonlinear propagation runs (seconds each)")


@pytest.fixture(autouse=True)
def _seed():
    """The pulse noise and NEE quantum noise use the global numpy RNG."""
    np.random.seed(12345)


@pytest.fixture
def grid():
    """Time grid centred on t=0: N=1024, dt=4 fs, reference at 2 um."""
    N = 1024
    dt = 4 * fs
    t = (np.arange(N) - N // 2) * dt
    f_ref = c / (2 * um)
    return t, f_ref


@pytest.fixture
def tutorial_grid():
    """The grid from tutorial 5: 0.8-3 um bandwidth, N=1024."""
    f_max, f_min = c / (800 * nm), c / (3 * um)
    BW = f_max - f_min
    N = 2 ** 10
    dt = 1 / BW
    t = -N * dt / 2 + np.arange(N) * dt
    f_ref = (f_max + f_min) / 2
    return t, f_ref


def make_ext_waveguide(material="LN_MgO_e"):
    """Waveguide whose index is a spline of a bulk material (fast to evaluate)."""
    wl = np.linspace(0.5 * um, 5 * um, 600)  # inside the Zelmon fit range
    return waveguides.waveguide(external=True, wl_array=wl,
                                n_array=materials.refractive_index(material, wl))


@pytest.fixture
def ext_wg():
    return make_ext_waveguide()
