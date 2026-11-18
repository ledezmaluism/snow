import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c

from snow import plotting, pulses
from conftest import make_ext_waveguide, fs, um


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def gauss(grid):
    t, f_ref = grid
    return pulses.gaussian_pulse(t, 100 * fs, f_ref, Energy=1e-12, Npwr_dB=300)


def test_units_and_labels(gauss):
    ax = plotting.plot_magsq(gauss.t, gauss.a, t_unit="fs")
    assert ax.get_xlabel() == "Time (fs)"
    x, y = ax.lines[0].get_data()
    np.testing.assert_allclose(x, gauss.t * 1e15)
    np.testing.assert_allclose(y, np.abs(gauss.a) ** 2)
    ax = plotting.plot_PSD(gauss.t, gauss.a, gauss.frep, gauss.f0, f_unit="nm", dBm=False)
    assert ax.get_xlabel() == "Wavelength (nm)" and "W / Hz" in ax.get_ylabel()


def test_spectral_axes_are_sorted(gauss):
    ax = plotting.plot_vs_freq(gauss.f_abs, np.abs(gauss.A) ** 2, f_unit="THz")
    x, _ = ax.lines[0].get_data()
    assert np.all(np.diff(x) > 0)


def test_draws_on_given_axis(gauss):
    _, ax = plt.subplots()
    assert gauss.plot_magsq(ax=ax) is ax
    assert gauss.plot_PSD(ax=ax, f_unit="THz") is ax
    assert gauss.plot_spectrum(ax=ax) is ax
    assert len(ax.lines) == 3


@pytest.mark.parametrize("bad", [dict(t_unit="s"), dict(f_unit="Hz")])
def test_bad_units_raise(gauss, bad):
    with pytest.raises(ValueError):
        if "t_unit" in bad:
            plotting.plot_magsq(gauss.t, gauss.a, **bad)
        else:
            plotting.plot_PSD(gauss.t, gauss.a, gauss.frep, gauss.f0, **bad)


def test_old_pulse_plot_functions_are_deprecated(gauss):
    with pytest.deprecated_call(match="snow.plotting.plot_magsq"):
        pulses.plot_magsq(gauss.t, gauss.a)
    with pytest.deprecated_call(match="plot_PSD"):
        pulses.plot_PSD(gauss.t, gauss.a, gauss.frep, gauss.f0)


@pytest.mark.parametrize("domain", ["time", "wavelength", "frequency"])
def test_plot_evolution(tutorial_grid, domain):
    t, f_ref = tutorial_grid
    wg = make_ext_waveguide()
    wg.set_length(1e-3)
    p = pulses.gaussian_pulse(t, 100 * fs, f_ref, f0=c / (1.5 * um), Energy=1e-12, Npwr_dB=300)
    res = wg.propagate(p, z_save=np.linspace(0, 1e-3, 6))
    ax = plotting.plot_evolution(res, domain=domain)
    assert ax.get_ylabel() == "z (mm)"
    mesh = ax.collections[0]
    assert mesh.get_array().shape == (6, t.size)
    with pytest.raises(ValueError, match="z_save"):
        plotting.plot_evolution(wg.propagate(p))
