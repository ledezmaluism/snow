import numpy as np
import pytest
from scipy.constants import c

from snow import crystals, materials, pulses
from conftest import make_ext_waveguide, fs, um


def n_bulk(wl):
    return materials.refractive_index("LN_MgO_e", wl)


def test_linear_crystal_matches_waveguide_with_same_index(tutorial_grid):
    t, f_ref = tutorial_grid
    L = 2e-3
    p = pulses.gaussian_pulse(t, 100 * fs, f_ref, f0=c / (1.5 * um), Energy=1e-12, Npwr_dB=300)

    wg = make_ext_waveguide()
    wg.set_length(L)
    wg.set_loss(0)
    wg.add_poling(lambda z: 1.0)
    wg.set_nonlinear_coeffs(N=1, X0=0.0)
    v_ref = 1 / wg.beta1(1.5 * um)
    out_wg, _ = wg.propagate(p, verbose=False, v_ref=v_ref)

    xtal = crystals.nonlinear_crystal(L, n_bulk, chi2=lambda z: 0.0)
    out_x, _ = xtal.propagate_NEE_fd(p, v_ref=v_ref, verbose=False)

    np.testing.assert_allclose(np.abs(out_x.a), np.abs(out_wg.a), atol=1e-4 * np.abs(p.a).max())
    assert out_x.frep == p.frep


def test_crystal_nonlinear_smoke(tutorial_grid):
    # Smoke test only: chi2 here multiplies a field normalised to sqrt(W), so a
    # physical chi2 gives negligible conversion (the bulk normalisation is unfinished).
    t, f_ref = tutorial_grid
    p = pulses.sech_pulse(t, 100 * fs, f_ref, f0=c / (2 * um), Energy=10e-12, Npwr_dB=300)
    xtal = crystals.nonlinear_crystal(1e-3, n_bulk, chi2=lambda z: 2 * 20e-12)
    out, steps = xtal.propagate_NEE_fd(p, verbose=False)
    assert steps.size > 0
    assert out.energy_td() / p.energy_td() == pytest.approx(1.0, abs=1e-6)


# ---- OPO (#47) ---------------------------------------------------------------

from scipy.constants import pi
from snow import opos

F_REP = 250e6
WL_P = 1 * um  # degenerate OPO: pump at 1 um, signal at 2 um


@pytest.fixture
def opo_grid():
    f_max, f_min = c / (0.8 * um), c / (3 * um)
    N = 2 ** 9
    dt = 1 / (f_max - f_min)
    return -N * dt / 2 + np.arange(N) * dt, (f_max + f_min) / 2


def make_opo(grid, chi2_amp, pins, dTs=(0.0,), roundtrips=15, numsamp=5):
    t, f_ref = grid
    # quasi-phase matched for 1 um -> 2 um; smooth grating (first harmonic only)
    dk = 2 * pi * n_bulk(WL_P) / WL_P - 2 * pi * n_bulk(2 * WL_P) / WL_P
    xtal = crystals.nonlinear_crystal(1e-3, n_bulk, chi2=lambda z: chi2_amp * np.cos(dk * z))
    return opos.opo(xtal, phis=[0.0], feedfunc=np.full(t.size, 0.7), outcoupling=0.7, t=t,
                    dTs=np.array(dTs), pins=np.array(pins), f_rep=F_REP,
                    fwhm=100 * fs, f_ref=f_ref, f_pump=c / WL_P, noise_dB=100,
                    roundtrips=roundtrips, numsamp=numsamp)


def test_opo_without_gain_outputs_only_noise(opo_grid):
    sim = make_opo(opo_grid, 0.0, [0.1], roundtrips=3)
    sim.run_opo()
    # the round-trip filter around 2 um removes the 1 um pump
    assert sim.get_outenergy(0.0, 0.1) < 1e-6 * 0.1 / F_REP


@pytest.mark.slow
def test_opo_threshold(opo_grid):
    below = make_opo(opo_grid, 1e-3, [1e-3])
    above = make_opo(opo_grid, 1e-3, [0.1])
    below.run_opo()
    above.run_opo()
    assert below.get_outenergy(0.0, 1e-3) < 1e-5 * 1e-3 / F_REP
    assert above.get_outenergy(0.0, 0.1) > 1e-2 * 0.1 / F_REP
    # the output is the degenerate signal at 2 um
    S = np.mean(np.abs(above.out_full_fd[-5:, 0, 0]) ** 2, axis=0)
    f = np.fft.fftshift(above.f_abs)
    assert np.sum(f * S) / np.sum(S) == pytest.approx(c / (2 * WL_P), rel=0.02)


def test_opo_outputs_are_consistent(opo_grid):
    sim = make_opo(opo_grid, 0.0, [0.05, 0.1], dTs=(0.0, 2 * fs), roundtrips=2, numsamp=1)
    sim.run_opo()
    assert sim.get_outpow(2 * fs, 0.1) == pytest.approx(sim.get_outenergy(2 * fs, 0.1) * F_REP)


def test_opo_plots(opo_grid):
    import matplotlib.pyplot as plt
    sim = make_opo(opo_grid, 0.0, [0.05, 0.1], dTs=(0.0, 2 * fs), roundtrips=2, numsamp=1)
    sim.run_opo()
    sim.plot_3d_spectrum(0.1)
    sim.plot_peak_structure(0.1)
    ax = sim.plot_outpow_inpow(np.array([0.0, 2 * fs]))
    assert ax.get_ylabel() == "Output Average Power (W)"
    ax = sim.plot_coneff(np.array([0.0, 2 * fs]))
    assert ax.get_ylabel() == "Power Conversion Efficiency (%)"
    eff = ax.collections[0].get_offsets()[:, 1]
    best = [max(sim.get_outpow(d, p) for d in (0.0, 2 * fs)) for p in (0.05, 0.1)]
    np.testing.assert_allclose(eff, 100 * np.array(best) / [0.05, 0.1], rtol=1e-5)
    plt.close("all")


def test_opo_pulse_params_still_work(opo_grid):
    t, f_ref = opo_grid
    xtal = crystals.nonlinear_crystal(1e-3, n_bulk, chi2=lambda z: 0.0)
    with pytest.deprecated_call():
        old = opos.opo(xtal, [0.0], np.ones(t.size), 0.5, t, np.array([0.0]), np.array([1e-3]),
                       F_REP, None, None, [100 * fs, f_ref, c / WL_P, 90], roundtrips=1, numsamp=1)
    assert (old.fwhm, old.f_ref, old.f_pump, old.noise_dB) == (100 * fs, f_ref, c / WL_P, 90)
    with pytest.raises(TypeError):
        opos.opo(xtal, [0.0], np.ones(t.size), 0.5, t, np.array([0.0]), np.array([1e-3]), F_REP)


def test_opo_rng_reproducible(opo_grid):
    runs = []
    for _ in range(2):
        sim = make_opo(opo_grid, 0.0, [0.1], roundtrips=2)
        sim.rng = 5
        sim.run_opo()
        runs.append(sim.out_full_td.copy())
    np.testing.assert_array_equal(runs[0], runs[1])


def test_opo_accepts_a_waveguide(opo_grid):
    # any medium with propagate(pulse, v_ref=...) works as the gain medium
    t, f_ref = opo_grid
    wg = make_ext_waveguide()
    wg.set_length(0.5e-3)
    wg.add_poling(lambda z: 1.0)
    wg.set_nonlinear_coeffs(N=1, X0=0.0)
    sim = opos.opo(wg, [0.0], np.full(t.size, 0.7), 0.7, t, np.array([0.0]), np.array([0.1]),
                   F_REP, fwhm=100 * fs, f_ref=f_ref, f_pump=c / WL_P, roundtrips=2, numsamp=1)
    sim.run_opo()
    assert sim.get_outenergy(0.0, 0.1) < 1e-6 * 0.1 / F_REP


def test_opo_h_is_deprecated(opo_grid):
    t, f_ref = opo_grid
    xtal = crystals.nonlinear_crystal(1e-3, n_bulk, chi2=lambda z: 0.0)
    with pytest.deprecated_call():
        opos.opo(xtal, [0.0], np.ones(t.size), 0.5, t, np.array([0.0]), np.array([1e-3]),
                 F_REP, 1e-6, None, [100 * fs, f_ref, c / WL_P, 100], roundtrips=1, numsamp=1)


def test_opo_plot_3d_spectrum_with_given_axis():
    import matplotlib.pyplot as plt
    N = 64
    out_fd = np.ones((2, 1, 2, N), complex)
    f_abs = 250e12 + np.fft.fftfreq(N, 1 / 300e12)  # FFT order, like opo.f_abs
    _, ax = plt.subplots()
    assert opos.plot_3d_spectrum(out_fd, f_abs, "sech", np.array([0.0, 1e-15]), np.array([1e-3]),
                                 1e-3, 1, 100e-15, 300e12, ax=ax) is ax
    plt.close("all")
