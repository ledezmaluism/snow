import numpy as np
import pytest
from scipy.constants import c, pi

from snow import pulses
from conftest import ps, fs

E0 = 1e-12      # 1 pJ
TAU = 100 * fs  # FWHM
QUIET = 300     # noise 300 dB below the peak, i.e. none


def intensity_fwhm(t, x):
    I = np.abs(x)**2
    above = t[I >= I.max() / 2]
    return above[-1] - above[0] + (t[1] - t[0])


@pytest.fixture
def gauss(grid):
    t, f_ref = grid
    return pulses.gaussian_pulse(t, TAU, f_ref, f0=f_ref, Energy=E0, Npwr_dB=QUIET)


# ---- pulse shapes ------------------------------------------------------------

@pytest.mark.parametrize("make", [pulses.gaussian_pulse, pulses.sech_pulse])
def test_requested_energy(grid, make):
    t, f_ref = grid
    p = make(t, TAU, f_ref, f0=f_ref, Energy=E0, Npwr_dB=QUIET)
    # the 0.94 / 0.88 peak-power prefactors are rounded, hence 0.2 %
    assert p.energy_td() == pytest.approx(E0, rel=2e-3)


@pytest.mark.parametrize("make", [pulses.gaussian_pulse, pulses.sech_pulse])
def test_requested_fwhm(grid, make):
    t, f_ref = grid
    p = make(t, TAU, f_ref, f0=f_ref, Energy=E0, Npwr_dB=QUIET)
    assert intensity_fwhm(t, p.a) == pytest.approx(TAU, abs=2 * (t[1] - t[0]))
    prominence = 0.25 * np.max(np.abs(p.a)**2)
    assert pulses.FWHM(t, p.a, prominence=prominence)[0] == pytest.approx(TAU, rel=1e-2)


@pytest.mark.parametrize("make, factor", [(pulses.gaussian_pulse, 0.94), (pulses.sech_pulse, 0.88)])
def test_peak_power_and_average_power_inputs(grid, make, factor):
    t, f_ref = grid
    p = make(t, TAU, f_ref, f0=f_ref, Ppeak=10.0, Npwr_dB=QUIET)
    assert np.max(np.abs(p.a)**2) == pytest.approx(10.0, rel=1e-3)
    p = make(t, TAU, f_ref, f0=f_ref, Pavg=1e-3, frep=100e6, Npwr_dB=QUIET)
    assert p.energy_td() == pytest.approx(1e-11, rel=2e-3)
    assert p.frep == 100e6


def test_default_energy_is_one_pj(grid, capsys):
    t, f_ref = grid
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=f_ref, Npwr_dB=QUIET)
    assert p.energy_td() == pytest.approx(1e-12, rel=2e-3)
    assert "default 1pJ" in capsys.readouterr().out


def test_noise_level(grid):
    t, f_ref = grid
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=f_ref, Energy=E0, Npwr_dB=30)
    clean = pulses.gaussian(t, E0, TAU)
    noise_power = np.mean(np.abs(p.a - clean)**2)
    assert noise_power == pytest.approx(np.max(np.abs(clean)**2) * 1e-3, rel=0.15)


def test_frequency_offset_sets_temporal_phase_slope(grid):
    t, f_ref = grid
    df = 10e12
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=f_ref + df, Energy=E0, Npwr_dB=QUIET)
    phase = p.temporal_phase()
    mask = ~np.isnan(phase)
    slope = np.polyfit(t[mask], phase[mask], 1)[0]
    assert slope == pytest.approx(2 * pi * df, rel=1e-6)


# ---- pulse class -------------------------------------------------------------

def test_grids(gauss, grid):
    t, f_ref = grid
    assert gauss.f0 == pytest.approx(f_ref)
    assert gauss.wl0 == pytest.approx(c / f_ref)
    assert gauss.df == pytest.approx(1 / (t.size * (t[1] - t[0])))
    np.testing.assert_allclose(gauss.f_abs, np.fft.fftfreq(t.size, t[1] - t[0]) + f_ref)
    np.testing.assert_allclose(gauss.wl, c / gauss.f_abs)


def test_parseval(gauss):
    assert gauss.energy_fd() == pytest.approx(gauss.energy_td(), rel=1e-12)


def test_time_center(gauss):
    assert abs(gauss.time_center()) < 1e-3 * fs


def test_transform_limited_spectral_phase_is_flat(gauss):
    phase = gauss.spectral_phase()
    assert np.nanmax(phase) - np.nanmin(phase) < 1e-6


def test_spectrum_is_referenced_to_t_zero(gauss):
    # pulse.A = fft(fftshift(a)) differs from fft(a) by (-1)^k on an even grid
    k = np.arange(gauss.NFFT)
    np.testing.assert_allclose(gauss.A, (-1.0)**k * np.fft.fft(gauss.a), atol=1e-9 * np.abs(gauss.A).max())


def test_update_td_and_fd_round_trip(gauss):
    a0 = gauss.a.copy()
    gauss.update_fd(gauss.A.copy())
    np.testing.assert_allclose(gauss.a, a0, atol=1e-12 * np.abs(a0).max())
    gauss.update_td(a0 * 2)
    assert gauss.energy_td() == pytest.approx(4 * np.sum(np.abs(a0)**2) * gauss.dt)


def test_update_with_wrong_size_raises(gauss):
    with pytest.raises(RuntimeError):
        gauss.update_td(np.zeros(10))
    with pytest.raises(RuntimeError):
        gauss.update_fd(np.zeros(10))


def test_arithmetic(gauss):
    e = gauss.energy_td()
    assert (gauss + gauss).energy_td() == pytest.approx(4 * e)
    assert (2 * gauss).energy_td() == pytest.approx(4 * e)
    assert (gauss * 2).energy_td() == pytest.approx(4 * e)
    assert (gauss * gauss).a == pytest.approx(gauss.a**2)


def test_add_t_offset_moves_pulse_earlier(grid):
    # Pins current behaviour: a positive offset moves the pulse to NEGATIVE time.
    t, f_ref = grid
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=f_ref, Energy=E0, Npwr_dB=QUIET)
    p = pulses.add_t_offset(p, 0.5 * ps)
    assert p.time_center() == pytest.approx(-0.5 * ps, abs=1e-3 * fs)


def test_psd_integrates_to_average_power(gauss):
    p_avg = np.sum(gauss.get_psd()) * gauss.df
    assert p_avg == pytest.approx(gauss.energy_td() * gauss.frep, rel=1e-12)
    np.testing.assert_allclose(gauss.get_psd_dBm(), 10 * np.log10(gauss.get_psd()) + 30)


def test_spectrum_in_dbm(gauss):
    res = 1e9
    np.testing.assert_allclose(gauss.spectrum(res), 10 * np.log10(gauss.get_psd() * res) + 30)


def test_wavelength_sanity_warning(grid, capsys):
    t, _ = grid
    pulses.pulse(t, np.zeros(t.size, complex), 100e-6, 1e8)
    assert "outside usual range" in capsys.readouterr().out


# ---- filters -----------------------------------------------------------------

def test_wide_ideal_filter_keeps_energy(gauss):
    out = gauss.apply_filter(gauss.f0, 100e12)
    assert out.energy_td() == pytest.approx(gauss.energy_td(), rel=1e-6)


def test_narrow_off_center_filter_removes_pulse(gauss):
    out = gauss.apply_filter(gauss.f0 + 60e12, 1e12)
    assert out.energy_td() < 1e-6 * gauss.energy_td()


def test_ideal_filter_response(gauss):
    _, h = pulses.filter_signal(gauss.f_abs, gauss.A, gauss.f0, 20e12)
    inside = np.abs(gauss.f_abs - gauss.f0) < 10e12
    np.testing.assert_array_equal(h, inside.astype(float))


# ---- coherence ---------------------------------------------------------------

def test_g1_of_identical_pulses_is_one(gauss):
    train = np.stack([gauss.a] * 4, axis=1)
    g1 = pulses.coherence_g1(train)
    strong = np.abs(np.fft.fft(gauss.a)) > 1e-3 * np.abs(np.fft.fft(gauss.a)).max()
    np.testing.assert_allclose(g1[strong], 1.0, rtol=1e-9)
    assert pulses.average_g1(gauss.a, g1) == pytest.approx(1.0, rel=1e-6)


def test_g1_of_independent_noise_is_low(grid):
    t, _ = grid
    train = np.stack([pulses.noise(t, 1.0) for _ in range(40)], axis=1)
    assert np.mean(pulses.coherence_g1(train)) < 0.3


# ---- plotting smoke tests (Agg backend) --------------------------------------

@pytest.mark.parametrize("f_unit", ["um", "nm", "THz"])
def test_plots_run(gauss, f_unit):
    import matplotlib.pyplot as plt
    assert gauss.plot_magsq() is not None
    assert gauss.plot_PSD(f_unit=f_unit) is not None
    assert gauss.plot_spectrum(f_unit=f_unit) is not None
    plt.close("all")


# ---- known bugs -------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="#41: filter_signal returns ifft(A*h) = fftshift(a): pulse lands at the window edge")
def test_filter_keeps_pulse_position(gauss):
    out = gauss.apply_filter(gauss.f0, 100e12)
    np.testing.assert_allclose(out.a, gauss.a, atol=1e-6 * np.abs(gauss.a).max())


@pytest.mark.xfail(strict=True, reason="#42: domain='Freq' sets a = ifft(A) without the fftshift used by update_fd")
def test_frequency_domain_constructor_round_trip(gauss):
    q = pulses.pulse(gauss.t, gauss.A, gauss.wl0, gauss.frep, domain="Freq")
    np.testing.assert_allclose(q.a, gauss.a, atol=1e-9 * np.abs(gauss.a).max())


@pytest.mark.xfail(strict=True, reason="#43: gaussian_pulse/sech_pulse default f0=0 means 0 Hz absolute, "
                                       "i.e. a -f_ref carrier offset that aliases on the grid")
@pytest.mark.parametrize("make", [pulses.gaussian_pulse, pulses.sech_pulse])
def test_default_center_frequency_is_reference(grid, make):
    t, f_ref = grid
    p = make(t, TAU, f_ref, Energy=E0, Npwr_dB=QUIET)
    S = np.abs(p.A)**2
    assert np.sum(p.f_abs * S) / np.sum(S) == pytest.approx(f_ref, rel=1e-3)


@pytest.mark.xfail(strict=True, reason="#44: width_FWHM returns an array (one entry per peak), not a number")
def test_width_fwhm_returns_scalar(gauss):
    w = gauss.width_FWHM()
    assert np.ndim(w) == 0
    assert w == pytest.approx(TAU, rel=1e-2)


@pytest.mark.xfail(strict=True, reason="#44: pulse.plot_magsq ignores its t_unit argument")
def test_plot_magsq_uses_t_unit(gauss):
    import matplotlib.pyplot as plt
    ax = gauss.plot_magsq(t_unit="fs")
    assert ax.get_xlabel() == "Time (fs)"
    plt.close("all")
