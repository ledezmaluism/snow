"""Propagation through snow.nlo.NEE, driven by waveguide.propagate_NEE.

All runs use a waveguide whose index is a spline of bulk MgO:LN (fast) and the
time/frequency grid from tutorial 5.
"""
import numpy as np
import pytest
from scipy.constants import c, h, pi

from snow import pulses, util
from conftest import make_ext_waveguide, fs, um

TAU = 100 * fs


def fwhm(p):
    I = np.abs(p.a)**2
    return pulses.FWHM(p.t, p.a, prominence=0.25 * I.max())[0]


def linear_wg(L, alpha=0.0):
    wg = make_ext_waveguide()
    wg.set_length(L)
    wg.set_loss(alpha)
    wg.add_poling(lambda z: 1.0)
    wg.set_nonlinear_coeffs(N=1, X0=0.0)
    return wg


def shg_wg(L=1e-3):
    wg = make_ext_waveguide()
    wg.set_length(L)
    wg.set_loss(0)
    wg.set_nonlinear_coeffs(N=1, X0=1.1e-12)
    return wg


def shg_run(tutorial_grid, energy, kg_sign=+1, L=1e-3, seed=1):
    t, f_ref = tutorial_grid
    wg = shg_wg(L)
    dk = float(wg.beta(1 * um)) - 2 * float(wg.beta(2 * um))
    np.random.seed(seed)
    p = pulses.sech_pulse(t, TAU, f_ref, f0=c / (2 * um), Energy=energy, Npwr_dB=300)
    out, steps = wg.propagate_NEE(p, verbose=False, Kg=kg_sign * dk, v_ref=1 / wg.beta1(2 * um)[0])
    return p, out, steps


def sh_fraction(out, f_ff=c / (2 * um)):
    S = np.abs(out.A)**2
    return S[out.f_abs > 1.7 * f_ff].sum() / S.sum()


# ---- linear propagation ------------------------------------------------------

def test_gvd_broadening_matches_gaussian_formula(tutorial_grid):
    t, f_ref = tutorial_grid
    wl = 1 * um
    wg = linear_wg(1.0)
    b2 = wg.beta2(wl)[0]
    LD = TAU**2 / (4 * np.log(2) * b2)
    wg.set_length(LD)
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=c / wl, Energy=1e-12, Npwr_dB=300)
    out, _ = wg.propagate_NEE(p, verbose=False, v_ref=1 / wg.beta1(wl)[0])
    # A Gaussian broadens by sqrt(1 + (L/LD)^2) = sqrt(2) at one dispersion length
    assert fwhm(out) == pytest.approx(np.sqrt(2) * fwhm(p), rel=0.02)
    assert out.energy_td() == pytest.approx(p.energy_td(), rel=1e-9)


def test_reference_frame_keeps_pulse_centred(tutorial_grid):
    t, f_ref = tutorial_grid
    wg = linear_wg(2e-3)
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=c / (1.5 * um), Energy=1e-12, Npwr_dB=300)
    out, _ = wg.propagate_NEE(p, verbose=False, v_ref=1 / wg.beta1(1.5 * um)[0])
    assert abs(out.time_center()) < 2 * fs


def test_group_velocity_walk_off(tutorial_grid):
    t, f_ref = tutorial_grid
    L = 0.5e-3
    wg = linear_wg(L)
    v_ref = 1 / wg.beta1(1.5 * um)[0]
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=c / (2 * um), Energy=1e-12, Npwr_dB=300)
    out, _ = wg.propagate_NEE(p, verbose=False, v_ref=v_ref)
    delay = L * (wg.beta1(2 * um)[0] - 1 / v_ref)
    assert out.time_center() == pytest.approx(delay, abs=2 * fs)


def test_loss(tutorial_grid):
    t, f_ref = tutorial_grid
    alpha = util.absorption_coeff(1.0)  # 1 dB/cm
    wg = linear_wg(10e-3, alpha)
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=c / (2 * um), Energy=1e-12, Npwr_dB=300)
    out, _ = wg.propagate_NEE(p, verbose=False, v_ref=1 / wg.beta1(2 * um)[0])
    assert out.energy_td() / p.energy_td() == pytest.approx(10**(-1.0 / 10), rel=1e-9)


def test_default_v_ref_is_group_velocity_at_reference(tutorial_grid):
    t, f_ref = tutorial_grid
    wg = linear_wg(1e-3)
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=f_ref, Energy=1e-12, Npwr_dB=300)
    out_default, _ = wg.propagate_NEE(p, verbose=False)
    out_explicit, _ = wg.propagate_NEE(p, verbose=False, v_ref=1 / wg.beta1(c / f_ref)[0])
    np.testing.assert_allclose(out_default.a, out_explicit.a, atol=1e-9 * np.abs(p.a).max())


def test_quantum_noise_is_half_a_photon_per_mode(tutorial_grid):
    t, f_ref = tutorial_grid
    wg = linear_wg(1e-4)
    vacuum = pulses.pulse(t, np.zeros(t.size, complex), c / f_ref, 250e6)
    out, _ = wg.propagate_NEE(vacuum, verbose=False, Qnoise=True, v_ref=1 / wg.beta1(c / f_ref)[0])
    expected = np.sum(h * vacuum.f_abs / 2)
    # NEE rebuilds its grid with BW = f_max - f_min = (N-1)*df, hence ~(N-1)/N
    assert out.energy_td() == pytest.approx(expected, rel=3e-3)


# ---- second-harmonic generation ---------------------------------------------

@pytest.mark.slow
def test_shg_phase_matching_sign(tutorial_grid):
    _, matched, _ = shg_run(tutorial_grid, 10e-12, +1)
    _, mismatched, _ = shg_run(tutorial_grid, 10e-12, -1)
    assert sh_fraction(matched) > 0.5
    assert sh_fraction(mismatched) < 1e-3


@pytest.mark.slow
def test_shg_low_conversion_is_quadratic(tutorial_grid):
    _, out1, _ = shg_run(tutorial_grid, 5e-15)
    _, out2, _ = shg_run(tutorial_grid, 10e-15)
    # SH energy ~ E^2, so the SH fraction doubles when the pump energy doubles
    assert sh_fraction(out2) / sh_fraction(out1) == pytest.approx(2.0, rel=0.02)


@pytest.mark.slow
def test_shg_regression(tutorial_grid):
    _, out, steps = shg_run(tutorial_grid, 10e-12)
    # 0.8646 before the NEE upsampled-grid fix
    assert sh_fraction(out) == pytest.approx(0.8705869926954777, rel=1e-3)
    assert steps.size > 0


@pytest.mark.slow
def test_shg_energy_roughly_conserved(tutorial_grid):
    p, out, _ = shg_run(tutorial_grid, 10e-12)
    assert out.energy_td() / p.energy_td() == pytest.approx(1.0, abs=1e-3)


# ---- known bugs -------------------------------------------------------------

@pytest.mark.slow
def test_shg_energy_conserved_tightly(tutorial_grid):
    p, out, _ = shg_run(tutorial_grid, 10e-12)
    assert out.energy_td() / p.energy_td() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.slow
def test_sh_centered_at_twice_fundamental(tutorial_grid):
    p, out, _ = shg_run(tutorial_grid, 10e-12)
    f_ff = c / (2 * um)
    f = out.f_abs
    S_in, S_out = np.abs(p.A)**2, np.abs(out.A)**2
    ff = (f > 0.7 * f_ff) & (f < 1.3 * f_ff)
    sh = (f > 1.7 * f_ff) & (f < 2.3 * f_ff)
    c_ff = np.sum(f[ff] * S_in[ff]) / np.sum(S_in[ff])
    c_sh = np.sum(f[sh] * S_out[sh]) / np.sum(S_out[sh])
    assert abs(c_sh - 2 * c_ff) < 0.25 * out.df


# ---- #46 ---------------------------------------------------------------------

def test_propagate_without_set_loss_is_lossless(tutorial_grid):
    t, f_ref = tutorial_grid
    wg = make_ext_waveguide()
    wg.set_length(1e-3)
    wg.add_poling(lambda z: 1.0)
    wg.set_nonlinear_coeffs(N=1, X0=0.0)
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=f_ref, Energy=1e-12, Npwr_dB=300)
    out, _ = wg.propagate_NEE(p, verbose=False)
    assert out.energy_td() == pytest.approx(p.energy_td(), rel=1e-9)


def test_zcheck_step_is_deprecated(tutorial_grid):
    t, f_ref = tutorial_grid
    wg = linear_wg(1e-4)
    p = pulses.gaussian_pulse(t, TAU, f_ref, Energy=1e-12, Npwr_dB=300)
    with pytest.deprecated_call():
        wg.propagate_NEE(p, verbose=False, zcheck_step=1e-3)


def test_default_v_ref_uses_temperature(tutorial_grid, monkeypatch):
    # beta1 must be evaluated once, at the reference wavelength and at T
    t, f_ref = tutorial_grid
    wg = linear_wg(1e-4)
    seen = []
    original = wg.beta1
    monkeypatch.setattr(wg, "beta1", lambda wl, T=24.5: seen.append(T) or original(wl, T=T))
    p = pulses.gaussian_pulse(t, TAU, f_ref, f0=f_ref, Energy=1e-12, Npwr_dB=300)
    wg.propagate_NEE(p, verbose=False, T=80)
    assert seen == [80]
