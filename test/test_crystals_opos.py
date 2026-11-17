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
    v_ref = 1 / wg.beta1(1.5 * um)[0]
    out_wg, _ = wg.propagate_NEE(p, verbose=False, v_ref=v_ref)

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


# ---- known bugs -------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="run_opo passes self.h positionally into propagate_NEE_fd "
                                       "(collides with v_ref) and builds pulses without frep")
def test_opo_single_roundtrip(tutorial_grid):
    from snow import opos
    t, f_ref = tutorial_grid
    xtal = crystals.nonlinear_crystal(1e-3, n_bulk, chi2=lambda z: 2 * 20e-12)
    f_sh = c / (1 * um)
    sim = opos.opo(xtal, phis=[0.0], feedfunc=np.ones(t.size), outcoupling=0.5, t=t,
                   dTs=np.array([0.0]), pins=np.array([1e-3]), f_rep=250e6, h=None, v_ref=None,
                   pulse_params=[100 * fs, f_ref, f_sh, 100], roundtrips=1, numsamp=1)
    sim.run_opo()
    assert sim.get_outpow(0.0, 1e-3) >= 0


@pytest.mark.xfail(strict=True, reason="plot_3d_spectrum uses undefined names dt and fig")
def test_opo_plot_3d_spectrum():
    from snow import opos
    N = 64
    out_fd = np.ones((2, 1, 2, N), complex)
    f_abs = np.linspace(100e12, 400e12, N)
    opos.plot_3d_spectrum(out_fd, f_abs, "sech", np.array([0.0, 1e-15]), np.array([1e-3]),
                          1e-3, 1, 100e-15, 300e12)
