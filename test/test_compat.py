"""The deprecated 0.1 API keeps working for one release, with DeprecationWarnings."""
import numpy as np
import pytest
from scipy.constants import c

from snow import materials, pulses, waveguides
from conftest import make_ext_waveguide, um, fs


def old(**kw):
    with pytest.deprecated_call():
        return waveguides.waveguide(**kw)


def test_old_waveguide_warns():
    with pytest.deprecated_call(match="Waveguide"):
        waveguides.waveguide()


@pytest.mark.parametrize("kw", [{}, {"w_top": 1.8e-6, "mode": "TM"}, {"h_etch": 700e-9, "theta": 90}])
def test_old_ridge_matches_new(kw):
    ridge_kw = dict(kw)
    w_old = old(**kw)
    w_new = waveguides.Waveguide(waveguides.RidgeIndex(**ridge_kw))
    # old class: arrays even for scalars
    assert w_old.neff(1e-6).shape == (1,)
    assert w_old.beta1(1e-6).shape == (1,)
    assert w_old.beta2(1e-6).shape == (1,)
    assert w_old.neff(1e-6)[0] == w_new.neff(1e-6)
    assert w_old.beta1(1e-6)[0] == w_new.beta1(1e-6)
    assert w_old.beta2(1e-6)[0] == w_new.beta2(1e-6)
    assert w_old.GVD(1e-6)[0] == w_new.GVD(1e-6)
    wl = np.array([1.0, 2.0]) * um
    np.testing.assert_array_equal(w_old.neff(wl), w_new.neff(wl))


def test_old_ridge_attributes_and_signatures():
    w = old(w_top=1.2e-6, mode="TE")
    assert w.w_top == 1.2e-6 and w.h_slab == pytest.approx(350e-9) and w.mode == "TE"
    # old positional order: neff(wl, mode, T)
    assert w.neff(1e-6, "TM")[0] == waveguides.RidgeIndex(w_top=1.2e-6).neff(1e-6, mode="TM")
    assert w.neff_physical(1e-6)[0] == w.neff(1e-6)[0]


def test_old_behavioral_and_external_match_new():
    b_old = old(behavioral=True, n_f0=2.1, GVM=1e-13)
    b_new = waveguides.Waveguide(waveguides.BehavioralIndex(n_f0=2.1, GVM=1e-13))
    assert b_old.neff(1.5e-6) == b_new.neff(1.5e-6)
    assert b_old.beta1(1.2e-6)[0] == b_new.beta1(1.2e-6)
    assert b_old.beh_n_f0 == 2.1

    wl = np.linspace(0.5, 5, 200) * um
    n = materials.refractive_index("LN_MgO_e", wl)
    e_old = old(external=True, wl_array=wl, n_array=n)
    e_new = waveguides.Waveguide(waveguides.TabulatedIndex(wl, n))
    assert e_old.neff(1.3e-6) == pytest.approx(e_new.neff(1.3e-6), rel=1e-15)
    assert e_old.tck is not None


def test_old_propagate_nee_matches_new(tutorial_grid):
    t, f_ref = tutorial_grid
    wl = np.linspace(0.5, 5, 600) * um
    w_old = old(external=True, wl_array=wl, n_array=materials.refractive_index("LN_MgO_e", wl))
    w_new = make_ext_waveguide()
    for w in (w_old, w_new):
        w.set_length(1e-3)
        w.set_nonlinear_coeffs(N=1, X0=1.1e-12)
    dk = w_new.beta(1 * um) - 2 * w_new.beta(2 * um)
    p = pulses.sech_pulse(t, 100 * fs, f_ref, f0=c / (2 * um), Energy=1e-12, Npwr_dB=300)
    with pytest.deprecated_call():
        a, _ = w_old.propagate_NEE(p, verbose=False, Kg=dk)
    b, _ = w_new.propagate(p, Kg=dk)
    # same model; the spline is evaluated through different code paths
    np.testing.assert_allclose(a.a, b.a, atol=1e-8 * np.abs(b.a).max())
    assert w_old.L == 1e-3


def test_zcheck_step_is_deprecated(tutorial_grid):
    t, f_ref = tutorial_grid
    w = make_ext_waveguide()
    w.set_length(1e-4)
    p = pulses.gaussian_pulse(t, 100 * fs, f_ref, Energy=1e-12, Npwr_dB=300)
    with pytest.deprecated_call(match="zcheck_step"), pytest.deprecated_call(match="propagate_NEE"):
        w.propagate_NEE(p, verbose=False, zcheck_step=1e-3)



# ---- Pulse, NonlinearCrystal and OPO renames -------------------------------------

def test_old_pulse_class(grid):
    t, f_ref = grid
    with pytest.deprecated_call(match="Pulse"):
        p = pulses.pulse(t, np.zeros(t.size, complex), c / f_ref, 1e8)
    assert isinstance(p, pulses.Pulse)


def test_old_crystal_class_and_method(tutorial_grid):
    from snow import crystals
    t, f_ref = tutorial_grid
    n = lambda wl: materials.refractive_index("LN_MgO_e", wl)
    with pytest.deprecated_call(match="NonlinearCrystal"):
        x_old = crystals.nonlinear_crystal(1e-4, n, chi2=lambda z: 0.0)
    x_new = crystals.NonlinearCrystal(1e-4, n, chi2=lambda z: 0.0)
    p = pulses.gaussian_pulse(t, 100 * fs, f_ref, Energy=1e-12, Npwr_dB=300)
    with pytest.deprecated_call(match="propagate_NEE_fd"):
        a, _ = x_old.propagate_NEE_fd(p, verbose=False)
    b, _ = x_new.propagate(p)
    np.testing.assert_array_equal(a.a, b.a)


def test_old_opo_signature(tutorial_grid):
    from snow import crystals, opos
    t, f_ref = tutorial_grid
    n = lambda wl: materials.refractive_index("LN_MgO_e", wl)
    xtal = crystals.NonlinearCrystal(1e-3, n, chi2=lambda z: 0.0)
    args = (xtal, [0.0], np.ones(t.size), 0.5, t, np.array([0.0]), np.array([1e-3]), 250e6)
    with pytest.deprecated_call(match="OPO"), pytest.deprecated_call(match="pulse_params"), \
            pytest.deprecated_call(match="h is ignored"):
        old_opo = opos.opo(*args, 1e-6, None, [100 * fs, f_ref, c / (1 * um), 90], roundtrips=1, numsamp=1)
    assert isinstance(old_opo, opos.OPO)
    assert old_opo.pulse_params == [100 * fs, f_ref, c / (1 * um), 90]
    with pytest.deprecated_call(), pytest.raises(TypeError):
        opos.opo(*args)



def test_top_level_lumerical_module_is_deprecated():
    import importlib
    import sys
    sys.modules.pop("lumerical", None)
    with pytest.deprecated_call(match="snow import lumerical"):
        old = importlib.import_module("lumerical")
    from snow import lumerical as new
    assert old.field_2D is new.field_2D
