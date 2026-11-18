import numpy as np
import pytest
from scipy.constants import c, pi

from snow import waveguides, materials
from conftest import make_ext_waveguide, um

k0 = lambda wl: 2 * pi / wl


# ---- slab solvers: roots satisfy the textbook dispersion relations ----------

@pytest.mark.parametrize("d", [0.3 * um, 0.7 * um, 2 * um])
def test_symmetric_slab_te0_dispersion_relation(d):
    n0, n1, wl = 1.44, 2.2, 1 * um
    neff = waveguides.neff_symmetric_slab(n0, n1, d, wl, "TE even")
    kx = k0(wl) * np.sqrt(n1**2 - neff**2)
    gamma = k0(wl) * np.sqrt(neff**2 - n0**2)
    assert kx * np.tan(kx * d / 2) == pytest.approx(gamma, rel=1e-8)
    assert n0 < neff < n1


def test_symmetric_slab_tm0_dispersion_relation():
    n0, n1, d, wl = 1.44, 2.2, 0.7 * um, 1 * um
    neff = waveguides.neff_symmetric_slab(n0, n1, d, wl, "TM even")
    kx = k0(wl) * np.sqrt(n1**2 - neff**2)
    gamma = k0(wl) * np.sqrt(neff**2 - n0**2)
    assert (n0 / n1)**2 * kx * np.tan(kx * d / 2) == pytest.approx(gamma, rel=1e-8)
    assert neff < waveguides.neff_symmetric_slab(n0, n1, d, wl, "TE even")


def test_symmetric_slab_te1_odd_dispersion_relation():
    n0, n1, d, wl = 1.0, 2.2, 1 * um, 1 * um
    neff = waveguides.neff_symmetric_slab(n0, n1, d, wl, "TE odd")
    kx = k0(wl) * np.sqrt(n1**2 - neff**2)
    gamma = k0(wl) * np.sqrt(neff**2 - n0**2)
    assert -kx / np.tan(kx * d / 2) == pytest.approx(gamma, rel=1e-8)
    assert neff < waveguides.neff_symmetric_slab(n0, n1, d, wl, "TE even")


@pytest.mark.parametrize("mode", ["TE", "TM"])
def test_asymmetric_slab_dispersion_relation(mode):
    n0, n1, n2, d, wl = 1.0, 2.14, 1.44, 0.6 * um, 1.55 * um
    neff = waveguides.neff_asymmetric_slab(n0, n1, n2, d, wl, mode)
    kx = k0(wl) * np.sqrt(n1**2 - neff**2)
    g0 = k0(wl) * np.sqrt(neff**2 - n0**2)
    g2 = k0(wl) * np.sqrt(neff**2 - n2**2)
    if mode == "TM":
        g0, g2 = g0 * (n1 / n0)**2, g2 * (n1 / n2)**2
    assert kx * d == pytest.approx(np.arctan(g0 / kx) + np.arctan(g2 / kx), rel=1e-8)
    assert n2 < neff < n1


def test_asymmetric_slab_reduces_to_symmetric():
    n0, n1, d, wl = 1.44, 2.2, 0.5 * um, 1 * um
    assert waveguides.neff_asymmetric_slab(n0, n1, n0, d, wl, "TE") == pytest.approx(
        waveguides.neff_symmetric_slab(n0, n1, d, wl, "TE even"), rel=1e-9)


def test_singlemode_symmetric_is_te1_cutoff():
    n0, n1, wl = 1.44, 2.2, 1 * um
    d = waveguides.singlemode_symmetric(n0, n1, wl)
    V = k0(wl) * d / 2 * np.sqrt(n1**2 - n0**2)
    assert V == pytest.approx(pi / 2)


def test_singlemode_asymmetric_is_te1_cutoff():
    n0, n1, n2, wl = 1.0, 2.2, 1.44, 1 * um
    d = waveguides.singlemode_asymmetric(n0, n1, n2, wl)
    V = k0(wl) * d * np.sqrt(n1**2 - n2**2)
    assert V == pytest.approx(pi + np.arctan(np.sqrt((n2**2 - n0**2) / (n1**2 - n2**2))))


# ---- effective-index-method ridge waveguide (regression) --------------------

# (wavelength, neff, beta1 [s/m], beta2 [s^2/m]) for the default waveguide
DEFAULT_WG = [
    (1.00e-6, 2.052248094393941, 7.592548006807788e-09, 1.145424289368537e-25),
    (1.55e-6, 1.9313807021597433, 7.58650442147929e-09, -6.670054536736873e-26),
    (2.00e-6, 1.831382679819164, 7.58613671650179e-09, 1.7108913065829764e-25),
]


def test_default_wg():
    wg = waveguides.waveguide()
    assert wg.neff(1e-6) == pytest.approx(2.052, rel=1e-3)
    assert wg.beta1(1e-6) * 1e9 == pytest.approx(7.592, rel=1e-3)
    assert wg.beta2(1e-6) * 1e25 == pytest.approx(1.145, rel=1e-3)


@pytest.mark.parametrize("wl, neff, b1, b2", DEFAULT_WG)
def test_default_wg_regression(wl, neff, b1, b2):
    wg = waveguides.waveguide()
    assert wg.neff(wl)[0] == pytest.approx(neff, rel=1e-10)
    assert wg.beta1(wl)[0] == pytest.approx(b1, rel=1e-8)
    assert wg.beta2(wl)[0] == pytest.approx(b2, rel=1e-5)


def test_other_geometry_regression():
    wg = waveguides.waveguide(w_top=1.8e-6, h_thinfilm=600e-9, h_etch=300e-9, theta=70,
                              box_material="Sapphire")
    assert wg.neff(2e-6)[0] == pytest.approx(1.8661625754684894, rel=1e-10)
    assert wg.beta1(2e-6)[0] == pytest.approx(7.308825870360717e-09, rel=1e-8)


def test_ridge_neff_between_cladding_and_core():
    wg = waveguides.waveguide()
    n_core = materials.refractive_index("LN_MgO_e", 1.55)
    n_box = materials.refractive_index("SiO2", 1.55)
    assert n_box < wg.neff(1.55e-6)[0] < n_core


def test_neff_increases_with_width():
    narrow = waveguides.waveguide(w_top=0.8e-6).neff(1.55e-6)[0]
    wide = waveguides.waveguide(w_top=2.0e-6).neff(1.55e-6)[0]
    assert wide > narrow


def test_neff_accepts_arrays():
    wg = waveguides.waveguide()
    wl = np.array([1.0, 1.55, 2.0]) * 1e-6
    np.testing.assert_allclose(wg.neff(wl), [r[1] for r in DEFAULT_WG], rtol=1e-10)


def test_derived_geometry():
    wg = waveguides.waveguide(w_top=1e-6, h_thinfilm=700e-9, h_etch=350e-9, theta=60)
    assert wg.h_slab == pytest.approx(350e-9)
    assert wg.w_base == pytest.approx(1e-6 + 2 * 350e-9 / np.tan(pi / 3))


def test_invalid_geometry_raises():
    with pytest.raises(ValueError):
        waveguides.waveguide(theta=120)


# ---- beta, beta1, beta2 are consistent derivatives --------------------------

@pytest.mark.parametrize("wl", [1.0e-6, 2.0e-6])
def test_beta1_is_dbeta_domega(wl):
    wg = waveguides.waveguide()
    dl = 2e-9
    dbeta = wg.beta(wl + dl)[0] - wg.beta(wl - dl)[0]
    domega = 2 * pi * c / (wl + dl) - 2 * pi * c / (wl - dl)
    assert wg.beta1(wl)[0] == pytest.approx(dbeta / domega, rel=1e-5)


@pytest.mark.parametrize("wl", [1.0e-6, 2.0e-6])
def test_beta2_is_dbeta1_domega(wl):
    wg = make_ext_waveguide()
    dl = 5e-9
    db1 = wg.beta1(wl + dl)[0] - wg.beta1(wl - dl)[0]
    domega = 2 * pi * c / (wl + dl) - 2 * pi * c / (wl - dl)
    assert wg.beta2(wl)[0] == pytest.approx(db1 / domega, rel=1e-3)


@pytest.mark.parametrize("wl_um", [1.0, 2.0])
def test_bulk_group_index_matches_material(wl_um):
    # n_g = n - wl dn/dwl, from the material formula directly
    wg = make_ext_waveguide()
    dl = 1e-3
    dn = materials.refractive_index("LN_MgO_e", wl_um + dl) - materials.refractive_index("LN_MgO_e", wl_um - dl)
    ng = materials.refractive_index("LN_MgO_e", wl_um) - wl_um * dn / (2 * dl)
    assert wg.beta1(wl_um * 1e-6)[0] * c == pytest.approx(ng, rel=1e-5)


# ---- alternative index models -----------------------------------------------

def test_external_index_reproduces_table():
    wl = np.linspace(0.5 * um, 5 * um, 600)
    n = materials.refractive_index("LN_MgO_e", wl)
    wg = make_ext_waveguide()
    np.testing.assert_allclose(wg.neff(wl[::37]), n[::37], rtol=1e-12)
    assert wg.neff(1.234e-6) == pytest.approx(materials.refractive_index("LN_MgO_e", 1.234e-6), rel=1e-7)


def test_behavioral_model_reproduces_its_inputs():
    z_wl, wl_1, wl_2 = 1.8e-6, 1.0e-6, 2.0e-6
    GVM, delta_n, n_f0, wl_f0 = 50e-15 / 1e-3, -0.05, 2.0, 1.5e-6
    wg = waveguides.waveguide(behavioral=True, z_wl=z_wl, wl_1=wl_1, wl_2=wl_2,
                              GVM=GVM, delta_n=delta_n, n_f0=n_f0, wl_f0=wl_f0)
    assert wg.neff(wl_f0) == pytest.approx(n_f0, rel=1e-12)
    assert wg.neff(wl_2) - wg.neff(wl_1) == pytest.approx(delta_n, rel=1e-9)
    assert wg.beta1(wl_2)[0] - wg.beta1(wl_1)[0] == pytest.approx(GVM, rel=1e-4)
    assert abs(wg.beta2(z_wl)[0]) < 1e-3 * abs(wg.beta2(wl_1)[0])


def test_behavioral_default_wl_f0_is_midpoint():
    wg = waveguides.waveguide(behavioral=True, n_f0=2.1)
    assert wg.neff(1.5e-6) == pytest.approx(2.1, rel=1e-12)


# ---- #49: TM, full etch, higher-order slab modes ----------------------------

def test_tm_mode():
    te = waveguides.waveguide().neff(1e-6)[0]
    tm = waveguides.waveguide().neff(1e-6, mode="TM")[0]
    n_box = materials.refractive_index("SiO2", 1.0)
    # thin film on SiO2 with air above: the TM-like mode is less confined
    assert n_box < tm < te


def test_tm_waveguide_option():
    wg = waveguides.waveguide(mode="TM")
    assert wg.neff(1e-6)[0] == waveguides.waveguide().neff(1e-6, mode="TM")[0]
    assert wg.beta1(1e-6)[0] > 0


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        waveguides.waveguide(mode="TEM")
    with pytest.raises(ValueError):
        waveguides.neff_ridge(1e-6, 2.1, mode="TEM")


def test_ridge_tm_uses_tm_vertical_and_te_lateral_slabs():
    wl, n_film, n_box, n_clad = 1e-6, 2.2, 1.44, 1.0
    w, h, hslab = 1.5e-6, 0.7e-6, 0.3e-6
    n_r = waveguides.neff_asymmetric_slab(n_clad, n_film, n_box, h, wl, "TM")
    n_s = waveguides.neff_asymmetric_slab(n_clad, n_film, n_box, hslab, wl, "TM")
    expected = waveguides.neff_symmetric_slab(n_s, n_r, w, wl, "TE even")
    assert waveguides.neff_ridge(wl, n_film, n_box, n_clad, w, h, hslab, "TM") == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["TE", "TM"])
def test_full_etch(mode):
    n_box = materials.refractive_index("SiO2", 1.0)
    # vertical sidewalls, so the etch depth does not change the mean width
    full = waveguides.waveguide(h_etch=700e-9, theta=90, mode=mode).neff(1e-6)[0]
    partial = waveguides.waveguide(h_etch=450e-9, theta=90, mode=mode).neff(1e-6)[0]
    assert n_box < full < partial
    # a 2 nm slab guides nothing either: continuous with the full etch
    near = waveguides.waveguide(h_etch=698e-9, theta=90, mode=mode).neff(1e-6)[0]
    assert near == pytest.approx(full, rel=1e-12)


def test_neff_derivative_helpers_are_deprecated():
    with pytest.deprecated_call():
        waveguides.neff_derivative(1.0, 2.2, 1.44, 1.0, 1.5, 0.7, 0.3)
    with pytest.deprecated_call():
        waveguides.neff_2derivative(1.0, 2.2, 1.44, 1.0, 1.5, 0.7, 0.3)


def test_slab_below_cutoff_uses_cladding_index():
    # a 5 nm slab guides no mode at 2 um; the slab region gets the box index
    n_box = 1.44
    assert waveguides._slab_index(1.0, 2.2, n_box, 5e-9, 2e-6, "TE") == n_box
    assert waveguides._slab_index(1.0, 2.2, n_box, 0.0, 2e-6, "TE") == n_box
    guided = waveguides._slab_index(1.0, 2.2, n_box, 0.5e-6, 2e-6, "TE")
    assert guided == waveguides.neff_asymmetric_slab(1.0, 2.2, n_box, 0.5e-6, 2e-6)


@pytest.mark.parametrize("parity, order", [("even", 1), ("even", 2), ("odd", 1)])
@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_symmetric_slab_higher_order_dispersion_relation(pol, parity, order):
    n0, n1, d, wl = 1.0, 2.2, 2 * um, 1 * um
    neff = waveguides.neff_symmetric_slab(n0, n1, d, wl, f"{pol} {parity}", order=order)
    kx = k0(wl) * np.sqrt(n1**2 - neff**2)
    gamma = k0(wl) * np.sqrt(neff**2 - n0**2)
    X = kx * d / 2
    f = np.tan(X) if parity == "even" else -1 / np.tan(X)
    pol_factor = (n0 / n1)**2 if pol == "TM" else 1.0
    assert pol_factor * kx * f == pytest.approx(gamma, rel=1e-8)
    # mode number 2*order (+1 for odd): X lies in its own branch
    m = 2 * order + (parity == "odd")
    assert m * pi / 2 <= X < (m + 1) * pi / 2


def test_symmetric_slab_modes_are_ordered():
    n0, n1, d, wl = 1.0, 2.2, 2 * um, 1 * um
    modes = [waveguides.neff_symmetric_slab(n0, n1, d, wl, f"TE {p}", order=m)
             for m in range(3) for p in ("even", "odd")]
    assert np.all(np.diff(modes) < 0)
    assert modes[-1] > n0


def test_symmetric_slab_cutoff_raises():
    with pytest.raises(ValueError, match="cut off"):
        waveguides.neff_symmetric_slab(1.0, 2.2, 0.2 * um, 1 * um, "TE odd")


# ---- linear propagation (#45) -----------------------------------------------

def test_propagate_linear_conserves_energy(grid):
    from snow import pulses
    t, f_ref = grid
    p = pulses.gaussian_pulse(t, 100e-15, f_ref, Energy=1e-12, Npwr_dB=300)
    wg = make_ext_waveguide()
    wg.set_length(1e-3)
    out = wg.propagate_linear(p)
    assert out.energy_td() == pytest.approx(p.energy_td(), rel=1e-9)
    assert out.frep == p.frep


def test_propagate_linear_loss(grid):
    from snow import pulses, util
    t, f_ref = grid
    p = pulses.gaussian_pulse(t, 100e-15, f_ref, Energy=1e-12, Npwr_dB=300)
    wg = make_ext_waveguide()
    wg.set_length(10e-3)
    wg.set_loss(util.absorption_coeff(1.0))
    assert wg.propagate_linear(p).energy_td() / p.energy_td() == pytest.approx(0.1**0.1, rel=1e-9)


@pytest.mark.parametrize("v_ref_wl", [None, 1.5e-6])
def test_propagate_linear_matches_nee_without_nonlinearity(tutorial_grid, v_ref_wl):
    from snow import pulses
    t, f_ref = tutorial_grid
    p = pulses.gaussian_pulse(t, 100e-15, f_ref, f0=c / (1.2 * um), Energy=1e-12, Npwr_dB=300)
    wg = make_ext_waveguide()
    wg.set_length(2e-3)
    wg.set_loss(5.0)
    wg.add_poling(lambda z: 1.0)
    wg.set_nonlinear_coeffs(N=1, X0=0.0)
    v_ref = None if v_ref_wl is None else 1 / wg.beta1(v_ref_wl)[0]
    lin = wg.propagate_linear(p, v_ref=v_ref)
    nee, _ = wg.propagate_NEE(p, v_ref=v_ref, verbose=False)
    np.testing.assert_allclose(lin.a, nee.a, atol=1e-6 * np.abs(p.a).max())


def test_propagate_linear_gvd_broadening(tutorial_grid):
    from snow import pulses
    t, f_ref = tutorial_grid
    tau, wl = 100e-15, 1 * um
    wg = make_ext_waveguide()
    LD = tau**2 / (4 * np.log(2) * wg.beta2(wl)[0])
    wg.set_length(LD)
    p = pulses.gaussian_pulse(t, tau, f_ref, f0=c / wl, Energy=1e-12, Npwr_dB=300)
    out = wg.propagate_linear(p, v_ref=1 / wg.beta1(wl)[0])
    assert out.width_FWHM() == pytest.approx(np.sqrt(2) * p.width_FWHM(), rel=0.02)
