import numpy as np
import pytest

from snow import materials

# Regression values from the current implementation (wavelengths in um).
# The coefficient sets were checked against the cited papers:
# Malitson (SiO2, sapphire), Zelmon 1997 (LN, MgO:LN), Gayer 2008 (MgO:LN, T),
# Brunner 2003 (SLT). LT_MgO_e and GaP have no source in the code.
REFERENCE = {
    "SiO2":       (1.450417409406875, 1.444023621703261, 1.4380853528795101),
    "Sapphire":   (1.7478052327016724, 1.7383814860324842, 1.7299322946395292),
    "LN_MgO_e":   (2.150867607307564, 2.129929809470047, 2.118045264226726),
    "LN_MgO_o":   (2.2329676383050847, 2.208166670822557, 2.193979665710777),
    "LN_o":       (2.2362967225511574, 2.2111110086535737, 2.1966950286033553),
    "LN_e":       (2.159140321285541, 2.1375596497855565, 2.1253328107976404),
    "GaP":        (3.117238468131533, 3.054845939828454, 3.0375281938517666),
    "LN_MgO_e_T": (2.151684703384064, 2.1305703312029918, 2.118636523058051),
    "LN_MgO_o_T": (2.2337188137507855, 2.208812688703256, 2.1945800697597027),
    "LT_MgO_e":   (2.137003632561021, 2.1162290282903045, 2.1037367824342614),
    "LT_SLT":     (2.1368509581407253, 2.1169092656125397, 2.1046014498754984),
}
WAVELENGTHS_UM = (1.0, 1.55, 2.0)


# Temperature the reference values were computed at (default 24.5 C).
# LT_SLT used to hard-code 25 C (#48), so its values are pinned at 25 C.
REFERENCE_T = {"LT_SLT": 25}


@pytest.mark.parametrize("material", sorted(REFERENCE))
def test_regression_values(material):
    T = REFERENCE_T.get(material, 24.5)
    for wl, n_ref in zip(WAVELENGTHS_UM, REFERENCE[material]):
        assert materials.refractive_index(material, wl, T=T) == pytest.approx(n_ref, rel=1e-12)


def test_sio2_matches_malitson_d_line():
    # Malitson 1965: n = 1.4585 at the sodium d line (587.6 nm)
    assert materials.refractive_index("SiO2", 0.5876) == pytest.approx(1.4585, abs=1e-4)


@pytest.mark.parametrize("material", sorted(REFERENCE))
def test_unit_detection_m_nm_um_agree(material):
    n_um = materials.refractive_index(material, 1.55)
    assert materials.refractive_index(material, 1.55e-6) == pytest.approx(n_um, rel=1e-12)
    assert materials.refractive_index(material, 1550.0) == pytest.approx(n_um, rel=1e-12)


@pytest.mark.parametrize("material", sorted(REFERENCE))
def test_array_input_matches_scalar(material):
    wl = np.array(WAVELENGTHS_UM) * 1e-6
    n = materials.refractive_index(material, wl, T=REFERENCE_T.get(material, 24.5))
    assert np.shape(n) == wl.shape
    np.testing.assert_allclose(n, REFERENCE[material], rtol=1e-12)


@pytest.mark.parametrize("material", ["SiO2", "Sapphire", "LN_MgO_e", "LN_e", "GaP"])
def test_normal_dispersion_in_near_ir(material):
    n = materials.refractive_index(material, np.linspace(1.0, 2.0, 11))
    assert np.all(np.diff(n) < 0)


@pytest.mark.parametrize("pair", [("LN_MgO_e_T", "LN_MgO_e"), ("LN_MgO_o_T", "LN_MgO_o")])
def test_gayer_model_close_to_zelmon_at_room_temperature(pair):
    tmodel, zelmon = pair
    for wl in WAVELENGTHS_UM:
        n_t = materials.refractive_index(tmodel, wl, T=24.5)
        assert n_t == pytest.approx(materials.refractive_index(zelmon, wl), abs=1.5e-3)


@pytest.mark.parametrize("material", ["LN_MgO_e_T", "LN_MgO_o_T"])
def test_index_increases_with_temperature(material):
    assert materials.refractive_index(material, 1.0, T=100) > materials.refractive_index(material, 1.0, T=24.5)


def test_temperature_ignored_by_room_temperature_models():
    assert materials.refractive_index("LN_MgO_e", 1.0, T=150) == materials.refractive_index("LN_MgO_e", 1.0)


def test_air():
    assert materials.refractive_index("Air", 1.0) == 1


def test_sellmeier_single_term():
    # n^2 = 1 + A wl^2/(wl^2 - B)
    assert materials.sellmeier(np.array([1.0]), np.array([0.0]), 1.0) == pytest.approx(np.sqrt(2))


# ---- registry ----------------------------------------------------------------

def test_registry_lists_every_material():
    assert set(materials.available()) == set(REFERENCE) | {"Air"}
    for name in materials.available():
        m = materials.MATERIALS[name]
        assert m.name == name and m.description


def test_temperature_flag_matches_behaviour():
    for name, m in materials.MATERIALS.items():
        n1 = materials.refractive_index(name, 1.0, T=24.5)
        n2 = materials.refractive_index(name, 1.0, T=120)
        assert (n1 != n2) == m.temperature, name


def test_range_warning():
    with pytest.warns(materials.MaterialRangeWarning, match="LN_MgO_e"):
        materials.refractive_index("LN_MgO_e", 6.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        materials.refractive_index("LN_MgO_e", 2.0)
        materials.refractive_index("GaP", 30.0 / 2)  # no known range: no warning


def test_unrecognised_wavelength_unit_warns():
    with pytest.warns(UserWarning, match="recognised"):
        materials.refractive_index("Air", 50.0)


def test_list_input():
    np.testing.assert_allclose(materials.refractive_index("SiO2", [1.0, 2.0]),
                               [REFERENCE["SiO2"][0], REFERENCE["SiO2"][2]], rtol=1e-12)


def test_register_new_material():
    @materials.register("_TestGlass", "constant", wl_range_um=(0.4, 2.0))
    def _(wl, T):
        return 1.5 + 0 * wl
    try:
        assert materials.refractive_index("_TestGlass", 1e-6) == 1.5
        assert "_TestGlass" in materials.available()
    finally:
        del materials.MATERIALS["_TestGlass"]


# ---- #48 ---------------------------------------------------------------------

def test_lt_slt_uses_temperature():
    n25 = materials.refractive_index("LT_SLT", 1.0, T=25)
    n100 = materials.refractive_index("LT_SLT", 1.0, T=100)
    # Brunner 2003: index rises with temperature, of order 1e-5 /K
    assert 1e-6 < (n100 - n25) / 75 < 1e-4


def test_unknown_material_raises_value_error():
    with pytest.raises(ValueError, match="Unobtainium"):
        materials.refractive_index("Unobtainium", 1.0)


def test_air_array_input():
    n = materials.refractive_index("Air", np.array([1e-6, 2e-6]))
    np.testing.assert_array_equal(n, [1.0, 1.0])
