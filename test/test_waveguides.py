import pytest
from snow import waveguides

def test_default_wg():
    wg = waveguides.waveguide()
    assert wg.neff(1e-6) == pytest.approx(2.052, rel=1e-3)
    assert wg.beta1(1e-6)*1e9 == pytest.approx(7.592, rel=1e-3)
    assert wg.beta2(1e-6)*1e25 == pytest.approx(1.145, rel=1e-3)
