# snow: Simulator for Nonlinear Optical Waveguides

<img src="img/snow-logo-1.png" title="snow" alt="snow logo" width="400">

snow is a Python toolbox for designing integrated nonlinear optical devices.
It was written to simulate thin-film lithium niobate (TFLN) waveguides, but the
material database and the waveguide models can describe other platforms.

It includes:

- **Materials**: refractive indices of LiNbO3 (congruent and MgO-doped, with
  temperature dependence), LiTaO3, SiO2, sapphire and GaP.
- **Waveguide dispersion**: effective index, group velocity and GVD of ridge
  waveguides, from the geometry (effective index method), from a "behavioral"
  dispersion model, or from your own mode-solver data.
- **Nonlinear pulse propagation**: a chi(2) nonlinear envelope equation solver
  for broadband SHG, SFG, DFG and parametric amplification, with arbitrary
  poling and optional vacuum noise.
- **Pulse tools**: Gaussian and sech pulses, energy, FWHM, spectra, filters,
  coherence and plotting.
- Experimental: bulk crystals (`snow.crystals`), synchronously pumped OPOs
  (`snow.opos`), and Lumerical MODE scripting helpers (`lumerical.py`).

## Installation

snow needs Python 3.10 or newer. From a clone of this repository:

```bash
git clone https://github.com/ledezmaluism/snow
cd snow
pip install -e .                 # library only
pip install -e ".[tutorials]"    # also colorcet and jupyter, for the tutorials
pip install -e ".[test]"         # also pytest
```

The editable install (`-e`) picks up changes when you `git pull`.

## Quick start

Second-harmonic generation of a 100 fs pulse at 2 um in a periodically poled
TFLN ridge:

```python
import numpy as np
from scipy.constants import c
from snow import pulses, waveguides

# Time/frequency grid covering 0.8-3 um
f_max, f_min = c/0.8e-6, c/3e-6
N = 2**10
dt = 1/(f_max - f_min)
t = (np.arange(N) - N//2) * dt
f_ref = (f_max + f_min)/2

# 100 fs, 1 pJ sech pump at 2 um (always give the carrier f0 explicitly)
pump = pulses.sech_pulse(t, 100e-15, f_ref, f0=c/2e-6, Energy=1e-12, frep=250e6)

# 700 nm MgO:LN film, 350 nm etch, 1.8 um wide, 1 mm long
wg = waveguides.waveguide(w_top=1.8e-6, h_thinfilm=700e-9, h_etch=350e-9)
wg.set_length(1e-3)
wg.set_loss(0)
wg.set_nonlinear_coeffs(N=1, X0=1.1e-12)

# First-order quasi-phase matching of 2 um -> 1 um
dk = wg.beta(1e-6)[0] - 2*wg.beta(2e-6)[0]
print(f"Poling period: {2*np.pi/dk*1e6:.3f} um")

out, steps = wg.propagate_NEE(pump, v_ref=1/wg.beta1(1e-6)[0], Kg=dk, verbose=False)

S = np.abs(out.A)**2
sh = out.f_abs > 1.5*c/2e-6           # everything above 1.5x the pump frequency
print(f"SH conversion: {S[sh].sum()/S.sum():.1%}")
print(f"Energy out/in: {out.energy_td()/pump.energy_td():.5f}")
```

```
Poling period: 5.179 um
SH conversion: 41.6%
Energy out/in: 0.99998
```

`Kg` replaces the square-wave poling by its first Fourier component (2/pi), so
the solver can take long steps. For non-uniform poling (chirps, apodization)
pass a function of z to `wg.add_poling` instead, and check convergence: the
adaptive step is not limited to a poling domain yet (#50).

## Tutorials

The notebooks in [`tutorials/`](tutorials) walk through a simulation step by
step:

1. [Introduction](tutorials/1%20Introduction.ipynb): workflow and references
2. [Time and frequency grids](tutorials/2%20Time%20and%20Frequency%20Grids.ipynb)
3. [Pulses](tutorials/3%20Pulses.ipynb): creating, analyzing and plotting pulses
4. [Waveguides](tutorials/4%20Waveguides.ipynb): effective index, GVM/GVD,
   dispersion engineering
5. [SHG](tutorials/5%20SHG.ipynb): a full nonlinear simulation

## Package overview

| Module | Contents |
|---|---|
| `snow.materials` | `refractive_index(material, wl, T)` for every material, with wavelength in m, nm or um |
| `snow.waveguides` | `waveguide` class (`neff`, `beta`, `beta1`, `beta2`/`GVD`, `propagate_NEE`) and the slab solvers |
| `snow.pulses` | `pulse` class, `gaussian_pulse`, `sech_pulse`, filters, energy/FWHM/PSD helpers, `coherence_g1`, plotting |
| `snow.nlo` | `NEE`, the adaptive chi(2) nonlinear envelope equation solver |
| `snow.util` | Richardson derivative, dB/cm to 1/m conversion |
| `snow.crystals` | `nonlinear_crystal` for bulk propagation (experimental) |
| `snow.opos` | `opo` class for synchronously pumped OPOs (currently broken, #47) |
| `lumerical.py` | Lumerical MODE geometry and mode-solver scripting, mode overlap helpers |

`crystals` and `opos` are not imported by `import snow`; import them directly.

## Conventions and units

- SI units throughout (m, s, Hz, W, J), except that `materials.refractive_index`
  also accepts nm and um.
- The time grid `t` is uniform and centred on t = 0.
- `pulse.a` is the complex envelope in sqrt(W), so `abs(a)**2` is power.
- `pulse.A = fft(fftshift(a))` is the spectrum, in FFT order. `pulse.f` are
  frequencies relative to the reference `pulse.f0`, and `pulse.f_abs` are
  absolute frequencies.
- Fields vary as exp(i(omega t - beta z)); loss `alpha` is a power attenuation
  coefficient (1/m).

[docs/physics.md](docs/physics.md) describes the propagation model, the
equation the solver integrates, and how the grids, frame and coupling are
defined.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

The suite runs in a few seconds. Known bugs are tests marked
`xfail(strict=True)` with the issue number in the reason: they fail today,
and once a bug is fixed the test passes, which strict mode reports as an
error until the marker is removed. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Known issues

Open bugs are tracked on the
[issue tracker](https://github.com/ledezmaluism/snow/issues?q=is%3Aissue+is%3Aopen+label%3Abug).
The ones most likely to affect results:

- [#41](https://github.com/ledezmaluism/snow/issues/41): `apply_filter` returns
  the pulse shifted by half a window.
- [#43](https://github.com/ledezmaluism/snow/issues/43): `gaussian_pulse` and
  `sech_pulse` default to a 0 Hz carrier; always pass `f0`.
- [#46](https://github.com/ledezmaluism/snow/issues/46): call `set_loss` before
  `propagate_NEE`, and pass `v_ref` when using a temperature.
- [#50](https://github.com/ledezmaluism/snow/issues/50): explicit square-wave
  poling is not converged by the adaptive solver.

Fixed: [#40](https://github.com/ledezmaluism/snow/issues/40), a mis-spaced
upsampled time grid in the NEE solver that affected every nonlinear result
before commit 2c1c17b.

## References

The engine solves a frequency-domain form of the nonlinear envelope equation
described in:

- M. Conforti, F. Baronio, and C. De Angelis, "Nonlinear envelope equation for
  broadband optical pulses in quadratic media," Phys. Rev. A 81, 053841 (2010).
- T. Brabec and F. Krausz, "Nonlinear optical pulse propagation in the
  single-cycle regime," Phys. Rev. Lett. 78, 3282 (1997).

The material models cite their sources in `snow/materials.py`.

## License

MIT, see [LICENSE](LICENSE).
