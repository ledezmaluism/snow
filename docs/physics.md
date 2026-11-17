# Physics and numerical model

This page describes what snow computes and the conventions behind it. The
code references are to `snow/pulses.py`, `snow/waveguides.py` and
`snow/nlo.py`.

## Grids

A simulation lives on a uniform time grid `t` of `N` points with spacing `dt`,
centred on t = 0. The frequency grid follows from the FFT:

- `f = fftfreq(N, dt)` are frequencies relative to a reference `f0`, in FFT
  order (zero first, then positive, then negative frequencies);
- `f_abs = f + f0` are absolute frequencies, `wl = c / f_abs` wavelengths;
- `df = 1/(N dt)`, and the grid spans a bandwidth `BW = N df = 1/dt`.

Pick the bandwidth first (it must contain every frequency the process
generates, e.g. the pump and its second harmonic), then `N`, which sets the
time window `T = N dt`. `f_abs` must stay positive; snow warns otherwise.
Tutorial 2 goes through this.

## Pulses

`pulse.a(t)` is the complex envelope relative to the carrier `f0`, scaled so
that `abs(a)**2` is the instantaneous power in W. The spectrum is

    A = fft(fftshift(a))

The `fftshift` puts the time origin at t = 0, so a transform-limited pulse
centred at t = 0 has a flat spectral phase. Derived quantities:

| quantity | definition | units |
|---|---|---|
| energy | `sum(abs(a)**2) * dt` | J |
| energy spectral density | `abs(fft(a) * dt)**2` | J/Hz |
| power spectral density | ESD x `frep` | W/Hz |

`gaussian_pulse` and `sech_pulse` take the intensity FWHM, an energy (or peak
or average power), the reference frequency `f_ref` and the absolute carrier
frequency `f0`. They add white noise `Npwr_dB` below the peak power.

## Waveguide dispersion

`waveguide.neff(wl)` comes from one of three models:

1. **Geometry** (default). The effective index method: the ridge region
   (film thickness) and the slab region (film minus etch) are each solved as
   asymmetric slabs (cladding / film / box) for their fundamental mode. The
   two indices then form a symmetric slab of width (top + base)/2, solved for
   its fundamental even mode. For the TE-like ridge mode (`mode='TE'`, the
   default) the vertical slabs are TE and the lateral slab TM; for the
   TM-like mode (`mode='TM'`) the other way round. A slab region that guides
   no mode (full etch, or a slab below cut-off) is given the larger of the
   box and cladding indices. With an anisotropic film, choose the material
   for the index along the field (e.g. `LN_MgO_o` for TM in x-cut LN).
2. **Behavioral**. The GVD is a quadratic in angular frequency,
   beta2(omega) = c1 (omega - z1)(omega - z2), integrated analytically to
   beta1, beta and n. The integration constants are fixed by a zero-GVD
   wavelength, a GVM between two wavelengths, an index difference and an
   index at one wavelength. This is useful for exploring dispersion
   regimes without committing to a geometry.
3. **External**. A cubic spline through tabulated (wavelength, n_eff) data,
   for example from a full-vectorial mode solver.

From n_eff:

- beta = 2 pi n_eff / wl,
- beta1 = d beta / d omega = (n_eff - wl dn_eff/dwl) / c (inverse group
  velocity),
- beta2 = d beta1 / d omega (GVD),

with derivatives from Richardson extrapolation of central differences
(`util.derivative`).

## Propagation equation

`waveguide.propagate_NEE` integrates, for the spectrum A(Omega, z),

    dA/dz = -i D(Omega) A  -  i k(z, Omega) F{ a^2 e^{i phi} + 2 |a|^2 e^{-i phi} }

with

    D(Omega) = beta(omega0 + Omega) - beta0 - Omega / v_ref - i alpha/2
    phi(t, z) = omega0 t - (beta0 - omega0 / v_ref + Kg) z
    k(z, omega) = d(z) X0 omega / (4 N)

- Fields vary as exp(i(omega t - beta z)).
- `beta0` is beta at the reference frequency `omega0 = 2 pi f0`.
- The time axis moves with velocity `v_ref` (by default the group velocity at
  `f0`). Choose it so the pulses of interest stay inside the window, e.g.
  `v_ref = 1/wg.beta1(wl_sh)[0]` keeps the second harmonic stationary.
- `alpha` is the power attenuation coefficient: power decays as
  exp(-alpha z). `util.absorption_coeff` converts from dB/cm.
- The term a^2 e^{i phi} generates sum frequencies (SHG, SFG). The term
  2 |a|^2 e^{-i phi} generates difference frequencies (DFG, OPA). Because a
  single envelope carries all fields, the same equation covers degenerate and
  non-degenerate processes and overlapping broadband spectra, without
  splitting the fields into separate waves.
- `d(z)` is the poling function set with `add_poling` (+1/-1 for periodic
  poling). `X0` is the effective chi(2) and `N` the mode normalization; with
  `|a|^2` in W they absorb the mode overlap.
- Quasi-phase matching shortcut: with `Kg != 0`, `d(z)` is replaced by the
  first Fourier coefficient of a 50 % duty-cycle grating, 2/pi, and the
  grating wavevector `Kg = 2 pi / period` enters the phase. The integrator can
  then take steps much longer than a domain. Use it for uniform gratings; for
  chirped or apodized poling pass the full `d(z)` and check convergence
  (#50).

## Solver

`nlo.NEE` works in the interaction picture, y = A e^{iDz}, so the linear part
is exact and only the nonlinear term is integrated, with scipy's adaptive
`RK45` (rtol = atol = 1e-4). At each evaluation the envelope is zero-padded
in frequency to a 4x finer time grid (8x if the bandwidth needs it), the
products are formed there, and the result is truncated back. This removes
aliasing of the sum-frequency products. The function returns the output
envelope and the list of step sizes.

`Qnoise=True` adds vacuum fluctuations to the input: half a photon,
h f / 2, of energy per frequency bin, with random phase. This seeds
spontaneous processes such as parametric generation.

## Bulk crystals

`crystals.nonlinear_crystal` uses the same solver with beta from a bulk index
function and k = chi2(z) omega / (4 n c). There is no transverse
normalization, so absolute conversion efficiencies are not meaningful yet.

## References

- M. Conforti, F. Baronio, and C. De Angelis, Phys. Rev. A 81, 053841 (2010).
- T. Brabec and F. Krausz, Phys. Rev. Lett. 78, 3282 (1997).
- O. Gayer et al., Appl. Phys. B 91, 343 (2008); erratum Appl. Phys. B 101,
  481 (2010).
- D. E. Zelmon, D. L. Small, and D. Jundt, J. Opt. Soc. Am. B 14, 3319 (1997).
- I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965).
