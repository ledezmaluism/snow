# Changelog

## 0.2.0

This release fixes every known calculation bug, adds a test suite, and gives
the package a cleaner API. The 0.1 names still work and raise a
`DeprecationWarning` naming their replacement; they will be removed in the
next release. The README has an upgrade table.

### Results that change

Nonlinear results from 0.1 should be recomputed. The fixes that change
numbers:

- The NEE solver's upsampled time grid was mis-spaced, so the carrier phase
  drifted across the window: energy was not conserved and the second harmonic
  was shifted (#40). This affected every nonlinear simulation.
- With a poling function, the solver stepped across domain walls; results were
  not converged (about 10 % too much SH in a 1 mm device) (#50).
- The solver's absolute tolerance was in FFT units, so weak components such as
  a noise-seeded parametric signal were integrated with 20-120 % errors. The
  state is now normalized by the input's peak amplitude and the default `atol`
  is 1e-10 on that scale (#51). An `atol` you pass yourself now has this
  relative meaning.
- `apply_filter` / `filter_signal` and `Pulse(..., domain='Freq')` returned the
  pulse shifted by half a window (#41, #42). This also broke the OPO round trip.
- `gaussian_pulse` and `sech_pulse` defaulted to a 0 Hz carrier; `f0` now
  defaults to `f_ref` (#43).
- `LT_SLT` ignored the temperature (#48).
- The default moving frame of `propagate` ignored the temperature (#46).

### Fixed

- `width_FWHM` returned an array; `plot_magsq` ignored `t_unit` (#44).
- `propagate_linear` could not run (#45).
- `propagate_NEE` needed `set_loss` to run (#46).
- `snow.opos` could not run; `get_outenergy` returned W instead of J; the
  output-power and efficiency plots were swapped (#47).
- Unknown material names crashed; `Air` returned a scalar for arrays (#48).
- The ridge model had no TM mode, failed on fully etched ridges, and the slab
  solver failed for higher-order modes (#49).
- Division, typo, import and deprecation fixes across modules (#38).

### Added

- `Waveguide` built from an index model: `RidgeIndex` (geometry, effective
  index method, TE or TM), `BehavioralIndex` (dispersion parameters) or
  `TabulatedIndex` (your own mode-solver data). Length, loss, poling and
  nonlinearity can be passed to the constructor. Dispersion methods return a
  float for a scalar wavelength.
- `Waveguide.propagate` and `NonlinearCrystal.propagate` on a shared core,
  `nlo.propagate`, returning a `PropagationResult` with optional z snapshots
  (`z_save=`); it still unpacks as `out, steps`.
- Explicit poling functions are integrated exactly: the domain walls are
  found and the integrator restarts at each one.
- `snow.plotting`, including `plot_evolution` for maps along z.
- `materials.available()`, `MATERIALS`, `register()`, and
  `MaterialRangeWarning` when a formula is used outside its fitted range.
- `rng=` for reproducible noise; problems are reported as Python warnings.
- `opos.OPO` with keyword arguments; the gain medium can be a waveguide or a
  crystal.
- `snow.lumerical` (the top-level `lumerical` module is deprecated).
- A pip-installable package with optional extras `opo`, `tutorials`, `test`
  and `dev` (#39).
- A test suite (227 tests, a few seconds), CI on GitHub Actions and ruff.
- A faster NEE right-hand side (about 35 %).
- Documentation: `docs/physics.md` (model and conventions), `CONTRIBUTING.md`,
  and eight tutorials. Tutorials 1-5 were rewritten with theory notes and
  checks against analytic results; 6-8 (broadband SHG, temperature tuning,
  chirped and apodized poling) are new.

### Known issues

- `apply_filter(type='butterworth')` applies the phase of a digital IIR
  filter, which delays the pulse (#52). Use the default `type='ideal'`.
- The default step limit for explicit poling is set by the shortest domain,
  which makes apodized gratings slow; pass `max_step` (#53).
- `snow.crystals` has no transverse beam normalization, so its chi(2) sets
  the gain only up to a scale.

### Contributors

Thanks to Nicolas Quesada (@nquesada) for making snow pip installable (#39)
and to Tom Soulanille (@pramasoul) for the bug fixes in #38.
