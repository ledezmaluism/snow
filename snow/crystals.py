# -*- coding: utf-8 -*-
"""
Bulk crystals (plane-wave propagation) instead of waveguides.

@author: Luis Ledezma
"""
import warnings

import numpy as np
from numpy.fft import fftshift
from scipy.constants import pi, c

from . import nlo

class NonlinearCrystal:
    '''
    A bulk crystal (plane-wave propagation).

    Parameters
    ----------
    L : float
        Length (m).
    n_func : callable
        n_func(wl) returns the refractive index at wavelength(s) wl (m),
        e.g. ``lambda wl: materials.refractive_index('LN_MgO_e', wl)``.
    chi2 : callable
        chi2(z) returns the chi(2) coefficient (m/V) at position z,
        including the poling sign.
    alpha : float
        Power attenuation coefficient (1/m).

    The coupling is k = chi2(z) omega / (4 n c). Note that it multiplies a
    field normalised to sqrt(W), with no beam-area normalization, so
    physical chi(2) values give negligible conversion; the bulk model is
    unfinished.
    '''

    def __init__(self, L, n_func, chi2, alpha=0):
        self.L = L
        self.n_func = n_func
        self.chi2 = chi2
        self.alpha = alpha

    @property
    def length(self):
        '''Length (m); same as `L`.'''
        return self.L

    def propagate(self, pulse, v_ref=None, verbose=False, z0=0, Qnoise=False,
                  rtol=1e-4, atol=1e-4, max_step=np.inf, rng=None, z_save=None):
        '''
        Propagate a pulse through the crystal with `nlo.propagate`.

        v_ref defaults to the group velocity at the first FFT bin (the
        reference frequency), from a finite difference of beta on the
        pulse grid. Returns an `nlo.PropagationResult`, which unpacks as
        (output pulse, integrator step sizes); `z_save` adds snapshots.
        '''
        f_abs = pulse.f_abs
        n = self.n_func(pulse.wl)
        beta = 2*pi*f_abs*n/c

        if v_ref is None:
            df = f_abs[1] - f_abs[0]
            beta_1 = fftshift(np.gradient(fftshift(beta), 2*pi*df))
            v_ref = 1/beta_1[0]

        chi2 = self.chi2
        omega_abs = 2*pi*pulse.f0 + pulse.Omega

        def k(z):
            return chi2(z)*omega_abs/(4*n*c)

        return nlo.propagate(pulse, self.L, beta, v_ref, k, alpha=self.alpha, z0=z0,
                             Qnoise=Qnoise, rtol=rtol, atol=atol, max_step=max_step,
                             verbose=verbose, rng=rng, z_save=z_save)

    def propagate_NEE_fd(self, pulse, v_ref=None, verbose=True, **kwargs):
        '''Deprecated: use `propagate` (which is quiet by default).'''
        warnings.warn('propagate_NEE_fd is deprecated; use propagate (verbose=False by default)',
                      DeprecationWarning, stacklevel=2)
        return self.propagate(pulse, v_ref=v_ref, verbose=verbose, **kwargs)


class nonlinear_crystal(NonlinearCrystal):
    '''Deprecated: use `NonlinearCrystal`.'''
    def __init__(self, *args, **kwargs):
        warnings.warn('crystals.nonlinear_crystal is deprecated; use crystals.NonlinearCrystal',
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
