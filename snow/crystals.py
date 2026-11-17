# -*- coding: utf-8 -*-
"""
@author: Luis Ledezma

Module to work with bulk crystals instead of waveguides
"""
import time
import numpy as np
from numpy.fft import fftshift
from scipy.constants import pi, c

from . import nlo
from . import pulses

class nonlinear_crystal():
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

    def propagate_NEE_fd(self, pulse, v_ref=None,
                         verbose=True, z0=0):
        '''
        Propagate a pulse through the crystal with `nlo.NEE`.

        v_ref defaults to the group velocity at the first FFT bin (the
        reference frequency). Returns (output pulse, integrator step sizes).
        '''

        #Timer
        tic_total = time.time()

        #Get stuff
        wl = pulse.wl
        f0 = pulse.f0
        f_abs = pulse.f_abs
        Omega  = pulse.Omega

        n = self.n_func(wl)
        beta = 2*pi*f_abs*n/c

        if v_ref == None:
            df = f_abs[1] - f_abs[0]
            beta_1 = fftshift(np.gradient(fftshift(beta), 2*pi*df))
            vg = 1/beta_1
            v_ref = vg[0]

        beta_ref = beta[0]
        beta_1_ref = 1/v_ref
        D = beta - beta_ref - Omega/v_ref - 1j*self.alpha/2

        chi2 = self.chi2
        omega_ref = 2*pi*f0
        omega_abs = omega_ref + Omega

        def k(z):
            return chi2(z)*omega_abs/(4*n*c)

        [a, a_evol] = nlo.NEE(t = pulse.t,
                          x = pulse.a,
                          Omega = Omega,
                          f0 = pulse.f0,
                          L = self.L,
                          D = D,
                          b0 = beta_ref,
                          b1_ref = beta_1_ref,
                          k = k,
                          z0 = z0,
                          verbose = verbose)

        tdelta = time.time() - tic_total
        if verbose:
            print('Total time = %0.1f s' %(tdelta))

        output_pulse = pulses.pulse(pulse.t, a, pulse.wl0, pulse.frep)
        return output_pulse, a_evol
