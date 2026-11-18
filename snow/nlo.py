# -*- coding: utf-8 -*-
"""
Nonlinear envelope equation (NEE) solver for chi(2) propagation.

`propagate` is the common entry point: a medium supplies its propagation
constant and coupling, and `propagate` builds the linear operator and calls
`NEE`. Most users go through `waveguide.propagate_NEE` or
`nonlinear_crystal.propagate_NEE_fd`. See docs/physics.md for the model and
conventions.

@author: Luis Ledezma
"""
import time
import warnings

import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.constants import pi, h

from scipy.integrate import RK45

from . import pulses

def NEE(t, x, Omega, f0,
        L, D, b0, b1_ref, k, 
        z0=0, verbose=True, Kg=0, Qnoise=False,
        rtol=1e-4, atol=1e-4, max_step=np.inf, z_breaks=None, rng=None):
    """
    Integrate the chi(2) nonlinear envelope equation from z0 to L.

    The spectrum A(Omega, z) of the envelope obeys

        dA/dz = -i D A - i k(z) F{ a^2 e^{i phi} + 2 |a|^2 e^{-i phi} },
        phi(t, z) = omega0 t - (b0 - omega0 b1_ref + Kg) z,

    where a = F^-1{A}, omega0 = 2 pi f0 and F is the FFT. The first nonlinear
    term generates sum frequencies (SHG/SFG), the second difference
    frequencies. The equation is solved in the interaction picture
    (y = A e^{iDz}) with scipy's adaptive RK45, restarted at every position
    in `z_breaks`. The
    nonlinear products are evaluated on a 4x upsampled grid (8x when the
    bandwidth requires it) to avoid aliasing.

    Parameters
    ----------
    t : array
        Time grid (s), uniformly spaced.
    x : array
        Input envelope a(t) in the time domain, in sqrt(W).
    Omega : array
        Angular frequencies relative to the reference, 2 pi fftfreq(N, dt).
    f0 : float
        Reference (carrier) frequency of the envelope (Hz).
    L : float
        End position (m).
    D : array
        Linear operator beta(Omega) - b0 - Omega b1_ref - i alpha/2 (1/m).
    b0 : float
        Propagation constant at the reference frequency (1/m).
    b1_ref : float
        Inverse velocity of the moving frame, 1/v_ref (s/m).
    k : callable
        k(z) returns the nonlinear coupling at position z, a scalar or an
        array over frequency (1/(m sqrt(W))).
    z0 : float, optional
        Start position (m).
    verbose : bool, optional
        Print the integrator status at the end.
    Kg : float, optional
        Grating wavevector for quasi-phase matching (1/m). With Kg != 0 the
        caller normally replaces the poling function by its first Fourier
        coefficient, 2/pi.
    Qnoise : bool, optional
        Add vacuum noise (half a photon per frequency bin, random phase)
        to the input.
    rtol, atol : float, optional
        Relative and absolute tolerances of the RK45 integrator.
    max_step : float, optional
        Largest step the integrator may take (m).
    z_breaks : array, optional
        Positions (m) where k(z) is discontinuous, e.g. poling domain walls.
        The integrator is restarted at each one, so no step crosses a
        discontinuity.
    rng : numpy Generator, int or None, optional
        Random generator (or seed) for the vacuum noise. None uses numpy's
        global generator, as before.

    Returns
    -------
    a : array
        Output envelope in the time domain.
    steps : array
        Step sizes (m) taken by the integrator. Callers call this
        `a_evol` for historical reasons; it is not the field evolution.
    """
    #Get stuff
    NFFT = t.size
    Omega_abs = Omega + 2*pi*f0
    f_max = np.amax(Omega_abs) / (2*pi)
    f_min = np.amin(Omega_abs) / (2*pi)
    BW = f_max - f_min
    Δt = 1/BW
    f = fftfreq(NFFT, Δt)
    Δf = f[1] - f[0]
    f_abs = f + f0
    
    #Calculate upsampling parameter, it usually will be Nup=4,
    #so, throw a warning if it needs to be 8
    Nup = 4
    if (3*f_max - f_min)/BW > Nup:
        Nup = 8
        warnings.warn('NEE: the bandwidth needs 8x upsampling (slower); '
                      'consider centring the reference frequency', RuntimeWarning, stacklevel=2)

    #Quantum noise
    if Qnoise:
        ϕ = random_source(rng).uniform( 0, 2*pi, NFFT )
        Xnoise = BW * np.sqrt(h*f_abs/2/Δf) * np.exp(1j * ϕ)
        xnoise = ifft(Xnoise)
    else:
        xnoise = np.zeros(NFFT)

    #Input signal to frequency domain
    A = fft(x + xnoise)
    Aup = np.zeros( Nup*NFFT ) * 1j
    
    #Upsampled time: same start and spacing dt/Nup as the input grid
    #(linspace(t[0], t[-1]) would stretch the spacing and drift the carrier phase)
    tup = t[0] + np.arange(Nup*NFFT) * (t[1] - t[0]) / Nup
    phi_1 = 2*pi*f0*tup
    phi_2 = b0 - b1_ref*2*pi*f0 + Kg
    
    #Upsampling stuff
    M = NFFT*Nup - A.size
    Xc = np.zeros(M)
    center = A.size // 2 + 1

    #Nonlinear function
    def fnl(z, y):
        phi = phi_1 - phi_2*z
        
        #get fast envelope
        y = y * np.exp(-1j*D*z)
        
        #Upsample
        Aup[:center] = y[:center]
        Aup[center:center+M] = Xc
        Aup[center+M:] = y[center:]
        aup = ifft(Aup) * Nup

        #Nonlinear stuff
        xup = aup*(np.cos(phi) + 1j*np.sin(phi))
        f1up = aup*(xup + 2*np.conj(xup))

        #Downsample
        F1 = np.zeros_like(y)
        F1up = fft(f1up)
        F1[:center] = F1up[:center]
        F1[center:] = F1up[center+M:]
        F1 = F1 / Nup

        return -1j * k(z) * F1 * np.exp(1j*D*z)
    
    #Integrate segment by segment, restarting at every discontinuity of k(z)
    #so that no step straddles a poling domain wall (#50)
    edges = [z0]
    if z_breaks is not None:
        edges += sorted(zb for zb in np.unique(z_breaks) if z0 < zb < L)
    edges.append(L)

    y = A
    steps = []
    status = 'finished'
    for za, zb in zip(edges[:-1], edges[1:]):
        Integrator = RK45( fnl, za, y, zb, rtol=rtol, atol=atol, max_step=max_step )
        while Integrator.status == "running":
            Integrator.step()
            steps.append( Integrator.step_size )
        if Integrator.status == 'failed':
            status = 'failed'
            warnings.warn(f'NEE: the integrator failed at z = {Integrator.t:.6g} m '
                          f'({Integrator.message}); the output is the field there',
                          RuntimeWarning, stacklevel=2)
            break
        y = Integrator.y
    steps = np.array(steps)

    if verbose:    
        print( status )
    
    A = y * np.exp(-1j*D*Integrator.t)
    a = ifft(A)
    
    return a, steps


def random_source(rng=None):
    '''
    Object with `normal` and `uniform` methods: numpy's global generator for
    rng=None (reproducible with np.random.seed, as before), otherwise
    np.random.default_rng(rng) (a Generator or a seed).
    '''
    if rng is None:
        return np.random
    return np.random.default_rng(rng)


def linear_operator(pulse, beta, alpha, v_ref):
    '''
    Linear operator in the moving frame,
    D = beta - beta_ref - Omega/v_ref - i alpha/2 (1/m), with beta_ref = beta
    at the reference frequency (first FFT bin). Returns (D, beta_ref).
    '''
    beta_ref = beta[0]
    D = beta - beta_ref - pulse.Omega/v_ref - 1j*alpha/2
    return D, beta_ref


def propagate(pulse, L, beta, v_ref, k, alpha=0, z0=0, Kg=0, Qnoise=False,
              rtol=1e-4, atol=1e-4, max_step=np.inf, z_breaks=None,
              verbose=True, rng=None):
    '''
    Propagate a pulse through any chi(2) medium with `NEE`.

    This is the common core of `waveguide.propagate_NEE` and
    `nonlinear_crystal.propagate_NEE_fd`; a medium only has to supply its
    propagation constant on the pulse grid and its coupling.

    Parameters
    ----------
    pulse : pulses.pulse
    L : float
        End position (m).
    beta : array
        Propagation constant at the pulse's absolute frequencies (1/m),
        FFT order.
    v_ref : float
        Velocity of the moving frame (m/s).
    k : callable
        k(z): nonlinear coupling at position z (scalar or array over
        frequency).
    alpha : float
        Power attenuation coefficient (1/m).
    The other arguments are passed to `NEE`.

    Returns
    -------
    output_pulse : pulses.pulse
    steps : array
        Integrator step sizes (m).
    '''
    tic = time.time()
    D, beta_ref = linear_operator(pulse, beta, alpha, v_ref)
    a, steps = NEE(t=pulse.t, x=pulse.a, Omega=pulse.Omega, f0=pulse.f0, L=L,
                   D=D, b0=beta_ref, b1_ref=1/v_ref, k=k, z0=z0, verbose=verbose,
                   Kg=Kg, Qnoise=Qnoise, rtol=rtol, atol=atol, max_step=max_step,
                   z_breaks=z_breaks, rng=rng)
    if verbose:
        print('Total time = %0.1f s' %(time.time() - tic))
    return pulses.pulse(pulse.t, a, pulse.wl0, pulse.frep), steps