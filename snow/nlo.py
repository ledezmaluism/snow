# -*- coding: utf-8 -*-
"""
Nonlinear envelope equation (NEE) solver for chi(2) propagation.

Most users call this through `waveguide.propagate_NEE` or
`nonlinear_crystal.propagate_NEE_fd`, which build the dispersion operator and
the coupling. See docs/physics.md for the model and conventions.

@author: Luis Ledezma
"""
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.constants import pi, h

from scipy.integrate import RK45

def NEE(t, x, Omega, f0,
        L, D, b0, b1_ref, k, 
        z0=0, verbose=True, Kg=0, Qnoise=False):
    """
    Integrate the chi(2) nonlinear envelope equation from z0 to L.

    The spectrum A(Omega, z) of the envelope obeys

        dA/dz = -i D A - i k(z) F{ a^2 e^{i phi} + 2 |a|^2 e^{-i phi} },
        phi(t, z) = omega0 t - (b0 - omega0 b1_ref + Kg) z,

    where a = F^-1{A}, omega0 = 2 pi f0 and F is the FFT. The first nonlinear
    term generates sum frequencies (SHG/SFG), the second difference
    frequencies. The equation is solved in the interaction picture
    (y = A e^{iDz}) with scipy's adaptive RK45 (rtol = atol = 1e-4). The
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
        print('Warning: large upsampling necessary!')
        print('Using %ix upsampling.' %(Nup))

    #Quantum noise
    if Qnoise:
        ϕ = np.random.uniform( 0, 2*pi, NFFT )
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
    
    rtol = 1e-4
    atol = 1e-4
    
    Integrator = RK45( fnl, z0, A, L, rtol=rtol, atol=atol )
    
    steps = np.array([])
    while Integrator.status == "running":
        Integrator.step()
        steps = np.append( steps, Integrator.step_size )

    if verbose:    
        print( Integrator.status )
    
    A[:] = Integrator.y * np.exp(-1j*D*Integrator.t)
    a = ifft(A)
    
    return a, steps


if __name__ == '__main__':
    pass