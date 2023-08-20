# -*- coding: utf-8 -*-
"""
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
    Nonlinear-envelope equation 
    Adaptive solver
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
    
    tup = np.linspace(t[0], t[-1], Nup*NFFT) #upsampled time
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