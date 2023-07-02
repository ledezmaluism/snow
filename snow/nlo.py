# -*- coding: utf-8 -*-
"""
@author: Luis Ledezma
"""
import numpy as np
from numpy.fft import fftshift, fft, ifft

import time
from scipy.constants import pi, c

from . import pulses

from scipy.integrate import RK45

class nonlinear_crystal():
    
    def __init__(self, L, n_func, chi2, alpha=0):
        self.L = L
        self.n_func = n_func
        self.chi2 = chi2
        self.alpha = alpha      
  
    def propagate_NEE_fd(self, pulse, h, v_ref=None,
                         verbose=True, zcheck_step = 0.5e-3, z0=0):
        
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

        [a, a_evol] = NEE(t = pulse.t, 
                          x = pulse.a,
                          Omega = Omega,
                          f0 = pulse.f0,
                          L = self.L,
                          D = D, 
                          b0 = beta_ref, 
                          b1_ref = beta_1_ref, 
                          k = k, 
                          h = h, 
                          zcheck_step = zcheck_step, 
                          z0 = 0,
                          verbose = verbose)
        
        tdelta = time.time() - tic_total
        print('Total time = %0.1f s' %(tdelta))
        
        output_pulse = pulses.pulse(pulse.t, a, pulse.wl0)
        return output_pulse, a_evol

def NEE(t, x, Omega, f0,
        L, D, b0, b1_ref, k, 
        zcheck_step, z0=0, verbose=True, Kg=0):
    """
    Adaptive solver

    """
    #Get stuff
    NFFT = t.size
    Omega_abs = Omega + 2*pi*f0
    f_max = np.amax(Omega_abs) / (2*pi)
    f_min = np.amin(Omega_abs) / (2*pi)
    BW = f_max - f_min
    
    #Calculate upsampling parameter, it usually will be Nup=4,
    #so, throw a warning if it needs to be 8
    Nup = 4
    if (3*f_max - f_min)/BW > Nup:
        Nup = 8
        print('Warning: large upsampling necessary!')
    print('Using %ix upsampling.' %(Nup))

    #Input signal to frequency domain
    A = fft(x)
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
        
    print( Integrator.status )
    print( )
    
    A[:] = Integrator.y * np.exp(-1j*D*Integrator.t)
    a = ifft(A)
    
    return a, steps

def test1():
    pass

if __name__ == '__main__':
    test1()
    