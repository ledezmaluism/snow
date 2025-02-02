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
        h, zcheck_step, z0=0, verbose=True):
    """
    Baseline method. Not adaptive. The step size h is used.
    The entire poling function is used (h << poling period)
    """

    h = float(h) #in case the input is a single element array
    
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
    
    #Initialize empty arrays
    Aup = np.zeros( Nup*NFFT, dtype='complex')
    F1 = np.zeros( NFFT, dtype='complex')

    #Input signal to frequency domain
    A = fft(x)

    #Pre-compute some stuff:
    Dh = np.exp(-1j*D*h) #Dispersion operator for step size h
    tup = np.linspace(t[0], t[-1], Nup*NFFT) #upsampled time
    phi_1 = 2*pi*f0*tup
    phi_2 = b0 - b1_ref*2*pi*f0
    
    #Upsampling stuff
    M = NFFT*Nup - A.size
    Xc = np.zeros(M)
    center = A.size // 2 + 1
    
    #Calculate number of steps needed
    Nsteps = int(L/h)
    
    #Print out some info
    print('Crystal length = %0.2f mm' %(L*1e3))
    print('Step size = %0.2f um' %(h*1e6))
    print('Number of steps = %i' %(Nsteps))
        
    #Let's inform the user after every zcheck_step
    zcheck = zcheck_step
    tic = time.time()
    
    #Initialize the array that will store the full pulse evolution
    a_evol = 1j*np.zeros([t.size, Nsteps])

    #Nonlinear function
    def fnl(z, A):
        phi = phi_1 - phi_2*z
        
        #Upsample
        Aup[:center] = A[:center]
        Aup[center:center+M] = Xc
        Aup[center+M:] = A[center:]
        aup = ifft(Aup) * Nup
        
        #Nonlinear stuff
        xup = aup*(np.cos(phi) + 1j*np.sin(phi))
        f1up = aup*(xup + 2*np.conj(xup))

        #Downsample
        F1up = fft(f1up)
        F1[:center] = F1up[:center]
        F1[center:] = F1up[center+M:]
        F1[:] = F1 / Nup

        return -1j * k(z) * F1 
    
    #Here we go, initialize z tracker and calculate first half dispersion step
    z = z0 + h/2
    A[:] = A * np.exp(-1j*D*h/2) #Half step
    for kz in range(Nsteps):     

        #Nonlinear step
        #Runge-Kutta 4th order
        k1 = fnl(z    , A       )
        k2 = fnl(z+h/2, A+h*k1/2)
        k3 = fnl(z+h/2, A+h*k2/2)
        k4 = fnl(z+h  , A+h*k3  )
        A[:] = A + (h/6)*(k1+2*k2+2*k3+k4) 
        z = z + h
        
        #Linear full step (two half-steps back to back)
        A[:] = Dh*A
        
        #Save evolution
        a = ifft(A)
        a_evol[:, kz] = a
        
        #Check for energy near the edges of time window
        r_begin = np.amax( np.abs( a[0:10] ) ) / np.amax( np.abs(a) )
        r_end = np.amax( np.abs( a[-10:] ) ) / np.amax( np.abs(a) )
        if r_begin>1e-2 or r_end>1e-2:
            print('Warning: signal seems to have reach time domain borders!')
            print('Aborting!!!')
            break
        
        #Let's inform the user now
        if verbose and round(z*1e3,3)==round(zcheck*1e3,3):
            tdelta = time.time() - tic
            print('Completed propagation along %0.2f mm (%0.1f s)' %(z*1e3, tdelta))
            tic = time.time()
            zcheck += zcheck_step

    A[:] = A * np.exp(1j*D*h/2) #Final half dispersion step back
    a = ifft(A)
    
    return a, a_evol 

def NEE2(t, x, Omega, f0,
        L, D, b0, b1_ref, k, 
        zcheck_step, z0=0, verbose=True, Kg=0):
    """
    Adaptive solver; h is not used.

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
    