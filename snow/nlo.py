# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.fft import fftshift
import time
from scipy.constants import pi, c
import pyfftw

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
    
    #Initialize the FFTW arrays
    a = pyfftw.empty_aligned(NFFT, dtype='complex128')
    A = pyfftw.empty_aligned(NFFT, dtype='complex128')
    aup = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    Aup = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    f1up = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    F1up = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    
    fft_a = pyfftw.FFTW(a, A)
    ifft_A = pyfftw.FFTW(A, a, direction='FFTW_BACKWARD')
    fft_f1up = pyfftw.FFTW(f1up, F1up)
    ifft_Aup = pyfftw.FFTW(Aup, aup, direction='FFTW_BACKWARD')

    #Input signal to frequency domain
    a[:] = x
    A = fft_a()

    #Pre-compute some stuff:
    Dh = np.exp(-1j*D*h) #Dispersion operator for step size h
    tup = np.linspace(t[0], t[-1], Nup*NFFT) #upsampled time
    phi_1 = 2*pi*f0*tup
    phi_2 = b0 - b1_ref*2*pi*f0
    F1 = np.zeros_like(A)
    
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
        aup[:] = ifft_Aup() * Nup
        
        #Nonlinear stuff
        xup = aup*(np.cos(phi) + 1j*np.sin(phi))
        f1up[:] = aup*(xup + 2*np.conj(xup))

        #Downsample
        F1up = fft_f1up()
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
        a = ifft_A()
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
    a = ifft_A()
    
    return a, a_evol 

def NEE2(t, x, Omega, f0,
        L, D, b0, b1_ref, k, 
        h, zcheck_step, z0=0, verbose=True):

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
    
    #Initialize the FFTW arrays
    a = pyfftw.empty_aligned(NFFT, dtype='complex128')
    A = pyfftw.empty_aligned(NFFT, dtype='complex128')
    aup = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    Aup = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    f1up = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    F1up = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    
    fft_a = pyfftw.FFTW(a, A)
    ifft_A = pyfftw.FFTW(A, a, direction='FFTW_BACKWARD')
    fft_f1up = pyfftw.FFTW(f1up, F1up)
    ifft_Aup = pyfftw.FFTW(Aup, aup, direction='FFTW_BACKWARD')

    #Input signal to frequency domain
    a[:] = x
    A = fft_a()
    
    tup = np.linspace(t[0], t[-1], Nup*NFFT) #upsampled time
    phi_1 = 2*pi*f0*tup
    phi_2 = b0 - b1_ref*2*pi*f0
    F1 = np.zeros_like(A)
    
    #Upsampling stuff
    M = NFFT*Nup - A.size
    Xc = np.zeros(M)
    center = A.size // 2 + 1

    z_prev = 0
    #Nonlinear function
    def fnl(z, A):
        phi = phi_1 - phi_2*z
        
        h = z - z_prev
        # print([z_prev*1e6, h*1e6])
        #get fast envelope
        A[:] = A * np.exp(-1j*D*h )
        
        #Upsample
        Aup[:center] = A[:center]
        Aup[center:center+M] = Xc
        Aup[center+M:] = A[center:]
        aup[:] = ifft_Aup() * Nup
        
        #Nonlinear stuff
        xup = aup*(np.cos(phi) + 1j*np.sin(phi))
        f1up[:] = aup*(xup + 2*np.conj(xup))

        #Downsample
        F1up = fft_f1up()
        F1[:center] = F1up[:center]
        F1[center:] = F1up[center+M:]
        F1[:] = F1 / Nup

        return -1j * k(z) * F1 * np.exp(1j*D*h)
    
    #Here we go, initialize z tracker and calculate first half dispersion step
    rtol = 1e-4
    atol = 1e-9
    
    Integrator = RK45( fnl, z0, A, L, rtol=rtol, atol=atol, first_step=h)
    # Integrator = BDF( fnl, z0, A, L, rtol=rtol, atol=atol, first_step=h)
    
    steps = np.array([])
    while Integrator.status == "running":
    
        z_prev = Integrator.t
        Integrator.step()
        # h = Integrator.step_size
        
        
        # z = np.append( z, Integrator.t )
        # A = np.append( A, Integrator.y )
        
        steps = np.append( steps, Integrator.step_size )
        
    print()
    print( Integrator.status )
    print()
    
    A[:] = Integrator.y
    a = ifft_A()
    
    return a, steps

def NEE3(t, x, Omega, f0,
        L, D, b0, b1_ref, k, 
        h, zcheck_step, z0=0, verbose=True, Kg=0):

    # h = float(h) #in case the input is a single element array
    
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
    
    #Initialize the FFTW arrays
    a = pyfftw.empty_aligned(NFFT, dtype='complex128')
    A = pyfftw.empty_aligned(NFFT, dtype='complex128')
    aup = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    Aup = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    f1up = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    F1up = pyfftw.empty_aligned(Nup*NFFT, dtype='complex128')
    
    fft_a = pyfftw.FFTW(a, A)
    ifft_A = pyfftw.FFTW(A, a, direction='FFTW_BACKWARD')
    fft_f1up = pyfftw.FFTW(f1up, F1up)
    ifft_Aup = pyfftw.FFTW(Aup, aup, direction='FFTW_BACKWARD')

    #Input signal to frequency domain
    a[:] = x
    A = fft_a()
    
    tup = np.linspace(t[0], t[-1], Nup*NFFT) #upsampled time
    phi_1 = 2*pi*f0*tup
    phi_2 = b0 - b1_ref*2*pi*f0 + Kg
    F1 = np.zeros_like(A)
    
    #Upsampling stuff
    M = NFFT*Nup - A.size
    Xc = np.zeros(M)
    center = A.size // 2 + 1

    z_prev = 0
    #Nonlinear function
    def fnl(z, A):
        phi = phi_1 - phi_2*z
        
        h = z - z_prev
        # print([z_prev*1e6, h*1e6])
        #get fast envelope
        A[:] = A * np.exp(-1j*D*h )
        
        #Upsample
        Aup[:center] = A[:center]
        Aup[center:center+M] = Xc
        Aup[center+M:] = A[center:]
        aup[:] = ifft_Aup() * Nup
        
        #Nonlinear stuff
        xup = aup*(np.cos(phi) + 1j*np.sin(phi))
        f1up[:] = aup*(xup + 2*np.conj(xup))

        #Downsample
        F1up = fft_f1up()
        F1[:center] = F1up[:center]
        F1[center:] = F1up[center+M:]
        F1[:] = F1 / Nup

        return -1j * k(z) * F1 * np.exp(1j*D*h)
    
    #Here we go, initialize z tracker and calculate first half dispersion step
    rtol = 1e-5
    atol = 1e-9
    
    Integrator = RK45( fnl, z0, A, L, rtol=rtol, atol=atol)
    # Integrator = BDF( fnl, z0, A, L, rtol=rtol, atol=atol, first_step=h)
    
    steps = np.array([])
    while Integrator.status == "running":
    
        z_prev = Integrator.t
        Integrator.step()
        # h = Integrator.step_size
        
        
        # z = np.append( z, Integrator.t )
        # A = np.append( A, Integrator.y )
        
        steps = np.append( steps, Integrator.step_size )
        
    print()
    print( Integrator.status )
    print()
    
    A[:] = Integrator.y
    a = ifft_A()
    
    return a, steps

def test1():
    pass

if __name__ == '__main__':
    test1()
    