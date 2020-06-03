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

class nonlinear_element():
    
    def __init__(self, L, n_func, chi2, alpha=0):
        self.L = L
        self.n_func = n_func
        self.chi2 = chi2
        self.alpha = alpha
        
    def prepare(self, pulse, v_ref=None):
        
        #Get the frequency info from the pulse
        wl = pulse.wl
        f_abs = pulse.f_abs
        df = f_abs[1] - f_abs[0]
        Omega  = pulse.Omega
        
        #Get the refractive index, beta, etc.
        n = self.n_func(wl)
        beta = 2*pi*f_abs*n/c
        beta_1 = fftshift(np.gradient(fftshift(beta), 2*pi*df))
        beta_2 = fftshift(np.gradient(fftshift(beta_1), 2*pi*df))
        vg = 1/beta_1
        if v_ref == None:
            v_ref = vg[0]
        
        f_ref = pulse.f0
        beta_ref = beta[0]
        self.beta_1_ref = 1/v_ref
        self.D = beta - beta_ref - Omega/v_ref - 1j*self.alpha/2
        
        self.Omega = Omega
        self.n = n
        self.beta = beta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.f_ref = f_ref
        self.beta_ref = beta_ref
    
    def propagate_NEE_fd(self, pulse, h, v_ref=None, method='bulk', 
                         verbose=True, Nup=1):
        
        #Timer
        tic_total = time.time()
         
        #Get stuff
        t = pulse.t
        NFFT = t.size
        
        self.prepare(pulse, v_ref)
        L = self.L
        D = self.D
        chi2 = self.chi2
        n = self.n
        omega_ref = 2*pi*self.f_ref
        omega_abs = omega_ref + self.Omega
        
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

        #Frequency domain input
        a[:] = pulse.a
        A = fft_a()

        #Pre-compute some stuff:
        Dh = np.exp(-1j*D*h) #Dispersion operator for step size h
        tup = np.linspace(t[0], t[-1], Nup*NFFT) #upsampled time
        phi_1 = omega_ref*tup
        phi_2 = self.beta_ref - self.beta_1_ref*omega_ref
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
            
        #Let's inform the user after every 0.5mm (hard-coded)
        zcheck_step = 0.5e-3
        zcheck = zcheck_step
        tic = time.time()
        
        #Initialize the array that will store the full pulse evolution
        a_evol = 1j*np.zeros([t.size, Nsteps+1])
        a_evol[:, 0] = a #Initial value
             
        def chi_bulk(z):
            return chi2(z)*omega_abs/(4*n*c) 
        
        def chi_wg(z):
            return chi2(z)
        
        if method=='bulk':
            chi = chi_bulk
            print("Using method = bulk")
        elif method=='waveguide':
            chi = chi_wg
            print("Using method = waveguide")
        else:
            print("Didn't understand method chosen. Using default bulk")
            chi = chi_bulk

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
    
            return -1j*chi(z)*F1 
        
        #Here we go, initialize z tracker and calculate first half dispersion step
        z = 0
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
            a_evol[:, kz+1] = a
            
            #Let's inform the user now
            if verbose and round(z*1e3,3)==round(zcheck*1e3,3):
                tdelta = time.time() - tic
                print('Completed propagation along %0.1f mm (%0.1f s)' %(z*1e3, tdelta))
                tic = time.time()
                zcheck += zcheck_step
    
        A[:] = A * np.exp(1j*D*h/2) #Final half dispersion step back
        a = ifft_A()
        
        tdelta = time.time() - tic_total
        print('Total time = %0.1f s' %(tdelta))
        return a, a_evol     
    
def test1():
    pass

if __name__ == '__main__':
    test1()
    