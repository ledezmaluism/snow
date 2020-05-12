# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import numpy as np
from numpy.fft import fft, ifft, fftshift
import time
import scipy.signal
from scipy.constants import pi, c

def complex_to_cuda(X):
    Xr = np.real(X)
    Xi = np.imag(X)
    X2 = np.array([Xr, Xi]).transpose()
    Y = torch.from_numpy(X2).cuda()
    return Y

def complex_multiply(x, y):
    re = x**2 + y**2
    im = 2*x*y
    return torch.stack([re, im])

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
        
    def propagate_NEE(self, pulse, h, v_ref=None, verbose=True):
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()
        
        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
            
        tic_total = time.time()
        
        #Get the pulse info:
        t = torch.from_numpy(pulse.t).cuda()
        A = complex_to_cuda(pulse.a)
        omega_ref = torch.tensor(2*pi*pulse.f0).cuda()
        NFFT = t.size
        h = torch.tensor(h).cuda()
        
        self.prepare(pulse, v_ref)
        
        #Unwrap the attributes for speed
        D = complex_to_cuda(self.D)
        L = self.L   
        chi2 = self.chi2
        omega_ref = torch.tensor(2*pi*self.f_ref).cuda()
        
        #Pre-compute some stuff:
        phi_1 = omega_ref*t
        phi_2 = torch.tensor(self.beta_ref - self.beta_1_ref*omega_ref).cuda()
        
        #Calculate number of steps needed
        Nsteps = int(L/h) + 1
        
        #Print out some info
        print('Crystal length = %0.2f mm' %(L*1e3))
        print('Step size = %0.2f um' %(h*1e6))
        print('Number of steps = %i' %(Nsteps))
            
        #Let's inform the user after every 0.5mm (hard-coded)
        zcheck_step = 0.5e-3
        zcheck = zcheck_step
        tic = time.time()
        
        #Initialize the array that will store the full pulse evolution
        A_evol = torch.zeros((NFFT, 2, Nsteps+1)).cuda()
        A_evol[:, :, 0] = A #Initial value
    

        def fnl(z, A): #Waveguides...
            phi = phi_1 - phi_2*z
            
            Ar = A[:,0]
            Ai = A[:,1]
            A2r = Ar**2 - Ai**2
            A2i = 2*Ar*Ai
            er = torch.cos(phi)
            ei = torch.sin(phi)
            
            f1r = 
            f1 = A*A*np.exp(1j*phi) + 2*A*np.conj(A)*np.exp(-1j*phi)

            f = -1j*ifft(chi2(z)*fft(f1))
            return f        

        
        #Dispersion operator for step size h
        Dh = torch.from_numpy(np.exp(-1j*D*h)).cuda()
        
        
        #Here we go, initialize z tracker and calculate first half dispersion step
        z = 0
        A = ifft(np.exp(-1j*D*h/2)*fft(A)) #Half step
        for kz in range(Nsteps):     
    
            #Nonlinear step
            #Runge-Kutta 4th order
            k1 = fnl(z    , A         )
            k2 = fnl(z+h/2, A + h*k1/2)
            k3 = fnl(z+h/2, A + h*k2/2)
            k4 = fnl(z+h  , A + h*k3  )
            A = A + (h/6)*(k1 + 2*k2 + 2*k3 + k4) 
            z = z + h
            
            #Linear full step (two half-steps back to back)
            A = torch.ifft(Dh*fft(A))
            
            #Save evolution
            A_evol[:, kz+1] = A
            
            #Let's inform the user now
            if verbose and round(z*1e3,3)==round(zcheck*1e3,3):
                tdelta = time.time() - tic
                print('Completed propagation along %0.1f mm (%0.1f s)' %(z*1e3, tdelta))
                tic = time.time()
                zcheck += zcheck_step
    
        A = ifft(np.exp(1j*D*h/2)*fft(A)) #Final half dispersion step back
        
        
        tdelta = time.time() - tic_total
        print('Total time = %0.1f s' %(tdelta))
        return A, A_evol  

def test1():
    pass

if __name__ == '__main__':
    test1()
    