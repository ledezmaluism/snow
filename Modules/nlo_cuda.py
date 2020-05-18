# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import time
from scipy.constants import pi, c, epsilon_0

#Complex unit
J = torch.stack([torch.tensor([0]),torch.tensor([1])])
J = torch.transpose(J, 0, 1)
J = J.cuda()

#Constants for Pytorch
PI = torch.tensor([pi]).cuda()
C = torch.tensor([c]).cuda()

def make_complex(X):
    if X.dim()==1:
        re = X
        im = torch.zeros_like(X)
        return torch.transpose(torch.stack([re, im]), 0, 1)
    elif X.dim()==2:
        return X
    else:
        print('Error in make_complex')

def complex_to_cuda(X):
    Xr = np.real(X)
    Xi = np.imag(X)
    X2 = np.array([Xr, Xi]).transpose()
    Y = torch.from_numpy(X2).cuda()
    return Y

def complex_multiply(Z1, Z2):
    x1 = Z1[:,0]
    y1 = Z1[:,1]
    x2 = Z2[:,0]
    y2 = Z2[:,1]
    re = x1*x2 - y1*y2
    im = x1*y2 + x2*y1
    return torch.transpose(torch.stack([re, im]), 0, 1)

def conjugate(Z):
    re = Z[:,0]
    im = -Z[:,1]
    return torch.transpose(torch.stack([re, im]), 0, 1)

def complex_expJ(phi):
    re = torch.cos(phi)
    im = torch.sin(phi)
    return torch.transpose(torch.stack([re, im]), 0, 1)

def complex_to_numpy(Z):
    x = Z[:,0].cpu().numpy()
    y = Z[:,1].cpu().numpy()
    z = x + 1j*y
    return z

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
                         verbose=True):
        
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
            print()
            
        tic_total = time.time()
        
        #Get the pulse info:
        t = torch.from_numpy(pulse.t).cuda()
        A = complex_to_cuda(pulse.a)
        omega_ref = torch.tensor(2*pi*pulse.f0).cuda()
        # h = torch.tensor(h).cuda()
        NFFT = pulse.t.size

        self.prepare(pulse, v_ref)
        
        #Unwrap the attributes for speed
        D = self.D
        L = self.L   
        chi2 = self.chi2
        omega_ref = torch.tensor([2*pi*self.f_ref]).cuda()
        Omega = torch.tensor([self.Omega]).cuda()
        n = torch.tensor(self.n).cuda()
        
        #Pre-compute some stuff:
        phi_1 = omega_ref*t
        phi_2 = self.beta_ref - self.beta_1_ref*omega_ref
        
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
        
             
        def chi_bulk(z):
            chi = make_complex( chi2(z)*(Omega+omega_ref)/(4*n*C) )
            return chi
        
        def chi_wg(z):
            return chi2(z)

        def fnl(z, A):
            phi = phi_1 - phi_2*z
            Ephi = complex_expJ(phi)
            
            A = torch.ifft(A, signal_ndim=1)
            A2 = complex_multiply(A, A)
            Amag = complex_multiply(A, conjugate(A))
            f1 = complex_multiply(A2, Ephi) + 2*complex_multiply(Amag, conjugate(Ephi))
            
            F1 = torch.fft(f1, signal_ndim=1)
            CHI = chi(z)
            F1 = complex_multiply(F1, CHI)

            f = -1*complex_multiply(J, F1)    
            return f

        if method=='bulk':
            chi = chi_bulk
        elif method=='waveguide':
            chi = chi_wg
        else:
            print("Didn't understand method chosen. Using default bulk")
            chi = chi_bulk
        
        #Change to frequency domain
        A = torch.fft(A, signal_ndim=1)
        
        #Dispersion operator for step size h
        Dh = complex_to_cuda(np.exp(-1j*D*h))
        Dh2 = complex_to_cuda(np.exp(-1j*D*h/2))
        Dh3 = complex_to_cuda(np.exp(1j*D*h/2))
        
        #Here we go, initialize z tracker and calculate first half dispersion step
        h = torch.tensor([h]).cuda()
        z = torch.tensor([0]).cuda()
        A = complex_multiply(Dh2, A) #Half step
        for kz in torch.arange(0, Nsteps).cuda():     
    
            #Nonlinear step
            #Runge-Kutta 4th order
            k1 = fnl(z    , A       )
            k2 = fnl(z+h/2, A+h*k1/2)
            k3 = fnl(z+h/2, A+h*k2/2)
            k4 = fnl(z+h  , A+h*k3  )
            A = A + (h/6)*(k1+2*k2+2*k3+k4) 
            z = z + h
            
            #Linear full step (two half-steps back to back)
            A = complex_multiply(Dh, A)
            
            #Save evolution
            A_evol[:, :, kz+1] = torch.ifft(A, signal_ndim=1)
            
            #Let's inform the user now
            # if verbose and round(z*1e3,3)==round(zcheck*1e3,3):
            # if torch.fmod(kz, 100) = 
            #     tdelta = time.time() - tic
            #     print('Completed propagation along %0.1f mm (%0.1f s)' %(z*1e3, tdelta))
            #     tic = time.time()
            #     zcheck += zcheck_step
    
        A = complex_multiply(Dh3, A) #Final half dispersion step back
        A = torch.ifft(A, signal_ndim=1)
        
        tdelta = time.time() - tic_total
        print('Total time = %0.1f s' %(tdelta))
        return complex_to_numpy(A), A_evol  
        
def test1():
    import materials
    import nlo
    nm = 1e-9
    um = 1e-6
    mm = 1e-3
    ps = 1e-12
    THz = 1e12
    
    wl_ff = 1400*nm #Fundamental wavelength
    f0_ff = c/wl_ff
    f0_sh = 2*f0_ff #SHG frequency
    
    #Let's push this to the limit
    f_min = -100*THz
    f_max = 1e3*THz
    BW = f_max - f_min

    #Now we can create the time and frequency arrays
    NFFT = 2**14
    dt = 1/BW
    t_start = -2.5*ps
    t_stop = t_start + NFFT*dt
    t = np.arange(t_start, t_stop, step=dt)
    f = fftfreq(NFFT, dt)
    Omega = 2*pi*f
    df = f[1]-f[0]

    #Reference frequency
    f_ref = (f_min + f_max)/2
    # f_ref = f0_sh
    wl_ref = c/f_ref

    #Absolute frequencies and wavelengths
    f_abs = f + f_ref
    wl_abs = c/f_abs
    omega_abs = 2*pi*f_abs
    f_min = np.amin(f_abs)
    f_max = np.amax(f_abs)
    
    #get the frequency indexes for fundamental and second harmonics
    f0_ff_index = np.abs(f + f_ref - f0_ff).argmin()
    f0_sh_index = np.abs(f + f_ref - f0_sh).argmin()

    
    wl_hfc = 300*nm
    wl_lfc = 5.5*um
    loss = 100*(np.exp(-2*wl_abs/wl_hfc) + np.exp(20*(wl_abs-wl_lfc)/wl_lfc)) #units: 1/m
    loss[loss>1e3] = 1e3

    def n_func(wl):
        wtrans1 = 350*nm
        wtrans2 = 5*um
        n_low = 1
        n_high = 3
        n_trans1 = materials.refractive_index('LT_MgO_e', wtrans1/um)
        n_trans2 = materials.refractive_index('LT_MgO_e', wtrans2/um)
        slope_trans1 = (materials.refractive_index('LT_MgO_e', wtrans1/um) - materials.refractive_index('LT_MgO_e', (wtrans1-1*nm)/um))/(1*nm)
        slope_trans2 = (materials.refractive_index('LT_MgO_e', wtrans2/um) - materials.refractive_index('LT_MgO_e', (wtrans2-1*nm)/um))/(1*nm)
        
        n = np.zeros_like(wl) 
        for k in range(wl.size):   
            if wtrans1 <= wl[k] <= wtrans2:
                n[k] = materials.refractive_index('LT_MgO_e', wl[k]/um)
            elif wl[k]<=wtrans1:
                a = n_trans1
                c = abs(slope_trans1) / (n_high - a)
                b = n_high*c
                x = wtrans1 - wl[k]
                n[k] = (a + b*x)/(1 + c*x)
            else:
                a = n_trans2
                c = slope_trans2 / (n_low - a)
                b = n_low*c
                x = wl[k] - wtrans2
                n[k] = (a + b*x)/(1 + c*x)
                
        return n
    
    nLN = n_func(wl_abs)
    nw = nLN[f0_ff_index]
    
    beta = omega_abs*nLN/c
    beta_1 = fftshift(np.gradient(fftshift(beta), 2*pi*df))
    vg = 1/beta_1
    v_ref = vg[f0_sh_index]

    poling_period = 17.4*um  
    d33 = 10.6e-12
    deff = torch.tensor([d33]).cuda()
    pp = torch.tensor([poling_period]).cuda()
    
    def chi2(z):
        poling = (2/PI)*torch.cos(z*2*PI/pp)
        return 2*deff*poling

    crystal_cuda = nonlinear_element(L=5*mm, n_func=n_func, chi2=chi2, alpha=loss)
    
    #Frequency offset from reference:
    Omega_0 = 2*pi*(f0_ff - f_ref)
    
    #Peak value:
    Intensity = 10e13 #10GW/cm^2
    n = nw
    Epeak = np.sqrt(2*Intensity/(n*c*epsilon_0))
    
    #Pulse width:
    tau = 60e-15 
    
    #Noise floor
    noise = 0.01*np.random.normal(size=NFFT)
    
    #Pulse creation
    x = Epeak*np.exp(-2*np.log(2)*(t/tau)**2)*np.exp(1j*Omega_0*t)
    x = x + noise
    pulse = nlo.pulse(t, x, wl_ref)

    h = 1*mm/500 #Step size
    [out_pulse, pulse_evol] = crystal_cuda.propagate_NEE_fd(pulse, h, v_ref=v_ref)

    ax1 = nlo.plot_mag(t, out_pulse)
    ax1 = pulse.plot_mag(ax=ax1)
    ax1.set_xlim(-2,0.5)
    ax1.set_ylim(0,2e8)
    ax1.set_ylabel('Pulse Amplitude (V/m)')
    ax1.set_xlabel('Time (ps)');

    
    
if __name__ == '__main__':
    test1()
    