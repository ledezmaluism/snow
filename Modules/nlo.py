# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
import time
import scipy.signal
from scipy.constants import pi, c
import pyfftw


def get_freq_domain(t, x):
    X = fft(x)
    dt = t[1]-t[0] #Sample spacing
    NFFT = t.size
    f = fftfreq(NFFT, dt)
    return f, X

def get_spectrum(t, x):
    dt = t[1]-t[0] #Sample spacing
    f, X = get_freq_domain(t, x)
    Xmag = abs(X)*dt
    return f, Xmag

def get_esd(t, x):
    f, Xmag = get_spectrum(t, x)
    Xesd = Xmag**2
    return f, Xesd

def get_esd_dB(t, x):
    f, Xesd = get_esd(t, x)
    Xesd = Xesd/np.amax(Xesd)
    Xesd_dB = 10*np.log10(Xesd)
    return f, Xesd_dB

def get_esd_dBJ(t, x):
    f, Xesd = get_esd(t, x)
    Xesd_dB = 10*np.log10(Xesd)
    return f, Xesd_dB

def FWHM(t, x, prominence=1):
    dt = t[1]-t[0] #Sample spacing
    x2 = abs(x)**2 #Intensity
    peaks = scipy.signal.find_peaks(x2, prominence=prominence)
    width = scipy.signal.peak_widths(x2, peaks[0])
    fwhm = width[0]*dt
    return fwhm

def plot_vs_time(t, x, ylabel='', t_unit='ps', ax=None,
                   xlim=None, ylim=None):
    '''
    Private function to plot stuff x vs time (fs)
    '''
    if t_unit=='ps':
        t = t*1e12
    elif t_unit=='fs':
        t = t*1e15
    elif t_unit=='ns':
        t = t*1e9
    else:
        print('What was that? Don\'t know that time unit...')
        
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(t, x)
    ax.grid(True)
    ax.set_xlabel('Time (' + t_unit + ')');
    ax.set_ylabel(ylabel);
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    return ax
        
def __plot_vs_wavelength(wl, x, ylabel='', wl_unit='um', 
                         ax=None, xlim=None, ylim=None):
    '''
    Private function to plot stuff x vs wavelength (microns)
    '''        
    if wl_unit=='um':
        wl = wl*1e6
    elif wl_unit=='nm':
        wl = wl*1e9
    else:
        print('What was that? Don\'t know that wavelength unit...')
        
    wl = fftshift(wl)
    x = fftshift(x)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(wl, x)
    ax.grid(True)
    ax.set_xlabel('Wavelength (' + wl_unit + ')');
    ax.set_ylabel(ylabel);
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        
    return ax
            
def __plot_vs_freq(f, x, ylabel='', f_unit='THz', 
                       ax=None, xlim=None, ylim=None):
    '''
    Private function to plot stuff x vs frequency (THz)
    '''
    if f_unit=='THz':
        f = f*1e-12
    elif f_unit=='GHz':
        f = f*1e-9
    elif f_unit=='PHz':
        f = f*1e-15
    else:
        print('What was that? Don\'t know that frequency unit...')
        
    f = fftshift(f)
    x = fftshift(x)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(f, x)
    ax.grid(True)
    ax.set_xlabel('Frequency (' + f_unit + ')');
    ax.set_ylabel(ylabel);
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        
    return ax
 

def plot_mag(t, x, label='Pulse Amplitude (W^1/2)', ax=None, 
                     xlim=None, ylim=None, t_unit='ps'):
    x_mag = abs(x)
    ax = plot_vs_time(t, x_mag, ylabel=label, ax=ax, xlim=xlim, ylim=ylim, t_unit=t_unit)
    
    return ax
        
def plot_magsq(t, x, label='Pulse Intensity (W)', ax=None, 
                       xlim=None, ylim=None, t_unit='ps'):
    x2 = abs(x)**2
    ax = plot_vs_time(t, x2, ylabel=label, ax=ax, xlim=xlim, ylim=ylim, t_unit=t_unit)
    
    return ax
   
def plot_mag_relative(t, x, label='Relative Pulse Amplitude', ax=None, 
                         xlim=None, ylim=None, t_unit='ps'):
    x_mag = abs(x)
    x_mag_rel = x_mag/np.amax(x_mag)
    ax = plot_vs_time(t, x_mag_rel, ylabel=label, ax=ax, xlim=xlim, ylim=ylim, t_unit=t_unit)
    
    return ax

def plot_magsq_relative(t, x, label='Relative Pulse Intensity', ax=None, 
                           xlim=None, ylim=None, t_unit='ps'):
    x2 = abs(x)**2
    x2_rel = x2/np.amax(x2)
    ax = plot_vs_time(t, x2_rel, ylabel=label, ax=ax, xlim=xlim, ylim=ylim, t_unit=t_unit)
    
    return ax
            
def plot_phase_vs_time(t, x, label='Pulse Phase (rads)', ax=None,
                       xlim=None, ylim=None, t_unit='ps'):
    x_phase = np.angle(x)
    ax = plot_vs_time(t, x_phase, ylabel=label, ax=ax, xlim=xlim, ylim=ylim, t_unit=t_unit)
    
    return ax

def plot_phase_vs_freq(t, x, label='Pulse Phase (rads)', ax=None,
                       xlim=None, ylim=None, f_unit='THz'):
    
    f, X = get_freq_domain(t, x)
    X_phase = np.angle(X)
    ax = plot_vs_time(f, X_phase, ylabel=label, ax=ax, 
                      xlim=xlim, ylim=ylim, f_unit=f_unit)
    
    return ax

def plot_spectrum(t, x, label='Spectrum Amplitude (W^1/2 / Hz)', ax=None,
                  xlim=None, ylim=None, f_unit='THz'):
    f, Xmag = get_spectrum(t, x)
    ax = __plot_vs_freq(f, Xmag, ylabel=label, ax=ax, 
                        xlim=xlim, ylim=ylim, f_unit=f_unit)
    return ax
        
def plot_ESD(t, x, label='Energy Spectral Density (W / Hz^2)', ax=None, 
             xlim=None, ylim=None, f_unit='THz'):
    f, Xesd = get_esd(t, x)
    ax = __plot_vs_freq(f, Xesd, ylabel=label, ax=ax, 
                        xlim=xlim, ylim=ylim, f_unit=f_unit)
    return ax
        
def plot_ESD_dB(t, x, label='Energy Spectral Density (dBJ / Hz^2)', ax=None,
                xlim=None, ylim=None, f_unit='THz'):
    f, Xesd = get_esd_dB(t, x)
    ax = __plot_vs_freq(f, Xesd, ylabel=label, ax=ax, 
                        xlim=xlim, ylim=ylim, f_unit=f_unit)
    return ax
        
def plot_spectrum_absfreq(t, x, f0, label='Spectrum Amplitude (W^1/2 / Hz)', 
                          ax=None, xlim=None, ylim=None, f_unit='THz'):
    f, Xmag = get_spectrum(t, x)
    f = f+f0
    ax = __plot_vs_freq(f, Xmag, ylabel=label, ax=ax, 
                        xlim=xlim, ylim=ylim, f_unit=f_unit)
    return ax
        
def plot_ESD_absfreq(t, x, f0, label='Energy Spectral Density (W / Hz^2)', 
                     ax=None, xlim=None, ylim=None, f_unit='THz'):
    f, Xesd = get_esd(t, x)
    f = f  + f0
    ax = __plot_vs_freq(f, Xesd, ylabel=label, ax=ax, 
                        xlim=xlim, ylim=ylim, f_unit=f_unit)
    return ax
        
def plot_ESD_dB_absfreq(t, x, f0, label='Energy Spectral Density (dBJ / Hz^2)', 
                        ax=None, xlim=None, ylim=None, f_unit='THz'):
    f, Xesd = get_esd_dB(t, x)
    f = f  + f0
    ax = __plot_vs_freq(f, Xesd, ylabel=label, ax=ax, 
                        xlim=xlim, ylim=ylim, f_unit=f_unit)
    return ax
        
def plot_spectrum_vs_wavelength(t, x, f0, label='Spectrum Amplitude (W^1/2 / Hz)', 
                                ax=None, xlim=None, ylim=None, wl_unit='um'):
    f, Xmag = get_spectrum(t, x)
    f = f  + f0
    wl = c/f*1e6 #Microns
    ax = __plot_vs_wavelength(wl, Xmag, ylabel=label, ax=ax, 
                              xlim=xlim, ylim=ylim, wl_unit=wl_unit)
    return ax
        
def plot_ESD_vs_wavelength(t, x, f0, label='Energy Spectral Density (W / Hz^2)', 
                           ax=None, xlim=None, ylim=None, wl_unit='um'):
    f, Xesd = get_esd(t, x)
    f = f  + f0
    wl = c/f*1e6
    ax = __plot_vs_wavelength(wl, Xesd, ylabel=label, ax=ax, 
                              xlim=xlim, ylim=ylim, wl_unit=wl_unit)
    return ax
        
def plot_ESD_dB_vs_wavelength(t, x, f0, label='Energy Spectral Density (dBJ / Hz^2)', 
                           ax=None, xlim=None, ylim=None, wl_unit='um'):
    f, Xesd = get_esd_dB(t, x)
    f = f  + f0
    wl = c/f
    ax = __plot_vs_wavelength(wl, Xesd, ylabel=label, ax=ax, 
                              xlim=xlim, ylim=ylim, wl_unit=wl_unit)
    return ax
        
def energy_td(t, x):
    dt = t[1] - t[0]
    pwr = abs(x)**2
    energy = np.sum(pwr)*dt #Joules
    return energy

def energy_fd(t, x):
    f, Xesd = get_esd(t, x)
    df = f[1]-f[0]
    energy = np.sum(Xesd)*df #Joules
    return energy

def pulse_center(t, x):
    dt = t[1] - t[0]
    I = abs(x)**2
    return np.sum(t*I*dt)/np.sum(I*dt)





class pulse:

    def __init__(self, t, x, wavelength, domain='Time'):

        if not 50e-9 <= wavelength <= 20e-6:
            print('Warning: Wavelength of %0.2f um '%(wavelength*1e6) + 
                  'for pulse looks outside usual range.')

        #Static atributes, they usually don't change
        self.t = t
        self.dt = t[1]-t[0] #Sample spacing
        self.NFFT = t.size
        
        #Relative frequencies
        self.f = fftfreq(self.NFFT, self.dt)
        self.Omega = 2*pi*self.f
        self.df = self.f[1]-self.f[0]
        
        #Reference frequency:
        self.wl0 = wavelength 
        self.f0 = c//wavelength 
        
        #Absolute frequencies
        self.f_abs = self.f + self.f0
        self.wl = c/self.f_abs

        #Dynamic atributes, they change as pulse propagates
        if domain=='Time':
            self.a = x
            self.A = fft(x, self.NFFT)
        elif domain=='Freq':
            self.A = x
            self.a = ifft(x, self.NFFT)
        else:
            print('Wrong domain option. Options are "Time" or "Freq"')

        if np.amin(self.f_abs)<=0:
            print('Warning: negative absolute frequencies found. ' + 
                  'Considering reducing the number of points, ' +  
                  'or increasing the total time')

    def update_td(self, a):
        if np.size(self.t) == np.size(a):
            self.a = a
            self.A = fft(a, self.NFFT)
            self.update_param()
        else:
            raise RuntimeError('Hmm, this pulse seems diffenrent. Cannot update.')

    def update_fd(self, A):
        if np.size(self.t) == np.size(A):
            self.A = A
            self.a = ifft(A, self.NFFT)
            self.update_param()
        else:
            raise RuntimeError('Hmm, this pulse seems diffenrent. Cannot update.') 

    def energy_td(self):
        t = self.t
        x = self.a
        return energy_td(t, x)

    def energy_fd(self):
        t = self.t
        x = self.a
        return energy_fd(t, x)
    
    def get_esd(self):
        return get_esd(self.t, self.a)
    
    def get_esd_dB(self):
        return get_esd_dB(self.t, self.a)

    def time_center(self):
        return pulse_center(self.t, self.a)

    def width_FWHM(self, prominence=None):
        if prominence is None:
            prominence = 0.25*np.amax(self.a)
        return FWHM(self.t, self.a, prominence=prominence)

    def plot_mag(self, label='Pulse Amplitude (W^1/2)', ax=None, 
                         xlim=None, ylim=None, t_unit='ps'):
        t = self.t
        x = self.a
        ax = plot_mag(t, x, label=label, ax=ax, xlim=xlim, ylim=ylim, t_unit=t_unit)
        return ax

    def plot_magsq(self, label='Pulse Intensity (W)', ax=None, 
                           xlim=None, ylim=None, t_unit='ps'):
        t = self.t
        x = self.a
        ax = plot_magsq(t, x, label, ax, xlim, ylim)
        return ax

    def plot_mag_relative(self, label='Pulse Amplitude (W^1/2)', ax=None, 
                             xlim=None, ylim=None, t_unit='ps'):
        t = self.t
        x = self.a
        ax = plot_mag_relative(t, x, label, ax, xlim, ylim)
        return ax

    def plot_magsq_relative(self, label='Pulse Intensity (W)', ax=None, 
                               xlim=None, ylim=None, t_unit='ps'):
        t = self.t
        x = self.a
        ax = plot_magsq_relative(t, x, label, ax, xlim, ylim)
        return ax          

    def plot_phase_vs_time(self, label='Pulse Phase (rads)', ax=None,
                           xlim=None, ylim=None, t_unit='ps'):
        t = self.t
        x = self.a
        ax = plot_phase_vs_time(t, x, label, ax, xlim, ylim)
        return ax

    def plot_phase_vs_freq(self, label='Pulse Phase (rads)', ax=None,
                           xlim=None, ylim=None, f_unit='THz'):
        t = self.t
        x = self.a
        ax = plot_phase_vs_freq(t, x, label, ax, xlim, ylim, f_unit=f_unit)
        return ax

    def plot_spectrum(self, label='Spectrum Amplitude (W^1/2 / Hz)', ax=None,
                      xlim=None, ylim=None, f_unit='THz'):
        t = self.t
        x = self.a
        ax = plot_spectrum(t, x, label, ax, xlim, ylim, f_unit=f_unit)
        return ax

    def plot_ESD(self, label='Energy Spectral Density (W / Hz^2)', ax=None, 
                 xlim=None, ylim=None, f_unit='THz'):
        t = self.t
        x = self.a
        ax = plot_ESD(t, x, label, ax, xlim, ylim, f_unit=f_unit)
        return ax

    def plot_ESD_dB(self, label='Energy Spectral Density (dB / Hz^2)', ax=None,
                    xlim=None, ylim=None, f_unit='THz'):
        t = self.t
        x = self.a
        ax = plot_ESD_dB(t, x, label, ax, xlim, ylim, f_unit=f_unit)
        return ax

    def plot_spectrum_absfreq(self, label='Spectrum Amplitude (W^1/2 / Hz)', 
                              ax=None, xlim=None, ylim=None, f_unit='THz'):
        t = self.t
        x = self.a
        f0 = self.f0
        ax = plot_spectrum_absfreq(t, x, f0, label, ax, xlim, ylim, f_unit=f_unit)
        return ax

    def plot_ESD_absfreq(self, label='Energy Spectral Density (W / Hz^2)', 
                         ax=None, xlim=None, ylim=None, f_unit='THz'):
        t = self.t
        x = self.a
        f0 = self.f0
        ax = plot_ESD_absfreq(t, x, f0, label, ax, xlim, ylim, f_unit=f_unit)
        return ax

    def plot_ESD_dB_absfreq(self, label='Energy Spectral Density (dB / Hz^2)', 
                            ax=None, xlim=None, ylim=None, f_unit='THz'):
        t = self.t
        x = self.a
        f0 = self.f0
        ax = plot_ESD_dB_absfreq(t, x, f0, label, ax, xlim, ylim, f_unit=f_unit)
        return ax

    def plot_spectrum_vs_wavelength(self, label='Spectrum Amplitude (W^1/2 / Hz)', 
                                    ax=None, xlim=None, ylim=None, wl_unit='um'):
        t = self.t
        x = self.a
        f0 = self.f0
        ax = plot_spectrum_vs_wavelength(t, x, f0, label, ax, xlim, ylim, wl_unit=wl_unit)
        return ax

    def plot_ESD_vs_wavelength(self, label='Energy Spectral Density (W / Hz^2)', 
                               ax=None, xlim=None, ylim=None, wl_unit='um'):
        t = self.t
        x = self.a
        f0 = self.f0
        ax = plot_ESD_vs_wavelength(t, x, f0, label, ax, xlim, ylim, wl_unit=wl_unit)
        return ax

    def plot_ESD_dB_vs_wavelength(self, label='Energy Spectral Density (dB / Hz^2)', 
                               ax=None, xlim=None, ylim=None, wl_unit='um'):
        t = self.t
        x = self.a
        f0 = self.f0
        ax = plot_ESD_dB_vs_wavelength(t, x, f0, label, ax, xlim, ylim, wl_unit=wl_unit)
        return ax
    
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
    