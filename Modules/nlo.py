# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy import exp, log10, abs
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
import copy
from util import check
from scipy.constants import pi, c


def get_freq_domain(t, x):
    X = fft(x)
    dt = t[1]-t[0] #Sample spacing
    NFFT = t.size
    f = fftfreq(NFFT, dt)
    return fftshift(f), fftshift(X)

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
    Xesd_rel = Xesd/np.amax(Xesd)
    Xesd_dB = 10*log10(Xesd_rel)
    return f, Xesd_dB

def plot_vs_time(t, x, ylabel='', xlabel = 'Time (fs)', ax=None,
                   xlim=None, ylim=None):
    '''
    Private function to plot stuff x vs time (fs)
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(t, x)
    ax.grid()
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel);
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    return ax
        
def __plot_vs_wavelength(wl, x, ylabel='', xlabel = 'Wavelength (um)', 
                         ax=None, xlim=None, ylim=None, shift=True):
    '''
    Private function to plot stuff x vs wavelength (microns)
    '''
    if shift:
        wl = fftshift(wl)
        x = fftshift(x)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(wl, x)
    ax.grid()
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel);
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        
        return ax
            
def __plot_vs_freq(f, x, ylabel='', xlabel = 'Frequency (THz)', 
                       ax=None, xlim=None, ylim=None):
    '''
    Private function to plot stuff x vs frequency (THz)
    '''
    
    #Check if frequencies are not shifted yet
    if f[0]==0:
        f = fftshift(f)
        x = fftshift(x)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(f, x)
    ax.grid()
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel);
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        
    return ax
 

def plot_mag(t, x, label='Pulse Amplitude (W^1/2)', ax=None, 
                     xlim=None, ylim=None):
    x_mag = abs(x)
    ax = plot_vs_time(t, x_mag, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    
    return ax
        
def plot_magsq(t, x, label='Pulse Intensity (W)', ax=None, 
                       xlim=None, ylim=None):
    x2 = abs(x)**2
    ax = plot_vs_time(t, x2, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    
    return ax
   
def plot_mag_relative(t, x, label='Relative Pulse Amplitude', ax=None, 
                         xlim=None, ylim=None):
    x_mag = abs(x)
    x_mag_rel = x_mag/np.amax(x_mag)
    ax = plot_vs_time(t, x_mag_rel, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    
    return ax

def plot_magsq_relative(t, x, label='Relative Pulse Intensity', ax=None, 
                           xlim=None, ylim=None):
    x2 = abs(x)**2
    x2_rel = x2/np.amax(x2)
    ax = plot_vs_time(t, x2_rel, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    
    return ax
            
def plot_phase_vs_time(t, x, label='Pulse Phase (rads)', ax=None,
                       xlim=None, ylim=None):
    x_phase = np.angle(x)
    ax = plot_vs_time(t, x_phase, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    
    return ax

def plot_phase_vs_freq(t, x, label='Pulse Phase (rads)', ax=None,
                       xlim=None, ylim=None):
    
    f, X = get_freq_domain(t, x)
    X_phase = np.angle(X)
    ax = plot_vs_time(f, X_phase, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    
    return ax

def plot_spectrum(t, x, label='Spectrum Amplitude (W^1/2 / Hz)', ax=None,
                  xlim=None, ylim=None):
    f, Xmag = get_spectrum(t, x)
    ax = __plot_vs_freq(f, Xmag, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    return ax
        
def plot_ESD(t, x, label='Energy Spectral Density (W / Hz^2)', ax=None, 
             xlim=None, ylim=None):
    f, Xesd = get_esd(t, x)
    ax = __plot_vs_freq(f, Xesd, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    return ax
        
def plot_ESD_dB(t, x, label='Energy Spectral Density (dB / Hz^2)', ax=None,
                xlim=None, ylim=None):
    f, Xesd = get_esd_dB(t, x)
    ax = __plot_vs_freq(f, Xesd, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    return ax
        
def plot_spectrum_absfreq(t, x, f0, label='Spectrum Amplitude (W^1/2 / Hz)', 
                          ax=None, xlim=None, ylim=None):
    f, Xmag = get_spectrum(t, x)
    f = f+f0
    ax = __plot_vs_freq(f, Xmag, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    return ax
        
def plot_ESD_absfreq(t, x, f0, label='Energy Spectral Density (W / Hz^2)', 
                     ax=None, xlim=None, ylim=None):
    f, Xesd = get_esd(t, x)
    f = f  + f0
    ax = __plot_vs_freq(f, Xesd, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    return ax
        
def plot_ESD_dB_absfreq(t, x, f0, label='Energy Spectral Density (dB / Hz^2)', 
                        ax=None, xlim=None, ylim=None):
    f, Xesd = get_esd_dB(t, x)
    f = f  + f0
    ax = __plot_vs_freq(f, Xesd, ylabel=label, ax=ax, xlim=xlim, ylim=ylim)
    return ax
        
def plot_spectrum_vs_wavelength(t, x, f0, label='Spectrum Amplitude (W^1/2 / Hz)', 
                                ax=None, xlim=None, ylim=None, shift=True):
    f, Xmag = get_spectrum(t, x)
    f = f  + f0
    wl = c/f*1e6 #Microns
    ax = __plot_vs_wavelength(wl, Xmag, ylabel=label, ax=ax, 
                              xlim=xlim, ylim=ylim, shift=shift)
    return ax
        
def plot_ESD_vs_wavelength(t, x, f0, label='Energy Spectral Density (W / Hz^2)', 
                           ax=None, xlim=None, ylim=None, shift=True):
    f, Xesd = get_esd(t, x)
    f = f  + f0
    wl = c/f*1e6 #Microns
    ax = __plot_vs_wavelength(wl, Xesd, ylabel=label, ax=ax, 
                              xlim=xlim, ylim=ylim, shift=shift)
        
def plot_ESD_dB_vs_wavelength(t, x, f0, label='Energy Spectral Density (dB / Hz^2)', 
                           ax=None, xlim=None, ylim=None, shift=True):
    f, Xesd = get_esd_dB(t, x)
    f = f  + f0
    wl = c/f*1e6 #Microns
    ax = __plot_vs_wavelength(wl, Xesd, ylabel=label, ax=ax, 
                              xlim=xlim, ylim=ylim, shift=shift)
        
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

class pulse:
    
    '''
    Some Atributes
    e: time-domain complex amplitude (W^1/2)
    t: time-domain support vector (fs)
    E: frequency-domain complex amplitude (W^1/2)
    f: frequency-domain support vector (baseband, THz)
    wl0: center wavelength (microns)
    
    '''
    
    def __init__(self, t, x, wavelength, domain='Time'):
        
        if not check(wavelength, 0.05, 10):
            print('Warning: Wavelength of %0.2f um '%(wavelength) + 
                  'for pulse looks outside usual range.')
        
        #Static atributes, they usually don't change
        self.t = t
        self.dt = t[1]-t[0] #Sample spacing
        self.NFFT = t.size
        self.f = fftfreq(self.NFFT, self.dt)*1e3 #THz
        self.omega = 2*pi*self.f
        self.df = self.f[1]-self.f[0]
        self.wl0 = wavelength #central wavelength in microns
        self.f0 = (c*1e6/1e12)/wavelength #central frequency in THz
        self.fabs = self.f + self.f0
        self.wl = (c*1e6/1e12)/self.fabs
        
        #Dynamic atributes, they change as pulse propagates
        if domain=='Time':
            self.e = x
            self.E = fft(x, self.NFFT)
        elif domain=='Freq':
            self.E = x
            self.e = ifft(x, self.NFFT)
        else:
            print('Wrong domain option. Options are "Time" or "Freq"')
        self.Emag = abs(self.E)*(self.dt*1e-15) # (W^1/2)/Hz
        self.esd = self.Emag**2 #Energy spectral density (J/Hz = W/Hz^2)
        esd_rel = self.esd/np.amax(self.esd)
        self.esd_dB = 10*log10(esd_rel)

        if np.amin(self.fabs)<=0:
            print('Warning: negative absolute frequencies found. ' + 
                  'Considering reducing the number of points, ' +  
                  'or increasing the total time')
            
    def update_td(self, e):
        if np.size(self.t) == np.size(e):
            self.e = e
            self.E = fft(e, self.NFFT)
            self.update_param()
        else:
            raise RuntimeError('Hmm, this pulse seems diffenrent. Cannot update.')
            
    def update_fd(self, E):
        if np.size(self.t) == np.size(E):
            self.E = E
            self.e = ifft(E, self.NFFT)
            self.update_param()
        else:
            raise RuntimeError('Hmm, this pulse seems diffenrent. Cannot update.') 
            
    def update_param(self):
        self.Emag = abs(self.E)*(self.dt*1e-15) # (W^1/2)/Hz
        self.esd = self.Emag**2 #Energy spectral density (J/Hz = W/Hz^2)
        esd_rel = self.esd/np.amax(self.esd)
        self.esd_dB = 10*log10(esd_rel)
        
    def plot_mag(self, label='Pulse Amplitude (W^1/2)', ax=None, 
                         xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_mag(t, x, label=label, ax=ax, xlim=xlim, ylim=ylim)
        return ax
    
    def plot_magsq(self, label='Pulse Intensity (W)', ax=None, 
                           xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_magsq(t, x, label, ax, xlim, ylim)
        return ax

    def plot_mag_relative(self, label='Pulse Amplitude (W^1/2)', ax=None, 
                             xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_mag_relative(t, x, label, ax, xlim, ylim)
        return ax
    
    def plot_magsq_relative(self, label='Pulse Intensity (W)', ax=None, 
                               xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_magsq_relative(t, x, label, ax, xlim, ylim)
        return ax          
            
    def plot_phase_vs_time(self, label='Pulse Phase (rads)', ax=None,
                           xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_phase_vs_time(t, x, label, ax, xlim, ylim)
        return ax
    
    def plot_phase_vs_freq(self, label='Pulse Phase (rads)', ax=None,
                           xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_phase_vs_freq(t, x, label, ax, xlim, ylim)
        return ax
    
    def plot_spectrum(self, label='Spectrum Amplitude (W^1/2 / Hz)', ax=None,
                      xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_spectrum(t, x, label, ax, xlim, ylim)
        return ax
        
    def plot_ESD(self, label='Energy Spectral Density (W / Hz^2)', ax=None, 
                 xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_ESD(t, x, label, ax, xlim, ylim)
        return ax
        
    def plot_ESD_dB(self, label='Energy Spectral Density (dB / Hz^2)', ax=None,
                    xlim=None, ylim=None):
        t = self.t
        x = self.e
        ax = plot_ESD_dB(t, x, label, ax, xlim, ylim)
        return ax
        
    def plot_spectrum_absfreq(self, label='Spectrum Amplitude (W^1/2 / Hz)', 
                              ax=None, xlim=None, ylim=None):
        t = self.t
        x = self.e
        f0 = self.f0
        ax = plot_spectrum_absfreq(t, x, f0, label, ax, xlim, ylim)
        return ax
        
    def plot_ESD_absfreq(self, label='Energy Spectral Density (W / Hz^2)', 
                         ax=None, xlim=None, ylim=None):
        t = self.t
        x = self.e
        f0 = self.f0
        ax = plot_ESD_absfreq(t, x, f0, label, ax, xlim, ylim)
        return ax
        
    def plot_ESD_dB_absfreq(self, label='Energy Spectral Density (dB / Hz^2)', 
                            ax=None, xlim=None, ylim=None):
        t = self.t
        x = self.e
        f0 = self.f0
        ax = plot_ESD_dB_absfreq(t, x, f0, label, ax, xlim, ylim)
        return ax
        
    def plot_spectrum_vs_wavelength(self, label='Spectrum Amplitude (W^1/2 / Hz)', 
                                    ax=None, xlim=None, ylim=None):
        t = self.t
        x = self.e
        f0 = self.f0
        ax = plot_spectrum_vs_wavelength(t, x, f0, label, ax, xlim, ylim)
        return ax
    
    def plot_ESD_vs_wavelength(self, label='Energy Spectral Density (W / Hz^2)', 
                               ax=None, xlim=None, ylim=None):
        t = self.t
        x = self.e
        f0 = self.f0
        ax = plot_ESD_vs_wavelength(t, x, f0, label, ax, xlim, ylim)
        return ax
        
    def plot_ESD_dB_vs_wavelength(self, label='Energy Spectral Density (dB / Hz^2)', 
                               ax=None, xlim=None, ylim=None):
        t = self.t
        x = self.e
        f0 = self.f0
        ax = plot_ESD_dB_vs_wavelength(t, x, f0, label, ax, xlim, ylim)
        return ax
        
    def energy_td(self):
        t = self.t
        x = self.e
        return energy_td(t, x)
    
    def energy_fd(self):
        t = self.t
        x = self.e
        return energy_fd(t, x)
    
    def photon_number(self):
        pass
    
    def time_center(self):
        pass
    
    def width_FWHM(self):
        pass
    
    def width_rms(self):
        pass
    
    
class nonlinear_element():
    '''
    Atributes
    D: Dispersion vs frequency
    kappa: nonlinear coefficient
    L: length (um)
    PP: poling period (um)
    h: split-step size (um)
    '''
    def __init__(self, L=1, PP=0, Da=None, Db=None, nlc=None, h=None):
        self.L = L
        self.PP = PP
        self.nlc = nlc
        if h is None:
            self.h = L/50
        else:
            self.h = h
        self.Da = exp(-self.h*Da)
        self.Db = exp(-self.h*Db)
        
    def update_h(self, h_new):
        h_old = self.h
        self.h = h_new
        self.Da = (self.Da)**(h_new/h_old)
        self.Db = (self.Db)**(h_new/h_old)
        
    def add_dispersion_functions(self, Da, Db):
        self.Da = np.exp(-self.h*Da)
        self.Db = np.exp(-self.h*Db)
        
    def add_dispersion_neff(self, neff):
        pass
    
    def add_nonlinear_coeff(self, nlc):
        self.nlc = nlc
    
    def propagate(self, a_input, b_input):
        Da = self.Da
        Db = self.Db
        nlc = self.nlc
        h = self.h
        
        a_output = copy.deepcopy(a_input)
        b_output = copy.deepcopy(b_input)
        #Get just the time domain part of the pulses
        x = a_output.e
        y = b_output.e
        
        for kz in range(int(self.L/self.h)):
            #Linear step
            x = ifft(Da*fft(x))
            y = ifft(Db*fft(y))
            
            #Nonlinear step
            #Runge-Kutta 4th order
            [k1, l1] = h*self._nonlinear_operator(x,y,nlc)
            [k2, l2] = h*self._nonlinear_operator(x+k1/2,y+l1/2,nlc)
            [k3, l3] = h*self._nonlinear_operator(x+k2/2,y+l2/2,nlc)
            [k4, l4] = h*self._nonlinear_operator(x+k3,y+l3,nlc)
                                                   
            x = x + (1/6)*(k1+2*k2+2*k3+k4)
            y = y + (1/6)*(l1+2*l2+2*l3+l4)
        
        #Update the pulses
        a_output.update_td(x)
        b_output.update_td(y)
        return a_output,b_output
        
    def _nonlinear_operator(self, a, b, nlc):
        f = ifft(nlc*fft(b*np.conj(a)))
        g = -ifft(nlc*fft(a*a))
        return np.array([f,g])


def test1():
    pass

if __name__ == '__main__':
    test1()
    