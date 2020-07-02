# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
import scipy.signal
from scipy.constants import pi, c

def coherence_g1(pulse_train):
    '''
    Calculates g1 coherence function for a train of pulses
    It uses all possible pairs of pulses in the train

    Parameters
    ----------
    pulse_train : ARRAY
        ARRAY OF PULSES. THERE SHOULD BE A PULSE PER COLUMN. 
        SO THE NUMBER OF COLUMNS WILL BE EQUAL TO THE NUMBER OF PULSES.
        IT USES ALL N*(N-1)/2 PAIR OF PULSES AVAILABLE

    Returns
    -------
    g : TYPE
        THE COHERENCE FUNCTION IN THE FREQUENCY DOMAIN.

    '''
    E12 = 0
    E11 = 0
    E22 = 0
    
    Npulses = pulse_train.shape[1]
    nn = 0 #Number of pairs

    for k1 in range(Npulses - 1):
        
        E1 = fft(pulse_train[:, k1])
        
        for k2 in np.arange(k1+1, Npulses):
            
            E2 = fft(pulse_train[:, k2])
    
            E12 += np.conj(E1) * E2
            E11 += np.conj(E1) * E1
            E22 += np.conj(E2) * E2
        
            nn += 1
    
    g = np.abs(E12 / np.sqrt(E11 * E22) )   
    print('Calculated coherence with %i pairs of pulses' %(nn) )
    return g

def average_g1(x, g1):
    Xesd = np.abs( fft(x) )**2
    return np.sum( g1 * Xesd ) / np.sum( Xesd ) 

def gaussian(t, Energy, FWHM, f0=0):
    '''
    Parameters
    ----------
    t : ARRAY
        TIME ARRAY.
    Energy : FLOAT
        PULSE ENERGY.
    FWHM : FLOAT
        PULSE WIDTH.
    f0 : FLOAT
        CENTER FREQUENCY (OFFSET FOURIER DOMAIN).

    Returns
    -------
    TYPE
        AMPLITUDE ARRAY.

    '''
    Ppeak = 0.94*Energy/FWHM
    x = np.sqrt(Ppeak) * np.exp(-2*np.log(2)*(t/FWHM)**2)
    return x * np.exp(1j*2*pi*f0*t)

def sech(t, Energy, FWHM, f0=0):
    '''
    Parameters
    ----------
    t : ARRAY
        TIME ARRAY.
    Energy : FLOAT
        PULSE ENERGY.
    FWHM : FLOAT
        PULSE WIDTH.
    f0 : FLOAT
        CENTER FREQUENCY (OFFSET FOURIER DOMAIN).

    Returns
    -------
    TYPE
        AMPLITUDE ARRAY.
    '''
    Ppeak = 0.88*Energy/FWHM
    x = np.sqrt(Ppeak) / np.cosh( 1.76 * t / FWHM )
    return x * np.exp(1j*2*pi*f0*t)

def noise(t, Npower):
    sigma = np.sqrt(Npower) / np.sqrt(2)
    x = np.random.normal(scale = sigma, size = t.size) + 1j*np.random.normal(scale = sigma, size = t.size)
    return x

def gaussian_pulse(t, FWHM, f_ref, Energy=None, Ppeak=None, f0=0, Npwr_dB=-100):
    '''
    Generates a gaussian pulse with noise with a given Energy or peak power
    '''
    
    if Energy != None:
        pass
    elif Ppeak != None:
        Energy = Ppeak * FWHM / 0.94
    else:
        print('Warning: Using default 1pJ of energy for the pulse')
        Energy = 1e-12
        
    x = gaussian(t, Energy, FWHM, f0)
    Npwr = np.amax(np.abs(x)**2) * 10**( -Npwr_dB/10 )
    n = noise(t, Npwr)
    return pulse(t, x+n, c/f_ref, domain='Time')

def sech_pulse(t, FWHM, f_ref, Energy=None, Ppeak=None, f0=0, Npwr_dB=-100):
    '''
    Generates a sech pulse with noise with a given Energy or peak power
    '''
    
    if Energy != None:
        pass
    elif Ppeak != None:
        Energy = Ppeak * FWHM / 0.88
    else:
        print('Warning: Using default 1pJ of energy for the pulse')
        Energy = 1e-12
        
    x = sech(t, Energy, FWHM, f0-f_ref)
    Npwr = np.amax(np.abs(x)**2) * 10**( -Npwr_dB/10 )
    n = noise(t, Npwr)
    return pulse(t, x+n, c/f_ref, domain='Time')

def filter_signal(f_abs, X, f0, bw):
    #Input in is the frequency domain already
    f_min = np.amin(f_abs)
    f_max = np.amax(f_abs)
    f1 = (f0 - bw/2 - f_min)/(f_max-f_min)
    f2 = (f0 + bw/2 - f_min)/(f_max-f_min)
    sos = scipy.signal.butter(15, [f1, f2], 'bandpass', output='sos')
    _, h = scipy.signal.sosfreqz(sos, worN = X.size)
    filtered = ifft( X * fftshift(h) )
    return filtered, h

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
            
    def __add__(self, pulse2):
        xsum = self.a + pulse2.a
        return pulse(self.t, xsum, self.wl0)

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
    
def test1():
    pass

if __name__ == '__main__':
    test1()
    