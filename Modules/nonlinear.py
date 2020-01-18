# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy.constants import pi, c

class pulse:
    
    '''
    Atributes
    e: time-domain complex amplitude (W^1/2)
    t: time-domain support vector (fs)
    E: frequency-domain complex amplitude
    f: frequency-domain support vector (baseband, THz)
    wl0: center wavelength (microns)
    
    '''
    
    def __init__(self, t, e, wavelength):
        
        dt = t[1]-t[0] #Sample spacing
        NFFT = t.size
        
        self.t = t
        self.e = e
        self.E = fft(e, NFFT)
        self.f = fftfreq(NFFT, dt)*1e3 #THz
        self.omega = 2*pi*self.f
        
        self.wl0 = wavelength #microns
        self.f0 = (c*1e6/1e12)/wavelength #THz
        self.fabs = self.f + self.f0
        self.wl = (c*1e6/1e12)/self.fabs
        
        if np.amin(self.fabs)<=0:
            print('Warning: negative absolute frequencies found. ' + 
                  'Considering reducing the number of points, ' +  
                  'or increasing the total time')
        
    def __plot_vs_time(self, x, ylabel='', xlabel = 'Time (fs)', 
                       xlim=None, ylim=None):
        '''
        Private function to plot stuff x vs time (fs)
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.t, x)
        ax1.grid()
        ax1.set_xlabel(xlabel);
        ax1.set_ylabel(ylabel);
        if xlim != None:
            ax1.set_xlim(xlim)
        if ylim != None:
            ax1.set_ylim(ylim)
            
    def __plot_vs_wavelength(self, x, ylabel='', xlabel = 'Wavelength (um)', 
                             xlim=None, ylim=None):
        '''
        Private function to plot stuff x vs wavelength (microns)
        '''
        wl = fftshift(self.wl)
        x = fftshift(x)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(wl, x)
        ax1.grid()
        ax1.set_xlabel(xlabel);
        ax1.set_ylabel(ylabel);
        if xlim != None:
            ax1.set_xlim(xlim)
        if ylim != None:
            ax1.set_ylim(ylim)
            
    def __plot_vs_freq(self, x, ylabel='', xlabel = 'Frequency (THz)', 
                           xlim=None, ylim=None, absfreq=False):
        '''
        Private function to plot stuff x vs frequency (THz)
        '''
        if absfreq:
            f = self.fabs
        else:
            f = self.f
        f = fftshift(f)
        x = fftshift(x)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(f, x)
        ax1.grid()
        ax1.set_xlabel(xlabel);
        ax1.set_ylabel(ylabel);
        if xlim != None:
            ax1.set_xlim(xlim)
        if ylim != None:
            ax1.set_ylim(ylim)
    
    def plot_magsq_rel_vs_time(self, label='Pulse Intensity (W)', xlim=None):
        e2 = np.abs(self.e)**2
        e2_rel = e2/np.amax(e2)
        self.__plot_vs_time(e2_rel, ylabel=label, xlim=xlim)
        
    def plot_magsq_vs_time(self, label='Pulse Intensity (W)', xlim=None):
        e2 = np.abs(self.e)**2
        self.__plot_vs_time(e2, ylabel=label, xlim=xlim)
            
    def plot_mag_rel_vs_time(self, label='Pulse Amplitude (W^1/2)', xlim=None):
        e_mag = np.abs(self.e)
        e_mag_rel = e_mag/np.amax(e_mag)
        self.__plot_vs_time(e_mag_rel, ylabel=label, xlim=xlim)
            
    def plot_mag_vs_time(self, label='Pulse Amplitude (W^1/2)', xlim=None):
        e_mag = np.abs(self.e)
        self.__plot_vs_time(e_mag, ylabel=label, xlim=xlim)
            
    def plot_phase_vs_time(self, label='Pulse Phase (rads)', xlim=None):
        e_phase = np.angle(self.e)
        self.__plot_vs_time(e_phase, ylabel=label, xlim=xlim)
        
    def plot_spectrum(self, label='Spectrum Amplitude (W^1/2 / Hz)', xlim=None):
        E_mag = np.abs(self.E)
        self.__plot_vs_freq(E_mag, ylabel=label, xlim=xlim)
        
    def plot_PSD(self, label='Energy Spectral Density (W / Hz^2)', xlim=None):
        psd = np.abs(self.E)**2
        self.__plot_vs_freq(psd, ylabel=label, xlim=xlim)
        
    def plot_PSD_dB(self, label='Energy Spectral Density (dB / Hz^2)', xlim=None):
        psd = np.abs(self.E)**2
        psd_rel = psd/np.amax(psd)
        psd_dB = 10*np.log10(psd_rel)
        self.__plot_vs_freq(psd_dB, ylabel=label, xlim=xlim)
        
    def plot_spectrum_absfreq(self, label='Spectrum Amplitude (W^1/2 / Hz)', xlim=None):
        E_mag = np.abs(self.E)
        self.__plot_vs_freq(E_mag, ylabel=label, xlim=xlim, absfreq=True)
        
    def plot_PSD_absfreq(self, label='Energy Spectral Density (W / Hz^2)', xlim=None):
        psd = np.abs(self.E)**2
        self.__plot_vs_freq(psd, ylabel=label, xlim=xlim, absfreq=True)
        
    def plot_PSD_dB_absfreq(self, label='Energy Spectral Density (dB / Hz^2)', xlim=None):
        psd = np.abs(self.E)**2
        psd_rel = psd/np.amax(psd)
        psd_dB = 10*np.log10(psd_rel)
        self.__plot_vs_freq(psd_dB, ylabel=label, xlim=xlim, absfreq=True)
        
    def plot_spectrum_vs_wavelength(self, label='Spectrum Amplitude (W^1/2 / Hz)', xlim=None):
        E_mag = np.abs(self.E)
        self.__plot_vs_wavelength(E_mag, ylabel=label, xlim=xlim)
        
    def plot_PSD_vs_wavelength(self, label='Energy Spectral Density (W / Hz^2)', xlim=None):
        psd = np.abs(self.E)**2
        self.__plot_vs_wavelength(psd, ylabel=label, xlim=xlim)
        
    def plot_PSD_dB_wavelength(self, label='Energy Spectral Density (dB / Hz^2)', xlim=None):
        psd = np.abs(self.E)**2
        psd_rel = psd/np.amax(psd)
        psd_dB = 10*np.log10(psd_rel)
        self.__plot_vs_wavelength(psd_dB, ylabel=label, xlim=xlim)
        
    def energy(self):
        pass
    
    def time_center(self):
        pass
    
    def photon_number(self):
        pass
            

def nonlinear_operator(a,b,kappa):
    f = ifft(kappa*fft(b*np.conj(a)))
    g = -ifft(kappa*fft(a*a))
    return np.array([f,g])

def single_pass(a,b,Da,Db,kappa,L,h):
    
    for kz in range(int(L/h)):
        #Linear step
        a = ifft(Da*fft(a))
        b = ifft(Da*fft(b))
        
        #Nonlinear step
        #Runge-Kutta
        [k1, l1] = h*nonlinear_operator(a,b,kappa)
        [k2, l2] = h*nonlinear_operator(a+k1/2,b+l1/2,kappa)
        [k3, l3] = h*nonlinear_operator(a+k2/2,b+l2/2,kappa)
        [k4, l4] = h*nonlinear_operator(a+k3,b+l3,kappa)
                                               
        a = a + (1/6)*(k1+2*k2+2*k3+k4)
        b = b + (1/6)*(l1+2*l2+2*l3+l4)

    return a,b

def opo(b, N, L, h, Da, Db, fb, kappa):
    '''
    Inputs:
    b: input pump pulse
    N: number of rountrips
    L: length of the crystal
    h: spatial step on the crystal
    Da: Dispersion operator of crystal at signal
    Db: Dispersion operator of crystal at pump
    fb: Feedback operator
    kappa: nonlinear coupling
    
    Outputs:
    a: signal pulse after N roundtrips
    b: pump pulse after last roundtrip
    '''
    
    #Initialize signal as random
    NFFT = b.size
    noise = 1e-10*np.random.normal(size=NFFT)
    a = noise
    
    evol = np.zeros([N, NFFT])
    for kn in range(N):
        a,b_output = single_pass(a,b,Da,Db,kappa,L,h)
            
        #Apply feedback
        a = ifft(fb*fft(a))
        
        evol[kn,:] = (np.abs(a)/np.max(np.abs(a)))**2; #Round-trip evolution
        
        if kn%50==0 and kn!=0:
            print('Completed roundtrip ' + str(kn))
        
    print('Completed roundtrip ' + str(kn+1)) 
    return a, b_output, evol

if __name__ == '__main__':
    NFFT = 2**10
    Tmax = 1400 #(fs) (window will go from -Tmax to Tmax)
    t = np.linspace(-Tmax, Tmax, NFFT) 
    tau = 100/1.76
    b = np.sqrt(0.88*4e6/100)/np.cosh(t/tau)
    
    a = pulse(t, b, 1.0)
    a.plot_PSD_dB_wavelength()