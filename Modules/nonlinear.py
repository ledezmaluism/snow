# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy import exp, log, log10, sqrt, abs
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
import copy
from util import check
from scipy.constants import pi, c


class pulse:
    
    '''
    Some Atributes
    e: time-domain complex amplitude (W^1/2)
    t: time-domain support vector (fs)
    E: frequency-domain complex amplitude (W^1/2)
    f: frequency-domain support vector (baseband, THz)
    wl0: center wavelength (microns)
    
    '''
    
    def __init__(self, t, e, wavelength):
        
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
        self.e = e
        self.E = fft(e, self.NFFT)
        self.Emag = abs(self.E)*(self.dt*1e-15) # (W^1/2)/Hz
        self.esd = self.Emag**2 #Energy spectral density (J/Hz = W/Hz^2)
        

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
        e2 = abs(self.e)**2
        e2_rel = e2/np.amax(e2)
        self.__plot_vs_time(e2_rel, ylabel=label, xlim=xlim)
        
    def plot_magsq_vs_time(self, label='Pulse Intensity (W)', xlim=None):
        e2 = abs(self.e)**2
        self.__plot_vs_time(e2, ylabel=label, xlim=xlim)
            
    def plot_mag_rel_vs_time(self, label='Pulse Amplitude (W^1/2)', xlim=None):
        e_mag = abs(self.e)
        e_mag_rel = e_mag/np.amax(e_mag)
        self.__plot_vs_time(e_mag_rel, ylabel=label, xlim=xlim)
            
    def plot_mag_vs_time(self, label='Pulse Amplitude (W^1/2)', xlim=None):
        e_mag = abs(self.e)
        self.__plot_vs_time(e_mag, ylabel=label, xlim=xlim)
            
    def plot_phase_vs_time(self, label='Pulse Phase (rads)', xlim=None):
        e_phase = np.angle(self.e)
        self.__plot_vs_time(e_phase, ylabel=label, xlim=xlim)
        
    def plot_spectrum(self, label='Spectrum Amplitude (W^1/2 / Hz)', xlim=None):
        self.__plot_vs_freq(self.Emag, ylabel=label, xlim=xlim)
        
    def plot_ESD(self, label='Energy Spectral Density (W / Hz^2)', xlim=None):
        self.__plot_vs_freq(self.esd, ylabel=label, xlim=xlim)
        
    def plot_ESD_dB(self, label='Energy Spectral Density (dB / Hz^2)', xlim=None):
        esd_rel = self.esd/np.amax(self.esd)
        esd_dB = 10*log10(esd_rel)
        self.__plot_vs_freq(esd_dB, ylabel=label, xlim=xlim)
        
    def plot_spectrum_absfreq(self, label='Spectrum Amplitude (W^1/2 / Hz)', xlim=None):
        self.__plot_vs_freq(self.Emag, ylabel=label, xlim=xlim, absfreq=True)
        
    def plot_ESD_absfreq(self, label='Energy Spectral Density (W / Hz^2)', xlim=None):
        self.__plot_vs_freq(self.esd, ylabel=label, xlim=xlim, absfreq=True)
        
    def plot_ESD_dB_absfreq(self, label='Energy Spectral Density (dB / Hz^2)', xlim=None):
        esd_rel = self.esd/np.amax(self.esd)
        esd_dB = 10*log10(esd_rel)
        self.__plot_vs_freq(esd_dB, ylabel=label, xlim=xlim, absfreq=True)
        
    def plot_spectrum_vs_wavelength(self, label='Spectrum Amplitude (W^1/2 / Hz)', xlim=None):
        self.__plot_vs_wavelength(self.Emag, ylabel=label, xlim=xlim)
        
    def plot_ESD_vs_wavelength(self, label='Energy Spectral Density (W / Hz^2)', xlim=None):
        self.__plot_vs_wavelength(self.esd, ylabel=label, xlim=xlim)
        
    def plot_ESD_dB_wavelength(self, label='Energy Spectral Density (dB / Hz^2)', xlim=None):
        esd_rel = self.esd/np.amax(self.esd)
        esd_dB = 10*log10(esd_rel)
        self.__plot_vs_wavelength(esd_dB, ylabel=label, xlim=xlim)
        
    def energy_td(self):
        pwr = abs(self.e)**2
        energy = np.sum(pwr)*self.dt*1e-15 #Joules
        return energy
    
    def energy_fd(self):
        energy = np.sum(self.esd)*self.df*1e12 #Joules
        return energy
    
    def photon_number(self):
        pass
    
    def time_center(self):
        pass
    
    def width_FWHM(self):
        pass
    
    def width_rms(self):
        pass
    
           
class linear_element():
    
    def __init__(self, D):
        self.D = D
        
    def propagate(self, signal):
        X = self.D*signal.E
        signal.update_fd(X)
        return signal
    
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
        
    def add_dispersion_functions(self, Da, Db):
        self.Da = np.exp(-self.h*Da)
        self.Db = np.exp(-self.h*Db)
        
    def add_dispersion_coeffs(self, Ca, Cb, Omega):
        alpha_a = Ca[0]
        b2a = Ca[1]
        b3a = Ca[2]
        b4a = Ca[3]
        alpha_b = Cb[0]
        u = Cb[1]
        b2b = Cb[2]
        b3b = Cb[3]
        b4b = Cb[4]
        
        Da = alpha_a/2 - 1j*b2a*Omega**2/2 - 1j*b3a*Omega**3/6 - 1j*b4a*Omega**4/24
        Db = alpha_b/2 - 1j*u*Omega - 1j*b2b*Omega**2/2 - 1j*b3b*Omega**3/6 - 1j*b4b*Omega**4/24
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
    