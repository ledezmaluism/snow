import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
import time
import math
from scipy import fftpack as sp
from scipy import signal
from matplotlib import cm
import colorcet as cc
from matplotlib.colors import Normalize

#This are my libraries
from . import nlo 
from . import pulses
from . import materials

from scipy.constants import pi, c, epsilon_0
nm = 1e-9
um = 1e-6
mm = 1e-3
ns = 1e-9
ps = 1e-12
fs = 1e-15
MHz = 1e6
GHz = 1e9
THz = 1e12
nJ = 1e-9
pJ = 1e-12

def get_outpow(out_full_td, dt, dTs, dT_val, pin_val,
               pins, f_rep, numsamp):
    '''
    Returns average output power at steady state for a given detuning and input power.
    '''
    pin_index = np.argmin(np.abs(pins-pin_val))
    dT_index = np.argmin(np.abs(dTs-dT_val))
    ave_pow = np.sum(np.average([x*x for x in np.abs(out_full_td[-numsamp:,pin_index,dT_index,:])],0))*dt*f_rep
    return ave_pow

def get_outenergy(out_full_td, dt, dTs, dT_val, pin_val,
                  pins, f_rep, numsamp):
    '''
    Returns average pulse energy at steady state for a given detuning and input power.
    '''
    pin_index = np.argmin(np.abs(pins-pin_val))
    dT_index = np.argmin(np.abs(dTs-dT_val))
    ave_energy = np.sum(np.average([x*x for x in np.abs(out_full_td[-numsamp:,pin_index,dT_index,:])],0))
    return ave_energy

def plot_3d_spectrum(out_full_fd, f_abs, pulsetype, dTs, pins,
                     pin_val, numsamp, tau, f0_sh, ax=None):
    '''
    Function for plotting 3D Spectrum at input power specified by pin_val
    '''
    if pulsetype == 'sech':
        ff_filter_bw = 7*0.315/tau
    elif pulsetype == 'gauss':
        ff_filter_bw = 7*0.44/tau
    f0_ff = f0_sh/2
    wl_array = c/fftshift(f_abs)*1/um
    index = np.argmin(np.abs(pins-pin_val))
    PSDs = 20*np.log10(np.average(np.abs(out_full_fd[-numsamp:,index,:,:]),0)/np.amax(np.abs(out_full_fd[:,index,:,:])))*dt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    myplot = ax.pcolormesh(wl_array, dTs/fs, PSDs, cmap = cc.cm["rainbow"], vmin=-25, vmax=0)
    cb = fig.colorbar(myplot, ax=ax)
    cb.set_label('Relative PSD (dB)')
    ax.set_xlim(c/(um*(f0_ff+ff_filter_bw/2.5)),c/(um*(f0_ff-ff_filter_bw/2.5)))
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Detuning (fs)')
    
    return ax
    
def plot_peak_structure(out_full_td, dt, dTs, pins, pin_val,
                        f_rep, numsamp, ax=None):
    '''
    Function for plotting peak structure at input power specified by pin_val
    '''
    index = np.argmin(np.abs(pins-pin_val))
    ave_pows_dT = np.sum(np.squeeze(np.average([x*x for x in np.abs(out_full_td[-numsamp:,index,:,:])],0)),1)*dt*f_rep
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(dTs/fs, ave_pows_dT)
    ax.set_xlabel('Detuning (fs)')
    ax.set_ylabel('Average Power (W)')
    
    return ax

def plot_outpow_inpow(out_full_td, dt, dTs, dT_range,
                      pins, f_rep, numsamp, ax=None):
    '''
    Function for plotting output power vs input power for a peak in a specified detuning range
    '''
    maxind = np.argmin(np.abs(dTs-np.amax(dT_range)))
    minind = np.argmin(np.abs(dTs-np.amin(dT_range)))
    ave_pows_pin = np.amax(np.sum(np.average([x*x for x in np.abs(out_full_td[-numsamp:,:,minind:maxind,:])],0),2),1)*dt*f_rep
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.scatter(pins, ave_pows_pin/pins*100)
    ax.set_xlabel('Input Average Power (W)')
    ax.set_ylabel('Power Conversion Efficiency (%)')
    
    return ax
    
def plot_coneff(out_full_td, dt, dTs, dT_range, pins,
                f_rep, numsamp, ax=None):
    '''
    Function for plotting conversion efficiency vs input power for a peak in a specified detuning range
    '''
    maxind = np.argmin(np.abs(dTs-np.amax(dT_range)))
    minind = np.argmin(np.abs(dTs-np.amin(dT_range)))
    ave_pows_pin = np.amax(np.sum(np.average([x*x for x in np.abs(out_full_td[-numsamp:,:,minind:maxind,:])],0),2),1)*dt*f_rep
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.scatter(pins, ave_pows_pin)
    ax.set_xlabel('Input Average Power (W)')
    ax.set_ylabel('Output Average Power (W)')
    
    return ax
    
class opo:
    def __init__(self, crystal, phis, feedfunc, outcoupling, t,
                 dTs, pins, f_rep, h, v_ref, pulse_params,
                 pulsetype='sech', roundtrips=100, numsamp=20):
        #pulse_params include tau, f_ref, f0_sh, and Npwr_dB
        #phis are cavity dispersion parameters [GDD, TOD, 4OD, ...]
        #feedfunc is any frequency-dependent loss function.
        self.pulse_params = pulse_params
        if (pulsetype != 'sech') and (pulsetype != 'gauss'):
            raise ValueError('pulsetype must be either sech or gauss.')
        self.pulsetype = pulsetype
        
        #reference OPO parameters
        self.crystal = crystal
        self.h = h
        self.v_ref = v_ref
        self.feedfunc = feedfunc
        self.outcoupling = outcoupling
        self.dTs = dTs
        self.pins = pins
        self.numsamp = numsamp
        self.roundtrips = roundtrips
        self.f_rep = f_rep
        self.phis = phis
        
        #temporal grid
        self.t = t
        self.dt = t[1]-t[0] #Sample spacing
        self.NFFT = t.size
        
        #frequency grid
        self.f = np.fft.fftfreq(self.NFFT, self.dt)
        self.Omega = 2*pi*self.f
        self.df = self.f[1]-self.f[0]
        
        #absolute frequencies
        self.f0 = pulse_params[1]
        self.f_abs = self.f + self.f0
        
        #OPO outputs
        self.out_full_td = np.zeros((roundtrips, np.size(pins), np.size(dTs),
                                   np.size(t)), dtype=np.complex64)
        self.out_full_fd = np.zeros((roundtrips, np.size(pins), np.size(dTs),
                                   np.size(t)), dtype=np.complex64)
    
    def run_opo(self):
        tau = self.pulse_params[0]
        if self.pulsetype == 'sech':
            ff_filter_bw = 7*0.315/tau
        elif self.pulsetype == 'gauss':
            ff_filter_bw = 7*0.44/tau
        f_ref = self.pulse_params[1]
        wl_ref = c/f_ref
        f0_sh = self.pulse_params[2]
        f0_ff = f0_sh/2
        Noise_dB = self.pulse_params[3]
        phi_base = 0
        for tt in range(np.size(self.phis)):
            phi_base = phi_base + self.phis[tt]/(math.factorial(tt+2))*(2*pi*(self.f_abs-f0_ff))**(tt+2)
        for kk in range(np.size(self.pins)):
            pin = self.pins[kk] #Average Power
            if self.pulsetype == 'sech':
                Ppeak = 1/self.f_rep*0.88*pin/tau #Peak Power
            elif self.pulsetype == 'gauss':
                Ppeak = 1/self.f_rep*2*np.sqrt(np.log(2)/pi)*pin/tau
            for jj in range(np.size(self.dTs)):
                dT = self.dTs[jj] # cavity detuning
                phi = phi_base + 2*pi*dT*self.f_abs
                if self.pulsetype == 'sech':
                    pulse = pulses.sech_pulse(self.t, FWHM=tau, Ppeak=Ppeak,
                                              f_ref=f_ref, f0=f0_sh, Npwr_dB=Noise_dB) #Sech pump pulse
                elif self.pulsetype == 'guass':
                    pulse = pulses.gassian_pulse(self.t, FWHM=tau, Ppeak=Ppeak,
                                                 f_ref=f_ref, f0=f0_sh, Npwr_dB=Noise_dB) #Gaussian pump pulse
                in_pulse = pulse # input pulse for the first roundtrip
                for ii in range(self.roundtrips):
                    [out_pulse, pulse_evol] = self.crystal.propagate_NEE_fd(in_pulse, self.h,
                                                                            v_ref=self.v_ref)
                    out_ff, h_ff = pulses.filter_signal(self.f_abs, out_pulse.A, f0_ff,
                                                        ff_filter_bw, type='ideal')
                    feedback_ff = np.fft.ifft(self.feedfunc*np.exp(-1j*phi)*fft(out_ff))
                    recycled_pulse = pulses.pulse(self.t, feedback_ff, wl_ref)
                    self.out_full_td[ii, kk, jj, :] = ifft(self.outcoupling*fft(out_ff))
                    self.out_full_fd[ii, kk, jj, :] = fftshift(self.outcoupling*fft(out_ff))
                    if self.pulsetype == 'sech':
                        pulse = pulses.sech_pulse(self.t, FWHM=tau, Ppeak=Ppeak,
                                                  f_ref=f_ref, f0=f0_sh, Npwr_dB=Noise_dB) #Sech pump pulse
                    elif self.pulsetype == 'gauss':
                        pulse = pulses.gassian_pulse(self.t, FWHM=tau, Ppeak=Ppeak,
                                                     f_ref=_ref, f0=f0_sh, Npwr_dB=Noise_dB) #Gaussian pump pulse
                    in_pulse = pulse+recycled_pulse
    
    def plot_3d_spectrum(self, pin_val, ax=None):
        if (np.size(self.dTs) < 2):
            raise ValueError('Plotting a 3d spectrum for a single detuning value does not make sense.')
        return plot_3d_spectrum(self.out_full_fd, self.f_abs, self.pulsetype, self.dTs, self.pins,
                                pin_val, self.numsamp, self.pulse_params[0], self.pulse_params[2], ax)

    def plot_peak_structure(self, pin_val, ax=None):
        if (np.size(self.dTs) < 2):
            raise ValueError('Plotting a 3d spectrum for a single detuning value does not make sense.')
        return plot_peak_structure(self.out_full_td, self.dt, self.dTs, self.pins, pin_val,
                                   self.f_rep, self.numsamp, ax)
    
    def plot_outpow_inpow(self, dT_range, ax=None):
        if (np.size(dT_range) != 2):
            raise ValueError('dT_range should be a np.array with two values.')
        elif (np.size(self.pins) < 2):
            raise ValueError('Plotting output power for a single input power does not make sense.')
        return plot_outpow_inpow(self.out_full_td, self.dt, self.dTs, dT_range,
                                 self.pins, self.f_rep, self.numsamp, ax)
    
    def plot_coneff(self, dT_range, ax=None):
        if (np.size(dT_range) != 2):
            raise ValueError('dT_range should be a np.array with two values.')
        elif (np.size(self.pins) < 2):
            raise ValueError('Plotting output power for a single input power does not make sense.')
        return plot_coneff(self.out_full_td, self.dt, self.dTs, dT_range,
                           self.pins, self.f_rep, self.numsamp, ax)
    
    def get_outpow(self, dT_val, pin_val):
        if (np.size(dT_val) > 1) or (np.size(pin_val) > 1):
            raise ValueError('dT_val and pin_val should be single valued inputs.')
        return get_outpow(self.out_full_td, self.dt, self.dTs, dT_val,
                           pin_val, self.pins, self.f_rep, self.numsamp)
    
    def get_outenergy(self, dT_val, pin_val):
        if (np.size(dT_val) > 1) or (np.size(pin_val) > 1):
            raise ValueError('dT_val and pin_val should be single valued inputs.')
        return get_outenergy(self.out_full_td, self.dt, self.dTs, dT_val,
                           pin_val, self.pins, self.f_rep, self.numsamp)