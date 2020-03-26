# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:03:45 2020

@author: luish
"""

import nlo #This is my library
from util import sech

import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import time
import scipy
from matplotlib import cm

from scipy.constants import pi, c


c = c*1e-12; #mm/fs, [Freq]=PHz
NFFT = 2**10 #Fourier size
Tmax = 2000 #(fs) (window will go from -Tmax to Tmax)
t = np.linspace(-Tmax, Tmax, NFFT, endpoint=False)
ts = t[1]-t[0] #Sampling period

#Pulses
wla = 2.090e-3 #signal wavelength (mm)
noise = 1e-10*np.random.normal(size=NFFT)
signal = nlo.pulse(t, noise, wla*1e3)
wlb = wla/2 #pump wavelength (mm)
pump_pwr = 0.5 #Average pump power (W)
Tp = 70 #Input pulse width (fs)
tau = Tp/1.76
pulse = np.sqrt(0.88*4e6/Tp*pump_pwr)*sech(t/tau)
pump = nlo.pulse(t, pulse, wlb*1e3)

#The pulses includes a natural FFT frequency axis
Omega = pump.omega*1e-3

#OPO parameters
wa = 2*pi*c/wla #Central angular frequency for pulse "a"
Co_loss = 1-0.65  #Output coupler loss
dT = 4 #Detuning in fs
l = c*dT/wlb #Detuning parameter l
Ws = 14e-3 #Beam waist of signal (mm)
Nrt = 200 #Round trips

#Cavity dispersion parameters
phi2 = 25*2
phi3 = 76
phi4 = -13020
# phi5 = 983328
phi5 = 0

#Feedback loop
deltaphi = (phi2/2)*Omega**2 + (phi3/6)*Omega**3 + (phi4/24)*Omega**4 + (phi5/120)*Omega**5
phi = pi*l + dT*Omega + deltaphi
fb = np.sqrt(Co_loss)*np.exp(1j*phi)

#Linear element representing this feedback path
fb = nlo.linear_element(fb)

#Crystal dispersion
alpha_a = 0.00691 #loss for signal in crystal (1/mm)
alpha_b = 0.00691 #loss for pump in crystal (1/mm)
u = 112.778 #Group velocity mismatch (fs/mm)
b2a = -53.64 #second order GVD signal (fs^2/mm)
b3a = 756.14 #third order GVD signal (fs^3/mm)
b4a = -2924.19 #fourth order GVD signal (fs^4/mm)
b2b = 240.92 #second order GVD pump (fs^2/mm)
b3b = 211.285 #second order GVD pump (fs^3/mm)
b4b = -18.3758 #second order GVD pump (fs^4/mm)
deff = 2/pi*16e-9 #effective nonlinear coefficient (mm/V)
na = 2.2333# refractive index at signal
nb = 2.1935# refractive index at pump

#Dispersion functions
Da = alpha_a/2 - 1j*b2a*Omega**2/2 - 1j*b3a*Omega**3/6 - 1j*b4a*Omega**4/24
Db = alpha_b/2 - 1j*u*Omega - 1j*b2b*Omega**2/2 - 1j*b3b*Omega**3/6 - 1j*b4b*Omega**4/24

#Crystal parameters
L = 1 #mm
h = L/50 #Distance step Size

#Nonlinear coupling
nlc = np.sqrt(2*377)*deff*(Omega+wa)/(Ws*na*c*np.sqrt(pi*nb))

#Nonlinear crystal
crystal = nlo.nonlinear_element(L=L, h=h, Da=Da, Db=Db, nlc=nlc)

def opo(signal, pump, nl_element, feedback):

    #Roundtrip numbers, to be optimized for convergence
    N = 200
    
    #Variables to save roundtrip evolution
    signal_evolution = np.zeros([N, pump.NFFT])
    signal_energy_evol = np.zeros(N)
    pump_energy_evol = np.zeros(N)
    
    for kn in range(N):
        [signal, pump_out] =  crystal.propagate(signal, pump)
            
        #Apply feedback
        fb.propagate(signal)
        
        signal_evolution[kn,:] = (np.abs(signal.e)/np.max(np.abs(signal.e)))**2
        signal_energy_evol = signal.energy_td()
        pump_energy_evol = pump_out.energy_td()
        
        if (kn+1)%50==0:
            print('Completed roundtrip ' + str(kn+1))
    
    return signal, pump_out, signal_evolution, signal_energy_evol, pump_energy_evol

signal, pump_out, signal_evolution, signal_energy_evol, pump_energy_evol = opo(signal, pump, crystal, fb)

# signal.plot_ESD_dB_wavelength(xlim=[1,3])