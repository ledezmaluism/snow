# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:12:11 2018

@author: luish
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.constants import c,pi,mu_0,epsilon_0

def FourierThis(x,dt,NFFT=2048,N=1):
    #Split the input signal in N parts (reducing the sampling rate N times)
    #Calculate the fft for each part
    #Average the results
    
    #The frequency resolution improves this way !??
    #No! Mayve the noise improves though
    Fs =  1/(N*dt)
    Nfreqs = int(NFFT/2+1)
    freqs = np.linspace(0.0,Fs/2,Nfreqs)
    X = np.zeros(freqs.size)

    L = x.size
    for k in range(N):
        tk = np.arange(k,L,N)
        xk = x[tk]
        X = X + np.fft.rfft(xk,NFFT)
    
    X = X/np.sqrt(N)    
    return X, freqs

def FourierEnergy(X):
    #Calculate the energy of a one-sided Fourier transform
    X = np.abs(X)
    NFFT = 2*len(X)
    Energy = (X[0]**2 + 2*np.sum(X[1:]**2))/NFFT
    return Energy

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def detector(y_field, fs, bw=100e9, order=6):
    y_detect = butter_lowpass_filter(y_field**2, bw, fs, order)
    return y_detect

###############################################################################
###############################################################################

# Test function for module  
def _test_FourierThis():
    
    #Create signal
    f0 = 1e9
    w = 2*pi*f0
    bw = 0.1e9; #Aprox
    Fs = 20*f0
    
    tau = 1/bw;
    t0 = 5*tau    
    dt = 1/Fs
    N = 2000
    t = np.arange(0,N*dt,dt)
    x = np.sin(w*t)*np.exp(-((t-t0)/tau)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,x)
    ax.set_title('Time-domain signal')
    
    #Analyze
    X,freqs = FourierThis(x,dt)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freqs,np.abs(X))
    ax.set_title('Raw analysis')
    
    #Split analysis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Split analysis')
    
    X,freqs = FourierThis(x,dt,N=1) 
    ax.plot(freqs,np.abs(X))
    
    X,freqs = FourierThis(x,dt,N=2)
    ax.plot(freqs,np.abs(X))
    
    X,freqs = FourierThis(x,dt,N=4)
    ax.plot(freqs,np.abs(X))
    
    ax.legend(['N=1', 'N=2', 'N=4'])
    ax.set_xlim(0.85e9,1.15e9)

if __name__ == '__main__':
    _test_FourierThis()