# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:12:11 2018

@author: luish
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.constants import pi

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

def FTX(x,dt,NFFT=2048):
    #Calculates the Fourier Transform for real time signals
    Fs =  1/dt
    Nfreqs = int(NFFT/2+1)
    freqs = np.linspace(0.0,Fs/2,Nfreqs)
    X = np.fft.rfft(x, NFFT)
    return X, freqs

def IFTX(X,df,NFFT=2048):
    #Calculates the Fourier Transform for real time signals
    dt =  1/df
    Nt = NFFT
    times = np.linspace(0.0, dt, Nt)
    X = np.fft.irfft(X, NFFT)
    return X, times

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

def derivative( f, x, n, h ):
    """Richardson's Extrapolation to approximate  f'(x) at a particular x.

    USAGE:
	d = richardson( f, x, n, h )

    INPUT:
	f	- function to find derivative of
	x	- value of x to find derivative at
	n	- number of levels of extrapolation
	h	- initial stepsize

    OUTPUT:
        numpy float array -  two-dimensional array of extrapolation values.
                             The [n,n] value of this array should be the
                             most accurate estimate of f'(x).

    NOTES:                             
        Based on an algorithm in "Numerical Mathematics and Computing"
        4th Edition, by Cheney and Kincaid, Brooks-Cole, 1999.

    AUTHOR:
        Jonathan R. Senning <jonathan.senning@gordon.edu>
        Gordon College
        February 9, 1999
        Converted ty Python August 2008
    """

    # d[n,n] will contain the most accurate approximation to f'(x).

    d = np.array( [[0] * (n + 1)] * (n + 1), float )

    for i in range( n + 1 ):
        d[i,0] = 0.5 * ( f( x + h ) - f( x - h ) ) / h

        powerOf4 = 1  # values of 4^j
        for j in range( 1, i + 1 ):
            powerOf4 = 4 * powerOf4
            d[i,j] = d[i,j-1] + ( d[i,j-1] - d[i-1,j-1] ) / ( powerOf4 - 1 )

        h = 0.5 * h

    return d[n,n]



###############################################################################
###############################################################################
def _test_():
    '''
    Test function for module  
    '''
    pass

if __name__ == '__main__':
    _test_()