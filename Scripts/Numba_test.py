# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:52:34 2020

@author: luish
"""

import numpy as np
from numba import vectorize, float64, jit
from timeit import default_timer as timer

NFFT = 2**14
N = 1000
Tmax = 10000

# #Numpy
start = timer()
t = np.linspace(-Tmax, Tmax, NFFT)
x = np.random.normal(size=t.size)
for k in range(N):
    X = np.fft.fft(x**2)
    y = np.fft.ifft(X)
    
elapsed = timer() - start
print('Numpy time= ' + str(round(elapsed,5)) + ' seconds')


# @vectorize([float64(float64)])
@jit([float64(float64)], nopython=True, parallel=True)
def f(x):
    X = np.fft.fft(x**2)
    y = np.fft.ifft(X)
    return y

k = np.arange(N)
start = timer()
y = f(k)
elapsed = timer() - start
print('Numba time= ' + str(round(elapsed,5)) + ' seconds')