# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:52:34 2020

@author: luish
"""

import numpy as np
import torch
import time
import pyfftw
# pyfftw.interfaces.cache.enable()
# pyfftw.interfaces.cache.set_keepalive_time(5)
    
NFFT = 2**12
N = 9000
# Tmax = 10000

#Numpy
y = np.random.normal(size=NFFT) + 1j*np.random.normal(size=NFFT)
tic = time.time()
for k in np.arange(N):
    Y = np.fft.fft(y)
    y = np.fft.ifft(Y)
    
elapsed = time.time() - tic
print('Numpy time= ' + str(round(elapsed,5)) + ' seconds')


#Pytorch GPU from numpy
# y = np.random.normal(size=NFFT) + 1j*np.random.normal(size=NFFT)
# y2 = np.array([np.real(y), np.imag(y)]).transpose()
# y = torch.tensor(y2).cuda()

# tic = time.time()
# for k in torch.arange(N).cuda():
#     Y = torch.fft(y, 1)
#     y = torch.ifft(Y, 1)


# elapsed = time.time() - tic
# print('Torch time= ' + str(round(elapsed,5)) + ' seconds')

#FFTW
# y = pyfftw.empty_aligned(NFFT, dtype='complex128', n=16)
# y[:] = np.random.normal(size=NFFT) + 1j*np.random.normal(size=NFFT)

# tic = time.time()
# for k in np.arange(N):
#     Y = pyfftw.interfaces.numpy_fft.fft(y)
#     y = pyfftw.interfaces.numpy_fft.fft(Y)
    
# elapsed = time.time() - tic
# print('FFTW time= ' + str(round(elapsed,5)) + ' seconds')



#FFTW pure
y = pyfftw.empty_aligned(NFFT, dtype='complex128')
Y = pyfftw.empty_aligned(NFFT, dtype='complex128')

fft_y  = pyfftw.FFTW(y, Y)
ifft_Y = pyfftw.FFTW(Y, y, direction='FFTW_BACKWARD')

y[:] = np.random.normal(size=NFFT) + 1j*np.random.normal(size=NFFT)

tic = time.time()
for k in np.arange(N):
    Y = fft_y()
    y = ifft_Y()
    
elapsed = time.time() - tic
print('FFTW time= ' + str(round(elapsed,5)) + ' seconds')