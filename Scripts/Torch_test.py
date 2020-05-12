# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:52:34 2020

@author: luish
"""

import numpy as np
import torch
import time

NFFT = 2**14
N = 50000
Tmax = 10000

#Numpy
y = np.random.normal(size=NFFT) + 1j*np.random.normal(size=NFFT)
tic = time.time()
for k in np.arange(N):
    Y = np.fft.fft(y)
    y = np.fft.ifft(Y)
    
elapsed = time.time() - tic
print('Numpy time= ' + str(round(elapsed,5)) + ' seconds')


#Pytorch CPU
# x = torch.randn(NFFT)
# X = torch.rfft(x, 1)

# tic = time.time()
# for k in torch.arange(N):
#     x = torch.ifft(X, 1)
#     X = torch.fft(x, 1)
    
# elapsed = time.time() - tic
# print('Torch time= ' + str(round(elapsed,5)) + ' seconds')

#Pytorch GPU
# x = torch.randn(NFFT).cuda()
# X = torch.rfft(x, 1)

# tic = time.time()
# for k in torch.arange(N).cuda():
#     x = torch.ifft(X, 1)
#     X = torch.fft(x, 1)
    
# elapsed = time.time() - tic
# print('Torch time= ' + str(round(elapsed,5)) + ' seconds')



#Pytorch GPU from numpy
y = np.random.normal(size=NFFT) + 1j*np.random.normal(size=NFFT)
y2 = np.array([np.real(y), np.imag(y)]).transpose()
y = torch.tensor(y2).cuda()

tic = time.time()
for k in torch.arange(N).cuda():
    Y = torch.fft(y, 1)
    y = torch.ifft(Y, 1)


elapsed = time.time() - tic
print('Torch time= ' + str(round(elapsed,5)) + ' seconds')