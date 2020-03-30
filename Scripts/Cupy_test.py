# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:52:34 2020

@author: luish
"""

import numpy as np
import cupy as cp
import time

NFFT = 2**15
N = 10000
Tmax = 10000

#Cupy
# memory_pool = cp.cuda.MemoryPool()
# cp.cuda.set_allocator(memory_pool.malloc)
# pinned_memory_pool = cp.cuda.PinnedMemoryPool()
# cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

tic = time.time()
t = cp.linspace(-Tmax, Tmax, NFFT)
x = cp.random.normal(size=t.size)
for k in cp.arange(N):
    X = cp.fft.fft(x**2)
    y = cp.fft.ifft(X)
    
elapsed = time.time() - tic
print('Cupy time= ' + str(round(elapsed,5)) + ' seconds')

# memory_pool = cp.cuda.MemoryPool()
# cp.cuda.set_allocator(memory_pool.malloc)
# pinned_memory_pool = cp.cuda.PinnedMemoryPool()
# cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

tic = time.time()
t2 = np.linspace(-Tmax, Tmax, NFFT)
x2 = np.random.normal(size=t.size)
for k in np.arange(N):
    X = cp.fft.fft(x**2)
    y = cp.fft.ifft(X)
    
elapsed = time.time() - tic
print('Cupy time= ' + str(round(elapsed,5)) + ' seconds')

# #Numpy
# tic = time.time()
# t = np.linspace(-Tmax, Tmax, NFFT)
# x = np.random.normal(size=t.size)
# for k in range(N):
#     X = np.fft.fft(x**2)
#     y = np.fft.ifft(X)
    
# elapsed = time.time() - tic
# print('Numpy time= ' + str(round(elapsed,5)) + ' seconds')