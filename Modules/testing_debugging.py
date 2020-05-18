# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:59:40 2020

@author: luish
"""

import nlo
import numpy as np
import cProfile
import profile
import pyfftw
import timeit
import cmath
 
nm = 1e-9
um = 1e-6
mm = 1e-3
ps = 1e-12
fs = 1e-15
GHz = 1e9
THz = 1e1

NFFT = 2**12
f_min = -100*THz
f_max = 1e3*THz
BW = f_max - f_min
dt = 1/BW
t_start = -2.5*ps
t_stop = t_start + NFFT*dt
t = np.arange(t_start, t_stop, step=dt)

wl_ref = 1*um
x = np.random.normal(size=NFFT)
p = nlo.pulse(t, x, wl_ref)
h = 1*mm/200

def chi2(z):
    # poling = (2/pi)*np.cos(z*2*pi/pp)
    # return 2*deff*poling
    return 1

def n_func(wl):
    return 1

crystal = nlo.nonlinear_element(L=5*mm, n_func=n_func, chi2=chi2)

def foo():
    # [out_pulse, pulse_evol] = crystal.propagate_NEE_fd(p, h)
    [out_pulse, pulse_evol] = crystal.propagate_NEE_fd_2(p, h)


phi_1 = t
phi_2 = t
z = 1*um
# def fnl(A):
#     phi = phi_1 - phi_2*z
    
#     a = ifft(A)
#     f1 = a*a*np.exp(1j*phi) + 2*a*np.conj(a)*np.exp(-1j*phi)
       
#     f = -1j*chi(z)*fft(f1)
#     return f

a = pyfftw.empty_aligned(NFFT, dtype='complex128')
A = pyfftw.empty_aligned(NFFT, dtype='complex128')
f = pyfftw.empty_aligned(NFFT, dtype='complex128')
F = pyfftw.empty_aligned(NFFT, dtype='complex128')

fft_a = pyfftw.FFTW(a, A)
ifft_A = pyfftw.FFTW(A, a, direction='FFTW_BACKWARD')
fft_f = pyfftw.FFTW(f, F)
A = x

def fexp(phi):
    # return np.exp(1j*phi) 
    # return np.cos(phi) + 1j*np.sin(phi)
    return cmath.exp(1j*phi)

def f1(a, phi):
    # return a*a*np.exp(1j*phi) + 2*a*np.conj(a)*np.exp(-1j*phi)
    x = fexp(phi)
    y = a*x
    return a*(y + 2*np.conj(y))

def f2():
    a = ifft_A()
    return a
    
def f3():
    F = fft_f()
    return F

def fnl():
    for k in range(1000):
        phi = phi_1 - phi_2*z 
        a = f2()
        f[:] = f1(a, phi)
        F = f3()
        Z = -1j*chi2(z)*F
    return Z

cProfile.run('fnl()')
# profile.run('fnl()')

# timeit.timeit('fnl()', number=10,  globals=globals())