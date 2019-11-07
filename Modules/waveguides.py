# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:57:17 2019
@author: luish

Module to calculate waveguide parameters semi-analytically
(using mainly the effective index method)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.constants import pi

def beta_f(kx, ky, n, k0):
    '''
    Propagation constant in the z-direction
    from kx, ky, n, and k0
    '''
    return np.sqrt(n**2*k0**2 - kx**2 - ky**2)

def symmetric_slab_equations(X, n0, n1, d, wl, mode):
    '''
    Nonlinear equations to be solved
    '''
    k0 = 2*pi/wl #wavevector
    Rsqr = (k0*d/2)**2*(n1**2-n0**2)

    Y = np.sqrt(abs(Rsqr - X**2))

    if mode=='TE even':
        F = X*np.tan(X) - Y
    elif mode=='TE odd':
        F = X/np.tan(X) + Y
    elif mode=='TM even':
        F = X*np.tan(X)*(n0/n1)**2 - Y
    elif mode=='TM odd':
        F = X/np.tan(X)*(n0/n1)**2 + Y
    else:
        print('Wrong mode selection....?')
        F = 0

    return F

def neff_symmetric_slab(n0, n1, d, wl, mode='TE even', order=0):
    '''
    n0: cladding index 
    n1: slab index 
    d: slab thickness
    wl: wavelength
    mode: desired mode 'TE even', 'TE odd', 'TM even', 'TM odd'
    order: mode number integer
    '''
    k0 = 2*pi/wl #wavevector
    R = np.sqrt((k0*d/2)**2*(n1**2-n0**2))
    #Solve for Normalized variables: X=kx*d/2, Y=a*d/2

    #Initial condition
    if mode=='TE even' or 'TM even':
        xmin = 0
        xmax = min(R, pi/2*(order+1))
    elif mode=='TE odd' or 'TM odd':
        xmin = pi/2
        xmax = min(R, pi*(order+1))
    else:
        print('Wrong mode selection....?')

    #Solve equations
    X = brentq(symmetric_slab_equations, xmin, xmax, (n0, n1, d, wl, mode))
    kx = X*2/d
    beta = beta_f(kx, 0, n1, k0)
    neff = beta/k0
    return neff

def asymmetric_slab_eqs(x, n0, n1, n2, d, wl, mode='TE', order=0):
    '''
    Nonlinear equations to be solved
    '''
    k0 = 2*pi/wl #wavevector
    Rsqr_1 = (k0*d)**2*(n1**2-n0**2)
    Rsqr_2 = (k0*d)**2*(n1**2-n2**2)

    X = x
    Y = np.sqrt(abs(Rsqr_1 - X**2))
    Y2 = np.sqrt(abs(Rsqr_2 - X**2))

    if mode=='TM':
        Y = Y*(n1/n0)**2
        Y2 = Y2*(n1/n2)**2

    F = np.arctan(Y/X) + np.arctan(Y2/X) - X + order*pi

    return F

def neff_asymmetric_slab(n0, n1, n2, d, wl, mode='TE', order=0):
    '''
    n0: cladding index 
    n1: slab index 
    n2: lower cladding index
    d: slab thickness
    wl: wavelength
    mode: desired mode 'TE' or 'TM'
    order: mode number integer
    '''
    k0 = 2*pi/wl #wavevector
    Rsqr_1 = np.sqrt((k0*d)**2*(n1**2-n0**2))
    Rsqr_2 = np.sqrt((k0*d)**2*(n1**2-n2**2))

    #Solve for Normalized variables: X=kx*d, Y=a*d, Y2=a2*d
    xmin = 1e-6 + order*pi
    xmax = min(Rsqr_1, Rsqr_2, pi*(order+1))


    #Solve equations
    X = brentq(asymmetric_slab_eqs, xmin, xmax, (n0, n1, n2, d, wl, mode, order))
    kx = X/d
    beta = beta_f(kx, 0, n1, k0)
    neff = beta/k0
    return neff


def singlemode_symmetric(n0, n1, wl):
    #Returns thickness for single mode operation
    d = wl/2
    if n1>n0:
        d = d/np.sqrt(n1**2 - n0**2)
    else:
        d = d/np.sqrt(n0**2 - n1**2)
    return d

def singlemode_asymmetric(n0, n1, n2, wl):
    #Returns thickness for single mode operation
    #n1>n2>n0
    d = wl/2
    d = d/np.sqrt(n1**2 - n2**2)
    d = d*( 1 + (1/pi)*np.arctan(np.sqrt(n2**2 - n0**2)/np.sqrt(n1**2 - n2**2)))
    return d


def number_of_modes():
    '''
    Returns number of modes supported by a waveguide geometry
    '''
    pass

def neff_ridge():
    '''
    Returns neff based on the effective index method
    '''
    pass
# =============================================================================
# #Geometric variables:
# start_time = time.time()
# h_ridge = 0.7
# width = 1.5
# hslab = 0.45
# n0 = 1
# freq_list = np.arange(140,310,5)
# #wl_list = np.arange(1,1.5,0.1)
# wl_list = (c/freq_list)*1e-6
# neff_te0 = np.zeros(wl_list.shape)
# for kw in range(wl_list.size):
#     wl = wl_list[kw]
#     n1e = mat.refractive_index('LN_MgO_e', wl)
#     n1o = mat.refractive_index('LN_MgO_o', wl)
#     n2 = mat.refractive_index('SiO2', wl)
    
#     neff_slab_te0 = neff_asymmetric_slab(n0, n1e, n2, hslab, wl)
#     neff_ridge_te0 = neff_asymmetric_slab(n0, n1e, n2, h_ridge, wl)
    
#     neff_te0[kw] = neff_symmetric_slab(neff_slab_te0, neff_ridge_te0, width, wl, 'TM even')
# elapsed_time_2 = time.time() - start_time


# #Comparison with Lumerical
# data = np.load('LNoI_freq_sweep_h_LN_0p7_width_1p5_1572550198.npz')
# neff_lumerical = data['neff']

# error = abs(neff_lumerical-neff_te0)/neff_lumerical*100

# fig, ax = plt.subplots()
# ax.plot(freq_list, neff_lumerical, label='lumerical')
# ax.plot(freq_list, neff_te0, label='python')
# ax.legend()
# ax.grid(True)
# ax.set_xlabel('Frequency (THz)')
# ax.set_ylabel('Effective index $n_{eff}$')
# ax.axis([140, 310, 1.8, 2.1])
# ax.set_title('h_slab = %0.0f nm, h_ridge = %0.0f nm, Width = %0.2f $\mu$m' %(hslab*1000, h_ridge*1000, width))

# fig2, ax2 = plt.subplots()
# ax2.plot(freq_list, error)
# ax2.set_xlabel('Frequency (THz)')
# ax2.set_ylabel('Relative error (%)')
# ax2.set_title('Relative Error = (100%)|n_lumerical - n_python|/n_lumerical')
# ax2.axis([140, 310, 0, 1])
# ax2.grid(True)



# #First TM
# #Vertical step
#neff_slab_tm0 = neff_asymmetric_slab(n0, n1o, n2, hslab, wl, mode='TM')
#neff_ridge_tm0 = neff_asymmetric_slab(n0, n1o, n2, h_ridge, wl, mode='TM')
# #Horizontal step
#neff_tm0 = neff_symmetric_slab(neff_slab_tm0, neff_ridge_tm0, width, wl, 'TE even')

#TE01
#Vertical step
#neff_slab_te1 = neff_asymmetric_slab(n0, n1e, n2, hslab, wl, order=1)
#neff_ridge_te1 = neff_asymmetric_slab(n0, n1e, n2, h_ridge, wl, order=1)
#Horizontal step
# neff_te01 = neff_symmetric_slab(neff_slab_te1, neff_ridge_te1, width, wl, 'TM even')

# TM10
# #Vertical step
# neff_slab_tm01 = neff_asymmetric_slab(n0, n1o, n2, hslab, wl, mode='TM')
# neff_ridge_tm01 = neff_asymmetric_slab(n0, n1o, n2, h_ridge, wl, mode='TM')
# Horizontal step
# neff_tm01 = neff_symmetric_slab(neff_slab_tm0, neff_ridge_tm0, width, wl, 'TE even', order=1)