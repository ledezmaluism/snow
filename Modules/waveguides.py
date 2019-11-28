# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:57:17 2019
@author: luish

Module to calculate waveguide parameters semi-analytically
(using mainly the effective index method)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.constants import pi, c

import analysis

class waveguide:

    def __init__(self, w_top=1, h_ridge=1, h_slab=0.5, theta=60):
        #validation
        etch = h_ridge - h_slab
        w_base = w_top + 2*etch/np.tan(theta*pi/180)
        if etch*(w_base-w_top)<0:
            raise ValueError("Something wrong with this geometry")

        #Attributes given
        self.w_top = w_top
        self.h_ridge = h_ridge
        self.h_slab = h_slab
        self.theta = theta

        #Attributes calculated
        self.etch = etch
        self.w_base = w_base

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

def neff_ridge(wl, nridge, nbox=1, nclad=1, w=1, h=0.7, hslab=0.1, mode='TE'):
    '''
    Returns neff based on the effective index method
    '''
    if mode=='TE':
        neff_slab_te0 = neff_asymmetric_slab(nclad, nridge, nbox, hslab, wl)
        neff_ridge_te0 = neff_asymmetric_slab(nclad, nridge, nbox, h, wl)
        # neff = neff_symmetric_slab(neff_slab_te0, neff_ridge_te0, w, wl, 'TM even')
        neff = neff_asymmetric_slab(neff_slab_te0, neff_ridge_te0, neff_slab_te0, w, wl, 'TM even')
    return neff

def neff_derivative(wl, nridge, nbox=1, nclad=1, w=1, h=0.7, hslab=0.1, mode='TE'):
    '''
    Finds the derivative of neff at given wavelength

    USAGE:
	d = neff_derivative(wl, nridge)

    INPUT:
	wl	- wavelength
	nridge	- ...


    OUTPUT:
        numpy float neff'

    '''
    n = 2 #number of extrapolation levels
    wl_step = 0.001 #Initial step size
    def f(wl):
        return neff_ridge(wl, nridge, nbox, nclad, w, h, hslab, mode)

    return analysis.derivative(f, wl, n, wl_step)


def neff_2derivative(wl, nridge, nbox=1, nclad=1, w=1, h=0.7, hslab=0.1, mode='TE'):
    '''
    Finds the second derivative of neff at given wavelength

    USAGE:
	d = neff_derivative(wl, nridge)

    INPUT:
	wl	- wavelength
	nridge	- ...


    OUTPUT:
        numpy float neff''

    '''
    n = 2 #number of extrapolation levels
    wl_step = 0.001 #Initial step size
    def f(wl):
        return neff_derivative(wl, nridge, nbox, nclad, w, h, hslab, mode)

    return analysis.derivative(f, wl, n, wl_step)

def ng():
    '''
    Group index
    '''
    pass

def gvd():
    '''
    GVD
    '''
    pass

###############################################################################
###############################################################################
def _test_():
    '''
    Test function for module  
    '''
    wl = 1
    nridge = 2.2
    nbox = 1.4
    hslab = 0.45
    ne = neff_ridge(wl, nridge, nbox, hslab=hslab)
    np = neff_derivative(wl,nridge,nbox, hslab=hslab)
    npp = neff_2derivative(wl,nridge,nbox, hslab=hslab)
    ng = ne - np
    print(ng)
    print(c/ng * 1e-8)
    print(npp)

if __name__ == '__main__':
    _test_()