# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:57:17 2019
@author: Luis Ledezma

Module to calculate waveguide parameters semi-analytically
(using mainly the effective index method)
"""

import numpy as np
import time
from numpy.fft import fftshift
from scipy.optimize import brentq
from scipy.constants import pi, c
from scipy import interpolate

from . import materials
from . import util
from . import nlo
from . import pulses

class waveguide:

    def __init__(self, w_top=1e-6, h_thinfilm=700e-9, h_etch=350e-9, theta=60,
                 tf_material = 'LN_MgO_e',
                 box_material = 'SiO2',
                 clad_material = 'Air',
                 behavioral = False,
                 z_wl=1e-6, wl_1=1e-6, wl_2=2e-6, GVM=0, delta_n=0.2, n_f0=2.0, wl_f0=None, c1=10**(-55),
                 external = False,
                 wl_array = np.array([1e-6]), n_array=np.array([1.0]) 
                 ):
        

        
        self.GVD = self.beta2
        
        if behavioral:
            self.neff = self.neff_behavioral
            self.beh_z_wl = z_wl
            self.beh_wl_1 = wl_1
            self.beh_wl_2 = wl_2
            self.beh_GVM = GVM
            self.beh_delta_n = delta_n
            self.beh_n_f0= n_f0
            self.wl_f0 = wl_f0
            self.c1 = c1
        elif external:
            self.tck = interpolate.splrep( wl_array, n_array, s=0 )
            self.neff = self.neff_interp
        else:
            self.neff = self.neff_physical
            #validation
            # etch = h_ridge - h_slab
            w_base = w_top + 2*h_etch/np.tan(theta*pi/180)
            if h_etch*(w_base-w_top)<0:
                raise ValueError("Something wrong with this geometry")
    
            #Attributes given
            self.w_top = w_top
            self.h_etch = h_etch
            self.h_thinfilm = h_thinfilm
            self.theta = theta
            self.tf_material = tf_material
            self.box_material = box_material
            self.clad_material = clad_material
        
            #Attributes calculated
            self.w_base = w_base
            self.h_slab = h_thinfilm - h_etch
        
    def neff_physical(self, wl, mode='TE', T=24.5):
        um = 1e-6
        w = (self.w_top + self.w_base)/2
        h = self.h_thinfilm
        hslab = self.h_slab
            
        if np.isscalar(wl):
            wl = np.asarray([wl])
            
        neff = np.zeros(wl.shape)
        for kw in range(wl.size):
            nridge = materials.refractive_index(self.tf_material, wl[kw]/um, T=T)
            nbox = materials.refractive_index(self.box_material, wl[kw]/um, T=T)
            nclad = materials.refractive_index(self.clad_material, wl[kw]/um, T=T)

            neff[kw] = neff_ridge(wl[kw], nridge, nbox, nclad, w, h, hslab, mode)
        return neff
    
    # def add_narray(self, wl_abs, T=24.5):
    #     neff = np.zeros(wl_abs.shape)
    #     for kw in range(wl_abs.size):
    #         neff[kw] = self.neff(wl_abs[kw], T=T)
    #     self.neff_array = neff
    #     self.beta = 2 * pi * self.neff_array / wl_abs
        
    #     f_abs = c / wl_abs
    #     df = f_abs[1] - f_abs[0]
    #     self.beta_1 = fftshift(np.gradient(fftshift(self.beta), 2*pi*df))
    
    def neff_interp(self, wl, T=25):
        return interpolate.splev(wl, self.tck, der=0)
    
    def neff_behavioral(self, wl, T=None):
        '''
        This method is equivalent to "neff_physical", but instead of using 
        the physical waveguide parameters to calculate the behavior of the 
        waveguide, it defines the behavior directly 
        (so it defines a black box model of the waveguide)
        Parameters
        ----------
        wl : TYPE
            Wavelength array
        z_wl : TYPE
            Zero GVD wavelength
        wl_1 : TYPE
            First wavelegnth for GVM calculation
        wl_2 : TYPE
            Second wavelegnth for GVM calculation
        GVM : TYPE, optional
            GVM between wl_1 and wl_2. The default is 0.
        n_ff : TYPE, optional
            Refractive index at wl_ff. The default is 2.0.
        wl_ff : TYPE, optional
            Fundamental frequency for refractive index. The default is None.

        Returns
        -------
        Waveguide object.

        '''          
        z_wl = self.beh_z_wl
        wl_1 = self.beh_wl_1
        wl_2 = self.beh_wl_2
        GVM = self.beh_GVM
        delta_n = self.beh_delta_n
        n_f0 = self.beh_n_f0
        wl_f0 = self.wl_f0
        
        
        if wl_f0 == None:
            wl_f0 = ( wl_1 + wl_2 )/2
        
        #z1, w1, w2 are wavelengths
        z1 = 2*pi*c/z_wl
        omega_1 = 2*pi*c/wl_1
        omega_2 = 2*pi*c/wl_2        
        omega0_f0 = 2*pi*c/wl_f0
        
        #Frequency array:
        omega = 2*pi*c/wl
        
        #Constants
        # c1 = 10 **(-55) #s^4/m
        c1 = self.c1
        
        # First we want to determine the remianing free parameters
        z2_num = 6*GVM - 2*c1*(omega_2**3-omega_1**3)+3*c1*z1*(omega_2**2-omega_1**2)
        z2_den = 3*c1*(2*z1-omega_2-omega_1)*(omega_2-omega_1)
        z2 = z2_num/z2_den
        
        
        c3 = (omega_2**3-omega_1**3)/12 - (z1+z2)*(omega_2**2-omega_1**2)/6 + z1*z2*(omega_2-omega_1)/2
        c3 = delta_n/c - c1*c3 
        c3 = c3 / (1/omega_2 - 1/omega_1)
        c2 = n_f0/c - c1*(omega0_f0**3/12 - (z1+z2)*(omega0_f0**2)/6 +z1*z2*omega0_f0/2) - c3/omega0_f0
        
        # Now we can solve for the remaing parameters
        # beta_2 = c1*(omega**2 -(z1+z2)*omega + z1*z2)
        # beta_1 = c1*(omega**3/3 - (z1+z2)*(omega**2)/2 + z1*z2*omega) + c2
        # beta = c1*(omega**4/12 - (z1+z2)*(omega**3)/6 + z1*z2*(omega**2)/2) + c2*omega + c3
        n = c*( c1*(omega**3/12 - (z1+z2)*(omega**2)/6 + z1*z2*omega/2) + c2 + c3/omega)
        
        return n
        
    def set_length(self, L):
        self.L = L
        
    def set_loss(self, alpha):
        self.alpha = alpha
    
    def beta(self, wl, T=24.5):
        beta = 2 * pi * self.neff(wl, T=T) / wl
        return beta
    
    def beta1(self, wl, T=24.5):
        
        if np.isscalar(wl):
            wl = np.asarray([wl])
        n = 2 #number of extrapolation levels
        wl_step = 1e-9 #Initial step size
        b1 = np.zeros_like(wl)
        
        def neff_T(wl):
            return self.neff(wl, T=T)
        
        for kw in range(wl.size):
            dndl = util.derivative(neff_T, wl[kw], n, wl_step) #Still need to add temp here
            neff = neff_T(wl[kw])
            b1[kw] = (neff - wl[kw] * dndl)/c
        return b1
    
    def beta2(self, wl):
        if np.isscalar(wl):
            wl = np.asarray([wl])
        n = 2 #number of extrapolation levels
        wl_step = 1e-9 #Initial step size
        gvd = np.zeros_like(wl)
        for kw in range(wl.size):
            db1dl = util.derivative(self.beta1, wl[kw], n, wl_step)
            gvd[kw] = - wl[kw]**2 / (2*pi*c) * db1dl
        return gvd
    
    def propagate_linear(self, pulse):
        t = pulse.t
        x = pulse.a
        wl_ref= pulse.wl0
        
        x = x * np.exp(-self.alpha*self.L) * np.exp(-1j * self.beta * self.L)
        
        output_pulse = pulses.pulse(t, x, wl_ref)
        return output_pulse
    
    def add_poling(self, poling):
        '''
        Adds poling function to waveguide. Poling is a function of z
        '''
        self.poling = poling
        
    def set_nonlinear_coeffs(self, N, X0):
        self.N = N
        self.X0 = X0
        
    def nonlinear_coupling(self, z):
        return self.poling(z) * self.X0 / (4*self.N)
    
    def propagate_NEE(self, pulse, v_ref=None, 
                         verbose=True, zcheck_step = 0.5e-3,
                         z0 = 0, T=24.5, Kg=0, Qnoise=False):
        #Timer
        tic_total = time.time()
         
        #Get pulse info
        f0 = pulse.f0
        Omega  = pulse.Omega
        
        beta = self.beta( pulse.wl, T=T)

        if v_ref == None:
            vg = 1/self.beta1( pulse.wl )
            v_ref = vg[0]
        
        beta_ref = beta[0]
        beta_1_ref = 1/v_ref
        D = beta - beta_ref - Omega/v_ref - 1j*self.alpha/2

        omega_ref = 2*pi*f0
        omega_abs = omega_ref + Omega
        
        def k(z): #nonlinear coupling
            if Kg == 0:
                p = self.poling(z)
            else:
                p = 2/pi #first order QPM
            return p * self.X0 * omega_abs / (4 * self.N)

        [a, a_evol] = nlo.NEE(t = pulse.t, 
                          x = pulse.a,
                          Omega = Omega,
                          f0 = pulse.f0,
                          L = self.L,
                          D = D, 
                          b0 = beta_ref, 
                          b1_ref = beta_1_ref, 
                          k = k,
                          z0 = z0,
                          verbose = verbose,
                          Kg = Kg,
                          Qnoise = Qnoise)
        
        tdelta = time.time() - tic_total

        if verbose:
            print('Total time = %0.1f s' %(tdelta))
            print()
            
        output_pulse = pulses.pulse(pulse.t, a, pulse.wl0, pulse.frep)
        
        return output_pulse, a_evol

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
        neff = neff_symmetric_slab(neff_slab_te0, neff_ridge_te0, w, wl, 'TM even')
        # neff = neff_asymmetric_slab(neff_slab_te0, neff_ridge_te0, neff_slab_te0, w, wl, 'TM even')
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

    return util.derivative(f, wl, n, wl_step)


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

    return util.derivative(f, wl, n, wl_step)

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