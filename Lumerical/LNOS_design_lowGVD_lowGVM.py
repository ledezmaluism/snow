# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish

Finds zeroGVM parameters given a range of h_slab
Inputs are:
    wavelength at pump and signal
    thin-film
    hslab range
Output:
    Design -> hslab and width combination for zero GVD and GVM
    
"""

import LNOI_functions as lumpy
import numpy as np
import matplotlib.pyplot as plt
import imp
import time
from scipy import optimize

from scipy.constants import pi, c
lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/MODE/api/python/lumapi.py")

mode = lumapi.MODE("Template_Luis.lms")

w_s = 3
w_p = w_s/2
h_LN = 1.2
h1 = h_LN - 0.8 
h2 = h_LN - 0.7
w1 = 1.5
w2 = 1.7

theta = 60
wg_length = 10

material_substrate = "Sapphire_analytic"
material_thinfilm = "LN_analytic_undoped_xne"

h_slab = np.array([h1,h2])

def setupvariables(wavelength):
    w_slab = 10*wavelength + 2
    h_margin = 4*wavelength
    h_substrate = 4*wavelength
    meshsize = wavelength/10
    finemesh = wavelength/50
    return w_slab, h_margin, h_substrate, meshsize, finemesh

def gvd_func(w, h_etch):
    #Calculates gvd as function of width and h_etch
    wavelength = w_s
    w_slab, h_margin, h_substrate, meshsize, finemesh = setupvariables(wavelength)
    lumpy.draw_wg(mode, material_thinfilm, material_substrate,
          h_LN, h_substrate, h_etch, w, w_slab, theta, wg_length)
    lumpy.add_fine_mesh(mode, finemesh, h_LN, w, x_factor=1.5, y_factor=1.5)
    lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                     w_slab, wg_length, h_margin)
    neff_result, tepf = lumpy.solve_mode(mode, wavelength, nmodes=10)
    index_TE_mode = np.where(tepf>0.5)
    n_TE_modes = (tepf>0.5).sum()
    if n_TE_modes==1:
        m = int(index_TE_mode[0])
        Vg, gvd = lumpy.dispersion_analysis(mode, wavelength, m+1)
    return gvd

def zerogvd_w(h_etch):
    #Calculates zero gvd width for a specific h_etch
    w_zero = optimize.brentq(gvd_func, w1, w2, args=h_etch, xtol=1e-3, rtol=1e-3)
    return w_zero

def gvm_func(h):
    #Calculates gvm as a function of h_slab
    h_etch = h_LN - h
    
    print('Finding zero GVD width for h_etch=%0.2f um' %(h_etch))
    #Find width for zero GVD
    w = zerogvd_w(h_etch)
    print('Found width=%0.2f um' %(w))
    
    #Signal first
    wavelength = w_s
    w_slab, h_margin, h_substrate, meshsize, finemesh = setupvariables(wavelength)
    lumpy.draw_wg(mode, material_thinfilm, material_substrate,
          h_LN, h_substrate, h_etch, w, w_slab, theta, wg_length)
    lumpy.add_fine_mesh(mode, finemesh, h_LN, w, x_factor=1.5, y_factor=1.5)
    lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                     w_slab, wg_length, h_margin)
    neff_result, tepf = lumpy.solve_mode(mode, wavelength, nmodes=10)
    index_TE_mode = np.where(tepf>0.5)
    m = int(index_TE_mode[0])
    vg_s, gvd = lumpy.dispersion_analysis(mode, wavelength, m+1)
    print('Vg at signal = %0.2f mm/fs' %(vg_s))
    print('GVD at signal = %0.2f fs^2/mm' %(gvd))
    
    #Pump next
    wavelength = w_p
    w_slab, h_margin, h_substrate, meshsize, finemesh = setupvariables(wavelength)
    lumpy.draw_wg(mode, material_thinfilm, material_substrate,
          h_LN, h_substrate, h_etch, w, w_slab, theta, wg_length)
    lumpy.add_fine_mesh(mode, finemesh, h_LN, w, x_factor=1.5, y_factor=1.5)
    lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                     w_slab, wg_length, h_margin)
    neff_result, tepf = lumpy.solve_mode(mode, wavelength, nmodes=10)
    for m in range(tepf.size):
        if tepf[m]>0.5:
            vg_p, gvd = lumpy.dispersion_analysis(mode, wavelength, m+1)
            break
    
    print('Vg at pump = %0.4f mm/fs' %(vg_p))    
    gvm = (1/vg_p - 1/vg_s) #fs/mm
    print('GVM = %0.4f fs/mm \n' %(gvm))
    return gvm

h_zero = optimize.brentq(gvm_func, h1, h2, xtol=1e-3, rtol=1e-3)
h_etch = h_LN - h_zero
width = zerogvd_w(h_etch)
GVD = gvd_func(width, h_etch)

#mode.close()
            
#Save data to file
#timestamp = str(round(time.time()))
#data_filename = 'LNoS_wl_%.1f_h_LN_%0.1f_' %(wavelength, h_LN)
#data_filename = data_filename.replace('.','p')
#data_filename += timestamp
#np.savez(data_filename, h_slab=h_slab,
#         width=width, h_LN=h_LN, wavelength=wavelength, 
#         theta=theta, h_substrate=h_substrate, w_slab=w_slab,
#         wg_length=wg_length, h_margin=h_margin, mesh_size=meshsize,
#         finemesh=finemesh, material_substrate=material_substrate,
#         material_thinfilm=material_thinfilm)
