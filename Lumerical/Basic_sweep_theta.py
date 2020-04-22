# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish

Sweep width at single frequency
Get first 4 modes and their effective indexes
"""

import LNOI_functions as lumpy
import numpy as np
import matplotlib.pyplot as plt
import imp
import time

from scipy.constants import pi, c
lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/MODE/api/python/lumapi.py")

mode = lumapi.MODE("Template_Luis.lms")

wavelength = 1.0 #pump

wg_length = 10

material_substrate = "SiO2_analytic"
#material_substrate = "Sapphire_analytic"
material_thinfilm = "LN_analytic_MgO_doped_xne"

theta_list = np.arange(50,80.1,2)
w_ridge = 1.5
h_etch = 0.25
h_LN = 0.7

n1 = theta_list.size

n_modes = 4
neff = np.empty([n1, n_modes])
tepf = np.empty([n1, n_modes])
neff[:] = np.nan
tepf[:] = np.nan


w_slab = 10*wavelength + 2*np.amax(w_ridge)
h_margin = 4*wavelength
h_substrate = 4*wavelength
meshsize = wavelength/10
finemesh = wavelength/50

TE_tepf_cutoff = 0.8

for kt in range(n1):
    theta = float(theta_list[kt])
    
    lumpy.draw_wg(mode, material_thinfilm, material_substrate,
          h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
    lumpy.add_fine_mesh(mode, finemesh, h_LN, w_ridge, x_factor=1.5, y_factor=1.5)
    lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                     w_slab, wg_length, h_margin)
    
    neff_result, tepf_result = lumpy.solve_mode(mode, wavelength, nmodes=20)
    
    print(neff_result)
    for n in range(neff_result.size):
        neff[kt,n] = neff_result[n]
        tepf[kt,n] = tepf_result[n]
        if n+1>=n_modes:
            break #Higher order mode found
        

#Save data to file
timestamp = str(round(time.time()))
data_filename = 'LNoI_wl_%.1f_theta_sweep_' %(wavelength)
data_filename = data_filename.replace('.','p')
data_filename += timestamp
np.savez(data_filename, neff=neff, tepf=tepf, h_etch=h_etch,
         w_ridge=w_ridge, h_LN=h_LN, wavelength=wavelength, 
         theta_list=theta_list, h_substrate=h_substrate, w_slab=w_slab,
         wg_length=wg_length, h_margin=h_margin, mesh_size=meshsize,
         finemesh=finemesh, material_substrate=material_substrate,
         material_thinfilm=material_thinfilm)
#
#mode.close()

#This is how you read the data
#data = np.load('GVD_wl_3um_hLN_thick_1p5um.npz')
#data.files
#data['GVD']
#x, y, Ex, Ey, Ez = lumpy.get_mode(mode, 1)
#Eabs = np.sqrt(x**2 + Ey**2 + Ex**2)
#lumpy.plot_2D_mode(Ex, x, y, h_LN, h_substrate, h_etch, w_ridge, w_slab, theta)