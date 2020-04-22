# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish

Finds zeroGVD parameters for a given thin-film.
I creates a vector [width] from a vector [hslab], which combination gives zeroGVD

Inputs are:
    wavelength
    thin-film
    hslab vector
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

wavelength = 3
h_LN = 1.3

theta = 60
wg_length = 10

material_substrate = "Sapphire_analytic"
material_thinfilm = "LN_analytic_undoped_xne"

h_slab = np.arange(0,0.61,0.1)
n = h_slab.size
width = np.empty([n])
width[:] = np.nan

w_slab = 10*wavelength + 2
h_margin = 4*wavelength
h_substrate = 4*wavelength
meshsize = wavelength/10
finemesh = wavelength/50

def gvd_func(w):
    #Calculates gvd as function of width
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

w_pos =  0.9 #GVD should be positive at this width
w_neg = 1.5 #GVD should be positive at this width
for ks in range(n):
    h_etch = h_LN - h_slab[ks]
    w_zero = optimize.brentq(gvd_func, w_pos, w_neg, xtol=1e-3, rtol=1e-3)
    width[ks] = w_zero

mode.close()
            
#plt.imshow(GVD, origin='lower', aspect='equal', interpolation='bicubic')

#Save data to file
timestamp = str(round(time.time()))
data_filename = 'LNoS_wl_%.1f_h_LN_%0.1f_' %(wavelength, h_LN)
data_filename = data_filename.replace('.','p')
data_filename += timestamp
np.savez(data_filename, h_slab=h_slab,
         width=width, h_LN=h_LN, wavelength=wavelength, 
         theta=theta, h_substrate=h_substrate, w_slab=w_slab,
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