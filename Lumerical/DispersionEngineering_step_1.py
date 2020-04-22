# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish

Parameters:
    GVD[width, h_slab, h_LN] - sweep all three, fixed wavelength
"""

import LNOI_functions as lumpy
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import imp
import time

from scipy.constants import pi, c
lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/MODE/api/python/lumapi.py")

mode = lumapi.MODE("Template_Luis.lms")

wavelength = 2

theta = 60
wg_length = 10

material_substrate = "SiO2_analytic"
#material_substrate = "Sapphire_analytic"
material_thinfilm = "LN_analytic_undoped_xne"

w_ridge_list = np.arange(0.5,2.01,0.1)
h_slab_list = np.arange(0,0.61,0.1)

#h_LN_list = np.arange(1,1.41,0.1)
#h_LN_list = np.arange(1.2,1.41,0.1)
h_LN_list = np.arange(0.7,0.71,0.1)

n1 = w_ridge_list.size
n2 = h_slab_list.size
n3 = h_LN_list.size

neff = np.empty([n1,n2,n3])
Vg = np.empty([n1,n2,n3])
GVD = np.empty([n1,n2,n3])
neff[:] = np.nan
Vg[:] = np.nan
GVD[:] = np.nan

w_slab = 10*wavelength + 2*np.amax(w_ridge_list)
h_margin = 4*wavelength
h_substrate = 4*wavelength
meshsize = wavelength/10
finemesh = wavelength/50

TE_tepf_cutoff = 0.8

for kLN in range(n3):
    h_LN = h_LN_list[kLN]
    for ke in range(n2): 
        h_etch = h_LN - h_slab_list[ke]
        for kw in range(n1):
            w_ridge = float(w_ridge_list[kw])
            
            lumpy.draw_wg(mode, material_thinfilm, material_substrate,
                  h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
            lumpy.add_fine_mesh(mode, finemesh, h_LN, w_ridge, x_factor=1.5, y_factor=1.5)
            lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                             w_slab, wg_length, h_margin)
            neff_result, tepf = lumpy.solve_mode(mode, wavelength, nmodes=6)
            index_TE_mode = np.where(tepf>TE_tepf_cutoff)
            n_TE_modes = (tepf>TE_tepf_cutoff).sum()
            if n_TE_modes==1:
                m = int(index_TE_mode[0])
                neff[kw,ke,kLN] = neff_result[m]
                Vg[kw,ke,kLN], GVD[kw,ke,kLN] = lumpy.dispersion_analysis(mode, wavelength, m+1)
            elif n_TE_modes>1:
                break #Higher order mode found
        
            
#plt.imshow(GVD, origin='lower', aspect='equal', interpolation='bicubic')

#Save data to file
timestamp = str(round(time.time()))
data_filename = 'LNoI_wl_%.1f_' %(wavelength)
data_filename = data_filename.replace('.','p')
data_filename += timestamp
np.savez(data_filename, GVD=GVD, Vg=Vg, neff=neff, h_slab_list=h_slab_list,
         w_ridge_list=w_ridge_list, h_LN_list=h_LN_list, wavelength=wavelength, 
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