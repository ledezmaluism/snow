# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish

Get poling period vs ridge width
"""

import LNOI_functions as lumpy
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import imp
import time

from scipy.constants import pi, c
lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2020a/api/python/lumapi.py")

mode = lumapi.MODE("Template_Luis.lms")

theta = 60
wg_length = 10

material_substrate = "SiO2_analytic"
#material_substrate = "Sapphire_analytic"
#material_thinfilm = "LN_analytic_undoped_xne"
material_thinfilm = "LN_analytic_MgO_doped_xne"

wavelength_list = np.arange(1,2.1,1)
#wavelength_list = np.array([1.55/2,1.55])
w_ridge_list = np.arange(1.0,2.0,0.1)
# w_ridge_list = np.array([1.2])
h_etch = 0.35
h_LN = 0.7

n1 = w_ridge_list.size
n2 = wavelength_list.size

neff = np.empty([n1,n2])
neff[:] = np.nan

TE_tepf_cutoff = 0.8

for kf in range(n2):
    wavelength = float(wavelength_list[kf])
    w_slab = 10*wavelength + 2*np.amax(w_ridge_list)
    h_margin = 4*wavelength
    h_substrate = 4*wavelength
    meshsize = wavelength/10
    finemesh = wavelength/50
    for kw in range(n1):
        w_ridge = float(w_ridge_list[kw])
        
        lumpy.draw_wg(mode, material_thinfilm, material_substrate,
              h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
        lumpy.add_fine_mesh(mode, finemesh, h_LN, w_ridge, x_factor=1.5, y_factor=1.5)
        lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                         w_slab, wg_length, h_margin)
        
        neff_result, tepf = lumpy.solve_mode(mode, wavelength, nmodes=10)
        index_TE_mode = np.where(tepf>TE_tepf_cutoff)
        index_TE_mode = index_TE_mode[0]
        n_TE_modes = (tepf>TE_tepf_cutoff).sum()
        if n_TE_modes>0:
            m = int(index_TE_mode[0])
            neff[kw,kf] = neff_result[m]
        # elif n_TE_modes>1:
        #     break #Higher order mode found
        

#Save data to file
timestamp = str(round(time.time()))
data_filename = 'LNoI_wl_%.1f_phasemismatch_' %(wavelength)
data_filename = data_filename.replace('.','p')
data_filename += timestamp
np.savez(data_filename, neff=neff, h_etch=h_etch, wavelength_list=wavelength_list,
          w_ridge_list=w_ridge_list, h_LN=h_LN,
          theta=theta, h_substrate=h_substrate, w_slab=w_slab,
          wg_length=wg_length, h_margin=h_margin, mesh_size=meshsize,
          finemesh=finemesh, material_substrate=material_substrate,
          material_thinfilm=material_thinfilm)

mode.close()

#This is how you read the data
#data = np.load('GVD_wl_3um_hLN_thick_1p5um.npz')
#data.files
#data['GVD']
#x, y, Ex, Ey, Ez = lumpy.get_mode(mode, 1)
#Eabs = np.sqrt(x**2 + Ey**2 + Ex**2)
#lumpy.plot_2D_mode(Ex, x, y, h_LN, h_substrate, h_etch, w_ridge, w_slab, theta)