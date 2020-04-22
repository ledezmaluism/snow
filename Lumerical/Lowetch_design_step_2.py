# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish

Sweep width and bending radius...
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

wavelength = 1 #pump

theta = 60
wg_length = 10

material_substrate = "SiO2_analytic"
#material_substrate = "Sapphire_analytic"
material_thinfilm = "LN_analytic_MgO_doped_xne"

w_ridge_list = np.arange(0.5,2.01,0.1)
bending_list = np.arange(1000,1001,100) #in microns
h_etch = 0.25
h_LN = 0.7

n1 = w_ridge_list.size
n2 = bending_list.size

neff = np.empty([n1,n2])
loss = np.empty([n1,n2])
neff[:] = np.nan
loss[:] = np.nan

w_slab = 10*wavelength + 2*np.amax(w_ridge_list)
h_margin = 4*wavelength
h_substrate = 4*wavelength
meshsize = wavelength/10
finemesh = wavelength/50

TE_tepf_cutoff = 0.8

for kb in range(n2):
    bending = float(bending_list[kb])*1e-6
    for kw in range(n1):
        w_ridge = float(w_ridge_list[kw])
        
        lumpy.draw_wg(mode, material_thinfilm, material_substrate,
              h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
        lumpy.add_fine_mesh(mode, finemesh, h_LN, w_ridge, x_factor=1.5, y_factor=1.5)
        lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                         w_slab, wg_length, h_margin)
        
        mode.set("bent waveguide",1)
        mode.set("bend radius", bending)
        
        neff_result, tepf = lumpy.solve_mode(mode, wavelength, nmodes=20)
        index_TE_mode = np.where(tepf>TE_tepf_cutoff)
        n_TE_modes = (tepf>TE_tepf_cutoff).sum()
        if n_TE_modes==1:
            m = int(index_TE_mode[0])
            neff[kw,kb] = neff_result[m]
            loss[kw,kb] = mode.getdata("FDE::data::mode"+str(m+1),"loss")
        elif n_TE_modes>1:
            break #Higher order mode found
        

#Save data to file
#timestamp = str(round(time.time()))
#data_filename = 'LNoI_wl_%.1f_bendingloss_' %(wavelength)
#data_filename = data_filename.replace('.','p')
#data_filename += timestamp
#np.savez(data_filename, neff=neff, loss=loss, h_etch=h_etch, bending_list=bending_list,
#         w_ridge_list=w_ridge_list, h_LN=h_LN, wavelength=wavelength, 
#         theta=theta, h_substrate=h_substrate, w_slab=w_slab,
#         wg_length=wg_length, h_margin=h_margin, mesh_size=meshsize,
#         finemesh=finemesh, material_substrate=material_substrate,
#         material_thinfilm=material_thinfilm)
#
#mode.close()

#This is how you read the data
#data = np.load('GVD_wl_3um_hLN_thick_1p5um.npz')
#data.files
#data['GVD']
#x, y, Ex, Ey, Ez = lumpy.get_mode(mode, 1)
#Eabs = np.sqrt(x**2 + Ey**2 + Ex**2)
#lumpy.plot_2D_mode(Ex, x, y, h_LN, h_substrate, h_etch, w_ridge, w_slab, theta)