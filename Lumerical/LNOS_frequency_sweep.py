# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish
"""

import LNOI_functions as lumpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imp
import time

from scipy.constants import pi, c

lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2020a/api/python/lumapi.py")
MODE = lumapi.MODE("Template_Luis.lms")

'''
Units
'''
um = 1e-6
nm = 1e-9
THz = 1e12

start_time = time.time()

h_LN = 0.7
h_etch = 0.250
w_ridge = 0.8
h_slab = h_LN - h_etch

theta = 45
wg_length = 10
#material_substrate = "Sapphire_analytic"
material_substrate = "SiO2_analytic"
#material_thinfilm = "LN_analytic_undoped_xne"
material_thinfilm = "LN_analytic_MgO_doped_xne"

def setupvariables(wavelength):
    w_slab = 10*wavelength + 2
    h_margin = 4*wavelength
    h_substrate = 4*wavelength
    meshsize = wavelength/10
    finemesh = wavelength/50
    return w_slab, h_margin, h_substrate, meshsize, finemesh

freq_start = 140 #THz
freq_stop = 310  #THz
freq_step = 5 #THz
freqs = np.arange(freq_start, freq_stop, freq_step)
n =  freqs.size
neff = np.empty([n])
beta = np.empty([n])
vg = np.empty([n])
gvd = np.empty([n])

for kf in range(n):
    wavelength = float(c/freqs[kf])
    w_slab, h_margin, h_substrate, meshsize, finemesh = setupvariables(wavelength)
    lumpy.draw_wg(mode, material_thinfilm, material_substrate,
                  h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
    lumpy.add_fine_mesh(mode, finemesh, h_LN, w_ridge, x_factor=1.5, y_factor=1.5)
    lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                             w_slab, wg_length, h_margin)
    neff_vector, tepf = lumpy.solve_mode(mode, wavelength, nmodes=6)
    for m in range(tepf.size):
        if tepf[m]>0.5:
            neff[kf] = neff_vector[m]
            vg[kf], gvd[kf] = lumpy.dispersion_analysis(mode, wavelength, m+1)
            break
    
beta = 2*pi*freqs*neff/c

##Save data to file
timestamp = str(round(time.time()))
data_filename = 'LNoI_freq_sweep_h_LN_%0.1f_width_%0.1f_' %(h_LN, w_ridge)
data_filename = data_filename.replace('.','p')
data_filename += timestamp
np.savez(data_filename, h_slab=h_slab, gvd=gvd, vg=vg, neff=neff, beta=beta,
         width=w_ridge, h_LN=h_LN, freqs=freqs, 
         theta=theta, wg_length=wg_length, material_substrate=material_substrate,
         material_thinfilm=material_thinfilm)

elapsed_time = time.time() - start_time