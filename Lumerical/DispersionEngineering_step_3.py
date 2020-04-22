# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish

Finds GVM at zeroGVD for a given thin-film.
Takes a pair of zeroGVD design vectors [width] and [hslab]
and calculates the GVM between pump and signal

Inputs are:
    wavelength pump and signal
    thin-film
    hslab vector
    width vector
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


'''
Import vectors [width] and [hslab]
'''
data = np.load('data\LNoS_zeroGVD_design_wl_3p0_h_LN_1p2_1556255055.npz')
#data = np.load('data\LNoS_zeroGVD_design_wl_3p0_h_LN_1p3_1556309685.npz')

signal = float(data['wavelength'])
h_LN = float(data['h_LN'])
h_slab = data['h_slab']
width = data['width']

theta = float(data['theta'])
wg_length = float(data['wg_length'])
material_substrate = str(data['material_substrate'])
material_thinfilm = str(data['material_thinfilm'])
'''
---------------------------------------------------------------------------
'''
pump = signal/2

def setupvariables(wavelength):
    w_slab = 10*wavelength + 2
    h_margin = 4*wavelength
    h_substrate = 4*wavelength
    meshsize = wavelength/10
    finemesh = wavelength/50
    return w_slab, h_margin, h_substrate, meshsize, finemesh
#
def gvm_func(wp, ws, w, h_etch):
    '''
    Calculates gvd at pump and gvm between signal and pump
    inputs:
        wavelength pump
        wavelength signal
        width
        h_etch
    '''
    #Signal first
    wavelength = ws
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
    
    #Pump next
    wavelength = wp
    w_slab, h_margin, h_substrate, meshsize, finemesh = setupvariables(wavelength)
    lumpy.draw_wg(mode, material_thinfilm, material_substrate,
          h_LN, h_substrate, h_etch, w, w_slab, theta, wg_length)
    lumpy.add_fine_mesh(mode, finemesh, h_LN, w, x_factor=1.5, y_factor=1.5)
    lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                     w_slab, wg_length, h_margin)
    neff_result, tepf = lumpy.solve_mode(mode, wavelength, nmodes=10)
    for m in range(tepf.size):
        if tepf[m]>0.5:
            vg_p, gvd_p = lumpy.dispersion_analysis(mode, wavelength, m+1)
            break
        
    gvm = 1/vg_p - 1/vg_s
    return gvd, vg_s, vg_p, gvm
#

n = h_slab.size
gvd = np.empty([n])
gvm = np.empty([n])
vg_p = np.empty([n])
vg_s = np.empty([n])
for ks in range(n):
    h_etch = h_LN - h_slab[ks]
    gvd[ks], vg_s[ks], vg_p[ks], gvm[ks] = gvm_func(pump, signal, width[ks], h_etch)
#
mode.close()
#            
##Save data to file
timestamp = str(round(time.time()))
data_filename = 'LNoS_GVM_At_zeroGVD_signal_%.1f_h_LN_%0.1f_' %(signal, h_LN)
data_filename = data_filename.replace('.','p')
data_filename += timestamp
np.savez(data_filename, h_slab=h_slab, gvm=gvm, gvd=gvd, vg_p=vg_p, vg_s=vg_s,
         width=width, h_LN=h_LN, w_signal=signal, w_pump=pump, 
         theta=theta, wg_length=wg_length, material_substrate=material_substrate,
         material_thinfilm=material_thinfilm)
