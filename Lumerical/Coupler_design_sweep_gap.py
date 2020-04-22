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

from scipy.constants import pi

if 'mode' not in vars():
    #lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/MODE/api/python/lumapi.py")
    lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2020a/api/python/lumapi.py")
    mode = lumapi.MODE("Template_Luis.lms")

h_LN = 0.7
h_etch = 0.35
#w_ridge = 1.5
w_ridge = 1.4
h_slab = h_LN - h_etch
print('slab = %0.1f nm' %(h_slab*1e3))

theta = 60
wg_length = 10
w_sidewall = h_etch/np.tan(theta*pi/180)
w_total = 2*w_sidewall + w_ridge

#Coupler gap at the base
gap_list = np.arange(0.5, 1.01, 0.1)
#gap_list = np.arange(0.1, 0.21, 0.1)

material_substrate = "SiO2_analytic"
#material_substrate = "Sapphire_analytic"
#material_thinfilm = "LN_analytic_undoped_xne"
#material_thinfilm = "LN_analytic_undoped_zne"
material_thinfilm = "LN_analytic_MgO_doped_xne"

def get_delta_vs_width(wavelength):
    delta_n = np.empty([gap_list.size])
    print('Wavelength = %0.1f um' %(wavelength))
    for kg in range(gap_list.size):
        gap = gap_list[kg]
        print('Calculating with gap = %0.1f nm' %(gap*1e3))
        mode.switchtolayout()
        mode.deleteall()
        w_slab = 10*wavelength + 2*w_ridge
        h_margin = 4*wavelength
        h_substrate = 4*wavelength
        meshsize = wavelength/10
        finemesh = wavelength/50
        lumpy.draw_substrate(mode, material_thinfilm, material_substrate, h_LN, h_substrate, 
                           h_etch, w_slab, wg_length, x0=0)
        lumpy.draw_ridge(mode, material_thinfilm, h_LN, h_etch, w_ridge, theta, wg_length, 
                       x0=w_total/2+gap/2, name='wg1')
        lumpy.draw_ridge(mode, material_thinfilm, h_LN, h_etch, w_ridge, theta, wg_length, 
                       x0=-w_total/2-gap/2, name='wg2')
        lumpy.add_fine_mesh(mode, finemesh, h_LN, 2*w_total+gap, x_factor=1.2, y_factor=1.5)
        mode.set('name','Outer mesh')
        lumpy.add_fine_mesh_lowlevel(mode, finemesh/2, 0, h_slab+h_etch/2, (gap+2*w_sidewall)*1.4, h_etch*1.4)
        mode.set('name','Inner mesh')
        
        lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                                 w_slab, wg_length, h_margin)
        neff, TEPF = lumpy.solve_mode(mode, wavelength, nmodes=20)
    
        '''
        Get first two TE modes,
        these are the even and odd modes
        '''
        nmodes = neff.size
        even = True
        for km in range(nmodes):
            if TEPF[km]>0.8:
                if even:
                    neff_sym = neff[km]
                    even = False
                else:
                    neff_asym = neff[km]
                    break #Found both modes, get out
        
        delta_n[kg] = neff_sym - neff_asym
    return delta_n

wl_pump = 1
wl_signal = 2

# wl_pump = 0.775 
# wl_signal = 2*wl_pump

deltaN_pump = get_delta_vs_width(wl_pump)
deltaN_signal = get_delta_vs_width(wl_signal)

#Save data to file
data_filename = 'Coupler_sweepgap_width_%.1f_pump_%.1f' %(w_ridge, wl_pump)
data_filename = data_filename.replace('.','p')
np.savez(data_filename, deltaN_pump=deltaN_pump, deltaN_signal=deltaN_signal,
         w_ridge=w_ridge, wl_pump=wl_pump, wl_signal=wl_signal, 
         gap_list=gap_list, h_etch=h_etch, 
         material_substrate=material_substrate, 
         material_thinfilm=material_thinfilm)