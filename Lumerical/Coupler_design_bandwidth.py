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
    lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2019b/api/python/lumapi.py")
    mode = lumapi.MODE("Template_Luis.lms")

#Coupler parameters
gap = 1.0 #This is defined on the bottom
L = 250
w_ridge = 1.5
# w_ridge = 1.0

#Technology
h_LN = 0.7
h_etch = 0.25
h_slab = h_LN - h_etch
print('slab = %0.1f nm' %(h_slab*1e3))
# theta = 60
theta = 45
wg_length = 10
w_sidewall = round(h_etch/np.tan(theta*pi/180),2)
w_total = 2*w_sidewall + w_ridge

gap_top = gap+ 2*w_sidewall


material_substrate = "SiO2_analytic"
#material_substrate = "Sapphire_analytic"
#material_thinfilm = "LN_analytic_undoped_xne"
#material_thinfilm = "LN_analytic_undoped_zne"
material_thinfilm = "LN_analytic_MgO_doped_xne"

def coupling(L, delta_n, wl):
    C = np.sin(pi*L*delta_n/wl)**2
    return C

def get_delta(wavelength):
    print('Wavelength = %0.3f um' %(wavelength))
    mode.switchtolayout()
    mode.deleteall()
    w_slab = 10*wavelength + 2*w_ridge
    h_margin = 4*wavelength
    h_substrate = 4*wavelength
    meshsize = wavelength/20
    finemesh = wavelength/40
#    meshsize = wavelength/40
#    finemesh = wavelength/80
    lumpy.draw_substrate(mode, material_thinfilm, material_substrate, h_LN, h_substrate, 
                       h_etch, w_slab, wg_length, x0=0)
    lumpy.draw_ridge(mode, material_thinfilm, h_LN, h_etch, w_ridge, theta, wg_length, 
                   x0=w_total/2+gap/2, name='wg1')
    lumpy.draw_ridge(mode, material_thinfilm, h_LN, h_etch, w_ridge, theta, wg_length, 
                   x0=-w_total/2-gap/2, name='wg2')
    lumpy.add_fine_mesh(mode, finemesh, h_LN, 2*w_total+gap, x_factor=1.2, y_factor=1.5)
    mode.set('name','Outer mesh')
#    lumpy.add_fine_mesh_lowlevel(mode, finemesh/2, 0, h_slab+h_etch/2, (gap+2*w_sidewall)*1.4, h_etch*1.4)
#    mode.set('name','Inner mesh')
    
    lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                             w_slab, wg_length, h_margin)
    neff, TEPF = lumpy.solve_mode(mode, wavelength, nmodes=20)

    '''
    Get first two TE modes,
    these are the even and odd modes
    '''
    nmodes = neff.size
    even = True
    found = False
    for km in range(nmodes):
        if TEPF[km]>0.8:
            if even:
                neff_sym = neff[km]
                even = False
            else:
                neff_asym = neff[km]
                found = True
                break #Found both modes, get out
    
    if found:
        delta_n = neff_sym - neff_asym
    else:
        delta_n = np.nan
    return delta_n

#wl_start = 0.7 
wl_start = 0.7
wl_stop = 2.2+0.1
#wl_stop = 1.2
wl_step = 0.01
wl = np.arange(wl_start, wl_stop, wl_step)

C = np.empty([wl.size])
for kf in range(wl.size):
    delta = get_delta(wl[kf])
    C[kf] = coupling(L, delta, wl[kf])

#Save data to file
data_filename = 'Coupler_bandwidth_w_%.2f_gap_%.2f_L_%.2f' %(w_ridge, gap, L)
data_filename = data_filename.replace('.','p')
np.savez(data_filename, h_LN=h_LN, h_etch=h_etch, theta=theta,
         w_ridge=w_ridge, L=L, gap=gap, C=C,
         material_substrate=material_substrate, 
         material_thinfilm=material_thinfilm)

#Plot
fig, ax = plt.subplots()
ax.plot(wl, C*100, label='Coupling Factor')
#ax.legend()
ax.grid(True)
ax.set_xlabel('Wavelength ($\mu$m)')
ax.set_ylabel('Coupling Factor (%)')
#ax.axis([0, max(L), 0, 100])
ax.set_title('Gap = %0.1f $\mu$m, Width = %0.1f $\mu$m, Length = %0.1f $\mu$m' %(gap, w_ridge, L))