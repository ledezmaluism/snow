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


# np.savez(data_filename, h_LN=h_LN, h_etch=h_etch, theta=theta,
#          w_ridge=w_ridge, L=L, gap=gap, C=C,
#          material_substrate=material_substrate, 
#          material_thinfilm=material_thinfilm)

# data = np.load('data\design_OPO_250nm\Coupler_bandwidth_w_1p50_gap_1p00_L_250p00.npz')
# data = np.load('data\design_OPO_250nm\Coupler_bandwidth_w_1p50_gap_1p00_L_200p00.npz')
data = np.load('data\design_OPO_250nm\Coupler_bandwidth_w_1p50_gap_1p00_L_150p00.npz')

C = data['C']
L = data['L']
width = data['w_ridge']
gap = data['gap']


wl_start = 0.7
wl_stop = 2.2+0.1
wl_step = 0.01
wl = np.arange(wl_start, wl_stop, wl_step)

#Plot
# fig, ax = plt.subplots()
label = 'Lc = %0.1f $\mu$m' %(L)
ax.plot(wl, C*100, label=label)
ax.legend()
ax.grid(True)
ax.set_xlabel('Wavelength ($\mu$m)')
ax.set_ylabel('Coupling Factor (%)')
ax.axis([0.7, 2.2, 0, 100])
ax.set_title('Gap = %0.1f $\mu$m, Width = %0.1f $\mu$m' %(gap, width))