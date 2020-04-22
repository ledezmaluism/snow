# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:13:59 2019

@author: luish

Parameters:
    GVD[width, h_slab, h_LN], fixed wavelength
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['mathtext.fontset'] = 'cm' #'dejavusans' #'stix'

data1 = np.load('data\LNoS_zeroGVD_design_wl_3p0_h_LN_1p2_1556255055.npz')
data2 = np.load('data\LNoS_zeroGVD_design_wl_3p0_h_LN_1p3_1556309685.npz')
wavelength = data1['wavelength']

def plot_width_vs_etch(data, ax):
    """
    Design plot: zero GVD curve
    """
    width = data['width']
    h_slab = data['h_slab']
    LN_thickness = data['h_LN']
    etch = LN_thickness - h_slab
    
    label = 'LN_thickness = %0.1f $\mu$m' %(LN_thickness)
    ax.plot(etch, width, label=label)
    

fig, ax = plt.subplots()
plot_width_vs_etch(data1, ax)
plot_width_vs_etch(data2, ax)

ax.legend()
ax.grid(True)
ax.set_xlabel('Etch depth ($\mu$m)')
ax.set_ylabel('Ridge width ($\mu$m)')
ax.axis([0.6, 1.3, 1, 1.8])

title = 'Iso-curve for $\mathrm{GVD} = 0$'
title += '\n $\lambda = %0.1f$ $\mu$m' %(wavelength)
ax.set_title(title)