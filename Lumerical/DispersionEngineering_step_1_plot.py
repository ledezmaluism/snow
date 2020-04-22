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

data = np.load('data\LNoI_wl_2p0_1558115759.npz')
#data = np.load('data\LNoS_wl_3p0_hLN_1p2_1p4_1556214482.npz')

GVD = data['GVD']
Vg = data['Vg']
neff = data['neff']

hslab = data['h_slab_list']
width = data['w_ridge_list'] 
LN_thickness = data['h_LN_list']
wavelength = data['wavelength']
  
def plot_vs_width(f, w, h, hLN_idx):
    """
    Plots vs width for several etch depths and a single hLN
    """
    f =  f[:,:,hLN_idx]
    LN_t = LN_thickness[hLN_idx]
    
    fig, ax = plt.subplots()
    ax.grid(True)
    for kh in range(h.size):
        etch = LN_t - h[kh]
        label = 'slab thickness = %0.1f $\mu$m, etch = %0.1f $\mu$m' %(h[kh], etch)
        ax.plot(w, f[:,kh], label=label)
    ax.legend()
    ax.set_xlabel('Ridge width ($\mu$m)')
    return fig, ax
    

"""
Plot GVD vs width
"""
w_min = 0.5
w_max = 2.0
w_plot_boolean = np.logical_and(width>w_min-0.01, width<w_max+0.01)
w_plot = width[w_plot_boolean]

hLN_idx = 0
fig, ax = plot_vs_width(GVD[w_plot_boolean,:,:], w_plot, hslab, hLN_idx)
ax.axis([w_min, w_max, -500, 500])
ax.set_ylabel('GVD (fs$^2$/mm)')
title = '$\mathrm{GVD} = \partial^2 k / \partial \omega^2$'
title += '\n $\lambda = %0.1f$ $\mu$m' %(wavelength)
title += '\n LN thickness = %0.1f $\mu$m' %(LN_thickness[hLN_idx])
ax.set_title(title)