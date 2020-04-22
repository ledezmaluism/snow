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

data = np.load('data\LNoI_wl_1p0_1558285121.npz')

neff = data['neff']
loss = data['loss']

hetch = data['h_etch']
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
        slab = LN_t - h[kh]
        label = 'etch = %0.1f nm, slab thickness = %0.1f nm' %(h[kh]*1000, slab*1000)
        ax.plot(w, f[:,kh], label=label)
    ax.legend()
    ax.set_xlabel('Ridge width ($\mu$m)')
    return fig, ax
    

"""
Plot Loss vs width
"""
w_min = 1.0
w_max = 2.0
w_plot_boolean = np.logical_and(width>w_min-0.01, width<w_max+0.01)
w_plot = width[w_plot_boolean]

hLN_idx = 0
fig, ax = plot_vs_width(loss[w_plot_boolean,:,:], w_plot, hetch, hLN_idx)
#ax.axis([w_min, w_max, -5, 5])
ax.set_ylabel('Loss (dB/cm)')
title = 'Loss (dB/cm)'
title += '\n $\lambda = %0.1f$ $\mu$m' %(wavelength)
title += '\n LN thickness = %0.1f $\mu$m' %(LN_thickness[hLN_idx])
ax.set_title(title)