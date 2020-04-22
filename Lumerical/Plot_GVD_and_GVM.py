# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:13:59 2019

@author: luish
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#plt.rcParams['mathtext.fontset'] = 'cm' #'dejavusans' #'stix'

#data = np.load('data\GVD_freq_sweep_hLN_1p4_pump_1p5_1554921162.npz')
#data = np.load('data\GVD_freq_sweep_hLN_1p3_pump_1p5_1554873856.npz')
#data = np.load('data\GVD_freq_sweep_hLN_1p2_pump_1p5_1554877415.npz')
data = np.load('data\GVD_freq_sweep_hLN_1p1_pump_1p5_1554919354.npz')

GVD = data['GVD']
Vg = data['Vg']

h = data['h_slab_list']
w = data['w_ridge_list'] 
wavelength = data['wavelength_list']
LN_thickness = data['h_LN']

kw_min = 4
kw_max = -2
w = w[kw_min:kw_max]
GVD = GVD[:,kw_min:kw_max,:]
Vg = Vg[:,kw_min:kw_max,:]

def plot_GVD_2D(kf):
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(GVD[:,:,kf]), origin='lower', aspect='equal', interpolation='bicubic',
               cmap=cm.jet, extent=[w.min(), w.max(), h.min(), h.max()])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label='$\mid \mathrm{GVD} \mid$ (fs$^2$/mm)')
    ax.set_xlabel('Ridge width ($\mu$m)')
    ax.set_ylabel('Etch depth ($\mu$m)')
    title = '$\mid \mathrm{GVD} \mid = \mid \partial^2 k / \partial \omega^2 \mid$'
    title += '\n $\lambda = %0.1f$ $\mu$m' %(wavelength[kf])
    title += '\n LN thickness = %0.1f $\mu$m' %(LN_thickness)
    ax.set_title(title)
    
def get_GVM(kp, ks):
    Vg_pump = Vg[:,:,kp]
    Vg_signal = Vg[:,:,ks]
    GVM = 1/Vg_pump - 1/Vg_signal
    return GVM*1e12 #fs/mm○

def plot_GVD_vs_with(kf):
    fig, ax = plt.subplots()
    ax.grid(True)
    for kh in range(h.size):
        etch = LN_thickness - h[kh]
        label = 'slab thickness = %0.1f $\mu$m, etch = %0.1f $\mu$m' %(h[kh], etch)
        ax.plot(w, GVD[kh,:,kf].transpose(), label=label)
    ax.legend()
    ax.set_xlabel('Ridge width ($\mu$m)')
    ax.set_ylabel('GVD (fs$^2$/mm)')
    title = '$\mid \mathrm{GVD} \mid = \mid \partial^2 k / \partial \omega^2 \mid$'
    title += '\n $\lambda = %0.1f$ $\mu$m' %(wavelength[kf])
    title += '\n LN thickness = %0.1f $\mu$m' %(LN_thickness)
    ax.set_title(title)

def plot_GVM_vs_with(kp, ks):
    gvm = get_GVM(kp, ks)
    fig, ax = plt.subplots()
    ax.grid(True)
    for kh in range(h.size):
        etch = LN_thickness - h[kh]
        label = 'slab thickness = %0.1f $\mu$m, etch = %0.1f $\mu$m' %(h[kh], etch)
        ax.plot(w, gvm[kh,:].transpose(), label=label)
    #ax.legend()
    ax.set_xlabel('Ridge width ($\mu$m)')
    ax.set_ylabel('GVM (fs/mm)')
    title = '$\mathrm{GVM} = 1/v_g(\omega_p) - 1/v_g(\omega_s)$'
    title += '\n $\lambda_p = %0.1f$ $\mu$m' %(wavelength[kp])
    title += '; $\lambda_s = %0.1f$ $\mu$m ' %(wavelength[ks])
    title += '\n LN thickness = %0.1f $\mu$m' %(LN_thickness)
    ax.set_title(title)

plot_GVD_vs_with(1)
plot_GVM_vs_with(0,1)