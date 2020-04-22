# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:13:59 2019

@author: luish

Parameters:
    gvm[width or h_slab], fixed wavelength
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['mathtext.fontset'] = 'cm' #'dejavusans' #'stix'

data1 = np.load('data\LNoS_GVM_At_zeroGVD_signal_3p0_h_LN_1p2_1556556603.npz')
data2 = np.load('data\LNoS_GVM_At_zeroGVD_signal_3p0_h_LN_1p3_1556556119.npz')
w_signal = data1['w_signal']
w_pump = data1['w_pump']

def plot_gvm_vs_etch(data, ax):
    gvm = data['gvm']*1e15/1e3 #fs/mm
    h_slab = data['h_slab']
    LN_thickness = data['h_LN']
    etch = LN_thickness - h_slab
    
    label = 'LN_thickness = %0.1f $\mu$m' %(LN_thickness)
    ax.plot(etch, gvm, label=label)
    

fig, ax = plt.subplots()
plot_gvm_vs_etch(data1, ax)
plot_gvm_vs_etch(data2, ax)

ax.legend()
ax.grid(True)
ax.set_xlabel('Etch depth ($\mu$m)')
ax.set_ylabel('GVM (fs/mm)')
#ax.axis([0.6, 1.3, 1, 1.8])

title = 'GVM for zero GVD$'
title += '\n $\lambda_s = %0.1f$ $\mu$m, $\lambda_p = %0.1f$ $\mu$m' %(w_signal, w_pump)
ax.set_title(title)