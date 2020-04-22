# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:13:59 2019

@author: luish

Parameters:
    gvm[width or h_slab], fixed wavelength
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
#plt.rcParams['mathtext.fontset'] = 'cm' #'dejavusans' #'stix'

data = np.load('data\design_OPO_250nm\LNoI_freq_sweep_h_LN_0p7_width_1p5_1560901665.npz')

freqs = data['freqs']
neff = data['neff']
#beta = data['beta']
vg = data['vg'] #Ginve in mm/fs
#gvd = data['gvd']
h_LN = data['h_LN']
h_slab = data['h_slab']
width = data['width']


#Assume racetrack resonator
L1 = 4 #mm
R = 0.2 #mm
L = 2*L1 + 2*pi*R

FSR = vg/L*1e6 #GHz

'''
beta1 plot
'''
fig, ax = plt.subplots()
ax.plot(freqs, FSR)
ax.grid(True)
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('FSR (GHz)')

title = 'Free Spectral Range \n'
title += 'LN_thickness = %0.1f $\mu$m \n' %(h_LN)
title += 'h_etch = %0.2f $\mu$m; width = %0.2f $\mu$m' %(h_LN-h_slab, width)
ax.set_title(title)
ax.autoscale(enable=True, tight=True)
