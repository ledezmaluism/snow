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

#data = np.load('data\LNoS_freq_sweep_h_LN_1p2_1556653732.npz')
#data = np.load('data\LNoS_freq_sweep_h_LN_1p2_1557164831.npz')
data = np.load('data\LNoS_freq_sweep_h_LN_1p2_1557245351.npz')
#data = np.load('data\LNoS_freq_sweep_h_LN_0p7_1559585332.npz')
#data = np.load('data\design_OPO_250nm\LNoI_freq_sweep_h_LN_0p7_width_1p5_1560901665.npz')

freqs = data['freqs']
neff = data['neff']
beta = data['beta']
vg = data['vg']
gvd = data['gvd']
h_LN = data['h_LN']
h_slab = data['h_slab']
width = data['width']

   
'''
neff plot
'''
fig, ax = plt.subplots()
ax.plot(freqs, neff)
ax.grid(True)
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('$n_{eff}$')
#ax.axis([0.6, 1.3, 1, 1.8])

title = 'Effective Index \n'
title += 'LN_thickness = %0.1f $\mu$m \n' %(h_LN)
title += 'h_etch = %0.2f $\mu$m; width = %0.2f $\mu$m' %(h_LN-h_slab, width)
ax.set_title(title)
ax.autoscale(enable=True, tight=True)

'''
beta1 plot
'''
fig, ax = plt.subplots()
ax.plot(freqs, 1/vg)
ax.grid(True)
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('$\\beta_1 = 1/v_g$ (fs/mm)')

title = 'Inverse of group velocity \n'
title += 'LN_thickness = %0.1f $\mu$m \n' %(h_LN)
title += 'h_etch = %0.2f $\mu$m; width = %0.2f $\mu$m' %(h_LN-h_slab, width)
ax.set_title(title)
ax.autoscale(enable=True, tight=True)

'''
beta2 plot
'''
fig, ax = plt.subplots()
ax.plot(freqs, gvd)
ax.grid(True)
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('$\\beta_2 = \mathrm{GVD} $ (fs$^2$/mm)')

title = 'Group velocity dispersion \n'
title += 'LN_thickness = %0.1f $\mu$m \n' %(h_LN)
title += 'h_etch = %0.2f $\mu$m; width = %0.2f $\mu$m' %(h_LN-h_slab, width)
ax.set_title(title)
ax.autoscale(enable=True, tight=True)