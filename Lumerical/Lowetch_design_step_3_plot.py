# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:13:59 2019

@author: luish

Parameters:

n1 = w_ridge_list.size
n2 = bending_list.size

neff = np.empty([n1,n2])
loss = np.empty([n1,n2])
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['mathtext.fontset'] = 'cm' #'dejavusans' #'stix'

# data = np.load('data\LNoI_wl_2p0_phasemismatch_1583886753.npz')
data = np.load('LNoI_wl_2p0_phasemismatch_1584122782.npz')

neff = data['neff']

etch = data['h_etch']
width = data['w_ridge_list'] 
LN_thickness = data['h_LN']
wavelength = data['wavelength_list']

delta_n = neff[:,0] - neff[:,1]
PP = wavelength[0]/delta_n

"""
Plot Poling Period vs width
"""
fig, ax = plt.subplots()
ax.grid(True)
ax.plot(width, PP)
ax.set_xlabel('Ridge width ($\mu$m)')

#ax.axis([np.amin(width), np.amax(width), 0, 5])
ax.set_ylabel('Poling Period ($\mu$m)')
title = '$\lambda_p = %0.1f$ $\mu$m; $\lambda_s = %0.1f$ $\mu$m' %(wavelength[0], wavelength[1])
title += '\n LN thickness = %0.1f $\mu$m, etch depth = %0.1f nm' %(LN_thickness, etch*1000)
ax.set_title(title)