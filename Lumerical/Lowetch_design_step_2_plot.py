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

#data = np.load('data\LNoI_wl_1p0_1558392399.npz')
#data = np.load('data\LNoI_wl_1p0_1558393660.npz')
#data = np.load('data\LNoI_wl_1p0_1558394765.npz')
data = np.load('data\design_OPO_250nm\LNoI_wl_1p0_bendingloss_1560809103.npz')

neff = data['neff']
loss = data['loss']

etch = data['h_etch']
width = data['w_ridge_list'] 
bending = data['bending_list']
LN_thickness = data['h_LN']
wavelength = data['wavelength']

"""
Choose range
"""
bending_min = 100
bending_max = 2000
bending_boolean = np.logical_and(bending>bending_min-0.01, bending<bending_max+0.01)
bending = bending[bending_boolean]
loss = loss[:,bending_boolean]

"""
Plot Loss vs width
"""
fig, ax = plt.subplots()
ax.grid(True)
for kb in range(bending.size):
    label = 'Bending radius = %0.1f um' %(bending[kb])
    ax.plot(width, loss[:,kb], label=label)
ax.legend()
ax.set_xlabel('Ridge width ($\mu$m)')

ax.axis([np.amin(width), np.amax(width), 0, 1])
ax.set_ylabel('Loss (dB/cm)')
title = 'Loss (dB/cm)'
title += '\n $\lambda = %0.1f$ $\mu$m' %(wavelength)
title += '\n LN thickness = %0.1f $\mu$m, etch depth = %0.1f nm' %(LN_thickness, etch*1000)
ax.set_title(title)

"""
Plot Loss vs radius
"""
fig, ax = plt.subplots()
ax.grid(True)
for kw in range(width.size):
    label = 'Width = %0.1f um' %(width[kw])
    ax.plot(bending, loss[kw,:], label=label)
ax.legend()
ax.set_xlabel('Bending Radius ($\mu$m)')

ax.axis([np.amin(bending), np.amax(bending), 0, 1])
ax.set_ylabel('Loss (dB/cm)')
title = 'Loss (dB/cm)'
title += '\n $\lambda = %0.1f$ $\mu$m' %(wavelength)
title += '\n LN thickness = %0.1f $\mu$m, etch depth = %0.1f nm' %(LN_thickness, etch*1000)
ax.set_title(title)