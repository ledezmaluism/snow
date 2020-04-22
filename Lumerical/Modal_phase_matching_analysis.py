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

#pump = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p0_modes_etch_300nm_1563839538.npz')
#signal = np.load('data\LNOI_modes_vs_width\LNoI_wl_2p0_modes_etch_300nm_1563914903.npz')

#pump = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p0_modes_etch_350nm_1564116289.npz')
#signal = np.load('data\LNOI_modes_vs_width\LNoI_wl_2p0_modes_etch_350nm_1564116748.npz')

#pump = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p0_modes_etch_400nm_1563857207.npz')
#signal = np.load('data\LNOI_modes_vs_width\LNoI_wl_2p0_modes_etch_400_nm_1563927557.npz')

#pump = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p0_modes_etch_650nm_1564157497.npz')
#signal = np.load('data\LNOI_modes_vs_width\LNoI_wl_2p0_modes_etch_650nm_1564156954.npz')

#pump = np.load('data\LNOI_modes_vs_width\LNoI_wl_0p78_modes_etch_650nm_1564616924.npz')
#signal = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p55_modes_etch_650nm_1564617227.npz')

pump = np.load('data\LNOI_modes_vs_width\LNoI_wl_0p78_modes_etch_350nm_1564637613.npz')
signal = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p55_modes_etch_350nm_1564617412.npz')

neff_p = pump['neff']
tepf_p = pump['tepf']

etch = pump['h_etch']
width = pump['w_ridge_list'] 
LN_thickness = pump['h_LN']
wavelength = pump['wavelength']


n_modes = neff_p.shape[1]
"""
Plot neff vs width
"""
fig1, ax1 = plt.subplots()

for n in range(n_modes):
    #label = 'Bending radius = %0.1f um' %(bending[kb])
    #ax.plot(width, loss[:,kb], label=label)
    ax1.plot(width, neff_p)

#ax.legend()
ax1.set_xlabel('Ridge width ($\mu$m)')
#
ax1.grid(True)
#
y_min = round(np.nanmin(neff_p)*10)/10
y_max = round((np.nanmax(neff_p)+0.05)*10)/10
#ax.axis([np.amin(width), np.amax(width), y_min, y_max])
ax1.set_ylabel('neff')
title = 'Effective Index, '
title += '$\lambda = %0.1f$ $\mu$m' %(wavelength)
title += '\n LN thickness = %0.1f $\mu$m, etch depth = %0.1f nm' %(LN_thickness, etch*1000)
ax1.set_title(title)

"""
Plot tepf vs width
"""
fig2, ax2 = plt.subplots()

for n in range(n_modes):
    #label = 'Bending radius = %0.1f um' %(bending[kb])
    #ax.plot(width, loss[:,kb], label=label)
    ax2.plot(width, tepf_p)

#ax.legend()
ax2.set_xlabel('Ridge width ($\mu$m)')
#
ax2.grid(True)
#
y_min = 0
y_max = 1
ax2.axis([np.amin(width), np.amax(width), y_min, y_max])
ax2.set_ylabel('tepf')
title = 'TE Polarization Factor, '
title += '$\lambda = %0.1f$ $\mu$m' %(wavelength)
title += '\n LN thickness = %0.1f $\mu$m, etch depth = %0.1f nm' %(LN_thickness, etch*1000)
ax2.set_title(title)


"""
Signal
"""

neff_s = signal['neff']
tepf_s = signal['tepf']
width = signal['w_ridge_list'] 
n_modes = neff_s.shape[1]

for n in range(n_modes):
    #label = 'Bending radius = %0.1f um' %(bending[kb])
    #ax.plot(width, loss[:,kb], label=label)
    ax1.plot(width, neff_s)





"""
Re-arrange stuff
At each width find wich modes are TE, and sort them by neff value
"""
#
#neff_TE = np.empty([width.size, n_modes])
#neff_TE[:] = np.nan
#neff_TM = np.empty([width.size, n_modes])
#neff_TM[:] = np.nan
#
##Separate by polarization
#for w in range(width.size):
#    for n in range(n_modes):
#        if tepf[w,n]>0.5:
#            neff_TE[w,n] = neff[w,n]
#        else:
#            neff_TM[w,n] = neff[w,n]
#
##Separate by mode order
#neff_TE_00 = np.empty([width.size])
#neff_TE_10 = np.empty([width.size])  
#neff_TE_00[:] = np.nan
#neff_TE_10[:] = np.nan          
#for w in range(width.size):
#    idx_TE_00 = np.nanargmax(neff_TE[w,:])
#    neff_TE_00[w] = neff_TE[w, idx_TE_00]
#    neff_TE[w, idx_TE_00] = np.nan #Remove TE00
#    try:
#        idx_TE_10 = np.nanargmax(neff_TE[w,:])
#        neff_TE_10[w] = neff_TE[w, idx_TE_10]
#        neff_TE[w, idx_TE_10] = np.nan #Remove TE10
#    except:
#        pass