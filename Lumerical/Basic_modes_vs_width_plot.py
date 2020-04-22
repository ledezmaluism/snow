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

#data = np.load('data\design_OPO_250nm\LNoI_wl_1p0_modes_1560813180.npz')
#data = np.load('data\design_OPO_250nm\LNoI_wl_2p0_modes_1560813556.npz')

#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p55_modes_etch_250nm_1564960391.npz')
#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p55_modes_etch_300nm_1564700829.npz')
#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p55_modes_etch_350nm_1564617412.npz')
#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p55_modes_etch_400nm_1564701215.npz')
#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_1p55_modes_etch_650nm_1564617227.npz')

#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_0p78_modes_etch_250nm_1564960680.npz')
#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_0p78_modes_etch_300nm_1564702559.npz')
#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_0p78_modes_etch_350nm_1564637613.npz')
#data = np.load('data\LNOI_modes_vs_width\LNoI_wl_0p78_modes_etch_400_nm_1564701953.npz')
#data = np.load('data\LNOI_modes_vs_width\LNoI_1p55um_LN400nm_fulletch_90walls.npz')
data = np.load('LNoI_wl_1p0_modes_etch_350p0_nm_1585065052.npz')

neff = data['neff']
tepf = data['tepf']

etch = data['h_etch']
width = data['w_ridge_list'] 
LN_thickness = data['h_LN']
wavelength = data['wavelength']


n_modes = neff.shape[1]
n_modes = 4
"""
Plot neff vs width
"""
fig, ax = plt.subplots()

for n in range(n_modes):
    #label = 'Bending radius = %0.1f um' %(bending[kb])
    #ax.plot(width, loss[:,kb], label=label)
    # ax.plot(width, neff[:,n])
    N = width.size
    for i in range(N-1):
        ax.plot(width[i:i+2], neff[i:i+2,n], color=plt.cm.winter(tepf[i+1,n]))

#ax.legend()
ax.set_xlabel('Ridge width ($\mu$m)')
#
ax.grid(True)
#
y_min = round(np.nanmin(neff)*10)/10
y_max = round((np.nanmax(neff)+0.05)*10)/10
#ax.axis([np.amin(width), np.amax(width), y_min, y_max])
ax.set_ylabel('neff')
title = 'Effective Index, '
title += '$\lambda = %0.1f$ $\mu$m' %(wavelength)
title += '\n LN thickness = %0.1f $\mu$m, etch depth = %0.1f nm' %(LN_thickness, etch*1000)
ax.set_title(title)

"""
Plot tepf vs width
"""
#fig, ax = plt.subplots()
#
#for n in range(n_modes):
#    #label = 'Bending radius = %0.1f um' %(bending[kb])
#    #ax.plot(width, loss[:,kb], label=label)
#    ax.plot(width, tepf[:,n])
#
##ax.legend()
#ax.set_xlabel('Ridge width ($\mu$m)')
##
#ax.grid(True)
##
#y_min = 0
#y_max = 1
#ax.axis([np.amin(width), np.amax(width), y_min, y_max])
#ax.set_ylabel('tepf')
#title = 'TE Polarization Factor, '
#title += '$\lambda = %0.1f$ $\mu$m' %(wavelength)
#title += '\n LN thickness = %0.1f $\mu$m, etch depth = %0.1f nm' %(LN_thickness, etch*1000)
#ax.set_title(title)

"""
Plot just TE modes
"""
# TE_mode_th = 0.8 #Threshold

# fig, ax = plt.subplots()

# for n in range(n_modes):
#     idx = tepf[:,n]>0.8
#     ax.plot(width[idx], neff[idx,n])
# #ax.plot(width, neff)

# #ax.legend()
# ax.set_xlabel('Ridge width ($\mu$m)')
# #
# ax.grid(True)
# #
# y_min = round(np.nanmin(neff)*10)/10
# y_max = round((np.nanmax(neff)+0.05)*10)/10
# #ax.axis([np.amin(width), np.amax(width), y_min, y_max])
# ax.set_ylabel('neff')
# title = 'TE Modes: Effective Index, '
# title += '$\lambda = %0.3f$ $\mu$m' %(wavelength)
# title += '\n LN thickness = %0.1f $\mu$m, etch depth = %0.1f nm' %(LN_thickness, etch*1000)
# ax.set_title(title)

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