# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:13:59 2019

@author: luish

PLots coupler design curves
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
#plt.rcParams['mathtext.fontset'] = 'cm' #'dejavusans' #'stix'

#data = np.load('data\design_OPO_250nm\Coupler_sweepgap_width_1p0_pump_0p775.npz')
data = np.load('Coupler_sweepgap_width_1p4_pump_1p0.npz')
# data = np.load('Coupler_sweepgap_width_1p9_pump_1p0.npz')

wl_pump = data['wl_pump']
wl_signal = data['wl_signal']
deltaN_pump = data['deltaN_pump']
deltaN_signal = data['deltaN_signal']

gap_list = data['gap_list']

hetch = data['h_etch']
width = data['w_ridge'] 


def coupling(L, delta_n, wl):
    C = np.sin(pi*L*delta_n/wl)**2
    return C

'''
Plot coupling vs coupler length for both signal and pump
'''
L = np.arange(10, 300, 1) #microns

gap_idx = 2

C_signal = coupling(L, deltaN_signal[gap_idx], wl_signal)
C_pump = coupling(L, deltaN_pump[gap_idx], wl_pump)

fig, ax = plt.subplots()
ax.plot(L, C_signal*100, label='signal')
ax.plot(L, C_pump*100, label='pump')
ax.legend()
ax.grid(True)
ax.set_xlabel('Coupler Length ($\mu$m)')
ax.set_ylabel('Coupling Factor (%)')
ax.axis([0, max(L), 0, 100])
ax.set_title('Gap = %0.1f $\mu$m' %(gap_list[gap_idx]))

'''
Plot coupling vs coupler length for both signal and pump
'''
L = np.arange(10, 300, 1) #microns

gap_idx = 2

C_signal = coupling(L, deltaN_signal[gap_idx], wl_signal)
C_pump = coupling(L, deltaN_pump[gap_idx], wl_pump)

fig, ax = plt.subplots()
ax.plot(L, C_signal*100, label='signal')
ax.plot(L, C_pump*100, label='pump')
ax.legend()
ax.grid(True)
ax.set_xlabel('Coupler Length ($\mu$m)')
ax.set_ylabel('Coupling Factor (%)')
ax.axis([0, 140, 0, 15])
ax.set_title('Gap = %0.1f $\mu$m' %(gap_list[gap_idx]))


'''
Plot ratio of coupling factor betweem signal and pump
'''
Lc_signal = wl_signal/(2*deltaN_signal) #microns
#ratio = deltaN_pump/deltaN_signal

x_pump = np.zeros(gap_list.size)
for kg in range(gap_list.size):
    Lc = Lc_signal[kg]
    x_pump[kg] = coupling(Lc, deltaN_pump[kg], wl_pump)

fig, ax = plt.subplots()
ax.plot(gap_list, x_pump*100)
ax.grid(True)
ax.set_xlabel('Coupler gap ($\mu$m)')
ax.set_ylabel('Pump Coupling Factor (%)')
ax.axis([0, max(gap_list), 0, 20])
ax.set_title('Pump coupling factor for 100% signal coupling')
