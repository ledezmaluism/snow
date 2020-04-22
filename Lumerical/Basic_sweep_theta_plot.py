# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019

@author: luish
"""

import LNOI_functions as lumpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imp

from scipy.constants import pi

data = np.load('data\design_OPO_250nm\LNoI_wl_1p0_theta_sweep_1560886482.npz')

width = data['w_ridge']
neff = data['neff']
tepf = data['tepf']
theta = data['theta_list']
wavelength = data=['wavelength']

#Plot
fig, ax = plt.subplots()
#label = 'Lc = %0.1f $\mu$m' %(L)
ax.plot(theta, neff[:,2])
#ax.legend()
#ax.grid(True)
#ax.set_xlabel('Wavelength ($\mu$m)')
#ax.set_ylabel('Coupling Factor (%)')
#ax.axis([0.7, 2.2, 0, 100])
#ax.set_title('Gap = %0.1f $\mu$m, Width = %0.1f $\mu$m' %(gap, width))

abs(neff[0,2] - neff[-1,2])/neff[-1,2]