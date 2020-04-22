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

if 'mode' not in vars():
    #lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/MODE/api/python/lumapi.py")
    lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2019b/api/python/lumapi.py")
    mode = lumapi.MODE("Template_Luis.lms")

wavelength = 1
h_LN = 0.7
h_etch = 0.25
w_ridge = 1.1
h_slab = h_LN - h_etch

theta = 60
wg_length = 10
w_ridge_base = w_ridge + 2*h_etch/np.tan(theta*pi/180)

print('slab = ', h_slab)
print('width at the base = %.3f um' %(w_ridge_base))

w_slab = 10*wavelength + 2*w_ridge
#w_slab = 20*wavelength + 2*w_ridge
h_margin = 4*wavelength
h_substrate = 4*wavelength
meshsize = wavelength/20
finemesh = wavelength/40

material_substrate = "SiO2_analytic"
#material_substrate = "Sapphire_analytic"

#material_thinfilm = "LN_analytic_undoped_xne"
#material_thinfilm = "LN_analytic_undoped_zne"
material_thinfilm = "LN_analytic_MgO_doped_xne"
#material_thinfilm = "LN_analytic_MgO_doped_zne"
#material_thinfilm = "LiNbO3 constant"

lumpy.draw_wg(mode, material_thinfilm, material_substrate,
              h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
lumpy.add_fine_mesh(mode, finemesh, h_LN, w_ridge_base, x_factor=1.2, y_factor=1.5)
lumpy.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                         w_slab, wg_length, h_margin)
#mode.save("waveguide.lms")
neff, TEPF = lumpy.solve_mode(mode, wavelength, nmodes=20)

x, y, Exa, Eya, Eza, Hxa, Hya, Hza = lumpy.get_mode(mode, 2)
x, y, Exb, Eyb, Ezb, Hxb, Hyb, Hzb = lumpy.get_mode(mode, 3)

Ete_x = Exa - Exb
Ete_y = Eya - Eyb

Etm_x = Exa + Exb
Etm_y = Eya + Eyb
