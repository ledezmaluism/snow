# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:31:26 2019
@author: luis ledezma

Goal:
    Draws a waveguide, set up the materials, simulation volume and mesh.
    Doesn't run a simulation.
"""


'''
Imports
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imp

import lumerical as lum

from scipy.constants import pi

'''
Load template:
    This is important as the template contains the materials information
'''
if 'mode' not in vars():
    lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2020a/api/python/lumapi.py")
    mode = lumapi.MODE("Template_Luis.lms")

'''
Units
'''
um = 1e-6
nm = 1e-9
THz = 1e12

'''
Input parameters
'''
wavelength = 1*um
h_LN = 0.7*nm
h_etch = 0.2*nm
w_ridge = 1000*nm
h_slab = h_LN - h_etch

theta = 60
wg_length = 10*um
w_ridge_base = w_ridge + 2*h_etch/np.tan(theta*pi/180)

print('slab = ', h_slab)
print('width at the base = %.3f um' %(w_ridge_base))

'''
Simulation volume
'''
w_slab = 10*wavelength + 2*w_ridge
# w_slab = 20*wavelength + 2*w_ridge
h_margin = 4*wavelength
h_substrate = 4*wavelength
# h_substrate = 8*wavelength
meshsize = wavelength/20
# finemesh = wavelength/40
finemesh = wavelength/80

'''
Materials
'''
# material_substrate = "SiO2_analytic"
material_substrate = "Sapphire_analytic"

# material_thinfilm = "LN_analytic_undoped_xne"
#material_thinfilm = "LN_analytic_undoped_zne"
material_thinfilm = "LN_analytic_MgO_doped_xne"
#material_thinfilm = "LN_analytic_MgO_doped_zne"
#material_thinfilm = "LiNbO3 constant"

'''
Drawing and setup
'''
lum.draw_wg(mode, material_thinfilm, material_substrate,
              h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
lum.add_fine_mesh(mode, finemesh, h_LN, w_ridge_base, x_factor=1.2, y_factor=1.5)
lum.add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                         w_slab, wg_length, h_margin)