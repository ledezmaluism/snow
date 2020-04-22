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
    lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2019b/api/python/lumapi.py")
    mode = lumapi.MODE("Template_Luis.lms")

'''
Input parameters
'''
wp = 1*1e-6
wavelength = 2*1e-6
h_LN = 700*1e-9
h_etch = 250*1e-9
w_ridge = 1.5*1e-6
# h_slab = h_LN - h_etch
h_slab=0

theta = 60
wg_length = 50*1e-6
# w_ridge_base = w_ridge + 2*h_etch/np.tan(theta*pi/180)
# print('slab = ', h_slab)
# print('width at the base = %.3f um' %(w_ridge_base))

'''
Simulation volume
'''
w_slab = 10*wavelength + 2*w_ridge
h_margin = 4*wavelength
h_substrate = 4*wavelength
meshsize = wavelength/20
finemesh = wavelength/80

'''
Materials
'''
mode.switchtolayout()
mode.deleteall()
material_substrate = "SiO2_analytic"
# material_substrate = "Sapphire_analytic"

# material_thinfilm = "LN_analytic_undoped_xne"
#material_thinfilm = "LN_analytic_undoped_zne"
material_thinfilm = "LN_analytic_MgO_doped_xne"
#material_thinfilm = "LN_analytic_MgO_doped_zne"
#material_thinfilm = "LiNbO3 constant"

'''
Drawing and setup
'''
#Draw substrate
mode.addrect()
mode.set("name","Substrate")
mode.set("material", material_substrate)
mode.set("x", 0)
mode.set("x span", w_slab)
mode.set("y", 0)
mode.set("y span", wg_length)
mode.set("z min", -h_substrate)
mode.set("z max", 0)
mode.set("alpha", 0.5)

#Draw slab
# mode.addrect()
# mode.set("name","LN slab")
# mode.set("material", material_thinfilm)
# mode.set("x", 0)
# mode.set("x span", w_slab)
# mode.set("y", 0)
# mode.set("y span", wg_length)
# mode.set("z min", 0)
# mode.set("z max", h_slab)
# mode.set("alpha", 0.5)

#Draw Ridge
mode.addrect()
mode.set("name","LN ridge")
mode.set("material", material_thinfilm)
mode.set("x", 0)
mode.set("x span", w_ridge)
mode.set("y", 0)
mode.set("y span", wg_length)
mode.set("z min", h_slab)
mode.set("z max", h_LN)
mode.set("alpha", 0.5)

##################################################################
#FDTD solver

#Geometry
mode.addvarfdtd()
mode.set("x", 0)
mode.set("x span", w_slab*0.9)
mode.set("y", 0)
mode.set("y span", wg_length*0.9)
mode.set("z min", -h_substrate*0.5)
mode.set("z max", h_LN*4)
#
##Effective index method
#Location of main slab mode
mode.set("x0", 0)
mode.set("y0", 0)
#
##Bandwidth
mode.set("bandwidth", "broadband")
mode.set("fit tolerance", 1e-5)
mode.set("max coefficients", 20)

##Test points
mode.set("number of test points", 1)
test_points = np.array([5e-6,0])
mode.set("test points", test_points)
#
##Boundary conditions
# bc = "PML"
bc = "Metal"
mode.set("x min bc", bc)
mode.set("x max bc", bc)
mode.set("y min bc", bc)
mode.set("y max bc", bc)
mode.set("z min bc", bc)
mode.set("z max bc", bc)
#
##Add source
mode.addmodesource()
mode.set("injection axis", "y-axis")
mode.set("x",0)
mode.set("x span", w_slab)
mode.set("y", -0.8*wg_length/2)
mode.set("wavelength start", 0.95*wp)
mode.set("wavelength stop", 1.05*wp)
#
##Add monitors
mode.addtime()
mode.set("monitor type","Linear Y")
mode.set("y span", wg_length*0.9)
mode.set("down sample Y",10)
mode.set("output Ex",1)
mode.set("output Ey",0)
mode.set("output Ez",0)
mode.set("output Hx",0)
mode.set("output Hy",0)
mode.set("output Hz",0)