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


if 'fdtd' not in vars():
    lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2019b/api/python/lumapi.py")
    #fdtd = lumapi.FDTD("Template_Luis_FDTD.fsp")
    fdtd = lumapi.FDTD()

wl_start = 1.0
wl_stop = 2.0

h_LN = 0.7
h_etch = 0.25
w_ridge = 1.5
h_slab = h_LN - h_etch
wg_length = 300

#material_substrate = "Sapphire_analytic"
#material_thinfilm = "LN_analytic_undoped_xne"
material_substrate = "SiO2_analytic"
material_thinfilm = "LN_analytic_MgO_doped_xne"

w_slab = wg_length #square slab
h_substrate = 5*h_LN
#meshsize = wavelength/10
#finemesh = wavelength/50

'''
Starts drawing
'''
fdtd.switchtolayout()
fdtd.deleteall()
#
##Change everything to meters
#h_LN = float(h_LN*1e-6)
#h_etch = float(h_etch*1e-6)
#h_slab = float(h_slab*1e-6)
#w_ridge = float(w_ridge*1e-6)
#w_slab = float(w_slab*1e-6)
#wg_length = float(wg_length*1e-6)
#h_substrate = float(h_substrate*1e-6)
#
##Calculate some extra geometric parameters
##Zmin = -wg_length/2
##Zmax = wg_length/2
#
lumpy.draw_substrate(fdtd, material_thinfilm, material_substrate, h_LN, h_substrate, 
                           h_etch, w_slab, wg_length, x0=0)

#Draw substrate
#mode.addrect()
#mode.set("name","Substrate")
#mode.set("material", material_substrate)
#mode.set("x", 0)
#mode.set("x span", w_slab)
#mode.set("y", 0)
#mode.set("y span", wg_length)
#mode.set("z min", -h_substrate)
#mode.set("z max", 0)
#mode.set("alpha", 0.5)

#Draw slab
#mode.addrect()
#mode.set("name","LN slab")
#mode.set("material", material_thinfilm)
#mode.set("x", 0)
#mode.set("x span", w_slab)
#mode.set("y", 0)
#mode.set("y span", wg_length)
#mode.set("z min", 0)
#mode.set("z max", h_slab)
#mode.set("alpha", 0.5)

#Draw Ridge
#mode.addrect()
#mode.set("name","LN ridge")
#mode.set("material", material_thinfilm)
#mode.set("x", 0)
#mode.set("x span", w_ridge)
#mode.set("y", 0)
#mode.set("y span", wg_length)
#mode.set("z min", h_slab)
#mode.set("z max", h_LN)
#mode.set("alpha", 0.5)

##################################################################
#FDTD solver

#Geometry
#mode.addvarfdtd()
#mode.set("x", 0)
#mode.set("x span", w_slab*0.9)
#mode.set("y", 0)
#mode.set("y span", wg_length*0.9)
#mode.set("z min", -h_substrate*0.8)
#mode.set("z max", h_LN*4)
#
##Effective index method
##Location of main slab mode
#mode.set("x0", 0)
#mode.set("y0", 0)
#
##Bandwidth
#mode.set("bandwidth", "broadband")
#mode.set("fit tolerance", 1e-5)
#mode.set("max coefficients", 20)
#mode.set("number of test points", 1)
#test_points = np.array([5e-6,0])
#mode.set("test points", test_points)
#
##Boundary conditions
#mode.set("x min bc", "PML")
#mode.set("x max bc", "PML")
#mode.set("y min bc", "PML")
#mode.set("y max bc", "PML")
#mode.set("z min bc", "PML")
#mode.set("z max bc", "PML")
#
##Add source
#mode.addmodesource()
#mode.set("injection axis", "y-axis")
#mode.set("x",0)
#mode.set("x span", w_slab)
#mode.set("y", -0.8*wg_length/2)
#mode.set("wavelength start", wavelength_start*1e-6)
#mode.set("wavelength stop", wavelength_stop*1e-6)
#
##Add monitors
#mode.addpower()