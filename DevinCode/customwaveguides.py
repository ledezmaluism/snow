# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:52:54 2020

@author: devin
"""
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
# import colorcet as cc
import waveguides

nm = 1e-9; um = 1e-6; cm = 1e-2
startwidth = 1.85*um #top width - really should be fct of z
hLN = 700*nm #LN_thickness
hetch = 340*nm
# =============================================================================
# class represents a longitudinally-varying waveguide
# by varying waveguide dimensions, can obtain varying index profiles

# =============================================================================

def constwidthfct(z):
    return startwidth

class customwaveguide:
    def __init__(self, width_fct = constwidthfct, length = 1*cm): 
        # 
        self.length = length #length of waveguide
        self.width_fct = width_fct # function of z that gives varying width
        # Add other parameter can tune later: thickness
        
    def neff(self,wl, z):
        width = self.width_fct(z) #either a scalar of array
        wg = waveguides.waveguide(w_top = width, h_ridge=hLN, h_slab=hLN-hetch)
        n = wg.neff(wl)
        return n
    def neffarray(self, wl, zarray):
        narray = np.zeros_like(zarray)
        for i in range(zarray.size):
            narray[i] = self.neff(wl, zarray[i])
        return narray
    def neffarrayarray(self, wlarray,zarray):
        narrayarray = np.zeros(wlarray.size, dtype = list)
        for i in range(wlarray.size):
            narrayarray[i] = self.neffarray(wlarray[i], zarray)
        return narrayarray #list of lists, each outer represents a wavelength, inner is index of that wavelength as a function of z

    def makez(self):
        N = 100
        z = np.linspace(0.0, self.length, N) # longitudinal distance in m
        return z
    
    #TODO add fct for group index, GVD, 
    # Idea: maybe implement for normal wg, then create and call on that wg for given z
    def groupindex(self, wl, z):
        pass
    
    def GVD(self, wl, z):
        pass
    
    def groupdelay(self, wl):
        pass

def linearwidthfct(startwidth, slope): #TEST
    def wfct(z):
        width = startwidth + slope*z 
        return width
    return wfct



# =============================================================================
# Plotting functions
# =============================================================================
def plotnarray(cwg, wl): #plots index of 1 wavelength over z
    zarray = cwg.makez()
    narray = cwg.neffarray(wl, zarray)
    
    fig = plt.figure();    ax1 = fig.add_subplot(111)
      
    ax1.ticklabel_format(useOffset = False)
    ax1.set_title('neff(z)') ;    ax1.set_xlabel("z (m)");    ax1.set_ylabel('Index n')    
    ax1.plot(zarray, narray)
    
    
def plotnarrayarray(cwg, wlarray): #produces contour lines of index of multiple wavelengths over z (same as plotnarray except diff. colors correspond to diff wl)
    #need legend to label different curves
    za = cwg.makez()
    naa = cwg.neffarrayarray(wlarray, za)
    
    fig = plt.figure();    ax1 = fig.add_subplot(111)
      
    ax1.ticklabel_format(useOffset = False)
    ax1.set_title('neff(z)') ;    ax1.set_xlabel("z (m)");    ax1.set_ylabel('Index n')  
    for i in range(naa.size):
        ax1.plot(za, naa[i])
        
def heatmapnaa(cwg, wlbounds, Nwl = 20): # plots heat map of n on axes z and wl, wl bounds is  2 el tuple
     # Nwl is number of wavelengths to include
    za = cwg.makez()
    wlarray = np.linspace(wlbounds[0], wlbounds[1], Nwl)
    naa = cwg.neffarrayarray(wlarray, za) 
    nmatr = np.zeros([wlarray.size, za.size]) #create nmatr of correct dimension
    for i in range(wlarray.size): #should be same as naa.size, gets wl index
        for j in range(za.size): # gets pos index
            nmatr[i][j] = naa[i][j] #gets index of refraction and stores in proper matrix     
    
    X,Y = np.meshgrid(za, wlarray)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X/cm, Y/um, nmatr) #, cmap = cc.cm["rainbow"])
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Wavelength (um)')
    plt.colorbar(im, ax=ax);
    
# =============================================================================
#     Functions for normal waveguides with no variance in z 
# =============================================================================
def plotnvswl(cwg, z, wlbounds, Nwl = 20): # plots n vs wavelength for given z value
    wlarray = np.linspace(wlbounds[0], wlbounds[1], Nwl)
    neff = np.zeros_like(wlarray)
    for k in range(wlarray.size):
        neff[k] = cwg.neff(wlarray[k], z)
    fig = plt.figure();    ax1 = fig.add_subplot(111)
      
    ax1.ticklabel_format(useOffset = False)
    ax1.set_title('neff(wl)') ;    ax1.set_xlabel("Wavelength (um)");    ax1.set_ylabel('Index n')    
    ax1.plot(wlarray/um, neff)
# =============================================================================
# TESTING
# =============================================================================
def testwidthfct(z):
    const = 100*nm / cm # for every cm in z, width changes by 100 nm
    width = startwidth + const*z 
    return width

def testclassfct():
    zarray = np.arange(0,1*cm, 100000*nm)
    print(testwidthfct(zarray[0]))
    print(testwidthfct(zarray))
    
    cwg = customwaveguide(testwidthfct, length = 2 * cm)
    wl = 1*um
    n = cwg.neff(wl,zarray[-1])
    print(n)
    narray = cwg.neffarray(wl,zarray)
    print(narray)
    
    wlarray = np.arange(1*um, 3*um, 1*um)
    narrayarray = cwg.neffarrayarray(wlarray,zarray)
    print(narrayarray)
    print(narrayarray[0])

# testclassfct()