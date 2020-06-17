# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:50:52 2020
@author: devin
"""
#%matplotlib inline
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt


# =============================================================================
# class grating represents a grating (PPLN). It must be passed an index of refraction n
# in order to convert to wavelength or frequency
# Outer class uses this one called "poledwaveguides"

# =============================================================================
class grating:
    def __init__(self, k0=1e6, kbw = 1e5, chirp = -2.5e6, length = 1e-2, dutycycle = 0.5): #maybe add apodization later
        # k0 is central grating vector that compensates for dK, kbw is the spread of k 
        # chirp is rate that k is changed (units 1/m^2), # length is length of ppln waveguide 
        self.k0 = k0
        self.bw = kbw; 
        self.chirp = chirp; 
        self.length = length; 
        self.dutycycle = dutycycle
        
        #varialbes to create z array 
        T = 1e-8 #spacing between sampled points, in m
        N = int(self.length / T) #number of sampled points, must be an integer
        self.T = T; self.N = N
        
    def makez(self):
        z = np.linspace(0.0, self.N*self.T, self.N) # longitudinal distance in m
        return z
    #instead of sampling in z make actual fct in z: 
    # z is a variable for long. position, can be a scalar or an array
    def gratingvector(self, z):
        k = self.k0 + self.chirp*(z-self.length/2)
        return k
    
    def gratingperiod(self, z):
        k = self.gratingvector(z)
        A = 2*np.pi/k
        return A  
    
    # turn period A into piecewise profile of actual grating
    # returns d, the normalized grating structure d(z) = chi_eff(z)/chi0
    def grating(self, z):
        D =self.dutycycle
        A = self.gratingperiod(z)
        d = np.sign(np.cos(2*np.pi/A *z) - np.cos(np.pi*D)) 
        return d 
    
    #returns G, the FT of the grating. There should be a peak at k0
    def RLV(self):
        z = self.makez()
        d = self.grating(z)
        G = fft(d) #not normalized ata ll, could use fftshift to get centered
        return G
    
um = 1e-6; 
# Static fcts , inputs have a grating g
def plotgratingvector(g): #plots 1st order grating vector Kg(z)
    z = g.makez(); k = g.gratingvector(z)
    fig, ax = plt.subplots()
    ax.plot(z, k)
    ax.set_title('Grating Vector')
    plt.xlabel("z (m)")
    plt.ylabel('Kg (rad/m)')
    
def plotgratingperiod(g):
    z = g.makez(); A = g.gratingperiod(z)
    fig, ax = plt.subplots()
    ax.ticklabel_format(style='sci', useOffset = False)
    ax.plot(z, A/um)
    ax.set_title('Grating Period ')
    plt.xlabel("z (m)")
    plt.ylabel('Local Grating Period (um)')
        
    # xbounds is a length 2 tuple indicating start and end of plotting in z
def plotgrating(g, xbounds = (0, 100*um)):
    z = g.makez(); d = g.grating(z)
    
    fig, ax = plt.subplots()
    plt.xlim(xbounds)
    ax.plot(z, d)
    ax.set_title('Grating Domain Alignment')
    plt.xlabel("z (m)")
    plt.ylabel('Domain alignment (normalized)')
    
#fct for plotting RLV of grating to see what dK_g can be obtained
def plotRLV(g, kbounds = (0,0)):
    N = g.N; T = g.T; z = g.makez()
    k = g.gratingvector(z)
    
    if kbounds == (0,0):
        kstart = min(k[1], k[-1])
        kend = max(k[1], k[-1])
        kbounds = (kstart, kend)
    
    G = fftshift(g.RLV());
    k = fftshift(2*np.pi*fftfreq(N,T)) # grating wave-vector, units rad/m
        # ignore 1st data pt to ignore constant offset (no problem b/c sin??)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(kbounds)
      
    ax1.ticklabel_format(style='sci', useOffset = False)
    ax1.set_title('Grating k ')    
    ax1.set_xlabel("k (rad/m)")
    ax1.set_ylabel('RLV weight')    
    ax1.plot(k, np.absolute(G))
    

    
# =============================================================================
#     TESTING
# =============================================================================

