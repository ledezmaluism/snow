# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:06:34 2019
@author: luis ledezma


Python module for simulations with Lumerical

to do: 
    replace functions for drawing waveguides for native lumerical function
    https://kb.lumerical.com/ref_sim_obj_structures_-_waveguide.html

"""
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import matplotlib.patches as patches

from scipy.constants import pi, c
from numpy import abs
from scipy.integrate import simps


def draw_ridge(mode, material_LN, h_LN, h_etch, w_ridge, theta, wg_length, 
               x0=0, name='ridge'):

    h_LN = float(h_LN)
    h_etch = float(h_etch)
    w_ridge = float(w_ridge)
    wg_length = float(wg_length)
    x0 = float(x0)
    
    #Calculate some extra geometric parameters
    h_slab = h_LN - h_etch
    w_sidewall = h_etch/np.tan(theta*pi/180)
    Zmin = -wg_length/2
    Zmax = wg_length/2
    
    #Main ridge
    mode.addrect()
    mode.set("name",name)
    mode.set("material", material_LN)
    mode.set("x", x0)
    mode.set("x span", w_ridge)
    mode.set("y min", h_slab)
    mode.set("y max", h_LN)
    mode.set("z min", Zmin)
    mode.set("z max", Zmax)
    mode.set("alpha", 0.5)
    
    #Draw left sidewall
    mode.addtriangle()
    mode.set("name",name+"_left_sidewall")
    mode.set("material", material_LN)
    mode.set("x", x0-w_ridge/2)
    mode.set("y", h_slab)
    mode.set("z min", Zmin)
    mode.set("z max", Zmax)
    V = np.array([[0, -w_sidewall, 0],[0, 0, h_etch]])
    mode.set("vertices", V)
    mode.set("alpha", 0.5)
    
    #Draw right sidewall
    mode.addtriangle()
    mode.set("name",name+"_right_sidewall")
    mode.set("material", material_LN)
    mode.set("x", x0+w_ridge/2)
    mode.set("y", h_slab)
    mode.set("z min", Zmin)
    mode.set("z max", Zmax)
    V = np.array([[0, w_sidewall, 0],[0, 0, h_etch]])
    mode.set("vertices", V)
    mode.set("alpha", 0.5)

def draw_substrate(mode, material_LN, material_substrate, h_LN, h_substrate, 
                   h_etch, w_slab, wg_length, x0=0):
    
    h_LN = float(h_LN)
    h_substrate = float(h_substrate)
    h_etch = float(h_etch)
    w_slab = float(w_slab)
    wg_length = float(wg_length)
    x0 = float(x0)
    
    #Calculate some extra geometric parameters
    h_slab = h_LN - h_etch
    Zmin = -wg_length/2
    Zmax = wg_length/2
    
    #Draw substrate
    mode.addrect()
    mode.set("name","Substrate")
    mode.set("material", material_substrate)
    mode.set("x", x0)
    mode.set("x span", w_slab)
    mode.set("y min", -h_substrate)
    mode.set("y max", 0)
    mode.set("z min", Zmin)
    mode.set("z max", Zmax)
    mode.set("alpha", 0.5)
    
    #Draw slab
    mode.addrect()
    mode.set("name","slab")
    mode.set("material", material_LN)
    mode.set("x", x0)
    mode.set("x span", w_slab)
    mode.set("y min", 0)
    mode.set("y max", h_slab)
    mode.set("z min", Zmin)
    mode.set("z max", Zmax)
    mode.set("alpha", 0.5)
    
def draw_wg(mode, material_LN, material_substrate, h_LN, h_substrate, h_etch, 
            w_ridge, w_slab, theta, wg_length, x0=0, delete=True):
    
    if delete:
        mode.switchtolayout()
        mode.deleteall()
    draw_substrate(mode, material_LN, material_substrate, h_LN, h_substrate,
                   h_etch, w_slab, wg_length, x0=0)
    draw_ridge(mode, material_LN, h_LN, h_etch, w_ridge, theta, wg_length,
               x0, name='ridge')
    time.sleep(0.1)
    
def add_fine_mesh(mode, finemesh, h_LN, w_ridge, x_factor=1.1, y_factor=1.1):
    x_span = w_ridge*x_factor
    y_span = h_LN*y_factor
    add_fine_mesh_lowlevel(mode, finemesh, 0, h_LN/2, x_span, y_span)
   

def add_fine_mesh_lowlevel(mode, finemesh, x0, y0, x_span, y_span):

    finemesh = float(finemesh)
    x0 = float(x0)
    y0 = float(y0)
    x_span = float(x_span)
    y_span = float(y_span)
    
    mode.addmesh()
    mode.set("x", x0)
    mode.set("x span", x_span)
    mode.set("y", y0)
    mode.set("y span", y_span)
    mode.set("override x mesh", 1)
    mode.set("override y mesh", 1)
    mode.set("override z mesh", 0)
    mode.set("dx", finemesh);
    mode.set("dy", finemesh);

def add_1D_mode_solver(mode, meshsize, h_LN, h_substrate, h_margin):
    
    #Change everything to meters
    meshsize = meshsize
    h_LN = h_LN
    h_substrate = h_substrate
    h_margin = h_margin
    
    #Calculate simulation volume
    Ymin = -h_substrate - h_margin
    Ymax = h_LN + h_margin
    
    mode.addfde()
    mode.set("solver type","1D Y:Z prop")
    mode.set("z", 0)
    mode.set("x", 0);
    mode.set("y max", Ymax); 
    mode.set("y min", Ymin);

    mode.set("define y mesh by","maximum mesh step");
    mode.set("dy", meshsize);

def add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, w_slab, wg_length,
                       h_margin):
           
    meshsize = float(meshsize)
    h_LN = float(h_LN)
    h_substrate = float(h_substrate)
    w_slab = float(w_slab)
    wg_length = float(wg_length)
    h_margin = float(h_margin)
    
    #Calculate simulation volume
    pml_margin = 12*meshsize
    X_span = w_slab - 2*pml_margin
    Ymin = -h_substrate + pml_margin
    Ymax = h_LN + h_margin
    
    # add 2D mode solver (waveguide cross-section)
    mode.addfde()  
    mode.set("solver type", "2D Z normal")
    mode.set("z", 0)
    mode.set("x", 0)
    mode.set("x span", X_span)  
    mode.set("y max", Ymax)
    mode.set("y min", Ymin)
    mode.set("solver type","2D Z normal")
    mode.set("define x mesh by","maximum mesh step")
    mode.set("dx", meshsize)
    mode.set("define y mesh by","maximum mesh step")
    mode.set("dy", meshsize)
    mode.set("x min bc", "PML")
    mode.set("y min bc", "PML")
    mode.set("x max bc", "PML")
    mode.set("y max bc", "PML")
    mode.set("mesh refinement", "conformal variant 0")
    
    time.sleep(0.1)

def solve_mode(mode, wavelength, nmodes=20):
    mode.set("wavelength", wavelength)
    mode.set("number of trial modes", nmodes)
    n = int(mode.findmodes())
    
    neff = np.zeros(n) #effective  index matrix
    TE = np.zeros(n) #TE polarization fraction
    for k in range(n):    
        neff[k] = np.float(np.real(mode.getdata("FDE::data::mode"+str(k+1), "neff")))
        TE[k] = np.float(mode.getdata("FDE::data::mode"+str(k+1), "TE polarization fraction"))

    return neff, TE

def dispersion_analysis(mode, wavelength, mode_number):
    mode.selectmode(mode_number)
    mode.setanalysis("track selected mode",1)
    mode.setanalysis("stop wavelength", wavelength)
    mode.setanalysis("detailed dispersion calculation",1)
    mode.setanalysis("number of points", 1)
    mode.setanalysis("number of trial modes", 5)
    mode.frequencysweep()
    vg = mode.getdata("frequencysweep","vg")*1e3/1e15 #mm/fs
    D = mode.getdata("frequencysweep","D")
    GVD = -D*(wavelength)**2/(2*pi*c)*1e27 #"fs^2/mm"
    return vg, GVD

def get_mode(mode, k):         
    #Get grid
    x = np.squeeze(mode.getdata("FDE::data::mode"+str(k),"x"))
    y = np.squeeze(mode.getdata("FDE::data::mode"+str(k),"y"))
    
    #Get fields
    Ex = mode.getdata("FDE::data::mode"+str(k),"Ex")
    Ey = mode.getdata("FDE::data::mode"+str(k),"Ey")
    Ez = mode.getdata("FDE::data::mode"+str(k),"Ez")
    Hx = mode.getdata("FDE::data::mode"+str(k),"Hx")
    Hy = mode.getdata("FDE::data::mode"+str(k),"Hy")
    Hz = mode.getdata("FDE::data::mode"+str(k),"Hz")
    
    #Get rid of singleton dimensions
    Ex = np.squeeze(Ex)
    Ey = np.squeeze(Ey)
    Ez = np.squeeze(Ez)
    Hx = np.squeeze(Hx)
    Hy = np.squeeze(Hy)
    Hz = np.squeeze(Hz)
    
    neff = np.float(np.real(mode.getdata("FDE::data::mode"+str(k), "neff")))
    
    return x, y, Ex, Ey, Ez, Hx, Hy, Hz, neff

def plot_2D_mode(F, x, y, h_LN, h_substrate, h_etch, w_ridge, w_slab, theta,
                 x_margin_view=1, y_margin_view=1, cmap=cm.jet):
    
    #Plot field in simulation coordinates
    fig,ax = plt.subplots(1)  
    Fabs = np.abs(F)
    im = ax.imshow(np.transpose(Fabs), aspect='equal', 
              interpolation='bicubic', cmap=cmap, origin='lower', 
              extent=[x.min(), x.max(), y.min(), y.max()],
              vmax=Fabs.max(), vmin=Fabs.min())
    
    #Show only interesting region
    Xmin = (-w_slab/2 - x_margin_view)
    Xmax = (w_slab/2 + x_margin_view)
    Ymin = (-h_substrate - y_margin_view)
    Ymax = (h_LN + y_margin_view)
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    
    #ax.set_xlabel('x ($\mu m$)')
    #ax.set_ylabel('y ($\mu m$)')
    ax.set_yticks([])
    ax.set_xticks([])

    #Overlay waveguide
    h_slab = (h_LN - h_etch)
    h_substrate = h_substrate
    h_LN = h_LN
    w_slab = w_slab
    w_ridge = w_ridge
    w_sidewall = h_etch/np.tan(theta*pi/180)
    
#    substrate = patches.Rectangle((-w_slab/2,-h_substrate), w_slab, h_substrate,
#                                  linewidth=1,edgecolor='w', facecolor='none')
#    v_ridge = np.array([[-w_ridge/2-w_sidewall,h_slab], [-w_ridge/2,h_LN],
#                        [w_ridge/2,h_LN], [w_ridge/2+w_sidewall,h_slab],
#                        [w_slab/2,h_slab], [w_slab/2,0], 
#                        [-w_slab/2,0], [-w_slab/2, h_slab] ] )
#    ridge = patches.Polygon(v_ridge, linewidth=1, 
#                            edgecolor='w', facecolor='none')
    
#    ax.add_patch(substrate)
#    ax.add_patch(ridge)
    
    fig.colorbar(im)


class mode():
    '''
    Full mode including both E and H fields
    '''
    def __init__(self, E, H):
        self.E = E
        self.H = H
        
    def poynting(self):
        E = self.E
        H = self.H
        S = E.cross(H.conj())
        return S
    
    def N(self):
        S = self.poynting()
        x = self.E.xx
        y = self.E.yy
        N = 0.5*simps(simps(S.z, y), x)
        return N

class field_2D():
    '''
    Transversal vector field
    '''
    def __init__(self, x, y, Ax, Ay, Az):
        self.xx = np.squeeze(x)
        self.yy = np.squeeze(y)
        self.x = np.squeeze(Ax)
        self.y = np.squeeze(Ay)
        self.z = np.squeeze(Az)
        
    def dot(self, E):
        Ax = self.x
        Ay = self.y
        Az = self.z
        Ex = E.x
        Ey = E.y
        Ez = E.z
        return Ex*Ax + Ey*Ay + Ez*Az
    
    def conj(self):
        Ax = np.conj(self.x)
        Ay = np.conj(self.y)
        Az = np.conj(self.z)
        xx = self.xx
        yy = self.yy
        A = field_2D(xx, yy, Ax, Ay, Az)
        return A
    
    def cross(self, E):
        Ax = self.x
        Ay = self.y
        Az = self.z
        Ex = E.x
        Ey = E.y
        Ez = E.z
        Cx = Ay*Ez - Az*Ey
        Cy = Az*Ex - Ax*Ez
        Cz = Ax*Ey - Ay*Ex
        C = field_2D(self.xx, self.yy, Cx, Cy, Cz)
        return C
    
    def magsq(self):
        return abs(self.dot(self.conj()))
    
    def overlap3(self, E2, E3, axis='xxx'):
        x = self.xx
        y = self.yy
        
        if axis=='xxx':
            integrand = self.x * E2.x * E3.x
        
        return simps(simps(integrand, y), x)

###############################################################################
###############################################################################
def _test_():
    '''
    Test function for module  
    '''
    import imp
    if 'mode' not in vars():
        lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2020a/api/python/lumapi.py")
        mode = lumapi.MODE("Template_Luis.lms")
    
    '''
    Geometry
    '''
    wavelength = 1
    h_LN = 0.7
    h_etch = 0.3
    w_ridge = 1.5
    h_slab = h_LN - h_etch
    theta = 60
    wg_length = 10
    w_ridge_base = w_ridge + 2*h_etch/np.tan(theta*pi/180)

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
    material_substrate = "SiO2_analytic"    
    material_thinfilm = "LN_analytic_MgO_doped_xne"
    
    '''
    Drawing and setup
    '''
    draw_wg(mode, material_thinfilm, material_substrate,
                  h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
    add_fine_mesh(mode, finemesh, h_LN, w_ridge_base, x_factor=1.2, y_factor=1.5)
    add_2D_mode_solver(mode, meshsize, h_LN, h_substrate, 
                             w_slab, wg_length, h_margin)

if __name__ == '__main__':
    # _test_()
    pass