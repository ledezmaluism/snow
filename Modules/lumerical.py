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


def draw_ridge(MODE, material_LN, h_LN, h_etch, w_ridge, theta, wg_length, 
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
    MODE.switchtolayout()
    MODE.addrect()
    MODE.set("name",name)
    MODE.set("material", material_LN)
    MODE.set("x", x0)
    MODE.set("x span", w_ridge)
    MODE.set("y min", h_slab)
    MODE.set("y max", h_LN)
    MODE.set("z min", Zmin)
    MODE.set("z max", Zmax)
    MODE.set("alpha", 0.5)
    
    #Draw left sidewall
    MODE.addtriangle()
    MODE.set("name",name+"_left_sidewall")
    MODE.set("material", material_LN)
    MODE.set("x", x0-w_ridge/2)
    MODE.set("y", h_slab)
    MODE.set("z min", Zmin)
    MODE.set("z max", Zmax)
    V = np.array([[0, -w_sidewall, 0],[0, 0, h_etch]])
    MODE.set("vertices", V)
    MODE.set("alpha", 0.5)
    
    #Draw right sidewall
    MODE.addtriangle()
    MODE.set("name",name+"_right_sidewall")
    MODE.set("material", material_LN)
    MODE.set("x", x0+w_ridge/2)
    MODE.set("y", h_slab)
    MODE.set("z min", Zmin)
    MODE.set("z max", Zmax)
    V = np.array([[0, w_sidewall, 0],[0, 0, h_etch]])
    MODE.set("vertices", V)
    MODE.set("alpha", 0.5)

def draw_substrate(MODE, material_LN, material_substrate, h_LN, h_substrate, 
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
    MODE.switchtolayout()
    MODE.addrect()
    MODE.set("name","Substrate")
    MODE.set("material", material_substrate)
    MODE.set("x", x0)
    MODE.set("x span", w_slab)
    MODE.set("y min", -h_substrate)
    MODE.set("y max", 0)
    MODE.set("z min", Zmin)
    MODE.set("z max", Zmax)
    MODE.set("alpha", 0.5)
    
    #Draw slab
    MODE.addrect()
    MODE.set("name","slab")
    MODE.set("material", material_LN)
    MODE.set("x", x0)
    MODE.set("x span", w_slab)
    MODE.set("y min", 0)
    MODE.set("y max", h_slab)
    MODE.set("z min", Zmin)
    MODE.set("z max", Zmax)
    MODE.set("alpha", 0.5)
    
def draw_wg(MODE, material_LN, material_substrate, h_LN, h_substrate, h_etch, 
            w_ridge, w_slab, theta, wg_length, x0=0, delete=True):
    
    MODE.switchtolayout()
    if delete:
        MODE.deleteall()
    draw_substrate(MODE, material_LN, material_substrate, h_LN, h_substrate,
                   h_etch, w_slab, wg_length, x0=0)
    draw_ridge(MODE, material_LN, h_LN, h_etch, w_ridge, theta, wg_length,
               x0, name='ridge')
    time.sleep(0.1)
    
def add_fine_mesh(MODE, finemesh, h_LN, w_ridge, x_factor=1.1, y_factor=1.1):
    x_span = w_ridge*x_factor
    y_span = h_LN*y_factor
    add_fine_mesh_lowlevel(MODE, finemesh, 0, h_LN/2, x_span, y_span)
   

def add_fine_mesh_lowlevel(MODE, finemesh, x0, y0, x_span, y_span):

    finemesh = float(finemesh)
    x0 = float(x0)
    y0 = float(y0)
    x_span = float(x_span)
    y_span = float(y_span)
    
    MODE.switchtolayout()
    MODE.addmesh()
    MODE.set("x", x0)
    MODE.set("x span", x_span)
    MODE.set("y", y0)
    MODE.set("y span", y_span)
    MODE.set("override x mesh", 1)
    MODE.set("override y mesh", 1)
    MODE.set("override z mesh", 0)
    MODE.set("dx", finemesh);
    MODE.set("dy", finemesh);

def add_1D_mode_solver(MODE, meshsize, h_LN, h_substrate, h_margin):
    
    #Change everything to meters
    meshsize = meshsize
    h_LN = h_LN
    h_substrate = h_substrate
    h_margin = h_margin
    
    #Calculate simulation volume
    Ymin = -h_substrate - h_margin
    Ymax = h_LN + h_margin
    
    MODE.switchtolayout()
    MODE.addfde()
    MODE.set("solver type","1D Y:Z prop")
    MODE.set("z", 0)
    MODE.set("x", 0);
    MODE.set("y max", Ymax); 
    MODE.set("y min", Ymin);

    MODE.set("define y mesh by","maximum mesh step");
    MODE.set("dy", meshsize);

def add_2D_mode_solver(MODE, meshsize, h_LN, h_substrate, w_slab, wg_length,
                       h_margin, symmetry=None):
           
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
    
    # add 2D MODE solver (waveguide cross-section)
    MODE.switchtolayout()
    MODE.addfde()  
    MODE.set("solver type", "2D Z normal")
    MODE.set("z", 0)
    MODE.set("x", 0)
    MODE.set("x span", X_span)  
    MODE.set("y max", Ymax)
    MODE.set("y min", Ymin)
    MODE.set("solver type","2D Z normal")
    MODE.set("define x mesh by","maximum mesh step")
    MODE.set("dx", meshsize)
    MODE.set("define y mesh by","maximum mesh step")
    MODE.set("dy", meshsize)
    MODE.set("x min bc", "PML")
    MODE.set("y min bc", "PML")
    MODE.set("x max bc", "PML")
    MODE.set("y max bc", "PML")
    MODE.set("mesh refinement", "conformal variant 0")
    
    if symmetry == 'TE':
        MODE.set("allow symmetry on all boundaries", True)
        MODE.set("x max bc", "Anti-Symmetric")
        MODE.set("x max", 0)
    elif symmetry == 'TM':
        MODE.set("allow symmetry on all boundaries", True)
        MODE.set("x max bc", "Symmetric")
        MODE.set("x max", 0)
    
    time.sleep(0.1)

def solve_mode(MODE, wavelength, nmodes=20):
    MODE.switchtolayout()
    MODE.set("wavelength", wavelength)
    MODE.set("number of trial modes", nmodes)
    n = int(MODE.findmodes())
    
    neff = np.zeros(n) #effective  index matrix
    ng = np.zeros(n) #group  index matrix
    loss = np.zeros(n) #group  index matrix
    TE = np.zeros(n) #TE polarization fraction
    for k in range(n):    
        neff[k] = np.float(np.real(MODE.getdata("FDE::data::mode"+str(k+1), "neff")))
        ng[k] = np.float(np.real(MODE.getdata('FDE::data::mode'+str(k+1), 'ng')))
        loss[k] = np.float(np.real(MODE.getdata('FDE::data::mode'+str(k+1), 'loss')))
        TE[k] = np.float(MODE.getdata("FDE::data::mode"+str(k+1), "TE polarization fraction"))
        
    return neff, ng, loss, TE

def dispersion_analysis(MODE, wavelength, mode_number):
    # MODE.switchtolayout()
    MODE.selectmode(mode_number)
    MODE.setanalysis("track selected mode",1)
    MODE.setanalysis("stop wavelength", wavelength)
    MODE.setanalysis("detailed dispersion calculation",1)
    MODE.setanalysis("number of points", 1)
    MODE.setanalysis("number of trial modes", 20)
    MODE.frequencysweep()
    vg = MODE.getdata("frequencysweep","vg")
    D = MODE.getdata("frequencysweep","D")
    GVD = -D*(wavelength)**2/(2*pi*c)
    return vg, GVD

class mode():
    '''
    Full mode including both E and H fields
    '''
    def __init__(self):
        pass
    
    def set_manually(self, E, H, neff):
        self.E = E
        self.H = H
        self.neff = neff
    
    def get_from_lumerical(self, MODE, k):
        #Get grid
        x = MODE.getdata("FDE::data::mode"+str(k),"x")
        y = MODE.getdata("FDE::data::mode"+str(k),"y")
        
        #Get fields
        Ex = MODE.getdata("FDE::data::mode"+str(k),"Ex")
        Ey = MODE.getdata("FDE::data::mode"+str(k),"Ey")
        Ez = MODE.getdata("FDE::data::mode"+str(k),"Ez")
        Hx = MODE.getdata("FDE::data::mode"+str(k),"Hx")
        Hy = MODE.getdata("FDE::data::mode"+str(k),"Hy")
        Hz = MODE.getdata("FDE::data::mode"+str(k),"Hz")
        
        #Get rid of singleton dimensions
        x = np.squeeze(x)
        y = np.squeeze(y)
        Ex = np.squeeze(Ex)
        Ey = np.squeeze(Ey)
        Ez = np.squeeze(Ez)
        Hx = np.squeeze(Hx)
        Hy = np.squeeze(Hy)
        Hz = np.squeeze(Hz)
        
        neff = np.float(np.real(MODE.getdata("FDE::data::mode"+str(k), "neff")))
        
        self.E = field_2D(x, y, Ex, Ey, Ez)
        self.H = field_2D(x, y, Hx, Hy, Hz)
        self.neff = neff

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
        return np.real(N)

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
        
    def rescale(self, factor):
        Ax = self.x * factor
        Ay = self.y * factor
        Az = self.z * factor
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
    

###############################################################################
###############################################################################
def _test_():
    '''
    Test function for module  
    '''
    import imp
    if 'MODE' not in vars():
        lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/2020a/api/python/lumapi.py")
        MODE = lumapi.MODE("Template_Luis.lms")
    
    '''
    Units
    '''
    um = 1e-6
    nm = 1e-9
    
    '''
    Input parameters
    '''
    wavelength = 1*um
    h_LN = 700*nm
    h_etch = 250*nm
    w_ridge = 1500*nm
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
    h_margin = 4*wavelength
    h_substrate = 4*wavelength
    meshsize = wavelength/20
    finemesh = wavelength/40
    
    '''
    Materials
    '''
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
    draw_wg(MODE, material_thinfilm, material_substrate,
                  h_LN, h_substrate, h_etch, w_ridge, w_slab, theta, wg_length)
    add_fine_mesh(MODE, finemesh, h_LN, w_ridge_base, x_factor=1.2, y_factor=1.5)
    add_2D_mode_solver(MODE, meshsize, h_LN, h_substrate, 
                             w_slab, wg_length, h_margin)
    
if __name__ == '__main__':
    # _test_()
    pass