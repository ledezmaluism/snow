# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:06:34 2019

@author: luish
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import matplotlib.patches as patches
#import imp

from scipy.constants import pi, c
#lumapi = imp.load_source("lumapi", "C:/Program Files/Lumerical/MODE/api/python/lumapi.py")
#Start lumerical MODE session
#mode = lumapi.MODE("Template_Luis.lms")

def draw_ridge(mode, material_LN, h_LN, h_etch, w_ridge, theta, wg_length, 
               x0=0, name='ridge'):
    '''
    Units: um
    '''
    #Change everything to meters
    h_LN = float(h_LN*1e-6)
    h_etch = float(h_etch*1e-6)
    w_ridge = float(w_ridge*1e-6)
    wg_length = float(wg_length*1e-6)
    x0 = float(x0*1e-6)
    
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
    '''
    Units: um
    '''
    #Change everything to meters
    h_LN = float(h_LN*1e-6)
    
    h_substrate = float(h_substrate*1e-6)
    h_etch = float(h_etch*1e-6)
    w_slab = float(w_slab*1e-6)
    wg_length = float(wg_length*1e-6)
    x0 = float(x0*1e-6)
    
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
    '''
    Units: um
    '''
    finemesh = float(finemesh*1e-6)
    x0 = float(x0*1e-6)
    y0 = float(y0*1e-6)
    x_span = float(x_span*1e-6)
    y_span = float(y_span*1e-6)
    
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
    meshsize = meshsize*1e-6
    h_LN = h_LN*1e-6
    h_substrate = h_substrate*1e-6
    h_margin = h_margin*1e-6
    
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
           
    #Change everything to meters
    meshsize = float(meshsize*1e-6)
    h_LN = float(h_LN*1e-6)
    h_substrate = float(h_substrate*1e-6)
    w_slab = float(w_slab*1e-6)
    wg_length = float(wg_length*1e-6)
    h_margin = float(h_margin*1e-6)
    
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

def solve_mode(mode, wavelength, nmodes=4):
    mode.set("wavelength", wavelength*1e-6)
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
    mode.setanalysis("stop wavelength", wavelength*1e-6)
    mode.setanalysis("detailed dispersion calculation",1)
    mode.setanalysis("number of points", 1)
    mode.setanalysis("number of trial modes", 5)
    mode.frequencysweep()
    vg = mode.getdata("frequencysweep","vg")*1e3/1e15 #mm/fs
    D = mode.getdata("frequencysweep","D")
    GVD = -D*(wavelength*1e-6)**2/(2*pi*c)*1e27 #"fs^2/mm"
    return vg, GVD

def get_mode(mode, k):         
    #Get grid
    x = mode.getdata("FDE::data::mode"+str(k),"x")
    y = mode.getdata("FDE::data::mode"+str(k),"y")
    
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
    
def mode_area(mode, n):
    pass

def sellmeier(A, B, wl):
    '''
    Parameters
    ----------
    A : ARRAY
        Ak coefficients
    B : ARRAY
        Bk coefficients
    wl : SCALAR
        Wavelegnth.

    Returns
    -------
    Refractive index from Sellmeiers expansion

    '''
    n2 = 1

    for k in range(A.size):
        n2 += A[k]*wl**2/(wl**2 - B[k])

    return np.sqrt(n2)

def refractive_index(material, wl):
    '''
    Parameters
    ----------
    material : STRING
        'SiO2', 'Sapphire', 'LN_MgO_o', 'LN_MgO_e', 'LN_o', 'LN_e'
    wl : SCALAR
        Wavelength

    Returns
    -------
    Refractive index at given wavelength

    '''
    A = 0
    B = 0
    if material=='SiO2':
        A = np.array([0.6961663, 0.4079426, 0.8974794])
        B = np.array([0.0684043, 0.1162414, 9.896161])
        B = B**2
    elif material=='Sapphire':
        A = np.array([1.5039759, 0.55069141, 6.5927379])
        B = np.array([0.0740288, 0.1216529, 20.072248])
        B = B**2
    elif material=='LN_MgO_e':
        A = np.array([2.2454, 1.3005, 6.8972])
        B = np.array([0.01242, 0.05313, 331.33])
    elif material=='LN_MgO_o':
        A = np.array([2.4272, 1.4617, 9.6536])
        B = np.array([0.01478, 0.05612, 371.216])
    elif material=='LN_o':
        A = np.array([2.6734, 1.2290, 12.614])
        B = np.array([0.01764, 0.05914, 474.6])
    elif material=='LN_e':
        A = np.array([2.9804, 0.5981, 8.9543])
        B = np.array([0.02047, 0.0666, 416.08])
    else:
        print('wrong material')

    return sellmeier(A, B, wl)
