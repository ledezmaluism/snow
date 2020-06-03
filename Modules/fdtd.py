# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:12:11 2018

@author: luish
"""

import numpy as np
from scipy.constants import c,pi,mu_0,epsilon_0

###############################################################################
###############################################################################
def fdtd(setup, material, MonitorsK):
    #Retrieve setup from dictionary
    N_T = setup['N_T']
    dt = setup['dt']
    dz = setup['dz']
    STEPS = setup['STEPS']
    k_source = setup['k_source']
    ey_source = setup['ey_source']
    
    #Retrieve Material properties
    w_1 = material['w_1']
    w_2 = material['w_2']
    w_3 = material['w_3']
    chi_1_1 = material['chi_1_1']
    chi_1_2 = material['chi_1_2']
    chi_1_3 = material['chi_1_3']
    chi_2_1 = material['chi_2_1']
    
    #Initialize fields
    ey = np.zeros(N_T+1)
    hx = np.zeros(N_T)
    jp = np.zeros(N_T+1)
    p1 = np.zeros(N_T+1)
    jp1 = np.zeros(N_T+1)
    p2 = np.zeros(N_T+1)
    jp2 = np.zeros(N_T+1)
    p3 = np.zeros(N_T+1)
    jp3 = np.zeros(N_T+1)

    #Initialize monitors
    ey_monitor = np.zeros((STEPS, len(MonitorsK)))

    #Initialize boundary conditions
    ey_0_2 = 0
    ey_0_1 = 0
    ey_N_2 = 0
    ey_N_1 = 0

    #Update factors
    alpha_h = dt/(dz*mu_0)
    alpha_e = dt/(dz*epsilon_0)
    alpha_e2 = dt/(epsilon_0)
    alpha_j1 = dt*w_1**2
    alpha_j2 = dt*w_2**2
    alpha_j3 = dt*w_3**2

    for n in range(0,STEPS):
    
        #Magnetic field update
        hx = hx +  alpha_h*(ey[1:]-ey[:N_T])
         
        #Polarization and polarization currents   
        jp1[1:N_T] = jp1[1:N_T] + alpha_j1*(epsilon_0*chi_1_1[1:N_T]*ey[1:N_T] + epsilon_0*chi_2_1[1:N_T]*ey[1:N_T]**2 - p1[1:N_T])
        p1[1:N_T] = p1[1:N_T] + dt*jp1[1:N_T]
        
        jp2[1:N_T] = jp2[1:N_T] + alpha_j2*(epsilon_0*chi_1_2[1:N_T]*ey[1:N_T] - p2[1:N_T])
        p2[1:N_T] = p2[1:N_T] + dt*jp2[1:N_T]
        
        jp3[1:N_T] = jp3[1:N_T] + alpha_j3*(epsilon_0*chi_1_3[1:N_T]*ey[1:N_T]- p3[1:N_T])
        p3[1:N_T] = p3[1:N_T] + dt*jp3[1:N_T]
        
        jp = jp1+jp2+jp3
        
        #Electric field update
        ey[1:N_T] = ey[1:N_T] + alpha_e*(hx[1:] - hx[:N_T-1]) - alpha_e2*jp[1:N_T]
    
        #Apply source
        ey[k_source] = ey[k_source] + ey_source[n]
        
        #Apply old boundary condition
        ey[0] = ey_0_2
        ey[N_T] = ey_N_2
        
        #Save boundary condition
        ey_0_2 = ey_0_1
        ey_0_1 = ey[1]
        ey_N_2 = ey_N_1
        ey_N_1 = ey[N_T-1]
        
        #Save field at the monitors locations
        ey_monitor[n,:] = ey[MonitorsK]
        
    return ey_monitor

# Test function for module
if __name__ == '__main__':
    pass