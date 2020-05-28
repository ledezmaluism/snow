#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pynlo
import numpy as np
from scipy.constants import pi, c, epsilon_0
km = 1e3
nm = 1e-9
um = 1e-6
mm = 1e-3
ps = 1e-12
fs = 1e-15
GHz = 1e9
THz = 1e12
pJ = 1e-12
mW = 1e-3

def dB(num):
    return 10 * np.log10(np.abs(num)**2)

Window  = 45   # simulation window (ps)
Steps   = 100     # simulation steps
Points  = 2**14  # simulation points

# D1 = -1
# D2 = -20
# D3 = 1

D1slope = 0.019
D2slope = 0.045
D3slope = 0.019

frep = 10*GHz
Pavg = 5
EPP = Pavg/frep
print('Energy = %0.1f pJ' %(EPP/pJ))

FWHM    = 0.670  # pulse duration (ps)
pulseWL = 1559.9   # pulse central wavelength (nm)
frep    = 10e3 # Repetition rate MHz
GDD     = 0.0    # Group delay dispersion (ps^2)
TOD     = 0.0    # Third order dispersion (ps^3)

# create the pulse
pulse = pynlo.light.DerivedPulses.SechPulse(power = 1, # Power will be scaled by set_epp
                                            T0_ps                   = FWHM/1.76,
                                            center_wavelength_nm    = pulseWL,
                                            time_window_ps          = Window,
                                            GDD=GDD, TOD=TOD,
                                            NPTS            = Points,
                                            frep_MHz        = frep,
                                            power_is_avg    = False)

# set the pulse energy
pulse.set_epp(EPP)

def get_betas(D, Dslope, wl):
    b2 = -D*wl**2/(2*pi*c)
    b3 = (wl**2/(2*pi*c))**2 * (Dslope - 2*D/wl)
    return b2,b3


def runsim():
    
    Raman   = True    # Enable Raman effect?
    Steep   = True    # Enable self steepening?
    
    ## Fiber 1 settings (Normal)
    Length  = Len1   # length in m
    beta2, beta3 = get_betas(D1*ps/nm/km, D1slope*ps/nm/nm/km, 1.55*um)
    beta2 = beta2/(ps**2/km) # (ps^2/km)
    beta3 = beta3/(ps**3/km) # (ps^3/km)
    beta4 = 0
    Alpha   = 0.9e-5     # attentuation coefficient (dB/cm)
    Gamma   = 10.5    # Gamma (1/(W km))
    fibWL   = pulseWL # Center WL of fiber (nm)
    alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m
    # create the fiber
    fiber1 = pynlo.media.fibers.fiber.FiberInstance()
    fiber1.generate_fiber(Length, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
                                  gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
    
    ## Fiber 2 settings (SMF)
    Length  = Len2   # length in m
    beta2, beta3 = get_betas(D2*ps/nm/km, D2slope*ps/nm/nm/km, 1.55*um)
    
    beta2 = beta2/(ps**2/km) # (ps^2/km)
    beta3 = beta3/(ps**3/km) # (ps^3/km)
    beta4 = 0
    Alpha   = 0.9e-5     # attentuation coefficient (dB/cm)
    Gamma   = 10.5    # Gamma (1/(W km))
    fibWL   = pulseWL # Center WL of fiber (nm)
    alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m
    # create the fiber
    fiber2 = pynlo.media.fibers.fiber.FiberInstance()
    fiber2.generate_fiber(Length, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
                                  gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
    
    
    # ## Fiber 3 settings (2m of anomalous dispersion HNLF)
    Length  = Len3  # length in m
    beta2, beta3 = get_betas(D3*ps/nm/km, D3slope*ps/nm/nm/km, 1.55*um)
    beta2 = beta2/(ps**2/km) # (ps^2/km)
    beta3 = beta3/(ps**3/km) # (ps^3/km)
    beta4 = 0
    Alpha   = 0.9e-5     # attentuation coefficient (dB/cm)
    Gamma   = 10.5    # Gamma (1/(W km))
    fibWL   = pulseWL # Center WL of fiber (nm)
    alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m
    
    # create the fiber
    fiber3 = pynlo.media.fibers.fiber.FiberInstance()
    fiber3.generate_fiber(Length, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
                                  gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
    
    
    # Propagation
    evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.005, USE_SIMPLE_RAMAN=True,
                     disable_Raman              = np.logical_not(Raman),
                     disable_self_steepening    = np.logical_not(Steep))
    
    y, AW, AT, pulse_out1 = evol.propagate(pulse_in=pulse, fiber=fiber1, n_steps=Steps)
    y, AW, AT, pulse_out2 = evol.propagate(pulse_in=pulse_out1, fiber=fiber2, n_steps=Steps)
    y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse_out2, fiber=fiber3, n_steps=Steps)
    
    # ### That's it! Physics complete. Just plotting commands from here!
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(pulse.wl_nm,    dB(pulse_out.AW),  color = 'r')
    ax.plot(pulse.wl_nm,    dB(pulse.AW),  color = 'b')
    ax.set_xlim([1100,2200])
    ax.set_ylim([-100,0])
    s = ('D1 = %0.1f \nD2=%0.1f \nD3=%0.1f' %(D1,D2,D3))
    ax.annotate(s,(2000,-20))


# D1a = np.arange(-1, -2.1, -0.5)
# D2a = np.arange(-8, -16.1, -1)
# D3a = np.arange(1, 2.1, 0.5) #anomalous

D1a = np.array([-1.5])
D2a = np.array([-12])
D3a = np.array([1])

L1a = np.array([1,2,3,4,5])
L2a = np.array([0.6])
L3a = np.array([2])


for k1 in range(D1a.size):
    for k2 in range(D2a.size):
        for k3 in range(D3a.size):
            for k4 in range(L1a.size):
                for k5 in range(L2a.size):
                    for k6 in range(L3a.size):
                        D1 = D1a[k1]
                        D2 = D2a[k2]
                        D3 = D3a[k3]
                        
                        Len1 = L1a[k4]
                        Len2 = L2a[k5]
                        Len3 = L3a[k6]
            
                        D1slope = 0.019
                        D2slope = 0.045
                        D3slope = 0.019
                        
                        runsim()