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

Window  = 60   # simulation window (ps)
Steps   = 100     # simulation steps
Points  = 2**14  # simulation points

# D1 = -1
# D2 = 20
# D3 = 1

# D1slope = 0.019
# D2slope = 0.045
# D3slope = 0.019

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
    ## Fiber 1 settings (Normal)
    Length  = 5   # length in m
    beta2, beta3 = get_betas(D1*ps/nm/km, D1slope*ps/nm/nm/km, 1.55*um)
    beta2 = beta2/(ps**2/km) # (ps^2/km)
    beta3 = beta3/(ps**3/km) # (ps^3/km)
    beta4 = 0
    Alpha   = 0.9e-5     # attentuation coefficient (dB/cm)
    Gamma   = 11.5    # Gamma (1/(W km))
    fibWL   = pulseWL # Center WL of fiber (nm)
    Raman   = True    # Enable Raman effect?
    Steep   = True    # Enable self steepening?
    alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m
    # create the fiber
    fiber1 = pynlo.media.fibers.fiber.FiberInstance()
    fiber1.generate_fiber(Length, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
                                  gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
    
    ## Fiber 2 settings (SMF)
    Length  = 0.6   # length in m
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
    Length  = 1   # length in m
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
    
    
    # In[13]:
    
    
    # Propagation
    evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.005, USE_SIMPLE_RAMAN=True,
                     disable_Raman              = np.logical_not(Raman),
                     disable_self_steepening    = np.logical_not(Steep))
    
    y, AW, AT, pulse_out1 = evol.propagate(pulse_in=pulse, fiber=fiber1, n_steps=Steps)
    y, AW, AT, pulse_out2 = evol.propagate(pulse_in=pulse_out1, fiber=fiber2, n_steps=Steps)
    y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse_out2, fiber=fiber3, n_steps=Steps)
    
    
    # ### That's it! Physics complete. Just plotting commands from here!
    
    # In[14]:
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(pulse.wl_nm,    dB(pulse_out.AW),  color = 'r')
    ax.plot(pulse.wl_nm,    dB(pulse.AW),  color = 'b')
    ax.set_xlim([1100,2200])
    ax.set_ylim([-100,0])
    s = ('D1 = %0.1f \nD2=%0.1f \nD3=%0.1f' %(D1,D2,D3))
    ax.annotate(s,(2000,-20))


D1a = np.arange(0,-2.1, -0.5)
D2a = np.arange(0, 25.1, 5)
D3a = np.arange(0,3.1,0.5)

for k1 in range(D1a.size):
    for k2 in range(D2a.size):
        for k3 in range(D3a.size):
            D1 = D1a[k1]
            D2 = D2a[k2]
            D3 = D3a[k3]
            
            D1slope = 0.019
            D2slope = 0.045
            D3slope = 0.019
            
            runsim()

# In[15]:

# # set up plots for the results:a
# fig = plt.figure(figsize=(20,20))
# ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
# ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
# ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=1, sharex=ax0)
# ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=1, sharex=ax1)

# F = pulse.F_THz     # Frequency grid of pulse (THz)

# zW = dB( np.transpose(AW)[:, (F > 0)] )
# zT = dB( np.transpose(AT) )

# y_mm = y * 1e3 # convert distance to mm

# ax0.plot(pulse_out.F_THz,    dB(pulse_out.AW),  color = 'r')
# ax1.plot(pulse_out.T_ps,     dB(pulse_out.AT),  color = 'r')

# ax0.plot(pulse.F_THz,    dB(pulse.AW),  color = 'b')
# ax1.plot(pulse.T_ps,     dB(pulse.AT),  color = 'b')

# extent = (np.min(F[F > 0]), np.max(F[F > 0]), 0, Length)
# ax2.imshow(zW, extent=extent,
#            vmin=np.max(zW) - 40.0, vmax=np.max(zW),
#            aspect='auto', origin='lower')

# extent = (np.min(pulse.T_ps), np.max(pulse.T_ps), np.min(y_mm), Length)
# ax3.imshow(zT, extent=extent,
#            vmin=np.max(zT) - 40.0, vmax=np.max(zT),
#            aspect='auto', origin='lower')


# ax0.set_ylabel('Intensity (dB)')
# ax0.set_ylim( - 80,  0)
# ax1.set_ylim( - 40, 40)

# ax2.set_ylabel('Propagation distance (mm)')
# ax2.set_xlabel('Frequency (THz)')
# ax2.set_xlim(0,400)

# ax3.set_xlabel('Time (ps)')

# plt.show()