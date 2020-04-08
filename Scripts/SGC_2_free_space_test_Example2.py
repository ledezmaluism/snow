#!/usr/bin/env python
# coding: utf-8

# # Supercontinuum and second harmonic generations
# 
# ## Single Envelope
# 
# Method from [1]. I plan to validate according to their second example, quoting:
# 
# ""
# ""
# 
# [1]M. Conforti, F. Baronio, and C. De Angelis, “Nonlinear envelope equation for broadband optical pulses in quadratic media,” Phys. Rev. A, vol. 81, no. 5, p. 053841, May 2010, doi: 10.1103/PhysRevA.81.053841.

# We begin by the typical Python imports.

# In[1]:


import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
import time
from scipy import fftpack as sp
from scipy import signal
import copy
from matplotlib import cm
# import colorcet as cc
from matplotlib.colors import Normalize

#This are my libraries
import nlo 
import materials
import waveguides
from util import sech


# ## Units
# We'll work in SI units. Mostly in base units.

# In[2]:


from scipy.constants import pi, c, epsilon_0


# ## Time and Frequency domain windows

# Now we need to determine an FFT size $N$, this will be the common size for the time and frequency axis in our simulations. We also need to select how large the time window $T_{\mathrm{max}}$ for our simulations will be. The time window needs to be large enough so that no part of the pulse reaches the edge of the domain at any point. We will pick a reference group velocity for as the moving reference frame, so how much the pulse travels depends on the respective group velocity mismatches of all the frequency components with respect to this reference velocity.
# 
# The FFT poses the following relations between the time domain and frequency domain windows,
# $$ T_\mathrm{max} = {N \Delta t} = \frac{1}{\Delta f }, \\
# BW = f_s = \frac{1}{\Delta t},$$
# or combining them
# $$T_\mathrm{max} BW = N.$$

# In[3]:


# Tmax = 2.5e-12

# wl_ff = 1580e-9 #Fundamental frequency
# f0_ff = c/wl_ff

# print('Fundamental wavelength = %0.1f nm' %(wl_ff*1e9))
# print('Fundamental frequency = %0.1f THz' %(f0_ff*1e-12))

# print('\n')

# NFFT = 2**14
# t = np.linspace(-5e-12, 17e-12, NFFT)

# Tmax = np.amax(t) - np.amin(t)
# dt = t[1]-t[0]
# f = fftfreq(NFFT, dt)
# omega = 2*pi*f
# df = f[1]-f[0]
# BW = np.amax(f) - np.amin(f)

# #Reference frequency
# wl_ref = 700e-9
# f_ref = c/wl_ref
# omega_ref = 2*pi*f_ref

# #Absolute frequencies and wavelengths
# f_abs = f + f_ref
# wl_abs = c/f_abs
# omega_abs = 2*pi*f_abs
# f_min = np.amin(f_abs)
# f_max = np.amax(f_abs)
# wl_max = c/f_min
# wl_min = c/f_max

# print('Time window size = %0.3f ps' %(Tmax*1e12))
# print('Sampling Rate = %0.3f fs' %(dt*1e15))
# print('Frequency Resolution = %0.3f GHz' %(df*1e-9))
# print('Total frequency span = %0.1f THz' %(BW*1e-12))
# print('Reference wavelength = %0.1f nm' %(wl_ref*1e9))
# print('Reference frequency = %0.1f THz' %(f_ref*1e-12))
# print('Minimum absolute frequency = %0.1f THz' %(f_min*1e-12))
# print('Maximum absolute frequency = %0.1f THz' %(f_max*1e-12))
# print('Minimum absolute wavelength = %0.1f nm' %(c/f_max*1e9))
# print('Maximum absolute wavelength = %0.1f um' %(c/f_min*1e6))
# print('\n')

# #get the frequency indexes for fundamental and second harmonics
# f0_ff_index = np.abs(f + f_ref - f0_ff).argmin()
# print('Array index for fundamental = %i' %(f0_ff_index))


# In[4]:


wl_ff = 1580e-9 #Fundamental frequency
f0_ff = c/wl_ff

#In this example we now what's the full bandwidth we want
f_max = c/500e-9
f_min = c/3500e-9
BW = f_max - f_min 

#Reference frequency
f_ref = (f_min + f_max)/2
wl_ref = c/f_ref
omega_ref = 2*pi*f_ref

#Time and frequency arrays
NFFT = 2**12
Tmax = NFFT/BW
dt = 1/BW
t_start = -1e-12
t_stop = t_start + NFFT*dt
t = np.arange(t_start, t_stop, step=dt)
f = fftfreq(NFFT, dt)
omega = 2*pi*f
df = f[1]-f[0]

#Absolute frequencies and wavelengths
f_abs = f + f_ref
wl_abs = c/f_abs
omega_abs = 2*pi*f_abs
f_min = np.amin(f_abs)
f_max = np.amax(f_abs)
wl_max = c/f_min
wl_min = c/f_max

print('Fundamental wavelength = %0.1f nm' %(wl_ff*1e9))
print('Fundamental frequency = %0.1f THz' %(f0_ff*1e-12))
print('Simulation bandwidth = %0.1f THz' %(BW*1e-12))
print('\n')
print('Reference wavelength = %0.1f nm' %(wl_ref*1e9))
print('Reference frequency = %0.1f THz' %(f_ref*1e-12))
print('\n')
print('Time window size = %0.3f ps' %(Tmax*1e12))
print('Sampling Rate = %0.3f fs' %(dt*1e15))
print('Frequency Resolution = %0.3f GHz' %(df*1e-9))
print('\n')
print('Minimum absolute frequency = %0.1f THz' %(f_min*1e-12))
print('Maximum absolute frequency = %0.1f THz' %(f_max*1e-12))
print('Minimum absolute wavelength = %0.1f nm' %(c/f_max*1e9))
print('Maximum absolute wavelength = %0.1f um' %(c/f_min*1e6))
print('\n')

#get the frequency indexes for fundamental and second harmonics
f0_ff_index = np.abs(f + f_ref - f0_ff).argmin()
print('Array index for fundamental = %i' %(f0_ff_index))


# ## Material properties
# 
# The higher order dispersion operator is
# $$D = \beta(\Omega) - \beta_0(\Omega=0) - \beta_1(\Omega=0) \Omega$$.

# In[5]:


nLN = materials.refractive_index('LN_MgO_e', wl_abs*1e6)
nw = nLN[f0_ff_index]

plt.rcParams['figure.figsize'] = [10, 5]
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(fftshift(wl_abs*1e6), fftshift(nLN))
ax1.scatter(wl_abs[f0_ff_index]*1e6, nw, c='r')
# ax1.set_xlim([0.4,2])
# ax1.set_ylim([1.8,2.4])
ax1.set_xlabel('Wavelength (um)')
ax1.set_ylabel('Refractive index')
ax1.grid('on')

print('Refractive index at fundamental = %0.3f' %(nw))


# In[6]:


beta = (omega_abs)*nLN/c
beta_1 = np.gradient(beta, 2*pi*df)
beta_2 = np.gradient(beta_1, 2*pi*df)

beta_ref = beta[0]
beta_1_ref = beta_1[f0_ff_index]
D = beta - beta_ref - beta_1_ref*omega

plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(fftshift(f_abs)*1e-12, fftshift(D));
plt.xlabel('Frequency (THz)');
plt.grid(True)
plt.ylabel('Higher order dispersion');

print('GVD at %0.3f um = %0.3f fs^2/mm' %(wl_ref*1e6, beta_2[0]*1e27))
print('GVD at %0.3f um = %0.3f fs^2/mm' %(wl_ff*1e6, beta_2[f0_ff_index]*1e27))


# ## Input pulse creation

# In[7]:


#Frequency offset from reference:
delta_omega = 2*pi*(f0_ff - f_ref)

#Peak value:
Intensity = 15e13 #15GW/cm^2
n = nLN[f0_ff_index]
Epeak = np.sqrt(2*Intensity/(n*c*epsilon_0))

#Pulse width:
tau = 50e-15 

#Noise floor
noise = 2e3*np.random.normal(size=NFFT)

#Pulse creation
pulse = Epeak*np.exp(-2*np.log(2)*(t/tau)**2)*np.exp(1j*delta_omega*t)
pulse = pulse + noise
pulse_energy = nlo.energy_td(t, pulse) #To check that everything went well


# print('Center Wavelength = %0.3f um' %(wl0*1e6))
# print('Center frequency = %0.3f THz' %(f0*1e-12))
print('FWHM = %0.3f fs' %(nlo.FWHM(t, pulse, prominence=1e8)*1e15))
print('Pulse Energy = %0.3f J' %(pulse_energy))
print('Pulse peak power = %0.3f GW' %(pulse_energy/tau*1e-9))

plt.rcParams['figure.figsize'] = [15, 10]
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

nlo.plot_mag(t*1e12, pulse, ax=ax1)
# nlo.plot_magsq(t*1e12, pulse, ax=ax2);
nlo.plot_ESD_dB(t*1e12, pulse, ax=ax2)
nlo.plot_ESD_dB_absfreq(t*1e12, pulse, f_ref*1e-12, ax=ax3)
nlo.plot_ESD_dB_vs_wavelength(t, pulse, f_ref, ax=ax4, xlim=[0,4])

ax2.scatter(f[f0_ff_index]*1e-12, 0, c='k');
ax3.scatter((f[f0_ff_index]+f_ref)*1e-12, 0, c='k');
ax4.scatter(c/(f[f0_ff_index]+f_ref)*1e6, 0, c='k');

ax1.set_xlim([-2, 0.5])
ax1.set_ylim([0,3*1e8])


# ## Nonlinear  Stuff

# In[19]:


# def poling_sinusoidal(z, pp):
#     return (2/pi)*np.cos(z*2*pi/pp)

def poling_rectangular(z, pp):
    return np.sign(np.cos(z*2*pi/pp))

def chi2(z):
    poling = poling_rectangular(z, pp)
    return 2*d33*poling

def chi(z):
    # return chi2(z)*omega_ref/(4*nLN[0]*c)
    return chi2(z)*(omega+omega_ref)/(4*nLN*c) #Freq dependent

# def nonlinear_operator(z, A):
#     phi = omega_ref*t - (beta_ref - beta_1_ref*omega_ref)*z
#     f1 = A*A*np.exp(1j*phi) + 2*A*np.conj(A)*np.exp(-1j*phi)
#     f1_deriv = np.gradient(f1, dt)
#     f = -1j*chi(z)*f1 - 1*(chi(z)/omega_ref)*f1_deriv
#     return f

def nonlinear_operator(z, A):

    Aup = signal.resample(A, Nup*NFFT) #upsampled signal
    tup = np.linspace(t[0], t[-1], Nup*NFFT) #upsampled time
    
    phi = omega_ref*tup - (beta_ref - beta_1_ref*omega_ref)*z
    f1up = Aup*Aup*np.exp(1j*phi) + 2*Aup*np.conj(Aup)*np.exp(-1j*phi)
    
    f1 = signal.resample(f1up, NFFT) #Downsample
    
    # f1_deriv = np.gradient(f1, dt)    
    # f = -1j*chi(z)*f1 #- 1*(chi(z)/omega_ref)*f1_deriv
    
    f = -1j*ifft(chi(z)*fft(f1))
    
    return f


# In[37]:


def propagate(A, D):
    
    A_evol = 1j*np.zeros([t.size, Nsteps+1])
    A_evol[:,0] = A #Initial value
    
    Da = np.exp(-1j*D*h)
    z = 0
    
    for kz in range(Nsteps):
        #Linear step
        A = ifft(Da*fft(A))

        #Nonlinear step
        #Runge-Kutta 4th order
        k1 = nonlinear_operator(z    , A       )
        k2 = nonlinear_operator(z+h/2, A+h*k1/2)
        k3 = nonlinear_operator(z+h/2, A+h*k2/2)
        k4 = nonlinear_operator(z+h  , A+h*k3  )

        z = z + h
        A = A + (h/6)*(k1+2*k2+2*k3+k4) 
        
        A_evol[:, kz+1] = A
        
        # completion = (kz+1)/Nsteps*100
        # if completion%10==0:
        #     print('Completion = %0.1f %%' %(completion))
    
    return A, A_evol


# In[46]:

Nup = 4

#Crystal parameters
L = 7e-3
h = 1e-3/100
Nsteps = int(L/h)+1
d33 = 27e-12
pp = 30e-6

print('Crystal length = %0.2f mm' %(L*1e3))
print('Step size = %0.2f um' %(h*1e6))
print('Number of steps = %i' %(Nsteps))
print('Nonlinear coefficient, d33 = %0.2f pm/V' %(d33*1e12))
print('Using periodic poling with period = %0.3f um' %(pp*1e6))


# ## Single pass

# In[39]:


start_time = time.time()
[a, a_evol] = propagate(pulse, D)
stop_time = time.time()
print('Time elapsed = %0.1f s' %(stop_time - start_time))


# In[40]:


ax1 = nlo.plot_mag(t*1e12, a)
ax1 = nlo.plot_mag(t*1e12, pulse, ax=ax1)
# ax1.set_xlim(-2.5,0.5)
ax1.set_ylim(0,3e8)


# In[41]:


X,Y = np.meshgrid(t*1e12, np.arange(Nsteps+1)*h*1e3)
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.pcolormesh(X, Y, (np.transpose(np.abs(a_evol))))
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Propagation distance (mm)')
ax.set_ylim([0,5])
plt.colorbar(im, ax=ax)


# ## Spectrum

# In[42]:


fig = plt.figure()
ax1 = fig.add_subplot(111)
nlo.plot_ESD_dB_vs_wavelength(t, pulse, f_ref, ax=ax1)
nlo.plot_ESD_dB_vs_wavelength(t, a, f_ref, ax=ax1)
ax1.set_xlim([0.5,1.8])
ax1.set_ylim([-80,0])


# In[43]:


fig = plt.figure()
ax1 = fig.add_subplot(111)
nlo.plot_ESD_dB_vs_wavelength(t, pulse, f_ref, ax=ax1)
nlo.plot_ESD_dB_vs_wavelength(t, a, f_ref, ax=ax1)
ax1.set_xlim([1,3.5])
ax1.set_ylim([-70,0])


# In[44]:
A_evol = np.zeros((t.size, Nsteps+1))

for k in range(Nsteps+1):
    x = a_evol[:,k]
    _ , Aesd = nlo.get_esd(t*1e15, x)
    A_evol[:,k] = 10*np.log10(Aesd)

A_evol = A_evol - np.amax(A_evol)
A_evol = fftshift(A_evol, axes=0)

wl_max = 3.5
wl_array = c/(fftshift(f)+f_ref)*1e6
wl_max_idx = np.abs(wl_array - wl_max).argmin()
wl_array = wl_array[wl_max_idx:]
Xwl,Ywl = np.meshgrid(wl_array, np.arange(Nsteps+1)*h*1e3)
plt.figure()
# plt.pcolormesh(Xwl, Ywl, (np.transpose(A_evol[wl_max_idx:, :])), cmap = cc.cm["fire"], vmin=-80, vmax=0)
plt.pcolormesh(Xwl, Ywl, (np.transpose(A_evol[wl_max_idx:, :])), cmap = cm.jet, vmin=-80, vmax=0)
plt.xlim(0.5,3.5)
cb = plt.colorbar()
cb.set_label('Relative PSD (dB)')
plt.xlabel('Wavelength (um)')
plt.ylabel('Propagation distance (mm)');
