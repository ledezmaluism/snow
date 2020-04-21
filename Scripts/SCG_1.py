#!/usr/bin/env python
# coding: utf-8

# # Super Continuum Generation in $\chi^{(2)}$ materials
# 
# Author: Luis Ledezma (ledezma@caltech.edu)
# 
# As a first step let's re-create the results from [1].

# ## The code
# 
# We begin by importing the modules needed, including my own. I also do some formatting. You can safely ignore all this.

# In[1]:


# import numpy as np
import cupy as np
# from numpy.fft import fft, ifft, fftshift, fftfreq
from cupy.fft import fft, ifft, fftshift, fftfreq
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

#Larger font for plots
plt.rcParams.update({'font.size': 18})


# ## Units
# We'll work in SI base units for the most part. Here we load some constants and also create some variables for scaling units conveniently.

# In[ ]:


from scipy.constants import pi, c, epsilon_0
nm = 1e-9
um = 1e-6
mm = 1e-3
ps = 1e-12
fs = 1e-15
GHz = 1e9
THz = 1e12


# ## Time and Frequency domain windows
# 
# ### General Comments
# We need to determine the FFT size $N$, as well the size of the time window $T$ and the total frequency bandwidth $F_\mathrm{BW}$. Both domain will have $N$ points, and these three parameters are related by the uncertainty relations:
# 
# $${\Delta t} {\Delta f } = \frac{1}{N}, \\
# T F_{\mathrm{BW}} = N.$$

# In[ ]:


wl_ff = 1580*nm #Fundamental wavelength
f0_ff = c/wl_ff

#In this example we now what's the full bandwidth we want
f_max = c/300e-9
f_min = c/10000e-9
BW = f_max - f_min 

#Now we can create the time and frequency arrays
NFFT = 2**14
Tmax = NFFT/BW
dt = 1/BW
t_start = -1*ps
t_stop = t_start + NFFT*dt
t = np.arange(t_start, t_stop, step=dt)
f = fftfreq(NFFT, dt)
Omega = 2*pi*f
df = f[1]-f[0]


# We also need to choose a reference frequency $f_{ref}$. Our simulation is a bandpass simulation centered around this reference frequency. So, the natural frequency variable for our simulation is $$\Omega = \omega - \omega_{ref}.$$
# I'll try to be consistent and use lower case variable names for absolute frequencies representing $\omega$'s, and variable names starting with an uppper case to represent $\Omega$. For, instance, the variable ```Omega``` that we just created represents $\Omega$, while in the next cell we'll define ```omega_ref``` and ```omega_abs``` which represents $\omega_{ref}$ and $\omega$.

# In[ ]:


#Reference frequency
f_ref = (f_min + f_max)/2
wl_ref = c/f_ref
omega_ref = 2*pi*f_ref

#Absolute frequencies and wavelengths
f_abs = f + f_ref
wl_abs = c/f_abs
omega_abs = 2*pi*f_abs
f_min = np.amin(f_abs)
f_max = np.amax(f_abs)
wl_max = c/f_min
wl_min = c/f_max

#get the frequency indexes
f0_ff_index = np.abs(f + f_ref - f0_ff).argmin()
f_ref_index = np.abs(f).argmin()

print('Fundamental wavelength = %0.1f nm' %(wl_ff/nm))
print('Fundamental frequency = %0.1f THz' %(f0_ff/THz))
print('Simulation bandwidth = %0.1f THz' %(BW/THz))
print('Time window size = %0.3f ps' %(Tmax/ps))
print('Sampling Rate = %0.3f fs' %(dt/fs))
print('Frequency Resolution = %0.3f GHz' %(df/GHz))
print('Reference wavelength = %0.1f nm' %(wl_ref/nm))
print('Reference frequency = %0.1f THz' %(f_ref/THz))
print('Minimum absolute frequency = %0.1f THz' %(f_min/THz))
print('Maximum absolute frequency = %0.1f THz' %(f_max/THz))
print('Minimum absolute wavelength = %0.1f nm' %(c/f_max/nm))
print('Maximum absolute wavelength = %0.1f um' %(c/f_min/um))
print('Array index for fundamental = %i' %(f0_ff_index))
print('Array index for reference = %i' %(f_ref_index))


# ## Material properties
# 
# I'm going to load the refractive index of lithium tantalate from my personal library named <font color=magenta> materials</font>. In that library I implement mostly Sellmeier's equations obtained from https://refractiveindex.info/ for a few common materials.

# In[ ]:


def n_func(wl):
    n = materials.refractive_index('LN_MgO_e', wl/um)
    return n

n = n_func(wl_abs)
nw = n[f0_ff_index]

plt.rcParams['figure.figsize'] = [10, 5]
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(fftshift(wl_abs/um), fftshift(n))
ax1.scatter(wl_abs[f0_ff_index]/um, nw, c='r')
ax1.set_xlim([wl_min/um, wl_max/um])
ax1.set_xlabel('Wavelength (um)')
ax1.set_ylabel('Refractive index')
ax1.grid(True)

print('Refractive index at fundamental = %0.3f' %(nw))


# Now we'll get the propagation constant as usual:
# $$ \beta = \frac{\omega n}{c} ,$$
# 
# as well as the group velocity and GVD:
# $$ 
# v_g = \frac{1}{\beta_1} = \left( \frac{\partial \beta}{\partial \omega} \right)^{-1} \\
# \mathrm{GVD} = \beta_2 = \frac{\partial^2 \beta}{\partial \omega^2} .
# $$
# 
# We also need to get the higher order dispersion operator; this can be obtained from the propagation constant as follows:
# $$D = \beta(\Omega) - \beta \big|_{\Omega=0} - \frac{\Omega}{v_{ref}}.$$
# 
# For this example we'll use $v_{ref} = v_g(2 \omega_0)$.
# 
# At this point we can also compute the GVM ($\Delta \beta^\prime $) between the fundamental signal and the moving reference frame:
# $$ \Delta \beta^\prime = \frac{1}{v_{ref}} - \frac{1}{v_g(\omega_0)}, $$
# and then we can compute how much a pulse at $\omega_0$ will travel in a crystal of length $L$,
# $$ \tau =  L \Delta \beta^\prime. $$

# In[ ]:


beta = omega_abs*n/c
beta_1 = fftshift(np.gradient(fftshift(beta), 2*pi*df))
beta_2 = fftshift(np.gradient(fftshift(beta_1), 2*pi*df))
vg = 1/beta_1

beta_ref = beta[0]
# f_v_ref = c/(700*nm)
f_v_ref = f0_ff
f_v_ref_index = np.abs(f + f_ref - f_v_ref).argmin()
# v_ref = vg[0]
# v_ref = 0.11*mm/ps
v_ref = vg[f_v_ref_index]
D = beta - beta_ref - Omega/v_ref
GVM = 1/v_ref - 1/vg

plt.rcParams['figure.figsize'] = [10, 18]
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.plot(fftshift(f_abs)/THz, fftshift(vg)/(mm/ps));
ax1.set_ylabel('Group Velocity (ps/mm)'); ax1.set_xlabel('Frequency (THz)');

ax2.plot(fftshift(f_abs)/THz, fftshift(GVM)/(ps/mm));
ax2.set_ylabel('GVM (ps/mm)'); ax2.set_xlabel('Frequency (THz)');

ax3.plot(fftshift(f_abs)/THz, fftshift(beta_2)/(fs**2/mm));
ax3.set_ylabel('GVD (fs$^2$/mm)'); ax3.set_xlabel('Frequency (THz)');

ax4.plot(fftshift(f_abs)/THz, fftshift(D));
ax4.set_xlabel('Frequency (THz)'); ax4.set_ylabel('High order dispersion (m$^{-1}$)');

ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True);


# ## Quasi Phase Matching

# In[ ]:


f1 = c/(2*um)
f2 = 2*f1
f1_index = np.abs(f + f_ref - f1).argmin()
f2_index = np.abs(f + f_ref - f2).argmin()

n1 = n[f1_index]
n2 = n[f2_index]

print(c/f2/(n2-n1)/um)


# In[ ]:


d33 = 27e-12
deff = d33
pp = 30*um

def chi2(z):
    poling = np.sign(np.cos(z*2*pi/pp))
    return 2*deff*poling


# ## Nonlinear  Component
# Now we create a nonlinear component based on this refractive index and propagation constant. This nonlinear component implements a method for propagation using the NEE.

# In[ ]:


L = 7*mm
crystal = nlo.nonlinear_element(L, n_func=n_func, chi2=chi2)

#Let's also check again the GVM now that we have the crystal length
plt.rcParams['figure.figsize'] = [10, 5]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(fftshift(f_abs)/THz, fftshift(GVM*L)/(ps));
ax1.set_ylabel('GVM*L (ps)'); ax2.set_xlabel('Frequency (THz)');


# ## Input pulse creation
# 
# We can find the peak value of the pulse from the intensity
# $$
# E_p = \sqrt{\frac{2 I}{n_\omega c \epsilon_0}}.
# $$
# 
# The reference frequency $\omega_{ref}$ is mapped to zero in $\Omega$. So, to generate a pulse at frequency $\omega_0$ we need to modulate it:
# $$
# p(t) e^{j (\omega_0 - \omega_{ref}) t} = p(t) e^{j \Omega_0 t}.
# $$
# 
# Let's now generate the pulse, add a bit of noise to it, and plot it to verify it is at the right frequency and looks like we expect it to look.
# 
# Here I introduce a class named ```pulse```, the goal is to pack everything relevant to the pulse into a single object, just like we did with ```nonlinear_element```. I also have a bunch of convenience methods associated with ```pulse``` for things like plotting, calculating FWHM, energy, etc.

# In[ ]:


#Frequency offset from reference:
Omega_0 = 2*pi*(f0_ff - f_ref)

#Peak value:
Intensity = 15e13 #15GW/cm^2
Epeak = np.sqrt(2*Intensity/(nw*c*epsilon_0))

#Pulse width:
tau = 50*fs

#Noise floor
noise = 2e3*np.random.normal(size=NFFT)

#Pulse creation
x = Epeak*np.exp(-2*np.log(2)*(t/tau)**2)*np.exp(1j*Omega_0*t)
x = x + noise
pulse = nlo.pulse(t, x, wl_ref)

plt.rcParams['figure.figsize'] = [18, 10]
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1 = pulse.plot_mag(ax=ax1, t_unit='ps'); ax1.set_ylabel('Pulse Amplitude (V/m)')
ax2 = pulse.plot_ESD_dB(ax=ax2, label='ESD (dB/Hz$^2$)', f_unit = 'THz')
ax3 = pulse.plot_ESD_dB_absfreq(ax=ax3, label='ESD (dB/Hz$^2$)', f_unit = 'THz')
ax4 = pulse.plot_ESD_dB_vs_wavelength(ax=ax4, label='ESD (dB/Hz$^2$)', wl_unit='um');


# ## Single pass
# 
# Finally, we can run the simulation. The method ```propagate_NEE``` takes a pulse object, a step size, and a reference velocity for the moving frame and outputs a pulse and it's evolution along the crystal.

# In[ ]:


h = 1*mm/200 #Step size
[out_pulse, pulse_evol_full] = crystal.propagate_NEE(pulse, h, v_ref=v_ref, method='v2')


# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
ax1 = nlo.plot_mag(t, out_pulse)
ax1 = pulse.plot_mag(ax=ax1)
# ax1.set_xlim(-2,0.5)
# ax1.set_ylim(0,2e8)
ax1.set_ylabel('Pulse Amplitude (V/m)')
ax1.set_xlabel('Time (ps)');


# In[ ]:


plt.rcParams['figure.figsize'] = [15, 5]
fig = plt.figure()
ax1 = fig.add_subplot(111)
pulse.plot_ESD_dB_vs_wavelength(ax=ax1)
nlo.plot_ESD_dB_vs_wavelength(t, out_pulse, f_ref, ax=ax1)
ax1.set_xlim([0.5, 3.5])
ax1.set_ylim([-80,0])
ax1.set_ylabel('')


# We can also look at the evolution of the pulse along the crystal. Here it also clear that the input pulse travels while it creates a faint second harmonic that is stationary in the simulation frame. 

# ## Pulse evolution along the crystal

# In[ ]:


#Downsample evolution for plotting
Ndown = 5
pulse_evol = pulse_evol_full[::,::Ndown]
Nsteps = pulse_evol.shape[1]-1


# In[ ]:


plt.rcParams['figure.figsize'] = [15, 5]
Nsteps = pulse_evol.shape[1]-1
X,Y = np.meshgrid(t/ps, np.arange(Nsteps+1)*h*1e3)
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.pcolormesh(X, Y, (np.transpose(np.abs(pulse_evol))), cmap = cc.cm["rainbow"])
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Propagation distance (mm)')
# ax.set_ylim([0,5])
# ax.set_xlim([-2,0.5])
plt.colorbar(im, ax=ax);


# In[ ]:


A_evol = np.zeros((t.size, Nsteps+1))

for k in range(Nsteps+1):
    x = pulse_evol[:,k]
    _ , Aesd = nlo.get_esd(t, x)
    A_evol[:,k] = 10*np.log10(Aesd)

A_evol = A_evol - np.amax(A_evol)
A_evol = fftshift(A_evol, axes=0)

plt.rcParams['figure.figsize'] = [15, 5]

X,Y = np.meshgrid(fftshift(f+f_ref)*1e-12, np.arange(Nsteps+1)*h*Ndown/mm)
plt.figure()
plt.pcolormesh(X, Y, (np.transpose(A_evol)), cmap = cc.cm["fire"], vmin=-80, vmax=0)
cb = plt.colorbar()
cb.set_label('Relative PSD (dB)')
plt.xlabel('Frequency (THz)')
plt.ylabel('Propagation distance (mm)');

# wl_array = c/(fftshift(f)+f_ref)*1e6
# Xwl,Ywl = np.meshgrid(wl_array, np.arange(Nsteps+1)*h*1e3)
# plt.figure()
# plt.pcolormesh(Xwl, Ywl, (np.transpose(A_evol)), cmap = cc.cm["fire"], vmin=-60, vmax=0)
# cb = plt.colorbar()
# cb.set_label('Relative PSD (dB)')
# plt.xlabel('Wavelength (um)')
# plt.ylabel('Propagation distance (mm)');


# In[ ]:


plt.rcParams['figure.figsize'] = [15, 5]
wl_min = 0.4
wl_max = 3.5
wl_array = c/(fftshift(f)+f_ref)*1e6
wl_max_idx = np.abs(wl_array - wl_max).argmin()
wl_min_idx = np.abs(wl_array - wl_min).argmin()
wl_array = wl_array[wl_max_idx:wl_min_idx]*1e3
Xwl,Ywl = np.meshgrid(wl_array, np.arange(Nsteps+1)*h*1e3)
plt.figure()
# plt.pcolormesh(Xwl, Ywl, (np.transpose(A_evol[wl_max_idx:, :])), cmap = cc.cm["fire"], vmin=-80, vmax=0)
plt.pcolormesh(Xwl[::4,::], Ywl[::4,::], (np.transpose(A_evol[wl_max_idx:wl_min_idx, ::4])), cmap = cm.jet, vmin=-80, vmax=0)
# plt.pcolormesh(Xwl, Ywl, (np.transpose(A_evol)), cmap = cm.jet, vmin=-80, vmax=0)
cb = plt.colorbar()
cb.set_label('Relative PSD (dB)')
plt.xlabel('Wavelength (um)')
plt.ylabel('Propagation distance (mm)')


# ## References
# [1] M. Conforti, F. Baronio, and C. De Angelis, “Nonlinear envelope equation for broadband optical pulses in quadratic media,” Phys. Rev. A, vol. 81, no. 5, p. 053841, May 2010, doi: 10.1103/PhysRevA.81.053841.

# # <center> END OF DOCUMENT </center>

# In[ ]:


# #Formatting stuff
# from IPython.core.display import HTML
# def css_styling():
#     styles = open("custom.css", "r").read()
#     return HTML(styles)
# css_styling()


# ### Software Versions

# In[ ]:


get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'scipy, numpy, matplotlib')


# ### Animation Generation

# In[ ]:


# %matplotlib inline
# from matplotlib import animation, rc
# # from IPython.display import HTML

# # First set up the figure, the axis, and the plot element we want to animate
# fig, ax = plt.subplots()

# ax.set_xlim(( -2, 0.5))
# ax.set_ylim((0, 2e8))
# ax.set_ylabel('Pulse Amplitude $\sqrt{W}$')
# ax.set_xlabel('Time (ps)')

# line, = ax.plot([], [], lw=2)

# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return (line,)

# # animation function. This is called sequentially
# def animate(i):
#     line.set_data(t*1e12, np.abs(a_evol[:,5*i]))
#     return (line,)

# # call the animator. blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=int(Nsteps/5), interval=20, blit=True)

# rc('animation', html='html5')
# anim

