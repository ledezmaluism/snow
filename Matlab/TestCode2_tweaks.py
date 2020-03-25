# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:22:00 2020

@author: luish

Based on code written by Arkadev Roy 10/25/2019
Modified by Luis 1/9/2020
"""
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
from numpy import sqrt, exp, log10
import matplotlib.pyplot as plt
from scipy.constants import pi,c

def sech(x):
    return 1/np.cosh(x)

def nonlinearopo(a, b, kappas):
    f = ifft((kappas)*fft(b*np.conj(a)));
    g = -ifft((kappas)*(fft(a*a)));
    return np.array([f,g])

def single_pass(a, b, L, h, Da, Db, kappas):

    for x in range(int(L/h)): # inner loop for evolution inside crystal
        a = ifft(fft(a)*Da); # signal converted to time domain
        b = ifft(fft(b)*Db); # pump converted to time domain
        
        [k1, l1] = h*nonlinearopo(a,b, kappas);
        [k2, l2] = h*nonlinearopo(a+k1/2,b+l1/2, kappas);
        [k3, l3] = h*nonlinearopo(a+k2/2,b+l2/2, kappas);
        [k4, l4] = h*nonlinearopo(a+k3,b+l3, kappas);
        
        a = a+(k1+2*k2+2*k3+k4)/6;
        b = b+(l1+2*l2+2*l3+l4)/6; # nonlinear split step
        
    return a,b

c = c*1e-12; #mm/fs

# Input Parameters

Nround_trips = 1
# Nround_trips = 200; # number of round-trips/ also a measure of slow time evolution
T=0.65; # Transmitivity of the output coupler
L=1; # Length of crystal in mm

frep = 250e6; #repetition rate
tw = 70; # Imput pump pulse width (fs)

dstep=L/50; # discretization in crystal i.e the crystal is divided into 50 segments

#Vectors for sweeping
pin = np.array([0.5]); # average pump input power (Watt)
#pin=[0.6,0.7,0.8,0.9,1];     ###### pin can be a matrix, in that case you can study the effect of input power variation

dT = np.array([4]); # detuning in fs
#dT=[0.75, 0.85, 0.87, 0.89, 0.91, 0.93]; ###### dT can be a matrix, in
#that case you can study the effect of detuning and find the peak structure

#---tau and omega arrays
NFFT = 2**12; # number of fft points/ number of discretizations of the fast time axis
Tmax = 1000; # Extent of the fast time axis in fs
dtau = (2*Tmax)/NFFT;# step size in fast axis
# tau = (-NFFT/2:NFFT/2-1)*dtau; # discretization of fast axis
tau = np.arange(-NFFT/2, NFFT/2)*dtau
# omega =  pi/Tmax* [(0:NFFT/2-1) (-NFFT/2:-1)]; #corresponding frequency grid
x1 = np.arange(0, NFFT/2)
x2 = np.arange(-NFFT/2, 0)
omega =  pi/Tmax*np.concatenate([x1, x2])

#Crystal dispersion
alphaa=0.00691; # loss for signal
alphab=0.00691; # loss for pump
u=112.778; # walk-off parameter (fs/mm)
beta2a=-53.64; # second order GVD signal (fs**2/mm)
beta3a=756.14; # third order GVD signal (fs**3/mm)
beta4a=-2924.19; # fourth order GVD signal (fs**4/mm)
beta2b=240.92; # second order GVD pump (fs**2/mm)
beta3b=211.285; # second order GVD pump (fs**3/mm)
beta4b=-18.3758; # second order GVD pump (fs**4/mm)

phi2=25*2; # cavity dispersion second order (fs**2/mm)
phi3=76;  # cavity dispersion third order (fs**3/mm)
phi4=-13020;

Da = (alphaa/2-1j*beta2a/2*omega**2-1j*beta3a/6*omega**3-1j*beta4a/24*omega**4); # dispersion of signal in fourier domain
Db = (alphab/2-1j*beta2b/2*omega**2-1j*beta3b/6*omega**3-1j*beta4b/24*omega**4-1j*u*omega); # dispersion of pump in fourier domain
Da = exp(Da*(-dstep)); #dispersion operator
Db = exp(Db*(-dstep)); #dispersion operator

phic = phi2/2*omega**2+phi3/6*(omega)**3+phi4/24*(omega)**4; # dispersion of cavity in fourier domain

#Nonlinear coupling
Wp= 10*10**-6; # beam waist of pump (m)
Ws= 14*10**-6; # beam waist of signal (m)
deff= 2/pi*16*10**-12; #2/pi*20*10**-12
ns=2.2333; # refractive index of signal
npp=2.1935; # refractive index of pump
Omegas = 2*pi*3*10**8/(2090*10**-9)+omega*10**15; #Absolute frequency
kappas = sqrt(2*377)*deff*Omegas/Ws/ns/sqrt(pi*npp)/(3*10**8)*10**-3;
# kappas = max(kappas,0);
kappas = np.clip(kappas, 0, None)

noise = 10**-10*np.ones([1,tau.size]);
#noise=10**-10*randn(1,length(tau)); # random noise of small amplitude is take as the initial condition for the signal
##

for kp in range(pin.size):
    for kd in range(dT.size):
        
        deltaT=dT[kd]; #detuning
        l=(3*10**8*deltaT*10**-15)/(1045*10**-9); #detuning converted from fs to l parameter
        phi=pi*l+(phic)+deltaT*(omega); # the feedback transfer function in fourier domain
        
        pavg=pin[kp];# average pump power
        p = 0.88*pavg/(frep*tw*1e-15); #calculation of peak pump power for sech pulse
        seed=sqrt(p)*sech(tau/tw*1.76); # input pump profile
        
        a = noise; # temporal profile of signal
        b = noise; # temporal profile of pump
        
        data_a = np.zeros([Nround_trips, NFFT]);
        data_b = np.zeros([Nround_trips, NFFT]);
        for n in  range(Nround_trips): # outer loop for roundtrip evolution dynamics
            b = seed;
            
            [a, b] = single_pass(a, b, L, dstep, Da, Db, kappas);
            
            a=ifft(fft(a)*sqrt(1-T)*exp(1j*(phi))); #feedback path
            data_a[n,:]=(abs(a)/np.amax(abs(a)))**2; #saving the normalized intensity profile of signal every roundtrip
            data_b[n,:]=(abs(b)/np.amax(abs(b)))**2;           
            
            if (n+1)%50==0:
                print('Input Power = %0.2f W, Detuning = %0.2f fs, Completed roundtrip %i\n' % (pavg, deltaT, n+1))
           
        A = fftshift(fft(a)); # spectral profile of the signal at steady state
        B = fftshift(fft(b)); # spectral profile of the pump after exiting the cavity
        temp3 = fftshift(fft(seed)); # spectral profile of the pump before entering the cavity
        
        
a = np.squeeze(a)
b = np.squeeze(b)
A = np.squeeze(A)

# Different Plots
fig1, ax1 = plt.subplots()
ax1.plot (tau, abs(a)**2,'b');
ax1.plot (tau, abs(b)**2,'r');
ax1.set_xlabel('Fast Time')
ax1.set_ylabel('Power(W)')
ax1.set_xlim([-7*tw,7*tw]);

fig2, ax2 = plt.subplots()
fconv=(3*10**8/2090/10**-9+fftshift(omega/2/pi*10**15));
lambdaconv=3*10**8./fconv*10**9;
ax2.plot((lambdaconv),(20*log10(abs(A/max(A)))),'b')
ax2.set_xlim([1500, 3000])
ax2.set_ylim([-30,0])
ax2.set_xlabel('wavelength (nm)')
ax2.set_ylabel('PSD (dB)')

fig3, ax3 = plt.subplots()
# mesh(1:1:Nround_trips,tau, data_a')
X,Y = np.meshgrid(np.arange(1,Nround_trips+1), tau)
ax3.contourf(X, Y, np.transpose(data_a), 100)
ax3.set_xlabel('Roundtrips')
ax3.set_ylabel('Fast Time (fs)')
ax3.set_ylim([-4*tw,4*tw])
ax3.set_title('Round-trip Evolution')

##
#Functions






