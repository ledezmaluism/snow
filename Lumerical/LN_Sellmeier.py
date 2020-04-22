import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.constants import c,pi,mu_0,epsilon_0
c = c*1e-6 #um/ps

def term(A,B,wavelength):
    return A*wavelength**2/(wavelength**2-B)

def n_calc(Coeff,wavelength):
    n2sq = 1
    n2sq = n2sq + term(Coeff[0], Coeff[1], wavelength)
    n2sq = n2sq + term(Coeff[2], Coeff[3], wavelength)
    n2sq = n2sq + term(Coeff[4], Coeff[5], wavelength)
    return np.sqrt(n2sq)


#Sweep setup
fstart = 60
fstop = 400
fstep = 5
f = np.arange(fstart, fstop+fstep, fstep)
w = 2*pi*f
wavelength = c/f

##################################################
#Doped lithium niobate

#Extraordinary waves
A_ne = 2.4272
B_ne = 0.01478
C_ne = 1.4617
D_ne = 0.05612
E_ne = 9.6536
F_ne = 371.216

#Ordinary waves
A_no = 2.2454
B_no = 0.01242
C_no = 1.3005
D_no = 0.05313
E_no = 6.8972
F_no = 331.33

#There seems to be an error in the paper, and no/ne are reversed
no_LN_doped = n_calc([A_ne,B_ne,C_ne,D_ne,E_ne,F_ne],wavelength)
ne_LN_doped = n_calc([A_no,B_no,C_no,D_no,E_no,F_no],wavelength)

##################################################
#Undoped lithium niobate
#Extraordinary waves
A_ne = 2.9804
B_ne = 0.02047
C_ne = 0.5981
D_ne = 0.0666
E_ne = 8.9543
F_ne = 416.08

#Ordinary waves
A_no = 2.6734
B_no = 0.01764
C_no = 1.2290
D_no = 0.05914
E_no = 12.614
F_no = 474.6

no_LN_undoped = n_calc([A_no,B_no,C_no,D_no,E_no,F_no],wavelength)
ne_LN_undoped = n_calc([A_ne,B_ne,C_ne,D_ne,E_ne,F_ne],wavelength)

##################################################
#Sapphire

#Extraordinary waves
A_ne = 1.5039759
B_ne = 0.0740288**2
C_ne = 0.55069141
D_ne = 0.1216529**2
E_ne = 6.5927379
F_ne = 20.072248**2

#Ordinary waves
A_no = 1.4313493
B_no = 0.0726631**2
C_no = 0.65054713
D_no = 0.1193242**2
E_no = 5.3414021
F_no = 18.028251**2

no_Sa = n_calc([A_no,B_no,C_no,D_no,E_no,F_no],wavelength)
ne_Sa = n_calc([A_ne,B_ne,C_ne,D_ne,E_ne,F_ne],wavelength)
n_Sa = np.sqrt(no_Sa*ne_Sa)

##################################################
#SiO2
A = 0.6961663
B = 0.0684043**2
C = 0.4079426
D = 0.1162414**2
E = 0.8974794
F = 9.896161**2

n_Si = n_calc([A,B,C,D,E,F],wavelength)

##################################################
#Plots
plt.rc('text', usetex=True)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(f, n_Si, 'r', label='Silica')
ax1.plot(f, no_Sa, 'b', label='Sapphire n(o)')
ax1.plot(f, ne_Sa, 'b', label='Sapphire n(e)')
ax1.plot(f, no_LN_undoped, 'm', label='LN undoped n(o)')
ax1.plot(f, ne_LN_undoped, 'm', label='LN undoped n(e)')
ax1.plot(f, no_LN_doped, 'k', label='LN MgO doped n(o)')
ax1.plot(f, ne_LN_doped, 'k', label='LN MgO doped n(e)')

lgd = ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#ax1.axis([70, 300, 500])
ax1.set_xmargin(0)
ax1.set_xlabel('Frequency (THz)')
ax1.set_ylabel('$n(\omega)$')
ax1.grid(True)
ax1.autoscale_view()

wavelength_ticks = np.array([1,1.5,2,3,4])
new_tick_locations = c/wavelength_ticks
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(wavelength_ticks)
ax2.set_xlabel('Wavelength ($\mu \mathrm{m}$)')
ax2.grid()
fig.savefig('RefractionCoeffs.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')