# -*- coding: utf-8 -*-
"""
@author: Luis Ledezma

Module to work with bulk crystals instead of waveguides
"""

class nonlinear_crystal():
    
    def __init__(self, L, n_func, chi2, alpha=0):
        self.L = L
        self.n_func = n_func
        self.chi2 = chi2
        self.alpha = alpha      
  
    def propagate_NEE_fd(self, pulse, h, v_ref=None,
                         verbose=True, zcheck_step = 0.5e-3, z0=0):
        
        #Timer
        tic_total = time.time()
         
        #Get stuff 
        wl = pulse.wl
        f0 = pulse.f0
        f_abs = pulse.f_abs
        Omega  = pulse.Omega
        
        n = self.n_func(wl)
        beta = 2*pi*f_abs*n/c

        if v_ref == None:
            df = f_abs[1] - f_abs[0]
            beta_1 = fftshift(np.gradient(fftshift(beta), 2*pi*df))
            vg = 1/beta_1
            v_ref = vg[0]
        
        beta_ref = beta[0]
        beta_1_ref = 1/v_ref
        D = beta - beta_ref - Omega/v_ref - 1j*self.alpha/2

        chi2 = self.chi2
        omega_ref = 2*pi*f0
        omega_abs = omega_ref + Omega
             
        def k(z):  
            return chi2(z)*omega_abs/(4*n*c)

        [a, a_evol] = NEE(t = pulse.t, 
                          x = pulse.a,
                          Omega = Omega,
                          f0 = pulse.f0,
                          L = self.L,
                          D = D, 
                          b0 = beta_ref, 
                          b1_ref = beta_1_ref, 
                          k = k, 
                          h = h, 
                          zcheck_step = zcheck_step, 
                          z0 = 0,
                          verbose = verbose)
        
        tdelta = time.time() - tic_total
        print('Total time = %0.1f s' %(tdelta))
        
        output_pulse = pulses.pulse(pulse.t, a, pulse.wl0)
        return output_pulse, a_evol