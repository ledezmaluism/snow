# -*- coding: utf-8 -*-
"""
Ridge waveguides: effective index, dispersion and pulse propagation.

The `waveguide` class gives n_eff(wl) from one of three models (geometry via
the effective index method, a "behavioral" dispersion polynomial, or an
interpolated table), derives beta, beta1 and beta2 from it, and propagates
pulses with the chi(2) nonlinear envelope equation (`propagate_NEE`).

The module-level functions solve the slab problems used by the effective
index method. All lengths are in meters.

@author: Luis Ledezma
"""

import numpy as np
import warnings
from scipy.optimize import brentq
from scipy.constants import pi, c
from scipy import interpolate

from . import materials
from . import util
from . import nlo
from . import pulses

class waveguide:
    '''
    A ridge waveguide (by default thin-film MgO:LN on SiO2, air cladding).

    The effective index model is chosen at construction:

    - default: geometry, solved with the effective index method for the
      fundamental TE-like or TM-like mode (`mode`);
    - ``behavioral=True``: a dispersion polynomial set directly by its
      zero-GVD wavelength, GVM, index difference and index (see
      `neff_behavioral`);
    - ``external=True``: a cubic spline through (wl_array, n_array), e.g.
      from a mode solver.

    Parameters
    ----------
    w_top : float
        Ridge top width (m).
    h_thinfilm : float
        Thin-film thickness (m).
    h_etch : float
        Etch depth (m); the slab is h_thinfilm - h_etch thick.
    theta : float
        Sidewall angle (degrees from horizontal). The effective index method
        uses the mean of the top and base widths.
    tf_material, box_material, clad_material : str
        Names accepted by `materials.refractive_index`.
    behavioral : bool
        Use the behavioral model with z_wl, wl_1, wl_2, GVM, delta_n, n_f0,
        wl_f0 and c1.
    external : bool
        Interpolate `n_array` versus `wl_array` (m).
    mode : {'TE', 'TM'}
        Polarization of the fundamental mode for the geometric model. With
        an anisotropic film, choose `tf_material` for the index along the
        field (e.g. 'LN_MgO_e' for TE in x-cut, 'LN_MgO_o' for TM).

    Before propagating, set the length, loss and nonlinearity with
    `set_length`, `add_poling` and `set_nonlinear_coeffs`, and optionally
    `set_loss` (lossless by default).
    '''

    def __init__(self, w_top=1e-6, h_thinfilm=700e-9, h_etch=350e-9, theta=60,
                 tf_material = 'LN_MgO_e',
                 box_material = 'SiO2',
                 clad_material = 'Air',
                 behavioral = False,
                 z_wl=1e-6, wl_1=1e-6, wl_2=2e-6, GVM=0, delta_n=0.2, n_f0=2.0, wl_f0=None, c1=10**(-55),
                 external = False,
                 wl_array = np.array([1e-6]), n_array=np.array([1.0]),
                 mode = 'TE'
                 ):
        if mode not in ('TE', 'TM'):
            raise ValueError(f"mode must be 'TE' or 'TM', not {mode!r}")
        self.mode = mode
        

        
        self.GVD = self.beta2
        self.alpha = 0 #lossless until set_loss is called
        
        if behavioral:
            self.neff = self.neff_behavioral
            self.beh_z_wl = z_wl
            self.beh_wl_1 = wl_1
            self.beh_wl_2 = wl_2
            self.beh_GVM = GVM
            self.beh_delta_n = delta_n
            self.beh_n_f0= n_f0
            self.wl_f0 = wl_f0
            self.c1 = c1
        elif external:
            self.tck = interpolate.splrep( wl_array, n_array, s=0 )
            self.neff = self.neff_interp
        else:
            self.neff = self.neff_physical
            #validation
            # etch = h_ridge - h_slab
            w_base = w_top + 2*h_etch/np.tan(theta*pi/180)
            if h_etch*(w_base-w_top)<0:
                raise ValueError("Something wrong with this geometry")
    
            #Attributes given
            self.w_top = w_top
            self.h_etch = h_etch
            self.h_thinfilm = h_thinfilm
            self.theta = theta
            self.tf_material = tf_material
            self.box_material = box_material
            self.clad_material = clad_material
        
            #Attributes calculated
            self.w_base = w_base
            self.h_slab = h_thinfilm - h_etch
        
    def neff_physical(self, wl, mode=None, T=24.5):
        '''
        Effective index of the fundamental TE-like or TM-like mode from the
        geometry (effective index method), at wavelength(s) wl (m) and
        temperature T (C). `mode` defaults to the waveguide's `mode`.
        Always returns an array.
        '''
        if mode is None:
            mode = self.mode
        um = 1e-6
        w = (self.w_top + self.w_base)/2
        h = self.h_thinfilm
        hslab = self.h_slab
            
        if np.isscalar(wl):
            wl = np.asarray([wl])
            
        neff = np.zeros(wl.shape)
        for kw in range(wl.size):
            nridge = materials.refractive_index(self.tf_material, wl[kw]/um, T=T)
            nbox = materials.refractive_index(self.box_material, wl[kw]/um, T=T)
            nclad = materials.refractive_index(self.clad_material, wl[kw]/um, T=T)

            neff[kw] = neff_ridge(wl[kw], nridge, nbox, nclad, w, h, hslab, mode)
        return neff
    
    def neff_interp(self, wl, T=25):
        '''Effective index from the spline of the external table (T is ignored).'''
        return interpolate.splev(wl, self.tck, der=0)
    
    def neff_behavioral(self, wl, T=None):
        '''
        Effective index of a "black box" waveguide defined by its dispersion
        rather than its geometry.

        The GVD is the quadratic beta2(omega) = c1 (omega - z1)(omega - z2),
        which is integrated to beta1, beta and n_eff. The free constants are
        fixed by the constructor arguments:

        - z_wl: zero-GVD wavelength (sets z1);
        - wl_1, wl_2, GVM: beta1(wl_2) - beta1(wl_1) = GVM (s/m), sets z2;
        - delta_n: n(wl_2) - n(wl_1);
        - n_f0, wl_f0: n(wl_f0) = n_f0 (wl_f0 defaults to (wl_1 + wl_2)/2);
        - c1: curvature of beta2 (s^4/m).

        Parameters
        ----------
        wl : float or array
            Wavelength (m).
        T : ignored

        Returns
        -------
        float or array
            Effective index.
        '''          
        z_wl = self.beh_z_wl
        wl_1 = self.beh_wl_1
        wl_2 = self.beh_wl_2
        GVM = self.beh_GVM
        delta_n = self.beh_delta_n
        n_f0 = self.beh_n_f0
        wl_f0 = self.wl_f0
        
        
        if wl_f0 == None:
            wl_f0 = ( wl_1 + wl_2 )/2
        
        #z1, w1, w2 are wavelengths
        z1 = 2*pi*c/z_wl
        omega_1 = 2*pi*c/wl_1
        omega_2 = 2*pi*c/wl_2        
        omega0_f0 = 2*pi*c/wl_f0
        
        #Frequency array:
        omega = 2*pi*c/wl
        
        #Constants
        # c1 = 10 **(-55) #s^4/m
        c1 = self.c1
        
        # First we want to determine the remianing free parameters
        z2_num = 6*GVM - 2*c1*(omega_2**3-omega_1**3)+3*c1*z1*(omega_2**2-omega_1**2)
        z2_den = 3*c1*(2*z1-omega_2-omega_1)*(omega_2-omega_1)
        z2 = z2_num/z2_den
        
        
        c3 = (omega_2**3-omega_1**3)/12 - (z1+z2)*(omega_2**2-omega_1**2)/6 + z1*z2*(omega_2-omega_1)/2
        c3 = delta_n/c - c1*c3 
        c3 = c3 / (1/omega_2 - 1/omega_1)
        c2 = n_f0/c - c1*(omega0_f0**3/12 - (z1+z2)*(omega0_f0**2)/6 +z1*z2*omega0_f0/2) - c3/omega0_f0
        
        # Now we can solve for the remaing parameters
        # beta_2 = c1*(omega**2 -(z1+z2)*omega + z1*z2)
        # beta_1 = c1*(omega**3/3 - (z1+z2)*(omega**2)/2 + z1*z2*omega) + c2
        # beta = c1*(omega**4/12 - (z1+z2)*(omega**3)/6 + z1*z2*(omega**2)/2) + c2*omega + c3
        n = c*( c1*(omega**3/12 - (z1+z2)*(omega**2)/6 + z1*z2*omega/2) + c2 + c3/omega)
        
        return n
        
    def set_length(self, L):
        '''Waveguide length (m).'''
        self.L = L
        
    def set_loss(self, alpha):
        '''
        Power attenuation coefficient alpha (1/m): power decays as
        exp(-alpha z). Use `util.absorption_coeff` to convert from dB/cm.
        '''
        self.alpha = alpha
    
    def beta(self, wl, T=24.5):
        '''Propagation constant 2 pi n_eff / wl (1/m).'''
        beta = 2 * pi * self.neff(wl, T=T) / wl
        return beta
    
    def beta1(self, wl, T=24.5):
        '''
        Inverse group velocity d beta/d omega = (n - wl dn/dwl)/c (s/m),
        from a Richardson-extrapolated derivative. Returns an array.
        '''
        
        if np.isscalar(wl):
            wl = np.asarray([wl])
        n = 2 #number of extrapolation levels
        wl_step = 1e-9 #Initial step size
        b1 = np.zeros_like(wl)
        
        def neff_T(wl):
            return self.neff(wl, T=T)
        
        for kw in range(wl.size):
            dndl = util.derivative(neff_T, wl[kw], n, wl_step) #Still need to add temp here
            neff = neff_T(wl[kw])
            val = (neff - wl[kw] * dndl)/c
            b1[kw] = val.item() if hasattr(val, 'item') else val
        return b1
    
    def beta2(self, wl):
        '''
        Group velocity dispersion d^2 beta/d omega^2 (s^2/m), at room
        temperature. Also available as `waveguide.GVD`. Returns an array.
        '''
        if np.isscalar(wl):
            wl = np.asarray([wl])
        n = 2 #number of extrapolation levels
        wl_step = 1e-9 #Initial step size
        gvd = np.zeros_like(wl)
        for kw in range(wl.size):
            db1dl = util.derivative(self.beta1, wl[kw], n, wl_step)
            gvd[kw] = - wl[kw]**2 / (2*pi*c) * db1dl
        return gvd
    
    def _linear_operator(self, pulse, v_ref=None, T=24.5):
        '''
        Dispersion operator in the moving frame, shared by the propagators:
        D = beta - beta_ref - Omega/v_ref - i alpha/2 (1/m).
        Returns (D, beta_ref, v_ref).
        '''
        beta, v_ref = self._beta_and_frame(pulse, v_ref, T)
        D, beta_ref = nlo.linear_operator(pulse, beta, self.alpha, v_ref)
        return D, beta_ref, v_ref

    def _beta_and_frame(self, pulse, v_ref=None, T=24.5):
        '''beta on the pulse grid, and v_ref (default: group velocity at wl0).'''
        beta = self.beta(pulse.wl, T=T)
        if v_ref is None:
            #group velocity at the reference wavelength only (pulse.wl[0])
            v_ref = 1/self.beta1(pulse.wl0, T=T)[0]
        return beta, v_ref

    def propagate_linear(self, pulse, v_ref=None, T=24.5):
        '''
        Propagate a pulse through the waveguide with dispersion and loss only.

        Uses the same moving frame and conventions as `propagate_NEE`, so for
        a zero nonlinear coefficient both give the same output. The phase
        beta_ref*L of the reference frequency is dropped, as in the NEE.

        Parameters
        ----------
        pulse : pulses.pulse
        v_ref : float, optional
            Velocity of the moving frame (m/s). Default: group velocity at
            the pulse's reference wavelength and temperature T.
        T : float
            Temperature (C).

        Returns
        -------
        pulses.pulse
        '''
        D, _, _ = self._linear_operator(pulse, v_ref=v_ref, T=T)
        output_pulse = pulses.pulse(pulse.t, pulse.a, pulse.wl0, pulse.frep)
        output_pulse.update_fd(pulse.A * np.exp(-1j*D*self.L))
        return output_pulse
    
    def add_poling(self, poling):
        '''
        Set the poling function d(z), the sign of chi(2) at position z (m),
        e.g. ``lambda z: np.sign(np.cos(2*pi*z/period))``.
        '''
        self.poling = poling
        
    def set_nonlinear_coeffs(self, N, X0):
        '''
        Nonlinear coupling: k(z, omega) = d(z) X0 omega / (4 N).

        X0 is the effective chi(2) (m/V) and N the mode normalization that
        makes |a|^2 the power; together they absorb the mode overlap.
        '''
        self.N = N
        self.X0 = X0
        
    def nonlinear_coupling(self, z):
        '''d(z) X0 / (4 N), without the omega factor.'''
        return self.poling(z) * self.X0 / (4*self.N)
    
    def propagate_NEE(self, pulse, v_ref=None, 
                         verbose=True, zcheck_step=None,
                         z0 = 0, T=24.5, Kg=0, Qnoise=False,
                         rtol=1e-4, atol=1e-4, max_step=None,
                         poling_resolution=10e-9, rng=None):
        '''
        Propagate a pulse through the waveguide with the chi(2) nonlinear
        envelope equation (`nlo.NEE`).

        Parameters
        ----------
        pulse : pulses.pulse
            Input pulse; its grid defines the simulation grid.
        v_ref : float, optional
            Velocity of the moving frame (m/s). Default: group velocity at
            the pulse's reference wavelength and temperature T.
            Use 1/beta1(wl)[0] to keep a given wavelength stationary.
        verbose : bool
            Print the integrator status and run time.
        zcheck_step : deprecated and ignored (left from the fixed-step
            solver); passing it raises a DeprecationWarning.
        z0 : float
            Start position (m), for the poling function.
        T : float
            Temperature (C) used for beta.
        Kg : float
            Grating wavevector (1/m). If nonzero, the poling function is
            replaced by its first Fourier coefficient 2/pi and the grating
            enters the phase instead (much faster for long devices).
        Qnoise : bool
            Add vacuum noise to the input (half a photon per bin).
        rng : numpy Generator, int or None
            Random generator or seed for the vacuum noise (None: numpy's
            global generator).
        rtol, atol : float
            Tolerances of the adaptive RK45 integrator. Tighten them to check
            that a result is converged.
        max_step : float or None
            Largest integration step (m). Default: a quarter of the shortest
            poling domain when domain walls are found, otherwise unlimited.
            Tightening `rtol` alone does not converge explicit poling; the
            step limit does.
        poling_resolution : float or None
            With Kg = 0, the poling function is sampled every
            `poling_resolution` (m) to locate its sign changes (domain
            walls), which are then refined to 1e-12 m. The integrator
            restarts at each wall, so no step crosses one (#50). Domains
            shorter than about twice this are missed. Only sign changes are
            treated as discontinuities; for other jumps set `max_step`.
            None disables the search.

        Returns
        -------
        output_pulse : pulses.pulse
        steps : array
            Integrator step sizes (m), not the field evolution.
        '''
        if zcheck_step is not None:
            warnings.warn('zcheck_step is ignored and will be removed',
                          DeprecationWarning, stacklevel=2)

        beta, v_ref = self._beta_and_frame(pulse, v_ref, T)
        omega_abs = 2*pi*pulse.f0 + pulse.Omega
        
        def k(z): #nonlinear coupling
            if Kg == 0:
                p = self.poling(z)
            else:
                p = 2/pi #first order QPM
            return p * self.X0 * omega_abs / (4 * self.N)

        z_breaks = None
        if Kg == 0 and poling_resolution is not None and self.X0 != 0:
            z_breaks = poling_domain_walls(self.poling, z0, self.L, poling_resolution)
        if max_step is None:
            #Within a domain the phase mismatch turns the nonlinear drive by
            #~pi, which RK45's error estimate misjudges: limit the step (#50)
            max_step = np.inf
            if z_breaks is not None and z_breaks.size >= 2:
                max_step = np.min(np.diff(z_breaks)) / 4

        return nlo.propagate(pulse, self.L, beta, v_ref, k, alpha=self.alpha, z0=z0,
                             Kg=Kg, Qnoise=Qnoise, rtol=rtol, atol=atol,
                             max_step=max_step, z_breaks=z_breaks,
                             verbose=verbose, rng=rng)

    def propagate(self, pulse, v_ref=None, verbose=False, **kwargs):
        '''
        Common propagation interface shared with `nonlinear_crystal` (used
        by `opos.opo`): same as `propagate_NEE`, quiet by default.
        '''
        return self.propagate_NEE(pulse, v_ref=v_ref, verbose=verbose, **kwargs)

def poling_domain_walls(poling, z0, L, resolution=10e-9):
    '''
    Positions in (z0, L) where the poling function changes sign.

    `poling` is sampled every `resolution` (m); each sign change is then
    refined with brentq to 1e-12 m. The function may be vectorized or not.
    '''
    n = int(np.ceil((L - z0)/resolution)) + 1
    zs = np.linspace(z0, L, n)
    try:
        vals = np.broadcast_to(np.asarray(poling(zs), dtype=float), zs.shape)
    except (TypeError, ValueError):
        vals = np.array([poling(z) for z in zs], dtype=float)
    s = np.sign(vals)
    walls = []
    for i in np.nonzero(s[:-1] != s[1:])[0]:
        if s[i] == 0:
            walls.append(zs[i])
        elif s[i+1] == 0:
            walls.append(zs[i+1])
        else:
            walls.append(brentq(lambda z: float(np.sign(poling(z))),
                                zs[i], zs[i+1], xtol=1e-12))
    return np.unique(walls)

def beta_f(kx, ky, n, k0):
    '''
    Propagation constant in the z-direction
    from kx, ky, n, and k0
    '''
    return np.sqrt(n**2*k0**2 - kx**2 - ky**2)

def symmetric_slab_equations(X, n0, n1, d, wl, mode):
    '''
    Nonlinear equations to be solved
    '''
    k0 = 2*pi/wl #wavevector
    Rsqr = (k0*d/2)**2*(n1**2-n0**2)

    Y = np.sqrt(abs(Rsqr - X**2))

    if mode=='TE even':
        F = X*np.tan(X) - Y
    elif mode=='TE odd':
        F = X/np.tan(X) + Y
    elif mode=='TM even':
        F = X*np.tan(X)*(n0/n1)**2 - Y
    elif mode=='TM odd':
        F = X/np.tan(X)*(n0/n1)**2 + Y
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return F

def neff_symmetric_slab(n0, n1, d, wl, mode='TE even', order=0):
    '''
    Effective index of a symmetric slab mode.

    n0: cladding index 
    n1: slab index 
    d: slab thickness
    wl: wavelength
    mode: desired mode 'TE even', 'TE odd', 'TM even', 'TM odd'
    order: index m among modes of that parity (m = 0 is the lowest). The
        m-th even mode is TE_2m / TM_2m, the m-th odd mode TE_2m+1 / TM_2m+1.

    Raises ValueError if the mode is cut off.
    '''
    k0 = 2*pi/wl #wavevector
    R = np.sqrt((k0*d/2)**2*(n1**2-n0**2))
    #Solve for Normalized variables: X=kx*d/2, Y=a*d/2

    #Root bracket: X tan X = Y has one root in [m pi, m pi + pi/2),
    #-X cot X = Y one root in [m pi + pi/2, (m+1) pi)
    if mode in ('TE even', 'TM even'):
        xmin = order*pi
        xmax = min(R, order*pi + pi/2)
    elif mode in ('TE odd', 'TM odd'):
        xmin = order*pi + pi/2
        xmax = min(R, (order+1)*pi)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if R <= xmin:
        raise ValueError(f"{mode} mode of order {order} is cut off")

    #Solve equations
    X = brentq(symmetric_slab_equations, xmin, xmax, (n0, n1, d, wl, mode))
    kx = X*2/d
    beta = beta_f(kx, 0, n1, k0)
    neff = beta/k0
    return neff

def asymmetric_slab_eqs(x, n0, n1, n2, d, wl, mode='TE', order=0):
    '''
    Nonlinear equations to be solved
    '''
    k0 = 2*pi/wl #wavevector
    Rsqr_1 = (k0*d)**2*(n1**2-n0**2)
    Rsqr_2 = (k0*d)**2*(n1**2-n2**2)

    X = x
    Y = np.sqrt(abs(Rsqr_1 - X**2))
    Y2 = np.sqrt(abs(Rsqr_2 - X**2))

    if mode=='TM':
        Y = Y*(n1/n0)**2
        Y2 = Y2*(n1/n2)**2

    F = np.arctan(Y/X) + np.arctan(Y2/X) - X + order*pi

    return F

def neff_asymmetric_slab(n0, n1, n2, d, wl, mode='TE', order=0):
    '''
    Effective index of an asymmetric slab mode.

    n0: cladding index 
    n1: slab index 
    n2: lower cladding index
    d: slab thickness
    wl: wavelength
    mode: desired mode 'TE' or 'TM'
    order: mode number integer
    '''
    k0 = 2*pi/wl #wavevector
    Rsqr_1 = np.sqrt((k0*d)**2*(n1**2-n0**2))
    Rsqr_2 = np.sqrt((k0*d)**2*(n1**2-n2**2))

    #Solve for Normalized variables: X=kx*d, Y=a*d, Y2=a2*d
    xmin = 1e-6 + order*pi
    xmax = min(Rsqr_1, Rsqr_2, pi*(order+1))


    #Solve equations
    X = brentq(asymmetric_slab_eqs, xmin, xmax, (n0, n1, n2, d, wl, mode, order))
    kx = X/d
    beta = beta_f(kx, 0, n1, k0)
    neff = beta/k0
    return neff

def singlemode_symmetric(n0, n1, wl):
    #Returns thickness for single mode operation
    d = wl/2
    if n1>n0:
        d = d/np.sqrt(n1**2 - n0**2)
    else:
        d = d/np.sqrt(n0**2 - n1**2)
    return d

def singlemode_asymmetric(n0, n1, n2, wl):
    #Returns thickness for single mode operation
    #n1>n2>n0
    d = wl/2
    d = d/np.sqrt(n1**2 - n2**2)
    d = d*( 1 + (1/pi)*np.arctan(np.sqrt(n2**2 - n0**2)/np.sqrt(n1**2 - n2**2)))
    return d


def neff_ridge(wl, nridge, nbox=1, nclad=1, w=1, h=0.7, hslab=0.1, mode='TE'):
    '''
    Effective index of a ridge with the effective index method.

    The ridge region (thickness h) and the slab regions (thickness hslab)
    are each solved as asymmetric slabs (cladding / film / box); their
    indices then form a symmetric slab of width w, solved for its
    fundamental even mode.

    - mode='TE' (field mostly horizontal): TE vertical slabs, then the
      lateral slab is solved as TM.
    - mode='TM' (field mostly vertical): TM vertical slabs, then the
      lateral slab is solved as TE. For an anisotropic film, pass the index
      along the field direction as `nridge`.

    A slab region with no guided mode (fully etched, hslab = 0, or below
    cut-off) is given the larger cladding index, max(nbox, nclad).
    Lengths in meters.
    '''
    if mode == 'TE':
        vertical, lateral = 'TE', 'TM even'
    elif mode == 'TM':
        vertical, lateral = 'TM', 'TE even'
    else:
        raise ValueError(f"mode must be 'TE' or 'TM', not {mode!r}")

    n_ridge = _slab_index(nclad, nridge, nbox, h, wl, vertical)
    n_slab = _slab_index(nclad, nridge, nbox, hslab, wl, vertical)
    return neff_symmetric_slab(n_slab, n_ridge, w, wl, lateral)

def _slab_index(n0, n1, n2, d, wl, mode):
    '''
    Fundamental-mode index of an asymmetric slab, or max(n0, n2) when the
    slab is absent or guides no mode.
    '''
    n_sub = max(n0, n2)
    if d <= 0:
        return n_sub
    #The fundamental mode exists if the dispersion function changes sign
    #inside the bracket used by neff_asymmetric_slab
    k0 = 2*pi/wl
    xmax = min(np.sqrt((k0*d)**2*(n1**2-n0**2)), np.sqrt((k0*d)**2*(n1**2-n2**2)), pi)
    if asymmetric_slab_eqs(xmax, n0, n1, n2, d, wl, mode) >= 0:
        return n_sub
    return neff_asymmetric_slab(n0, n1, n2, d, wl, mode)

def neff_derivative(wl, nridge, nbox=1, nclad=1, w=1, h=0.7, hslab=0.1, mode='TE'):
    '''
    d n_eff / d wl of `neff_ridge`. Deprecated: use `waveguide.beta1`.

    The step is 1e-3 in the units of `wl`, so this was only meaningful for
    wavelengths in microns, while `neff_ridge` works in meters.
    '''
    warnings.warn('neff_derivative is deprecated and will be removed; '
                  'use waveguide.beta1 instead', DeprecationWarning, stacklevel=2)
    n = 2 #number of extrapolation levels
    wl_step = 0.001 #Initial step size
    def f(wl):
        return neff_ridge(wl, nridge, nbox, nclad, w, h, hslab, mode)

    return util.derivative(f, wl, n, wl_step)


def neff_2derivative(wl, nridge, nbox=1, nclad=1, w=1, h=0.7, hslab=0.1, mode='TE'):
    '''
    d^2 n_eff / d wl^2 of `neff_ridge`. Deprecated: use `waveguide.beta2`.
    '''
    warnings.warn('neff_2derivative is deprecated and will be removed; '
                  'use waveguide.beta2 instead', DeprecationWarning, stacklevel=2)
    n = 2 #number of extrapolation levels
    wl_step = 0.001 #Initial step size
    def f(wl):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return neff_derivative(wl, nridge, nbox, nclad, w, h, hslab, mode)

    return util.derivative(f, wl, n, wl_step)