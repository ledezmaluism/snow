"""
Ridge waveguides: effective index, dispersion and pulse propagation.

A `Waveguide` combines an effective-index model with a length, a loss and a
chi(2) nonlinearity. The index models are:

- `RidgeIndex`: from the geometry, with the effective index method;
- `BehavioralIndex`: a dispersion polynomial set by its zero-GVD
  wavelength, GVM, index difference and index;
- `TabulatedIndex`: a cubic spline through your own (wavelength, n_eff)
  data, e.g. from a mode solver.

Any object with a method ``neff(wl, T=...)`` works as an index model.

    wg = Waveguide(RidgeIndex(w_top=1.8e-6), length=1e-3)
    wg.neff(2e-6)           # float for a float, array for an array
    out, steps = wg.propagate(pulse, Kg=...)

The module-level functions solve the slab problems used by the effective
index method. All lengths are in meters.

The old all-in-one `waveguide` class still works for this release but is
deprecated.

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


def _item(v):
    #python/numpy scalar from a scalar or size-1 array
    return v.item() if hasattr(v, 'item') else v


# ---- effective-index models ------------------------------------------------------

class RidgeIndex:
    '''
    Effective index of a ridge waveguide from its geometry (effective index
    method, see `neff_ridge`), by default thin-film MgO:LN on SiO2 with air
    cladding.

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
    mode : {'TE', 'TM'}
        Polarization of the fundamental mode. With an anisotropic film,
        choose `tf_material` for the index along the field (e.g. 'LN_MgO_e'
        for TE in x-cut, 'LN_MgO_o' for TM).
    '''
    def __init__(self, w_top=1e-6, h_thinfilm=700e-9, h_etch=350e-9, theta=60,
                 tf_material='LN_MgO_e', box_material='SiO2', clad_material='Air',
                 mode='TE'):
        if mode not in ('TE', 'TM'):
            raise ValueError(f"mode must be 'TE' or 'TM', not {mode!r}")
        w_base = w_top + 2*h_etch/np.tan(theta*pi/180)
        if h_etch*(w_base-w_top)<0:
            raise ValueError("Something wrong with this geometry")
        self.w_top = w_top
        self.h_etch = h_etch
        self.h_thinfilm = h_thinfilm
        self.theta = theta
        self.tf_material = tf_material
        self.box_material = box_material
        self.clad_material = clad_material
        self.mode = mode
        self.w_base = w_base
        self.h_slab = h_thinfilm - h_etch

    def neff(self, wl, T=24.5, mode=None):
        '''
        Effective index at wavelength(s) wl (m) and temperature T (C).
        `mode` overrides the polarization. Float for a scalar wl, array
        otherwise.
        '''
        if mode is None:
            mode = self.mode
        um = 1e-6
        w = (self.w_top + self.w_base)/2
        h = self.h_thinfilm
        hslab = self.h_slab

        scalar = np.ndim(wl) == 0
        wls = np.atleast_1d(np.asarray(wl, dtype=float))
        neff = np.zeros(wls.shape)
        for kw in range(wls.size):
            wk = wls.flat[kw]
            nridge = materials.refractive_index(self.tf_material, wk/um, T=T)
            nbox = materials.refractive_index(self.box_material, wk/um, T=T)
            nclad = materials.refractive_index(self.clad_material, wk/um, T=T)
            neff.flat[kw] = neff_ridge(wk, nridge, nbox, nclad, w, h, hslab, mode)
        return float(neff[0]) if scalar else neff


class BehavioralIndex:
    '''
    Effective index of a "black box" waveguide defined by its dispersion
    rather than its geometry.

    The GVD is the quadratic beta2(omega) = c1 (omega - z1)(omega - z2),
    integrated to beta1, beta and n_eff. The free constants are fixed by:

    Parameters
    ----------
    z_wl : float
        Zero-GVD wavelength (m); sets z1.
    wl_1, wl_2 : float
        Wavelengths (m) between which the GVM and index difference are set.
    GVM : float
        beta1(wl_2) - beta1(wl_1) (s/m); sets z2.
    delta_n : float
        n(wl_2) - n(wl_1).
    n_f0 : float
        Index at wl_f0.
    wl_f0 : float or None
        Wavelength (m) where n = n_f0; default (wl_1 + wl_2)/2.
    c1 : float
        Curvature of beta2 (s^4/m).
    '''
    def __init__(self, z_wl=1e-6, wl_1=1e-6, wl_2=2e-6, GVM=0, delta_n=0.2, n_f0=2.0,
                 wl_f0=None, c1=10**(-55)):
        self.z_wl = z_wl
        self.wl_1 = wl_1
        self.wl_2 = wl_2
        self.GVM = GVM
        self.delta_n = delta_n
        self.n_f0 = n_f0
        self.wl_f0 = wl_f0
        self.c1 = c1

    def neff(self, wl, T=None):
        '''Effective index at wl (m); T is ignored. Float for a scalar wl.'''
        n = self._n(np.asarray(wl, dtype=float) if np.ndim(wl) else wl)
        return float(n) if np.ndim(wl) == 0 else n

    def _n(self, wl):
        z_wl = self.z_wl
        wl_1 = self.wl_1
        wl_2 = self.wl_2
        GVM = self.GVM
        delta_n = self.delta_n
        n_f0 = self.n_f0
        wl_f0 = self.wl_f0

        if wl_f0 is None:
            wl_f0 = ( wl_1 + wl_2 )/2

        #z1, w1, w2 are wavelengths
        z1 = 2*pi*c/z_wl
        omega_1 = 2*pi*c/wl_1
        omega_2 = 2*pi*c/wl_2
        omega0_f0 = 2*pi*c/wl_f0

        #Frequency array:
        omega = 2*pi*c/wl

        c1 = self.c1

        # First we want to determine the remaining free parameters
        z2_num = 6*GVM - 2*c1*(omega_2**3-omega_1**3)+3*c1*z1*(omega_2**2-omega_1**2)
        z2_den = 3*c1*(2*z1-omega_2-omega_1)*(omega_2-omega_1)
        z2 = z2_num/z2_den

        c3 = (omega_2**3-omega_1**3)/12 - (z1+z2)*(omega_2**2-omega_1**2)/6 + z1*z2*(omega_2-omega_1)/2
        c3 = delta_n/c - c1*c3
        c3 = c3 / (1/omega_2 - 1/omega_1)
        c2 = n_f0/c - c1*(omega0_f0**3/12 - (z1+z2)*(omega0_f0**2)/6 +z1*z2*omega0_f0/2) - c3/omega0_f0

        # beta_2 = c1*(omega**2 -(z1+z2)*omega + z1*z2)
        # beta_1 = c1*(omega**3/3 - (z1+z2)*(omega**2)/2 + z1*z2*omega) + c2
        # beta = c1*(omega**4/12 - (z1+z2)*(omega**3)/6 + z1*z2*(omega**2)/2) + c2*omega + c3
        return c*( c1*(omega**3/12 - (z1+z2)*(omega**2)/6 + z1*z2*omega/2) + c2 + c3/omega)


class TabulatedIndex:
    '''
    Effective index interpolated (cubic spline) through tabulated data.

    Parameters
    ----------
    wl : array
        Wavelengths (m), increasing.
    n : array
        Effective index at each wavelength.
    '''
    def __init__(self, wl, n):
        self.wl = np.asarray(wl, dtype=float)
        self.n = np.asarray(n, dtype=float)
        self.tck = interpolate.splrep(self.wl, self.n, s=0)

    def neff(self, wl, T=None):
        '''Interpolated index at wl (m); T is ignored. Float for a scalar wl.'''
        n = interpolate.splev(wl, self.tck, der=0)
        return float(n) if np.ndim(wl) == 0 else n


def _uniform_poling(z):
    return 1.0


# ---- waveguide -------------------------------------------------------------------

class Waveguide:
    '''
    A waveguide: an effective-index model plus length, loss and chi(2)
    nonlinearity.

    Parameters
    ----------
    index : RidgeIndex, BehavioralIndex, TabulatedIndex or similar
        Anything with ``neff(wl, T=...)``.
    length : float, optional
        Length (m); needed to propagate.
    loss : float, optional
        Power attenuation coefficient alpha (1/m): power decays as
        exp(-alpha z). `util.absorption_coeff` converts from dB/cm.
    poling : callable, optional
        d(z), the sign of chi(2) at position z (m), e.g.
        ``lambda z: np.sign(np.cos(2*pi*z/period))``. Default: uniform (+1).
    X0 : float, optional
        Effective chi(2) (m/V). Default 0 (linear waveguide).
    N : float, optional
        Mode normalization; the coupling is k = d(z) X0 omega / (4 N).

    All dispersion methods return a float for a scalar wavelength and an
    array for an array.
    '''
    def __init__(self, index, length=None, loss=0.0, poling=None, X0=0.0, N=1.0):
        self.index = index
        self.length = length
        self.alpha = loss
        self.poling = _uniform_poling if poling is None else poling
        self.X0 = X0
        self.N = N

    # -- parameters --
    @property
    def L(self):
        '''Length (m); same as `length`.'''
        return self.length

    @L.setter
    def L(self, value):
        self.length = value

    @property
    def loss(self):
        '''Power attenuation coefficient (1/m); same as `alpha`.'''
        return self.alpha

    @loss.setter
    def loss(self, value):
        self.alpha = value

    def set_length(self, L):
        '''Waveguide length (m).'''
        self.length = L

    def set_loss(self, alpha):
        '''
        Power attenuation coefficient alpha (1/m): power decays as
        exp(-alpha z). Use `util.absorption_coeff` to convert from dB/cm.
        '''
        self.alpha = alpha

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

    # -- dispersion --
    def neff(self, wl, T=24.5, **kwargs):
        '''Effective index at wl (m) and T (C); extra arguments go to the index model.'''
        return self.index.neff(wl, T=T, **kwargs)

    def beta(self, wl, T=24.5):
        '''Propagation constant 2 pi n_eff / wl (1/m).'''
        if np.ndim(wl):
            wl = np.asarray(wl, dtype=float)
        return 2 * pi * self.neff(wl, T=T) / wl

    def beta1(self, wl, T=24.5):
        '''
        Inverse group velocity d beta/d omega = (n - wl dn/dwl)/c (s/m),
        from a Richardson-extrapolated derivative.
        '''
        scalar = np.ndim(wl) == 0
        wls = np.atleast_1d(np.asarray(wl, dtype=float))
        n = 2 #number of extrapolation levels
        wl_step = 1e-9 #Initial step size
        b1 = np.zeros(wls.shape)

        def neff_T(w):
            return self.neff(w, T=T)

        for kw in range(wls.size):
            wk = wls.flat[kw]
            dndl = util.derivative(neff_T, wk, n, wl_step)
            b1.flat[kw] = _item((neff_T(wk) - wk * dndl)/c)
        return float(b1[0]) if scalar else b1

    def beta2(self, wl, T=24.5):
        '''Group velocity dispersion d^2 beta/d omega^2 (s^2/m).'''
        scalar = np.ndim(wl) == 0
        wls = np.atleast_1d(np.asarray(wl, dtype=float))
        n = 2 #number of extrapolation levels
        wl_step = 1e-9 #Initial step size
        gvd = np.zeros(wls.shape)

        def beta1_T(w):
            return self.beta1(w, T=T)

        for kw in range(wls.size):
            wk = wls.flat[kw]
            db1dl = util.derivative(beta1_T, wk, n, wl_step)
            gvd.flat[kw] = - wk**2 / (2*pi*c) * db1dl
        return float(gvd[0]) if scalar else gvd

    def GVD(self, wl, T=24.5):
        '''Same as `beta2`.'''
        return self.beta2(wl, T=T)

    # -- propagation --
    def _beta_and_frame(self, pulse, v_ref=None, T=24.5):
        '''beta on the pulse grid, and v_ref (default: group velocity at wl0).'''
        beta = self.beta(pulse.wl, T=T)
        if v_ref is None:
            #group velocity at the reference wavelength only (pulse.wl[0])
            v_ref = 1/_item(np.ravel(self.beta1(pulse.wl0, T=T))[0])
        return beta, v_ref

    def _linear_operator(self, pulse, v_ref=None, T=24.5):
        '''
        Dispersion operator in the moving frame, shared by the propagators:
        D = beta - beta_ref - Omega/v_ref - i alpha/2 (1/m).
        Returns (D, beta_ref, v_ref).
        '''
        beta, v_ref = self._beta_and_frame(pulse, v_ref, T)
        D, beta_ref = nlo.linear_operator(pulse, beta, self.alpha, v_ref)
        return D, beta_ref, v_ref

    def _require_length(self):
        if self.length is None:
            raise ValueError('set the waveguide length first (length=... or set_length)')

    def propagate_linear(self, pulse, v_ref=None, T=24.5):
        '''
        Propagate a pulse through the waveguide with dispersion and loss only.

        Uses the same moving frame and conventions as `propagate`, so for a
        zero nonlinear coefficient both give the same output. The phase
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
        self._require_length()
        D, _, _ = self._linear_operator(pulse, v_ref=v_ref, T=T)
        output_pulse = pulses.pulse(pulse.t, pulse.a, pulse.wl0, pulse.frep)
        output_pulse.update_fd(pulse.A * np.exp(-1j*D*self.length))
        return output_pulse

    def propagate(self, pulse, v_ref=None, verbose=False, z0=0, T=24.5, Kg=0,
                  Qnoise=False, rtol=1e-4, atol=1e-4, max_step=None,
                  poling_resolution=10e-9, rng=None):
        '''
        Propagate a pulse through the waveguide with the chi(2) nonlinear
        envelope equation (`nlo.propagate`).

        Parameters
        ----------
        pulse : pulses.pulse
            Input pulse; its grid defines the simulation grid.
        v_ref : float, optional
            Velocity of the moving frame (m/s). Default: group velocity at
            the pulse's reference wavelength and temperature T.
            Use 1/wg.beta1(wl) to keep a given wavelength stationary.
        verbose : bool
            Print the integrator status and run time.
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
        self._require_length()
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
            z_breaks = poling_domain_walls(self.poling, z0, self.length, poling_resolution)
        if max_step is None:
            #Within a domain the phase mismatch turns the nonlinear drive by
            #~pi, which RK45's error estimate misjudges: limit the step (#50)
            max_step = np.inf
            if z_breaks is not None and z_breaks.size >= 2:
                max_step = np.min(np.diff(z_breaks)) / 4

        return nlo.propagate(pulse, self.length, beta, v_ref, k, alpha=self.alpha, z0=z0,
                             Kg=Kg, Qnoise=Qnoise, rtol=rtol, atol=atol,
                             max_step=max_step, z_breaks=z_breaks,
                             verbose=verbose, rng=rng)

    def propagate_NEE(self, pulse, v_ref=None, verbose=True, zcheck_step=None, **kwargs):
        '''Deprecated: use `propagate` (which is quiet by default).'''
        warnings.warn('propagate_NEE is deprecated; use propagate (verbose=False by default)',
                      DeprecationWarning, stacklevel=2)
        if zcheck_step is not None:
            warnings.warn('zcheck_step is ignored and will be removed',
                          DeprecationWarning, stacklevel=2)
        return self.propagate(pulse, v_ref=v_ref, verbose=verbose, **kwargs)


# ---- deprecated all-in-one class -------------------------------------------------

class waveguide(Waveguide):
    '''
    Deprecated: use ``Waveguide(RidgeIndex(...))``, ``Waveguide(BehavioralIndex(...))``
    or ``Waveguide(TabulatedIndex(...))``.

    Same constructor as before (geometry by default, ``behavioral=True`` or
    ``external=True`` for the other models). Unlike `Waveguide`, `neff` of
    the geometric model, `beta1` and `beta2` return arrays even for a scalar
    wavelength, as they always did.
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
        warnings.warn('waveguides.waveguide is deprecated; use Waveguide(RidgeIndex(...)), '
                      'Waveguide(BehavioralIndex(...)) or Waveguide(TabulatedIndex(...)). '
                      'The new class returns floats for scalar wavelengths.',
                      DeprecationWarning, stacklevel=2)
        if mode not in ('TE', 'TM'):
            raise ValueError(f"mode must be 'TE' or 'TM', not {mode!r}")
        self.mode = mode
        if behavioral:
            self._kind = 'behavioral'
            index = BehavioralIndex(z_wl, wl_1, wl_2, GVM, delta_n, n_f0, wl_f0, c1)
            self.beh_z_wl = z_wl
            self.beh_wl_1 = wl_1
            self.beh_wl_2 = wl_2
            self.beh_GVM = GVM
            self.beh_delta_n = delta_n
            self.beh_n_f0 = n_f0
            self.wl_f0 = wl_f0
            self.c1 = c1
        elif external:
            self._kind = 'external'
            index = TabulatedIndex(wl_array, n_array)
            self.tck = index.tck
        else:
            self._kind = 'ridge'
            index = RidgeIndex(w_top, h_thinfilm, h_etch, theta, tf_material,
                               box_material, clad_material, mode)
            for name in ('w_top', 'h_etch', 'h_thinfilm', 'theta', 'tf_material',
                         'box_material', 'clad_material', 'w_base', 'h_slab'):
                setattr(self, name, getattr(index, name))
        super().__init__(index)

    def neff(self, wl, *args, **kwargs):
        '''Old signature: neff(wl, mode=None, T=24.5) for the geometric model, neff(wl, T) otherwise.'''
        if self._kind == 'ridge':
            return self.neff_physical(wl, *args, **kwargs)
        if self._kind == 'behavioral':
            return self.neff_behavioral(wl, *args, **kwargs)
        return self.neff_interp(wl, *args, **kwargs)

    def neff_physical(self, wl, mode=None, T=24.5):
        '''Geometric effective index; always an array.'''
        return np.atleast_1d(self.index.neff(wl, T=T, mode=mode))

    def neff_interp(self, wl, T=25):
        '''Effective index from the spline of the external table (T is ignored).'''
        return interpolate.splev(wl, self.tck, der=0)

    def neff_behavioral(self, wl, T=None):
        '''Behavioral-model effective index.'''
        return self.index._n(wl)

    def beta1(self, wl, T=24.5):
        '''Inverse group velocity (s/m); always an array.'''
        return np.atleast_1d(super().beta1(wl, T=T))

    def beta2(self, wl, T=24.5):
        '''Group velocity dispersion (s^2/m); always an array.'''
        return np.atleast_1d(super().beta2(wl, T=T))

    def GVD(self, wl, T=24.5):
        return self.beta2(wl, T=T)


# ---- slab and ridge solvers --------------------------------------------------------

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
    d n_eff / d wl of `neff_ridge`. Deprecated: use `Waveguide.beta1`.

    The step is 1e-3 in the units of `wl`, so this was only meaningful for
    wavelengths in microns, while `neff_ridge` works in meters.
    '''
    warnings.warn('neff_derivative is deprecated and will be removed; '
                  'use Waveguide.beta1 instead', DeprecationWarning, stacklevel=2)
    n = 2 #number of extrapolation levels
    wl_step = 0.001 #Initial step size
    def f(wl):
        return neff_ridge(wl, nridge, nbox, nclad, w, h, hslab, mode)

    return util.derivative(f, wl, n, wl_step)


def neff_2derivative(wl, nridge, nbox=1, nclad=1, w=1, h=0.7, hslab=0.1, mode='TE'):
    '''
    d^2 n_eff / d wl^2 of `neff_ridge`. Deprecated: use `Waveguide.beta2`.
    '''
    warnings.warn('neff_2derivative is deprecated and will be removed; '
                  'use Waveguide.beta2 instead', DeprecationWarning, stacklevel=2)
    n = 2 #number of extrapolation levels
    wl_step = 0.001 #Initial step size
    def f(wl):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return neff_derivative(wl, nridge, nbox, nclad, w, h, hslab, mode)

    return util.derivative(f, wl, n, wl_step)
