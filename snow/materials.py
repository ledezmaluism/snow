"""
Bulk refractive indices.

`refractive_index(material, wl, T)` is the single entry point. Wavelengths may
be given in meters, nanometers or microns; the unit is guessed from the value.

Every material is a `Material` in the `MATERIALS` registry, which records its
formula, source, the wavelength range the formula was fitted over (when
known) and whether it depends on temperature. `available()` lists them and
`register` adds new ones.

@author: Luis Ledezma
"""
import warnings

import numpy as np


class MaterialRangeWarning(UserWarning):
    '''A refractive index was evaluated outside the formula's fitted range.'''


class Material:
    '''
    A refractive-index model.

    Parameters
    ----------
    name : str
        Registry key.
    formula : callable
        formula(wl_um, T) returns n, with wl in microns and T in Celsius.
    description : str
    source : str or None
        Reference for the coefficients.
    wl_range_um : (float, float) or None
        Wavelength range (um) of the fit; None if not known.
    temperature : bool
        Whether the formula uses T.
    '''
    def __init__(self, name, formula, description='', source=None,
                 wl_range_um=None, temperature=False):
        self.name = name
        self.formula = formula
        self.description = description
        self.source = source
        self.wl_range_um = wl_range_um
        self.temperature = temperature

    def n(self, wl_um, T=24.5):
        '''Refractive index at wavelength(s) wl_um (um) and T (C).'''
        if self.wl_range_um is not None:
            lo, hi = self.wl_range_um
            if np.min(wl_um) < lo or np.max(wl_um) > hi:
                warnings.warn(f'{self.name}: wavelength outside the fitted range '
                              f'{lo}-{hi} um ({np.min(wl_um):.3g}-{np.max(wl_um):.3g} um requested)',
                              MaterialRangeWarning, stacklevel=3)
        return self.formula(wl_um, T)

    def __repr__(self):
        return f'Material({self.name!r}: {self.description})'


MATERIALS = {}


def register(name, description='', source=None, wl_range_um=None, temperature=False):
    '''
    Decorator adding a formula(wl_um, T) to the registry under `name`.

    >>> @register('MyGlass', 'constant index', wl_range_um=(0.4, 2.0))
    ... def _(wl, T):
    ...     return 1.5 + 0*wl
    '''
    def wrap(formula):
        MATERIALS[name] = Material(name, formula, description, source, wl_range_um, temperature)
        return formula
    return wrap


def available():
    '''Names of all registered materials.'''
    return sorted(MATERIALS)


def sellmeier(A, B, wl):
    '''
    Sellmeier expansion n^2 = 1 + sum_k A_k wl^2 / (wl^2 - B_k).

    Parameters
    ----------
    A : array
        A_k coefficients (dimensionless).
    B : array
        B_k coefficients, in the square of the units of `wl`.
    wl : float or array
        Wavelength, in the units the coefficients were fitted for (microns
        for every material in this module).

    Returns
    -------
    float or array
        Refractive index.
    '''
    n2 = 1

    for k in range(A.size):
        n2 += A[k]*wl**2/(wl**2 - B[k])

    return np.sqrt(n2)

def poly_LT(A, wl):
    '''
    Polynomial dispersion formula used for MgO:LT,
    n^2 = A0 + A1 wl^2 + A2/wl^2 + A3/wl^4 + A4/wl^6, with wl in microns.
    '''
    n2 =  A[0] + A[1]*wl**2 + A[2]/wl**2 + A[3]/wl**4 + A[4]/wl**6
    return np.sqrt(n2)


def _to_microns(wl):
    '''Guess the unit of wl from its (first) value and convert to microns.'''
    if isinstance(wl, (list, tuple)):
        wl = np.asarray(wl, dtype=float)
    wl_test = wl[0] if isinstance(wl, np.ndarray) and wl.ndim > 0 else wl
    if 100e-9<wl_test<20e-6:
        return wl*1e6 #units were probably meters
    elif 100<wl_test<20000:
        return wl*1e-3 #Units were probably nm
    elif 0.1<wl_test<20:
        return wl #Units were microns, do nothing
    warnings.warn(f'wavelength {wl_test!r} is not in a recognised range for m, nm or um; '
                  'treating it as microns', UserWarning, stacklevel=3)
    return wl


def refractive_index(material, wl, T=24.5):
    '''
    Refractive index of a bulk material.

    Parameters
    ----------
    material : str
        A name from `available()`:

        ============== ====================================================
        'SiO2'         fused silica (Malitson 1965)
        'Sapphire'     Al2O3, extraordinary (Malitson)
        'LN_e', 'LN_o' congruent LiNbO3 (Zelmon 1997)
        'LN_MgO_e',    5% MgO:LiNbO3 (Zelmon 1997)
        'LN_MgO_o'
        'LN_MgO_e_T',  5% MgO:LiNbO3, temperature dependent
        'LN_MgO_o_T'   (Gayer et al. 2008, Appl. Phys. B 91, 343)
        'LT_MgO_e'     MgO:LiTaO3, extraordinary
        'LT_SLT'       stoichiometric LiTaO3 (Brunner et al. 2003)
        'GaP'          gallium phosphide
        'Air'          n = 1
        ============== ====================================================

    wl : float or array
        Wavelength in m, nm or um. The unit is inferred from the (first)
        value: 100 nm - 20 um in meters, 100 - 20000 in nm, 0.1 - 20 in um.
    T : float, optional
        Temperature in Celsius, used by the '_T' models and 'LT_SLT'
        (default 24.5).

    Returns
    -------
    float or array
        Refractive index at `wl`.

    Notes
    -----
    An unknown material name raises ValueError. Evaluating a formula outside
    its fitted wavelength range raises a `MaterialRangeWarning`.
    The Gayer model's absolute dn/dT is about 6x larger than Jundt's 1997
    congruent-LN model; phase-matching temperatures are what it was fitted to.
    '''
    try:
        model = MATERIALS[material]
    except KeyError:
        raise ValueError(f'unknown material {material!r}; available: {", ".join(available())}') from None
    return model.n(_to_microns(wl), T)


# ---- registry -----------------------------------------------------------------
# Wavelength ranges: as given with each source (refractiveindex.info for the
# Malitson fits); None where the code has no source for the coefficients.

@register('SiO2', 'fused silica', 'Malitson, JOSA 55, 1205 (1965)', (0.21, 6.7))
def _sio2(wl, T):
    "https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson"
    A = np.array([0.6961663, 0.4079426, 0.8974794])
    B = np.array([0.0684043, 0.1162414, 9.896161])
    return sellmeier(A, B**2, wl)

@register('Sapphire', 'Al2O3, extraordinary', 'Malitson and Dodge (1972)', (0.2, 5.0))
def _sapphire(wl, T):
    "https://refractiveindex.info/?shelf=main&book=Al2O3&page=Malitson-e"
    A = np.array([1.5039759, 0.55069141, 6.5927379])
    B = np.array([0.0740288, 0.1216529, 20.072248])
    return sellmeier(A, B**2, wl)

_ZELMON = 'Zelmon, Small and Jundt, JOSA B 14, 3319 (1997)'

@register('LN_MgO_e', '5% MgO:LiNbO3, extraordinary', _ZELMON, (0.4, 5.0))
def _ln_mgo_e(wl, T):
    A = np.array([2.2454, 1.3005, 6.8972])
    B = np.array([0.01242, 0.05313, 331.33])
    return sellmeier(A, B, wl)

@register('LN_MgO_o', '5% MgO:LiNbO3, ordinary', _ZELMON, (0.4, 5.0))
def _ln_mgo_o(wl, T):
    A = np.array([2.4272, 1.4617, 9.6536])
    B = np.array([0.01478, 0.05612, 371.216])
    return sellmeier(A, B, wl)

@register('LN_o', 'congruent LiNbO3, ordinary', _ZELMON, (0.4, 5.0))
def _ln_o(wl, T):
    A = np.array([2.6734, 1.2290, 12.614])
    B = np.array([0.01764, 0.05914, 474.6])
    return sellmeier(A, B, wl)

@register('LN_e', 'congruent LiNbO3, extraordinary', _ZELMON, (0.4, 5.0))
def _ln_e(wl, T):
    A = np.array([2.9804, 0.5981, 8.9543])
    B = np.array([0.02047, 0.0666, 416.08])
    return sellmeier(A, B, wl)

@register('GaP', 'gallium phosphide')
def _gap(wl, T):
    A = np.array([1.39, 4.131, 2.57, 2.056])
    B = np.array([0.172**2, 0.234**2, 0.345**2, 27.5**2])
    return sellmeier(A, B, wl)

def _gayer(wl, T, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4):
    #Gayer et al. 2008, eq. (2)
    T0 = 24.5
    f = ( T - T0 ) * ( T + 570.82 )
    n2 = a1 + b1*f + (a2+b2*f)/(wl**2-(a3+b3*f)**2) + (a4+b4*f)/(wl**2-a5**2) - a6*wl**2
    return np.sqrt(n2)

_GAYER = 'Gayer, Sacks, Galun and Arie, Appl. Phys. B 91, 343 (2008); erratum 101, 481 (2010)'

@register('LN_MgO_e_T', '5% MgO:LiNbO3, extraordinary, temperature dependent', _GAYER,
          (0.5, 4.0), temperature=True)
def _ln_mgo_e_t(wl, T):
    return _gayer(wl, T, 5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2,
                  2.86e-6, 4.7e-8, 6.113e-8, 1.516e-4)

@register('LN_MgO_o_T', '5% MgO:LiNbO3, ordinary, temperature dependent', _GAYER,
          (0.5, 4.0), temperature=True)
def _ln_mgo_o_t(wl, T):
    return _gayer(wl, T, 5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97e-2,
                  7.941e-7, 3.134e-8, -4.641e-9, -2.188e-6)

@register('LT_MgO_e', 'MgO:LiTaO3, extraordinary')
def _lt_mgo_e(wl, T):
    #This one uses a different form of Sellmeier equation...
    A = np.array([4.49361274, -0.02299905, 0.09659086, -1.0505035e-3, 6.30479079e-4])
    return poly_LT(A, wl)

@register('LT_SLT', 'stoichiometric LiTaO3, extraordinary, temperature dependent',
          'Bruner et al., Opt. Lett. 28, 194 (2003)', (0.39, 4.1), temperature=True)
def _lt_slt(wl, T):
    A = 4.502483
    B = 0.007294
    C = 0.185087
    D = -0.02357
    E = 0.073423
    F = 0.199595
    G = 0.001
    H = 7.99724
    b = 3.483933e-8*(T+273.15)**2
    c = 1.607839e-8*(T+273.15)**2
    n2 = A + (B+b)/(wl**2 - (C+c)**2) + E/(wl**2 - F**2) + G/(wl**2-H**2) + D*wl**2
    return np.sqrt(n2)

@register('Air', 'n = 1')
def _air(wl, T):
    return 1 if np.isscalar(wl) else np.ones(np.shape(wl))
