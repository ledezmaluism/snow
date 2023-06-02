"""
@author: Luis Ledezma
"""
import numpy as np

def sellmeier(A, B, wl):
    '''
    Parameters
    ----------
    A : ARRAY
        Ak coefficients
    B : ARRAY
        Bk coefficients
    wl : SCALAR
        Wavelegnth.

    Returns
    -------
    Refractive index from Sellmeiers expansion

    '''
    n2 = 1

    for k in range(A.size):
        n2 += A[k]*wl**2/(wl**2 - B[k])

    return np.sqrt(n2)

def poly_LT(A, wl):
    n2 =  A[0] + A[1]*wl**2 + A[2]/wl**2 + A[3]/wl**4 + A[4]/wl**6
    return np.sqrt(n2)

def refractive_index(material, wl, T=24.5):
    '''
    Parameters
    ----------
    material : STRING
        'SiO2', 'Sapphire', 'LN_MgO_o', 'LN_MgO_e', 'LN_o', 'LN_e', 'GaP'
    wl : SCALAR
        Wavelength
    T : SCALAR
        Temperature

    Returns
    -------
    Refractive index at given wavelength

    '''
    
    #Let's check for the right units
    if isinstance(wl, (list, tuple, np.ndarray)):
        wl_test = wl[0]
    else:
        wl_test = wl
    if 100e-9<wl_test<20e-6:
        wl = wl*1e6 #units were probably meters
    elif 100<wl_test<20000:
        wl = wl*1e-3 #Units were probably nm
    elif 0.1<wl_test<20:
        pass #Units were microns, do nothing
    else:
        print('Warning = Wavelength value seems out of range!')
            
    
    A = 0
    B = 0
    if material=='SiO2':
        "https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson"
        A = np.array([0.6961663, 0.4079426, 0.8974794])
        B = np.array([0.0684043, 0.1162414, 9.896161])
        B = B**2
        n = sellmeier(A, B, wl)
    elif material=='Sapphire':
        "https://refractiveindex.info/?shelf=main&book=Al2O3&page=Malitson-e"
        A = np.array([1.5039759, 0.55069141, 6.5927379])
        B = np.array([0.0740288, 0.1216529, 20.072248])
        B = B**2
        n = sellmeier(A, B, wl)
    elif material=='LN_MgO_e':
        A = np.array([2.2454, 1.3005, 6.8972])
        B = np.array([0.01242, 0.05313, 331.33])
        n = sellmeier(A, B, wl)
    elif material=='LN_MgO_o':
        A = np.array([2.4272, 1.4617, 9.6536])
        B = np.array([0.01478, 0.05612, 371.216])
        n = sellmeier(A, B, wl)
    elif material=='LN_o':
        A = np.array([2.6734, 1.2290, 12.614])
        B = np.array([0.01764, 0.05914, 474.6])
        n = sellmeier(A, B, wl)
    elif material=='LN_e':
        A = np.array([2.9804, 0.5981, 8.9543])
        B = np.array([0.02047, 0.0666, 416.08])
        n = sellmeier(A, B, wl)
    elif material=='GaP':
        A = np.array([1.39, 4.131, 2.57, 2.056])
        B = np.array([0.172**2, 0.234**2, 0.345**2, 27.5**2])
        n = sellmeier(A, B, wl)
    elif material=='LN_MgO_e_T':
        #Temperature dependent model 
        #from o. gayer, z. sacks, e. galun, a. arie 2008
        a1 = 5.756
        a2 = 0.0983
        a3 = 0.2020
        a4 = 189.32
        a5 = 12.52
        a6 = 1.32e-2
        b1 = 2.86e-6
        b2 = 4.7e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
        T0 = 24.5
        f = ( T - T0 ) * ( T + 570.82 )
        n2 = a1 + b1*f + (a2+b2*f)/(wl**2-(a3+b3*f)**2) + (a4+b4*f)/(wl**2-a5**2) - a6*wl**2
        n = np.sqrt(n2)
    elif material=='LN_MgO_o_T':
        #Temperature dependent model 
        #from o. gayer, z. sacks, e. galun, a. arie 2008
        a1 = 5.653
        a2 = 0.1185
        a3 = 0.2091
        a4 = 89.61
        a5 = 10.85
        a6 = 1.97e-2
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6
        T0 = 24.5
        f = ( T - T0 ) * ( T + 570.82 )
        n2 = a1 + b1*f + (a2+b2*f)/(wl**2-(a3+b3*f)**2) + (a4+b4*f)/(wl**2-a5**2) - a6*wl**2
        n = np.sqrt(n2)
    elif material=='LT_MgO_e':
        #This one uses a different form of Sellmeier equation...
        A = np.array([4.49361274, -0.02299905, 0.09659086, -1.0505035e-3, 6.30479079e-4])
        n = poly_LT(A, wl)
    elif material=='LT_SLT':
        #From Brunner et. al Optics Letters, Vol 28, No 3, 2003
        A = 4.502483
        B = 0.007294
        C = 0.185087
        D = -0.02357
        E = 0.073423
        F = 0.199595
        G = 0.001
        H = 7.99724
        T = 25
        b = 3.483933e-8*(T+273.15)**2
        c = 1.607839e-8*(T+273.15)**2
        n2 = A + (B+b)/(wl**2 - (C+c)**2) + E/(wl**2 - F**2) + G/(wl**2-H**2) + D*wl**2
        n = np.sqrt(n2)
    elif material=='Air':
        n = 1
    else:
        print('wrong material...?')

    return n

def extrapolate():
    pass