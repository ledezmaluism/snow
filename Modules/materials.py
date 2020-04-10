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

def refractive_index(material, wl):
    '''
    Parameters
    ----------
    material : STRING
        'SiO2', 'Sapphire', 'LN_MgO_o', 'LN_MgO_e', 'LN_o', 'LN_e'
    wl : SCALAR
        Wavelength

    Returns
    -------
    Refractive index at given wavelength

    '''
    A = 0
    B = 0
    if material=='SiO2':
        A = np.array([0.6961663, 0.4079426, 0.8974794])
        B = np.array([0.0684043, 0.1162414, 9.896161])
        B = B**2
        n = sellmeier(A, B, wl)
    elif material=='Sapphire':
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
    elif material=='LT_MgO_e':
        #This one uses a different form of Sellmeier equation...
        A = np.array([4.49361274, -0.02299905, 0.09659086, -1.0505035e-3, 6.30479079e-4])
        n = poly_LT(A, wl)
    else:
        print('wrong material...?')

    return n