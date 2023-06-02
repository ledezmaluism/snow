# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:29:38 2020

@author: Luis Ledezma
"""

import numpy as np

def sech(x):
    return 1/np.cosh(x)

def check(value, ll, ul):
    if ll <= value <= ul:
        return True
    return False

def absorption_coeff(Alpha):
    # convert from dB/cm to 1/m
    alpha = np.log((10**(Alpha * 0.1))) * 100
    return alpha

def derivative( f, x, n, h ):
    """Richardson's Extrapolation to approximate  f'(x) at a particular x.

    USAGE:
	d = richardson( f, x, n, h )

    INPUT:
	f	- function to find derivative of
	x	- value of x to find derivative at
	n	- number of levels of extrapolation
	h	- initial stepsize

    OUTPUT:
        numpy float array -  two-dimensional array of extrapolation values.
                             The [n,n] value of this array should be the
                             most accurate estimate of f'(x).

    NOTES:                             
        Based on an algorithm in "Numerical Mathematics and Computing"
        4th Edition, by Cheney and Kincaid, Brooks-Cole, 1999.

    AUTHOR:
        Jonathan R. Senning <jonathan.senning@gordon.edu>
        Gordon College
        February 9, 1999
        Converted ty Python August 2008
    """

    # d[n,n] will contain the most accurate approximation to f'(x).

    d = np.array( [[0] * (n + 1)] * (n + 1), float )

    for i in range( n + 1 ):
        d[i,0] = 0.5 * ( f( x + h ) - f( x - h ) ) / h

        powerOf4 = 1  # values of 4^j
        for j in range( 1, i + 1 ):
            powerOf4 = 4 * powerOf4
            d[i,j] = d[i,j-1] + ( d[i,j-1] - d[i-1,j-1] ) / ( powerOf4 - 1 )

        h = 0.5 * h

    return d[n,n]