# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:29:38 2020

@author: luish
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