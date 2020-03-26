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