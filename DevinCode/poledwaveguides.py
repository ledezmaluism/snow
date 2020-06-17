# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:47:23 2020

@author: devin
"""
import numpy as np
import matplotlib.pyplot as plt

import customwaveguides
import gratings

# =============================================================================
# class represents a PPLN waveguide. It has both a grating profile k(z) as 
#  well as a dimension profile captured in the customwaveguide
#Use class fct to engineer a poledwaveguide
# =============================================================================
class poledwaveguide:
    def __init__(self, g, cwg, length = 1e-2): 
        
        self.length = length
        self.grating = g
        self.customwaveguide = cwg

    # TODO Add plots of things vs. wavelength (i.e reuse plotting fct from prev modules but pass in neff)
    # TODO Add nonlinear fcts to evaluate this waveguide