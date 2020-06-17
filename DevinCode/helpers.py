# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:19:37 2020

@author: devin
"""
import numpy as np

# =============================================================================
# Module containing relevant helper fcts
# =============================================================================
c = 3e8
def wtok(w,n):
    k = n*w/c; return k
def ktow(k,n):
    w = c*k/n; return w
def ktowl(k,n):
    wl = 2*np.pi*n/k; return wl
def wltok(wl, w):
    k 2*np.pi*n/wl; return k