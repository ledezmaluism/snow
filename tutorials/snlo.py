# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:19:37 2020

@author: ledezmaluism

Helper module to import Modules/layout.py
"""

# Import a module with the same name from a different directory.

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# Temporarily hijack __file__ to avoid adding names at module scope;
# __file__ will be overwritten again during the reload() call.
__file__ = {'sys': sys, 'importlib': importlib}

del importlib
del os
del sys

__file__['importlib'].reload(__file__['sys'].modules[__name__])