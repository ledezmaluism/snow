# -*- coding: utf-8 -*-
"""
snow: Simulator for Nonlinear Optical Waveguides.

Submodules imported with the package:

- materials:  bulk refractive indices (Sellmeier models)
- waveguides: ridge-waveguide dispersion and pulse propagation
- pulses:     pulses on a time/frequency grid, analysis and plotting
- nlo:        the chi(2) nonlinear envelope equation solver
- util:       numerical helpers

Import explicitly when needed:

- snow.plotting: plotting helpers, including plot_evolution of z snapshots
- snow.crystals: bulk-crystal propagation
- snow.opos:     synchronously pumped OPO simulation

`import snow` does not import matplotlib; `snow.plotting` and `snow.opos` do.
"""

from . import materials
from . import waveguides
from . import pulses
from . import nlo
from . import util

__all__ = ["materials", "waveguides", "pulses", "nlo", "util"]
