"""Deprecated location: use ``from snow import lumerical``."""
import warnings

warnings.warn("the top-level lumerical module is deprecated; use 'from snow import lumerical'",
              DeprecationWarning, stacklevel=2)

from snow.lumerical import *  # noqa: E402,F401,F403
from snow.lumerical import _test_  # noqa: E402,F401
