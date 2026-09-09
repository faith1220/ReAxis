# ReAxis modifications, 2026-09-09: public naming, compatibility and release packaging.
"""Backward-compatible STAR-CLIP/HPRF module path; implementation is in reaxis."""
from . import reaxis as _reaxis
from .reaxis import *


def __getattr__(name):
    # Preserve explicit historical imports, including private utility names.
    return getattr(_reaxis, name)
