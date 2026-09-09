"""Paper-aligned ReAxis modules (implementation: adaptcliplib.reaxis)."""
from adaptcliplib import reaxis as _reaxis
from adaptcliplib.reaxis import *


def __getattr__(name):
    return getattr(_reaxis, name)
