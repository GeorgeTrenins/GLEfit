from ._base import BaseProperty
from .properties import MemoryKernel, MemorySpectrum

PROPERTY_MAP = {
    "kernel" : MemoryKernel,
    "spectrum" : MemorySpectrum
}