#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   intervals.py
@Time    :   2025/10/09 09:48:06
@Author  :   George Trenins
@Desc    :   Impose inequality constraints of the type
a < p, p < b, and a < p < b
'''

from __future__ import print_function, division, absolute_import
from ._base import BaseMapper
import numbers
import numpy as np


def _check_finite(p: float) -> None:
    """Raise ValueError for NaN or infinite p.
    """
    if not np.isfinite(p):
        raise ValueError(
            f"Invalid parameter value p = {float(p)}, must be finite")


def _checked_bound(value: float, name: str) -> float:
    """Return the bound `name` as a float.

    Raises TypeError unless it is a real number and ValueError if it is NaN or infinite.
    """
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"`{name}` must be a real number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value}")
    return value


class LowerBoundMapper(BaseMapper):

    def __init__(self, a: float = 0.0) -> None:
        """Enforce a < p by mapping p = exp(x) + a

        Args:
            a (float, optional): Lower bound on optimizable parameter.
                Defaults to 0.0. Must be a finite real number.
        """
        self.a = _checked_bound(a, "a")

    def forward(self, p: float) -> float:
        _check_finite(p)
        diff = p - self.a
        if diff <= 0:
            raise ValueError(f"Invalid parameter value p = {float(p)}, must be greater than  a = {self.a}")
        x = np.log(diff)
        return x
    
    def inverse(self, x: float) -> float:
        return np.exp(x) + self.a
    
    def grad(self, x: float) -> float:
        return np.exp(x)
    
    def hess(self, x: float) -> float:
        return np.exp(x)
    

class UpperBoundMapper(BaseMapper):

    def __init__(self, b: float = 0.0) -> None:
        """Enforce p < b by mapping p = b - exp(x)

        Args:
            b (float, optional): Upper bound on optimizable parameter.
                Defaults to 0.0. Must be a finite real number.
        """
        self.b = _checked_bound(b, "b")

    def forward(self, p: float) -> float:
        _check_finite(p)
        diff = self.b - p
        if diff <= 0:
            raise ValueError(f"Invalid parameter value p = {float(p)}, must be smaller than  b = {self.b}")
        x = np.log(diff)
        return x
    
    def inverse(self, x: float) -> float:
        return self.b - np.exp(x)
    
    def grad(self, x: float) -> float:
        return -np.exp(x)
    
    def hess(self, x: float) -> float:
        return -np.exp(x)
    

class IntervalMapper(BaseMapper):

    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        """Enforce a < p < b by mapping p = a + (b-a)/(1+exp(-x))

        Args:
            a (float, optional): lower bound. Defaults to 0.0.
            b (float, optional): upper bound. Defaults to 1.0.
                Both must be finite real numbers, with a < b.
        """
        a, b = _checked_bound(a, "a"), _checked_bound(b, "b")
        if not a < b:
            raise ValueError(f"Bounds must satisfy a < b, got a = {a}, b = {b}")
        self.a, self.b = a, b
        self.diff = self.b - self.a
        
    def forward(self, p: float) -> float:
        _check_finite(p)
        diff = p - self.a
        if diff <= 0:
            raise ValueError(f"Invalid parameter value p = {float(p)}, must be greater than  a = {self.a}")
        ans = np.log(diff)
        diff = self.b - p
        if diff <= 0:
            raise ValueError(f"Invalid parameter value p = {float(p)}, must be smaller than  b = {self.b}")
        ans -= np.log(diff)
        return ans
    
    def inverse(self, x: float) -> float:
        return self.a + self.diff / (1 + np.exp(-x))
    
    def grad(self, x: float) -> float:
        return self.diff / (2 * np.cosh(x/2))**2
    
    def hess(self, x: float) -> float:
        x2 = x/2
        return -self.diff * np.tanh(x2) / (2 * np.cosh(x2))**2
