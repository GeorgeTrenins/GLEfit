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
from typing import Optional
import numpy as np


class LowerBoundMapper(BaseMapper):

    def __init__(self, a: Optional[float] = 0.0) -> None:
        """Enforce a < p by mapping p = exp(x) + a

        Args:
            a (float, optional): Lower bound on optimizable parameter. Defaults to 0.0.
        """
        if not isinstance(a, (int, float)):
            raise ValueError(f"Expecting a float value for `a`, instead got {type(a).__name__}")
        self.a = float(a)

    def forward(self, p: float) -> float:
        diff = p - self.a
        x = np.log(diff)
        return x
    
    def inverse(self, x: float) -> float:
        return np.exp(x) + self.a
    
    def grad(self, x: float) -> float:
        return np.exp(x)
    
    def hess(self, x: float) -> float:
        return np.exp(x)
    

class UpperBoundMapper(BaseMapper):

    def __init__(self, b: Optional[float] = 0.0) -> None:
        """Enforce p < b by mapping p = b - exp(x)

        Args:
            b (float, optional): Upper bound on optimizable parameter. Defaults to 0.0.
        """
        if not isinstance(b, (int, float)):
            raise ValueError(f"Expecting a float value for `b`, instead got {type(b).__name__}")
        self.b = float(b)

    def forward(self, p: float) -> float:
        diff = self.b - p
        x = np.log(diff)
        return x
    
    def inverse(self, x: float) -> float:
        return self.b - np.exp(x)
    
    def grad(self, x: float) -> float:
        return -np.exp(x)
    
    def hess(self, x: float) -> float:
        return -np.exp(x)
    

class IntervalMapper(BaseMapper):

    def __init__(
            self, 
            a: Optional[float] = 0.0, 
            b: Optional[float] = 1.0
    ) -> None:
        """Enforce a < p < b by mapping p = a + (b-a)/(1+exp(-x))

        Args:
            a (float, optional): lower bound. Defaults to 0.0.
            b (float, optional): upper bound. Defaults to 1.0.
        """
        for var, label in zip([a, b], 'ab'):
            if not isinstance(var, (int, float)):
                raise ValueError(f"Expecting a float value for `{label}`, instead got {type(var).__name__}")
        self.a, self.b = sorted([float(a), float(b)])
        self.diff = self.b - self.a
        
    def forward(self, p: float) -> float:
        return np.log(p - self.a) - np.log(self.b - p)
    
    def inverse(self, x: float) -> float:
        return self.a + self.diff / (1 + np.exp(-x))
    
    def grad(self, x: float) -> float:
        return self.diff / (2 * np.cosh(x/2))**2
    
    def hess(self, x: float) -> float:
        x2 = x/2
        return -self.diff * np.tanh(x2) / (2 * np.cosh(x2))**2
        

        

    
