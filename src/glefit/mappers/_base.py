#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/10/09 09:35:47
@Author  :   George Trenins
@Desc    :   Variable transformations to impose constraints on the optimizable parameters. Here, `p` denotes the conventional parameter and `x` is the reparametrized variable
'''


from __future__ import print_function, division, absolute_import
from abc import ABC, abstractmethod
import numpy as np


class BaseMapper(ABC):

    @abstractmethod
    def forward(self, p: float) -> float:
        pass

    @abstractmethod
    def inverse(self, x: float) -> float:
        pass

    @abstractmethod
    def grad(self, x: float) -> float:
        pass

    @abstractmethod
    def hess(self, x: float) -> float:
        pass


class IdentityMapper(BaseMapper):
    """A do-nothing mapper, does not impose any constraints."""

    def forward(self, p: float) -> float:
        return np.copy(p)
    
    def inverse(self, x: float) -> float:
        return np.copy(x)
    
    def grad(self, x: float) -> float:
        return 1.0
    
    def hess(self, x: float) -> float:
        return 0.0 

