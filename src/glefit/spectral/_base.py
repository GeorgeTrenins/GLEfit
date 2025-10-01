#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/09/22 08:38:08
@Author  :   George Trenins
@Desc    :   Base class for calculating model spectral densities and corresponding memory-friction kernels
'''


from __future__ import print_function, division, absolute_import
from abc import ABC, abstractmethod


class BaseSpectralDensity(ABC):
    """Base class for calculating model spectral densities. Defines the 
    following methods:

        * J(omega): spectral density
        * Lambda(omega): spectrum, Lambda = J/omega
        * K(t): memory-friction kernel
        
    """

    @abstractmethod
    def J(self, omega):
        raise NotImplementedError
    
    @abstractmethod
    def Lambda(self, omega):
        raise NotImplementedError
    
    @abstractmethod
    def K(self, t):
        raise NotImplementedError