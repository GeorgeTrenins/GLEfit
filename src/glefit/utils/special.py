#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   special.py
@Time    :   2025/10/15 13:16:17
@Author  :   George Trenins
@Desc    :   Special functions
'''

from __future__ import print_function, division, absolute_import
import numpy as np


def coscosh(x, nu=0):
    """Return cos(x) if x < 0 and cosh(x) otherwise

    Args:
        x : float
            function argument
        nu : int
             derivative order

    Return:
       Value of piecewise function at x if nu == 0 and its nu'th derivative w.r.t. x 
       if nu > 0
    """

    if not isinstance(nu, int):
        raise TypeError(f"The derivative order `nu` must be integer, instead got {nu.__class__.__name__}")
    if nu < 0:
        raise ValueError(f"The derivative order must be non-negative, instead got {nu = }.")
    mask = x < 0
    if nu%2 == 0:
        k = nu//2
        f, g = np.cos, np.cosh
    else:
        k = (nu+1)//2
        f, g = np.sin, np.sinh
    sgn = np.where(mask, (-1)**k, 1)
    ans = np.where(mask, sgn*f(x), g(x))
    return ans

def sincsinhc(x, nu=0):
    """Return sinc(x) if x < 0 and sinhc(x) otherwise

    Args:
        x : float
            function argument
        nu : int
             derivative order (0, 1, or 2)

    Return:
       Value of piecewise function at x if nu == 0 and its nu'th derivative w.r.t. x 
       if nu > 0.
    """
    if not isinstance(nu, int):
        raise TypeError(f"The derivative order `nu` must be integer, instead got {nu.__class__.__name__}")    
    mask = x < 0
    eps = 1.0e-2
    sgn = np.where(mask, -1, 1)
    cond0 = x > eps
    cond1 = x > -eps
    small = np.logical_and(np.logical_not(cond0), cond1)
    den = np.where(small, 1.0, x)
    if nu == 0:
        ans = np.where( 
            x > eps, 
            np.sinh(x)/den, 
            np.where(x > -eps, 
                     1 + sgn*x**2/6 + x**4/120 + sgn*x**6/5040,
                     np.sinc(x/np.pi)) )
    elif nu == 1:
        ans = np.where( 
            x > eps, 
            np.exp(x)/(2*den) * (1-1/den) + np.exp(-x)/(2*den) * (1+1/den),
            np.where(x > -eps, 
                     sgn*x/3 + x**3/30 + sgn*x**5/840,
                     (np.cos(x) - np.sinc(x/np.pi))/den))
    elif nu == 2:
        ans = np.where( 
            x > eps, 
            np.exp(x)/den * (0.5 - 1/den + 1/den**2) - np.exp(-x)/den * (0.5 + 1/den + 1/den**2),
            np.where(x > -eps, 
                     sgn/3 + x**2/10 + sgn*x**4/168,
                     np.sinc(x/np.pi) * (2/x**2 - 1) - np.cos(x) * (2/den**2)) )
    else:
        raise ValueError(f"The derivative order must be 0, 1, or 2, instead got {nu = }.")
    return ans