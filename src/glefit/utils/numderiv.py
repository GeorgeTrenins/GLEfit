#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   numderiv.py
@Time    :   2025/09/16 11:22:28
@Author  :   George Trenins
@Desc    :   Tools for numerical differentiation
'''


from __future__ import print_function, division, absolute_import
import numpy as np
from typing import Callable, Iterable


# Central stencils for a first derivative, keyed by their order of accuracy:
# the weights of f(x + k h) for k = -a, ..., a.
STENCILS = {
    2 : [ -1/2, 0, 1/2 ],
    4 : [ 1/12, -2/3, 0, 2/3, -1/12 ],
}


def jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: Iterable[float],
    *,
    order: int = 2,
) -> np.ndarray:
    """
    Return the Jacobian of f: Rⁿ → Rᵐ at x by central finite differences.

    Each x_i is stepped by h_i = ε^(1/(1+order))·max(1, |x_i|), where ε is
    the machine epsilon. This balances the truncation error, of order
    h^order, against the rounding error of f, of order ε/h (Numerical
    Recipes, 3rd edition, section 5.7). h_i is then adjusted so that
    x_i + h_i is exactly representable.

    Args:
        f (callable): maps an array of shape (n,) to one of shape (m,)
        x (array_like): point of shape (n,)
        order (int, optional): order of accuracy, 2 (3 points) or
            4 (5 points). Defaults to 2.

    Returns:
        ndarray: J of shape (m, n), with J[j, i] = ∂f_j/∂x_i
    """
    try:
        stencil = STENCILS[order]
    except KeyError:
        raise ValueError(f"order must be one of {list(STENCILS)}, "
                         f"got {order!r}") from None
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array-like (shape (n,)), "
                         f"instead got shape = {x.shape}")
    n = x.size
    # evaluate once to get the output size
    y0 = np.asarray(f(x), dtype=float)
    if y0.ndim != 1:
        raise ValueError(f"f(x) must return a 1D array-like (shape (m,)), "
                         f"instead got shape = {y0.shape}.")
    m = y0.size
    h = np.finfo(float).eps**(1/(1+order)) * np.maximum(1.0, np.abs(x))
    # make sure (x + h) - x is exactly representable
    h = (x + h) - x
    J = np.zeros((m, n), dtype=float)
    a = (len(stencil) - 1)//2
    for i in range(n):
        for k, c in zip(range(-a, a+1), stencil):
            if c == 0:
                continue
            xh = x.copy()
            xh[i] += k*h[i]
            J[:, i] += c*np.asarray(f(xh), dtype=float)
    J /= h
    return J
