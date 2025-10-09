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
from typing import Callable, Iterable, Optional, Union, Tuple


STENCILS = {
    1 : # first-order derivative
    {
        2 : [ -1/2, 0, 1/2 ],  
        4 : [ 1/12, -2/3, 0, 2/3, -1/12 ]
    },
    2 : # second-order derivatives
    {
        2 : [ 1, -2,  1],
        4 : [-1/12, 4/3, -5/2, 4/3, -1/12],
        6 : [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
    }
}


def _central_finite_difference_hrel(x:float, nu: int, order: int) -> float:
    """Return the optimal relative finite-difference step according to the generalization of the prescription in 
    Numerical Recipes (3rd edition), Chapter 5.7

    Args:
        x (float): function argument - used to determine the correct machine epsilon
        nu (int): order of the derivative
        order (int): order of the finite difference stencil

    """

    eps = np.finfo(np.asarray(x, dtype=float).dtype).eps
    hrel = eps**(1/(nu+order))
    return hrel

def refine_step_nearest(x: float, h: float) -> float:
    """
    Return a refined step `href` so that y = round_to_float(x + h) and href = y - x is exactly representable.
    """
    
    temp = x + h
    return temp - x


def diff(
    f: Callable[[float], float],
    x: float,
    *,
    h: Optional[float] = None,
    rel_step: Optional[float] = None,
    max_step: Optional[float] = None,
    order: Optional[int] = 2, 
    nu: Optional[int] = 1,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None
) -> float:
    """
    Compute the derivative of scalar function f: R -> R at point x
    using central finite differences.

    Parameters
    ----------
    f : callable
        Function mapping R -> R. Signature: f(x, *args, **kwargs) and returns a float
    x : float
        Point at which to compute the derivative.
    h : float, optional
        Absolute step size. If None, chosen automatically.
    rel_step : float, optional
        Relative step factor. If provided and `h` is None, the step is h = rel_step * max(1, |x_i|). Default is chosen as eps^(1/(nu+order)) where eps is the fractional accuracy of the float type. 
    max_step : float, optional
        Upper bound for any |h_i|. No upper bound applied by default
    order : {2, 4}, default=2
        Stencil order. 2 -> 3-point central (O(h^2)); 4 -> 5-point central (O(h^4)).
    nu : {1, 2}, default=1
        Order of derivative.
    args, kwargs : extra arguments passed to f.

    Returns
    -------
    deriv : float
        nu-th order derivative of f(x) 

    Notes
    -----
    - Ensures x ± h are representable (avoids zero-perturbation due to limited precision).
    """
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = {}
    if not isinstance(x, float):
        raise TypeError(f"`x` is expected to be a float, instead got {type(x).__name__}")
    if h is None:
        if rel_step is None:
            rel_step = _central_finite_difference_hrel(x, nu, order) 
        h = rel_step * np.maximum(1.0, np.abs(x)) 
    else:
        h = abs(float(h))
    if max_step is not None:
        max_step = abs(float(max_step))
        h = np.clip(h, a_max=max_step)
    # ensure (x+h) - x is exactly representable
    h = refine_step_nearest(x, h)
    try:
        derivative_order_nu = STENCILS[nu]
    except KeyError as e:
        raise e(f"Requested derivative order {nu = }, currently only supporting {list(STENCILS.keys())}")
    try:
        stencil = derivative_order_nu[order]
    except KeyError as e:
        raise e(f"Requested stencil order = {order} for derivative order {nu = }, currently only supporting {list(derivative_order_nu.keys())}")
    ans = 0.0
    # stencil amplitude
    a = (len(stencil) - 1)//2
    for k, c in zip(range(-a, a+1), stencil):
        if c == 0: continue
        y = x + k*h
        ans += c*f(y)
    ans /= h**nu
    return ans

def jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: Iterable[float],
    *,
    fshape: Optional[Tuple[int, ...]] = None,
    h: Optional[Union[float, Iterable[float]]] = None,
    rel_step: Optional[float] = None,
    max_step: Optional[float] = None,
    order: Optional[int] = 2,
    value: Optional[bool] = False,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Compute the Jacobian J of a vector-valued function f: R^n -> R^m at point x
    using central finite differences.

    Parameters
    ----------
    f : callable
        Function mapping R^n -> R^m. Signature: f(x, *args, **kwargs) and returns 1D array-like of length m.
    x : array_like, shape (n,)
        Point at which to compute the Jacobian.
    fshape: tuple of int, optional
        Shape of the output of f(x). If absent, one call is made to f() before the finite-difference calculation to determine the shape.
    h : float or array_like, shape (n,), optional
        Absolute step size(s) per coordinate. If None, chosen automatically.
    rel_step : float, optional
        Relative step factor. If provided and `h` is None, steps are:
        h_i = rel_step * max(1, |x_i|). Default is eps**(1/3) for central differences.
    max_step : float, optional
        Upper bound for any |h_i|.
    order : {2, 4}, default=2
        Stencil order. 2 -> 3-point central (O(h^2)); 4 -> 5-point central (O(h^4)).
    value: bool, optional
        If true, in addition to the jacobian return f(x) as the second output
    args, kwargs : extra arguments passed to f.

    Returns
    -------
    J : ndarray, shape (m, n)
        Jacobian matrix where J[j, i] = df_j/dx_i at x.

    Notes
    -----
    - Uses per-coordinate step sizes scaled to x to balance truncation vs rounding error.
    - Ensures x ± h_i are representable (avoids zero-perturbation due to limited precision).
    """
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = {}
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array-like (shape (n,)), instead got shape = {x.shape}")
    n = x.size
    # Evaluate once to infer output size and dtype
    if fshape is None or value:
        y0 = np.asarray(f(x, *args, **kwargs), dtype=float)
        if y0.ndim != 1:
            raise ValueError(f"f(x) must return a 1D array-like (shape (m,)), instead got shape = {y0.shape}.")
        m = y0.size
    else:
        if len(fshape) != 1:
            raise ValueError(f"f(x) must return a 1D array-like (shape (m,)), instead got shape = {fshape}.")
        m = fshape[0]
    if h is None:
        # optimal size for 2nd order central difference
        if rel_step is None:
            # estimate of fractional error in evaluating f()
            rel_step = _central_finite_difference_hrel(x, 1, order)
        h = rel_step * np.maximum(1.0, np.abs(x))
    else:
        try:
            h = np.abs(h) * np.ones_like(x)
        except ValueError as e:
            raise e("h must be either a float or a 1D arrays of the same shape as x")
        if h.ndim != 1:
            raise ValueError("If provided, h must have the same shape as x.")
    if max_step is not None:
        max_step = abs(float(max_step))
        h = np.clip(h, a_max=max_step)
    # make sure (x + h) - x is exactly representable
    h = refine_step_nearest(x, h)
    J = np.zeros((m, n), dtype=float)
    # get the stencil for a first-order derivative
    try:
        stencil = STENCILS[1][order]
    except KeyError as e:
        raise e(f"Requested stencil order = {order} for derivative order nu = 1, currently only supporting {list(STENCILS[1].keys())}")
    a = (len(stencil) - 1)//2
    for i in range(n):
        hi = h[i]
        for k, c in zip(range(-a, a+1), stencil):
            if c == 0:
                continue
            else:
                xh = x.copy()
                xh[i] += k*hi
                J[:, i] += c*np.asarray(f(xh, *args, **kwargs), dtype=float)
    J /= h
    if value:
        return J, y0
    else:
        return J
