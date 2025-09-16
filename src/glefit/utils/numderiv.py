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
from typing import Callable, Iterable, Optional, Union


def jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: Iterable[float],
    *,
    h: Optional[Union[float, Iterable[float]]] = None,
    rel_step: Optional[float] = None,
    max_step: Optional[float] = None,
    order: int = 2,
    value: Optional[bool] = False,
    args: tuple = None,
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
        raise ValueError("x must be a 1D array-like (shape (n,)).")
    n = x.size
    # Evaluate once to infer output size and dtype
    y0 = np.asarray(f(x, *args, **kwargs), dtype=float)
    if y0.ndim != 1:
        raise ValueError("f(x) must return a 1D array-like (shape (m,)).")
    m = y0.size
    # Machine epsilon for dtype
    eps = np.finfo(x.dtype).eps
    # Step size selection
    if h is None:
        if rel_step is None:
            rel_step = eps ** (1.0 / 3.0)
        h = rel_step * np.maximum(1.0, np.abs(x))
    else:
        try:
            h = np.asarray(h, dtype=float) * np.ones_like(x)
        except ValueError as e:
            raise e("h must be either a float or a 1D arrays of the same shape as x")
        if h.ndim != 1:
            raise ValueError("If provided, h must have the same shape as x.")
    if max_step is not None:
        max_step = abs(float(max_step))
        h = np.clip(h, a_min=-max_step, a_max=max_step)
    # Ensure perturbations are not swallowed by floating-point granularity
    # Make h at least a couple of units of least precision of x in each direction.
    ulp_plus  = np.nextafter(x,  np.inf) - x
    ulp_minus = x - np.nextafter(x, -np.inf)
    min_h = 2.0 * np.maximum(np.abs(ulp_plus), np.abs(ulp_minus))
    h = np.where(np.abs(h) < min_h, np.sign(h + (h == 0)) * min_h, h)
    J = np.empty((m, n), dtype=float)
    if order == 2:
        for i in range(n):
            hi = h[i]
            xp = x.copy()
            xp[i] = x[i] + hi
            xm = x.copy()
            xm[i] = x[i] - hi
            fp = np.asarray(f(xp, *args, **kwargs), dtype=float)
            fm = np.asarray(f(xm, *args, **kwargs), dtype=float)
            if fp.shape != (m,) or fm.shape != (m,):
                raise ValueError("f(x) must return a 1D array-like of length m.")
            J[:, i] = (fp - fm) / (2.0 * hi)
    elif order == 4:
        for i in range(n):
            hi = h[i]
            xp  = x.copy()
            xp[i]  = x[i] + hi
            xm  = x.copy()
            xm[i]  = x[i] - hi
            xp2 = x.copy()
            xp2[i] = x[i] + 2.0 * hi
            xm2 = x.copy()
            xm2[i] = x[i] - 2.0 * hi
            fp  = np.asarray(f(xp,  *args, **kwargs), dtype=float)
            fm  = np.asarray(f(xm,  *args, **kwargs), dtype=float)
            fp2 = np.asarray(f(xp2, *args, **kwargs), dtype=float)
            fm2 = np.asarray(f(xm2, *args, **kwargs), dtype=float)
            J[:, i] = (-fp2 + 8.0 * fp - 8.0 * fm + fm2) / (12.0 * hi)
    else:
        raise ValueError("order must be 2 or 4.")
    if value:
        return J, y0
    else:
        return J
