#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   conftest.py
@Time    :   2026/11/03 12:00:00
@Author  :   George Trenins
@Desc    :   Exact reference values for the embedder tests
'''


from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import warnings
import mpmath
import numpy as np
import pytest
import sympy as sp


@contextmanager
def _ignore_sympy_bitcount_warning():
    """SymPy 1.14 calls mpmath.libmp.bitcount (sympy/core/evalf.py), deprecated in mpmath 1.4,
    from assumption checks that run in random order. The returned value is unaffected."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="bitcount function is deprecated", category=DeprecationWarning)
        yield


class SympyReference:
    """Exact memory kernel and spectrum of the drift matrix Ap = [[0, θᵀ], [-θ, A]].

    K(t) = θᵀ exp(-tA) θ and Λ(ω) = θᵀ A (A² + ω²)⁻¹ θ are built from θ and A,
    differentiated with SymPy and evaluated with mpmath at 30 significant digits.
    Valid for t, ω ≥ 0.
    """

    t, omega = sp.symbols("t omega", nonnegative=True)

    @_ignore_sympy_bitcount_warning()
    def __init__(self, theta: sp.Matrix, A: sp.Matrix, params: tuple) -> None:
        """`theta` (column) and `A` (square) are expressions in the symbols `params`."""
        self.params = tuple(params)
        self.kernel_expr = (theta.T * (-self.t * A).exp() * theta)[0, 0]
        resolvent = (A**2 + self.omega**2 * sp.eye(A.rows)).inv()
        self.spectrum_expr = (theta.T * A * resolvent * theta)[0, 0]

    def kernel(self, values, time, nu=0) -> np.ndarray:
        """K(t), or its gradient (nu=1) or Hessian (nu=2) with respect to `params`,
        at `params` = `values`. The grid axis comes last."""
        return self._derivatives(self.kernel_expr, self.t, values, time, nu)

    def spectrum(self, values, frequency, nu=0) -> np.ndarray:
        """Λ(ω), or its derivatives, as for `kernel`."""
        return self._derivatives(self.spectrum_expr, self.omega, values, frequency, nu)

    @_ignore_sympy_bitcount_warning()
    def _derivatives(self, expr, var, values, grid, nu):
        if nu == 0:
            return self._evaluate(expr, var, values, grid)
        if nu == 1:
            return np.array([
                self._evaluate(sp.diff(expr, p), var, values, grid)
                for p in self.params])
        if nu == 2:
            return np.array([[
                self._evaluate(sp.diff(expr, p, q), var, values, grid)
                for q in self.params] for p in self.params])
        raise ValueError(f"Expecting nu = 0, 1 or 2, instead got {nu = }")

    def _evaluate(self, expr, var, values, grid):
        f = sp.lambdify((*self.params, var), expr, modules="mpmath")
        with mpmath.workdps(30):
            args = [mpmath.mpf(float(v)) for v in values]
            return np.array([float(f(*args, mpmath.mpf(float(g)))) for g in grid])


@pytest.fixture(scope="session")
def sympy_reference():
    return SympyReference
