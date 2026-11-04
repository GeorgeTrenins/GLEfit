#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   conftest.py
@Time    :   2026/11/04 18:00:00
@Author  :   George Trenins
@Desc    :   Exact reference values shared by all test packages
'''


from __future__ import print_function, division, absolute_import
import mpmath
import numpy as np
import pytest


class DriftMatrixReference:
    """K(t) and Λ(ω) of a numerical drift matrix Ap = [[0, θᵀ], [-θ, A]], with exact
    derivatives along given directions, evaluated with mpmath at 30 significant digits."""

    @classmethod
    def kernel(cls, Ap, time, grad_Ap=None):
        """K(t) = θᵀ exp(-tA) θ"""
        return cls._evaluate(
            lambda row, A, col, t: row * mpmath.expm(-t * A) * col, Ap, grad_Ap, time)

    @classmethod
    def spectrum(cls, Ap, frequency, grad_Ap=None):
        """Λ(ω) = θᵀ A (A² + ω²)⁻¹ θ"""
        return cls._evaluate(
            (lambda row, A, col, w:
                row * A * mpmath.inverse(A**2 + w**2 * mpmath.eye(A.rows)) * col),
            Ap, grad_Ap, frequency)

    @staticmethod
    def _dual(X, dX):
        """Block matrix [[X, dX], [0, X]] in mpmath. Such matrices multiply by the product rule,
        and a matrix function f gives f([[X, dX], [0, X]]) = [[f(X), Df(X)[dX]], [0, f(X)]]
        (R. Mathias, SIAM J. Matrix Anal. Appl. 17, 1996)."""
        m, n = X.shape
        ans = mpmath.zeros(2 * m, 2 * n)
        for i in range(m):
            for j in range(n):
                ans[i, j] = ans[m + i, n + j] = mpmath.mpf(float(X[i, j]))
                ans[i, n + j] = mpmath.mpf(float(dX[i, j]))
        return ans

    @classmethod
    def _evaluate(cls, f, Ap, grad_Ap, grid):
        """Evaluate a quantity of the drift matrix on a grid, with its exact
        derivatives along each direction in grad_Ap, at 30 significant digits.

        f(row, A, col, x): the quantity at grid point x in terms of θᵀ, A and θ.
            It receives dual block matrices (see _dual) and returns
            [[F, dF], [0, F]].
        Ap: drift matrix, shape (n+1, n+1). θᵀ is read from the top row and θ
            from the negated left column, as in the dynamics.
        grad_Ap: k directions dAp, e.g. from drift_matrix_param_grad,
            shape (k, n+1, n+1).  None gives the value only.
        grid: times or frequencies, shape (ngrid,).

        Returns the values, shape (ngrid,), and the derivatives,
            shape (k, ngrid), in the layout of kernel(t, nu=1).
        """
        if grad_Ap is None:
            grad_Ap = [np.zeros_like(Ap)]
        if not (np.all(np.isfinite(Ap)) and np.all(np.isfinite(grad_Ap))):
            raise ValueError("Drift matrix or its gradient is not finite; mpmath.expm would not return")
        with mpmath.workdps(30):
            value, grad = [], []
            for dAp in grad_Ap:
                row = cls._dual(Ap[:1, 1:], dAp[:1, 1:])
                col = cls._dual(-Ap[1:, :1], -dAp[1:, :1])
                A = cls._dual(Ap[1:, 1:], dAp[1:, 1:])
                blocks = [f(row, A, col, mpmath.mpf(float(x))) for x in grid]
                value = [float(b[0, 0]) for b in blocks]
                grad.append([float(b[0, 1]) for b in blocks])
        return np.array(value), np.array(grad)


@pytest.fixture(scope="session")
def drift_matrix_reference():
    return DriftMatrixReference
