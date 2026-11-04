#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   conftest.py
@Time    :   2026/11/03 12:00:00
@Author  :   George Trenins
@Desc    :   Exact reference values for the embedder tests
'''


from __future__ import print_function, division, absolute_import
import mpmath
import numpy as np
import pytest
import sympy as sp


def pytest_configure(config):
    # SymPy 1.14 calls mpmath.libmp.bitcount (sympy/core/evalf.py), deprecated in mpmath 1.4,
    # from assumption checks that run in random order while expressions are built, including
    # in test fixtures. The returned value is unaffected.
    config.addinivalue_line(
        "filterwarnings", "ignore:bitcount function is deprecated:DeprecationWarning")


class SympyReference:
    """Exact memory kernel K(t) and spectrum Λ(ω), differentiated with SymPy and
    evaluated with mpmath at 30 significant digits. Valid for t, ω ≥ 0.
    """

    t, omega = sp.symbols("t omega", nonnegative=True)

    def __init__(self, kernel: sp.Expr, spectrum: sp.Expr, params: tuple) -> None:
        """`kernel` and `spectrum` are expressions in `t` and `omega` respectively,
        and in the symbols `params`."""
        self.params = tuple(params)
        self.kernel_expr = kernel
        self.spectrum_expr = spectrum
        self._compiled = {}

    @classmethod
    def from_drift_matrix(cls, theta: sp.Matrix, A: sp.Matrix, params: tuple) -> "SympyReference":
        """K(t) = θᵀ exp(-tA) θ and Λ(ω) = θᵀ A (A² + ω²)⁻¹ θ of the drift matrix
        Ap = [[0, θᵀ], [-θ, A]]. `theta` (column) and `A` (square) are expressions
        in the symbols `params`."""
        kernel = (theta.T * (-cls.t * A).exp() * theta)[0, 0]
        resolvent = (A**2 + cls.omega**2 * sp.eye(A.rows)).inv()
        spectrum = (theta.T * A * resolvent * theta)[0, 0]
        return cls(kernel, spectrum, params)

    def kernel(self, values, time, nu=0) -> np.ndarray:
        """K(t), or its gradient (nu=1) or Hessian (nu=2) with respect to `params`,
        at `params` = `values`. The grid axis comes last."""
        return self._derivatives("kernel", self.t, values, time, nu)

    def spectrum(self, values, frequency, nu=0) -> np.ndarray:
        """Λ(ω), or its derivatives, as for `kernel`."""
        return self._derivatives("spectrum", self.omega, values, frequency, nu)

    def _derivatives(self, name, var, values, grid, nu):
        if nu not in (0, 1, 2):
            raise ValueError(f"Expecting nu = 0, 1 or 2, instead got {nu = }")
        if (name, nu) not in self._compiled:
            self._compiled[name, nu] = self._compile(getattr(self, f"{name}_expr"), var, nu)
        ans = [self._evaluate(f, values, grid) for f in self._compiled[name, nu]]
        return np.reshape(ans, nu * (len(self.params),) + (len(grid),))

    def _compile(self, expr, var, nu):
        derivatives = [expr]
        for _ in range(nu):
            derivatives = [sp.diff(d, p) for d in derivatives for p in self.params]
        # differentiating |x| gives DiracDelta(x), which vanishes away from x = 0
        derivatives = [d.replace(sp.DiracDelta, lambda *args: sp.S.Zero) for d in derivatives]
        return [sp.lambdify((*self.params, var), d, modules="mpmath") for d in derivatives]

    def _evaluate(self, f, values, grid):
        with mpmath.workdps(30):
            args = [mpmath.mpf(float(v)) for v in values]
            return [self._real(f(*args, mpmath.mpf(float(x)))) for x in grid]

    @staticmethod
    def _real(value):
        """Complex intermediates, such as sqrt(-g) for g > 0, can leave a negligible imaginary part."""
        value = mpmath.mpmathify(value)
        if abs(mpmath.im(value)) > mpmath.mpf(10)**-20 * max(1, abs(value)):
            raise ValueError(f"Reference value {value} is not real")
        return float(mpmath.re(value))


@pytest.fixture(scope="session")
def sympy_reference():
    return SympyReference
