#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_special.py
@Time    :   2026/11/04 20:00:00
@Author  :   George Trenins
@Desc    :   Test the special functions
'''


from __future__ import print_function, division, absolute_import
import mpmath
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from glefit.utils.special import sinc


RTOL = 1.0e-13
# away from x = 0, which has its own test; negative x checks the parity
REAL_X = np.array([-2.0, 1.0e-3, 0.7, 3.0, 25.0])
# the gen2x2 kernel evaluates sinc at z = t·sqrt(-g), which is imaginary for g > 0
COMPLEX_X = np.array([1.5j, 30.0j, -2.0j, 1.0 + 2.0j])


def sympy_derivative(nu, x):
    """nu'th derivative of sin(x)/x from SymPy, evaluated with mpmath at 60 significant digits:
    its terms cancel at small x, losing 22 digits for nu = 5 at x = 1e-3."""
    s = sp.symbols("s")
    f = sp.lambdify(s, sp.diff(sp.sin(s) / s, s, nu), modules="mpmath")
    with mpmath.workdps(60):
        return np.array([complex(f(mpmath.mpmathify(complex(xi)))) for xi in x])


@pytest.mark.parametrize("nu", range(6), ids=lambda nu: f"nu={nu}")
def test_sinc_matches_sympy_derivative_for_real_argument(nu):
    expected = sympy_derivative(nu, REAL_X).real

    actual = sinc(REAL_X, nu=nu)

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("nu", range(6), ids=lambda nu: f"nu={nu}")
def test_sinc_matches_sympy_derivative_for_complex_argument(nu):
    expected = sympy_derivative(nu, COMPLEX_X)

    actual = sinc(COMPLEX_X, nu=nu)

    assert_allclose(actual, expected, rtol=RTOL)


def test_sinc_at_zero_gives_the_taylor_coefficients():
    # sin(x)/x = 1 - x²/3! + x⁴/5! - ..., so the nu'th derivative at 0 is
    # (-1)^(nu/2)/(nu+1) for even nu and 0 for odd nu
    expected = [1.0, 0.0, -1.0 / 3.0, 0.0, 1.0 / 5.0, 0.0]

    actual = [sinc(0.0, nu=nu) for nu in range(6)]

    assert_allclose(actual, expected, rtol=RTOL)


def test_sinc_of_integer_array_returns_floats():
    expected = sinc(np.array([0.0, 1.0, 2.0]), nu=1)

    actual = sinc(np.array([0, 1, 2]), nu=1)

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("nu", [-1, 1.5, True], ids=["negative", "non-integer", "bool"])
def test_sinc_rejects_invalid_derivative_order(nu):
    with pytest.raises(ValueError, match="non-negative integer"):
        sinc(1.0, nu=nu)
