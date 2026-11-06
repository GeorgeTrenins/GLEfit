#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_numderiv.py
@Time    :   2026/11/06 11:05:00
@Author  :   George Trenins
@Desc    :   Test the finite-difference Jacobian
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from glefit.utils.numderiv import jacobian


EPS = np.finfo(float).eps
# exact result of each stencil for sin(ωx): ω cos(ωx) times R(ωh)
RESPONSE = {
    2: lambda a: np.sin(a) / a,
    4: lambda a: (8 * np.sin(a) - np.sin(2 * a)) / (6 * a),
}
RTOL=1e-9
ATOL=1e-9

def powers(v, n):
    """f: R² → R³, a polynomial of degree n in each argument"""
    return np.array([v[0]**n, v[0] * v[1]**n, v[1]])


@pytest.mark.parametrize("order, expected", [
    # at v = (1.5, -0.5): [[2 v₀, 0], [v₁², 2 v₀ v₁], [0, 1]]
    (2, [[3.0, 0.0], [0.25, -1.5], [0.0, 1.0]]),
    # [[4 v₀³, 0], [v₁⁴, 4 v₀ v₁³], [0, 1]]
    (4, [[13.5, 0.0], [0.0625, -0.75], [0.0, 1.0]]),
], ids=["order-2", "order-4"])
def test_jacobian_is_exact_for_polynomials_up_to_the_stencil_order(
        order, expected):
    # away from v = 0, every power of the step contributes, so this
    # checks every stencil weight
    actual = jacobian(lambda v: powers(v, order), [1.5, -0.5], order=order)

    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("order, x, omega, h", [
    (2, 1.0e-3, 1.0e4, EPS ** (1 / 3)),
    (2, 4.0, 2.5e3, 4.0 * EPS ** (1 / 3)),
    (4, 1.0e-3, 100.0, EPS ** (1 / 5)),
    (4, 4.0, 25.0, 4.0 * EPS ** (1 / 5)),
], ids=["order-2-small-x", "order-2-large-x",
        "order-4-small-x", "order-4-large-x"])
def test_jacobian_default_step(
        order, x, omega, h):
    # with ωh ≈ 0.07, R(ωh) differs from 1 by (ωh)²/6 or (ωh)⁴/30,
    # far more than the rounding error, so the result shows the step
    expected = [[omega * np.cos(omega * x) * RESPONSE[order](omega * h)]]

    actual = jacobian(lambda v: np.sin(omega * v), [x], order=order)

    assert_allclose(actual, expected, rtol=RTOL)


def test_jacobian_leaves_x_unchanged():
    x = np.array([1.5, -0.5])

    jacobian(lambda v: v**2, x, order=4)

    assert_array_equal(x, [1.5, -0.5])


def test_jacobian_rejects_x_that_is_not_one_dimensional():
    with pytest.raises(ValueError, match="x must be a 1D"):
        jacobian(lambda v: v, [[1.0]])


def test_jacobian_rejects_f_that_does_not_return_a_one_dimensional_array():
    with pytest.raises(ValueError, match=r"f\(x\) must return a 1D"):
        jacobian(lambda v: np.atleast_2d(v), [1.0])


@pytest.mark.parametrize("order", [3, 6])
def test_jacobian_rejects_unsupported_order(order):
    with pytest.raises(ValueError, match="order must be one of"):
        jacobian(lambda v: v, [1.0], order=order)
