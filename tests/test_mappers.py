#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_mappers.py
@Time    :   2026/11/05 20:00:00
@Author  :   George Trenins
@Desc    :   Test the parameter mappers
'''


from __future__ import print_function, division, absolute_import
import pytest
import numpy as np
import sympy as sp
from numpy.testing import assert_allclose
from glefit.mappers import LowerBoundMapper, UpperBoundMapper, IntervalMapper


RTOL = 1.0e-14
X = sp.Symbol("x")
MAPS = [
    (LowerBoundMapper(-2.5), -2.5 + sp.exp(X)),
    (UpperBoundMapper(3.0), 3 - sp.exp(X)),
    (IntervalMapper(-1.0, 4.0), -1 + 5 / (1 + sp.exp(-X))),
]


@pytest.mark.parametrize("mapper, p_of_x", MAPS,
                         ids=["lower-bound", "upper-bound", "interval"])
@pytest.mark.parametrize("nu, method",
                         [(0, "inverse"), (1, "grad"), (2, "hess")],
                         ids=["inverse", "grad", "hess"])
@pytest.mark.parametrize("x", [-3.0, 0.0, 5.0])
def test_inverse_and_its_derivatives_match_sympy(
        mapper, p_of_x, nu, method, x):
    expected = float(sp.diff(p_of_x, X, nu).subs(X, x).evalf(30))

    actual = getattr(mapper, method)(x)

    assert_allclose(actual, expected, rtol=RTOL, atol=0)


@pytest.mark.parametrize("mapper, p, match", [
    (LowerBoundMapper(1.0), 1.0, "greater than"),
    (UpperBoundMapper(1.0), 1.0, "smaller than"),
    (IntervalMapper(0.0, 1.0), 0.0, "greater than"),
    (IntervalMapper(0.0, 1.0), 1.0, "smaller than"),
], ids=["lower-bound", "upper-bound", "interval-lower", "interval-upper"])
def test_forward_rejects_p_on_its_bound(mapper, p, match):
    # the bound itself would map to x = ±inf
    with pytest.raises(ValueError, match=match):
        mapper.forward(p)


@pytest.mark.parametrize("mapper", [
    LowerBoundMapper(-2.5), UpperBoundMapper(3.0), IntervalMapper(-1.0, 4.0),
], ids=["lower-bound", "upper-bound", "interval"])
@pytest.mark.parametrize("p", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "inf", "-inf"])
def test_forward_rejects_non_finite_p(mapper, p):
    with pytest.raises(ValueError, match="must be finite"):
        mapper.forward(p)


# each builds a mapper with `bound` in one place and valid other bounds
BUILDERS = pytest.mark.parametrize("build", [
    lambda bound: LowerBoundMapper(bound),
    lambda bound: UpperBoundMapper(bound),
    lambda bound: IntervalMapper(bound, 10.0),
    lambda bound: IntervalMapper(-10.0, bound),
], ids=["lower-bound", "upper-bound", "interval-a", "interval-b"])


@pytest.mark.parametrize("cls, bounds, expected", [
    (LowerBoundMapper, (np.int64(-2),), -1.0),
    (UpperBoundMapper, (np.float32(3.5),), 2.5),
    (IntervalMapper, (np.float32(-1.0), np.int64(4)), 1.5),
], ids=["lower-bound", "upper-bound", "interval"])
def test_bounds_accept_numpy_scalars(cls, bounds, expected):
    p = cls(*bounds).inverse(0.0)

    assert_allclose(p, expected, rtol=RTOL, atol=0)


@BUILDERS
@pytest.mark.parametrize("bound", [None, "1.0", 1j],
                         ids=["none", "str", "complex"])
def test_bounds_reject_non_numbers(build, bound):
    with pytest.raises(TypeError, match="must be a real number"):
        build(bound)


@BUILDERS
@pytest.mark.parametrize("bound", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "inf", "-inf"])
def test_bounds_reject_non_finite_values(build, bound):
    with pytest.raises(ValueError, match="must be finite"):
        build(bound)


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (1.0, 0.0)],
                         ids=["equal", "reversed"])
def test_interval_rejects_bounds_out_of_order(a, b):
    with pytest.raises(ValueError, match="a < b"):
        IntervalMapper(a, b)
