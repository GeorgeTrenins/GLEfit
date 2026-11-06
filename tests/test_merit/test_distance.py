#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_distance.py
@Time    :   2026/11/06 13:00:00
@Author  :   George Trenins
@Desc    :   Test the merit functions: values, distance and derivatives
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from glefit.embedding import (MultiEmbedder, PronyCosineEmbedder,
                              PronyEmbedder, TwoAuxEmbedder)
from glefit.merit import MemoryKernel, MemorySpectrum


RTOL = 1.0e-12
# the Hessian is a finite difference of the gradient
RTOL_HESS, ATOL_HESS = 1.0e-7, 1.0e-8
TIME = np.array([0.0, 0.4, 1.3, 3.0])
FREQ = np.array([0.0, 0.7, 2.0, 5.0])
# a target that no embedder below matches, and non-uniform weights
TARGET = np.array([0.3, -0.2, 0.1, 0.05])
WEIGHT = np.array([0.5, 2.0, 1.0, 3.0])
EMBEDDERS = {
    "prony": lambda: PronyEmbedder(0.8, 1.3),
    "pronycos": lambda: PronyCosineEmbedder(0.9, 0.7, 2.1),
    "gen2x2": lambda: TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0),
    "multi": lambda: MultiEmbedder([
        PronyEmbedder(0.5, 2.0),
        TwoAuxEmbedder([1.0, -0.5], 1.0, 0.2, 0.7)]),
}
PROPERTIES = {
    "kernel": (MemoryKernel, TIME),
    "spectrum": (MemorySpectrum, FREQ),
}


def trial_x(emb):
    """A point that differs from the embedder's x in every component"""
    return emb.x + 0.1 * (-1.0) ** np.arange(len(emb.x))


@pytest.fixture(params=list(EMBEDDERS))
def emb_name(request):
    return request.param


@pytest.fixture(params=list(PROPERTIES))
def kind(request):
    return request.param


@pytest.fixture
def trial_emb(emb_name):
    """An embedder moved to the trial point"""
    emb = EMBEDDERS[emb_name]()
    emb.x = trial_x(emb)
    return emb


@pytest.fixture
def exact(trial_emb, kind):
    """h, ∂h/∂x and ∂²h/∂x² at the trial point, from the closed forms"""
    f, grid = getattr(trial_emb, kind), PROPERTIES[kind][1]
    return f(grid), f(grid, nu=1, mapped=True), f(grid, nu=2, mapped=True)


@pytest.fixture(params=["moved-embedder", "x-argument"])
def prop_and_x(request, emb_name, kind):
    """A merit function and the x to pass to it, which together evaluate
    at the trial point: either the embedder was moved there and x is
    None, or the embedder was left alone and x is the trial point"""
    prop_class, grid = PROPERTIES[kind]
    emb = EMBEDDERS[emb_name]()
    x = trial_x(emb)
    if request.param == "moved-embedder":
        emb.x, x = x, None
    return prop_class(grid, TARGET, emb, "squared", weight=WEIGHT), x


def expected_distance(h):
    return np.sum(WEIGHT * (h - TARGET)**2)


def expected_gradient(h, g):
    return 2 * g @ (WEIGHT * (h - TARGET))


def expected_hessian(h, g, H):
    return 2 * (g * WEIGHT) @ g.T + 2 * H @ (WEIGHT * (h - TARGET))


def test_function_matches_drift_matrix_reference(
        prop_and_x, trial_emb, kind, drift_matrix_reference):
    prop, x = prop_and_x
    expected, _ = getattr(drift_matrix_reference, kind)(
        trial_emb.A, PROPERTIES[kind][1])

    actual = prop.function(x=x)

    assert_allclose(actual, expected, rtol=RTOL)


def test_distance_is_the_weighted_sum_of_squared_deviations(
        trial_emb, kind, exact):
    prop_class, grid = PROPERTIES[kind]
    prop = prop_class(grid, TARGET, trial_emb, "squared", weight=WEIGHT)

    actual = prop.distance

    assert_allclose(actual, expected_distance(exact[0]), rtol=RTOL)


def test_gradient_matches_closed_form(prop_and_x, exact):
    prop, x = prop_and_x

    actual = prop.gradient(x=x)

    assert_allclose(actual, expected_gradient(*exact[:2]), rtol=RTOL)


def test_both_returns_distance_and_gradient(prop_and_x, exact):
    prop, x = prop_and_x

    distance, gradient = prop.both(x=x)

    assert_allclose(distance, expected_distance(exact[0]), rtol=RTOL)
    assert_allclose(gradient, expected_gradient(*exact[:2]), rtol=RTOL)


def test_hessian_matches_closed_form_to_finite_difference_accuracy(
        prop_and_x, exact):
    prop, x = prop_and_x

    actual = prop.hessian(x=x)

    assert_allclose(actual, expected_hessian(*exact),
                    rtol=RTOL_HESS, atol=ATOL_HESS)


def test_all_returns_distance_gradient_and_hessian(prop_and_x, exact):
    prop, x = prop_and_x

    distance, gradient, hessian = prop.all(x=x)

    assert_allclose(distance, expected_distance(exact[0]), rtol=RTOL)
    assert_allclose(gradient, expected_gradient(*exact[:2]), rtol=RTOL)
    assert_allclose(hessian, expected_hessian(*exact),
                    rtol=RTOL_HESS, atol=ATOL_HESS)


@pytest.mark.parametrize("prop_class, method, grid, scale", [
    (MemoryKernel, "kernel", TIME / 1.0e-3, [1.0, 1.0e-3, 1.0e-3, 1.0]),
    (MemorySpectrum, "spectrum", FREQ * 1.0e-3,
     [np.sqrt(1.0e-3), 1.0e-3, 1.0e-3, 1.0]),
], ids=["kernel", "spectrum"])
def test_gen2x2_hessian_is_accurate_in_a_large_time_unit(
        prop_class, method, grid, scale):
    emb = EMBEDDERS["gen2x2"]()
    emb.primitive_params = emb.primitive_params * scale
    prop = prop_class(grid, TARGET, emb, "squared", weight=WEIGHT)
    f = getattr(emb, method)
    expected = expected_hessian(
        f(grid), f(grid, nu=1, mapped=True), f(grid, nu=2, mapped=True))

    actual = prop.hessian()

    assert_allclose(actual, expected, rtol=RTOL_HESS, atol=ATOL_HESS)


@pytest.mark.parametrize(
    "method", ["function", "gradient", "both", "hessian", "all"])
def test_passing_x_leaves_the_embedder_unchanged(method, kind):
    prop_class, grid = PROPERTIES[kind]
    emb = EMBEDDERS["multi"]()
    x0 = emb.x
    prop = prop_class(grid, TARGET, emb, "squared", weight=WEIGHT)

    getattr(prop, method)(x=trial_x(emb))

    assert_array_equal(emb.x, x0)
