#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_prony.py
@Time    :   2026/11/03 12:00:00
@Author  :   George Trenins
@Desc    :   Test the Prony embedder, K(t) = θ² exp(-γt)
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from glefit.embedding import PronyEmbedder
from glefit.mappers import IdentityMapper


RTOL = 1.0e-12
THETA, GAMMA = 1.3, 0.7
TIME = np.array([0.0, 0.3, 1.7, 5.0])
FREQ = np.array([0.0, 0.9, 2.5, 10.0])


@pytest.fixture
def emb():
    return PronyEmbedder(THETA, GAMMA)


@pytest.fixture
def reference(sympy_reference):
    theta, gamma = sp.symbols("theta gamma", positive=True)
    return sympy_reference.from_drift_matrix(sp.Matrix([theta]), sp.Matrix([[gamma]]), (theta, gamma))


def test_drift_matrix_couples_theta_skew_symmetrically(emb):
    assert_array_equal(emb.drift_matrix, [[0.0, THETA], [-THETA, GAMMA]])


def test_compute_drift_matrix_uses_its_argument_not_stored_params(emb):
    actual = emb.compute_drift_matrix(np.array([3.0, 0.25]))

    assert_array_equal(actual, [[0.0, 3.0], [-3.0, 0.25]])


def test_drift_matrix_param_grad_is_constant(emb):
    expected = [[[0.0, 1.0], [-1.0, 0.0]],   # d/dθ
                [[0.0, 0.0], [0.0, 1.0]]]    # d/dγ

    actual = emb.drift_matrix_param_grad(np.array([3.0, 0.25]))

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
def test_kernel_matches_drift_matrix_definition(emb, reference, nu):
    actual = emb.kernel(TIME, nu=nu)

    assert_allclose(actual, reference.kernel([THETA, GAMMA], TIME, nu=nu), rtol=RTOL)


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
def test_spectrum_matches_drift_matrix_definition(emb, reference, nu):
    actual = emb.spectrum(FREQ, nu=nu)

    assert_allclose(actual, reference.spectrum([THETA, GAMMA], FREQ, nu=nu), rtol=RTOL)


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
@pytest.mark.parametrize("method", ["kernel", "spectrum"])
def test_closed_forms_are_even_in_their_argument(emb, method, nu):
    f = getattr(emb, method)

    assert_array_equal(f(-TIME, nu=nu), f(TIME, nu=nu))


def test_default_mappers_reject_negative_params():
    with pytest.raises(ValueError, match="greater than"):
        PronyEmbedder(-THETA, GAMMA)


def test_non_numeric_param_is_rejected():
    with pytest.raises(TypeError, match="must be a float"):
        PronyEmbedder("1.3", GAMMA)


def test_from_dict_forwards_extra_keys_to_constructor():
    mappers = [IdentityMapper(), IdentityMapper()]

    emb = PronyEmbedder.from_dict({"theta": -THETA, "gamma": GAMMA, "mappers": mappers})

    assert_array_equal(emb.x, [-THETA, GAMMA])


def test_from_dict_leaves_its_input_unchanged():
    params = {"theta": THETA, "gamma": GAMMA}

    PronyEmbedder.from_dict(params)

    assert params == {"theta": THETA, "gamma": GAMMA}
