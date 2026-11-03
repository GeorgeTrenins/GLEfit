#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_pronycos.py
@Time    :   2026/11/03 15:00:00
@Author  :   George Trenins
@Desc    :   Test the oscillatory Prony embedder, K(t) = θ² exp(-γt) cos(ωt)
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from glefit.embedding import PronyCosineEmbedder
from glefit.mappers import IdentityMapper


RTOL = 1.0e-12
THETA, GAMMA, OMEGA = 1.3, 0.7, 2.0
# away from the zeros of cos(ωt) and sin(ωt), where a relative tolerance is meaningless
TIME = np.array([0.0, 0.3, 1.1, 4.0])
# includes the resonance at ω = OMEGA
FREQ = np.array([0.0, 0.9, OMEGA, 10.0])


@pytest.fixture
def emb():
    return PronyCosineEmbedder(THETA, GAMMA, OMEGA)


@pytest.fixture
def reference(sympy_reference):
    # "omega" is the frequency variable of SympyReference
    theta, gamma, Omega = sp.symbols("theta gamma Omega", positive=True)
    A = sp.Matrix([[gamma, Omega], [-Omega, gamma]])
    return sympy_reference.from_drift_matrix(sp.Matrix([theta, 0]), A, (theta, gamma, Omega))


def test_drift_matrix_has_rotating_auxiliary_block(emb):
    expected = [[0.0, THETA, 0.0],
                [-THETA, GAMMA, OMEGA],
                [0.0, -OMEGA, GAMMA]]

    assert_array_equal(emb.drift_matrix, expected)


def test_compute_drift_matrix_uses_its_argument_not_stored_params(emb):
    expected = [[0.0, 3.0, 0.0],
                [-3.0, 0.25, 1.5],
                [0.0, -1.5, 0.25]]

    actual = emb.compute_drift_matrix(np.array([3.0, 0.25, 1.5]))

    assert_array_equal(actual, expected)


def test_drift_matrix_param_grad_is_constant(emb):
    expected = [[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],   # d/dθ
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],    # d/dγ
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]]   # d/dω

    actual = emb.drift_matrix_param_grad(np.array([3.0, 0.25, 1.5]))

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
def test_kernel_matches_drift_matrix_definition(emb, reference, nu):
    actual = emb.kernel(TIME, nu=nu)

    assert_allclose(actual, reference.kernel([THETA, GAMMA, OMEGA], TIME, nu=nu), rtol=RTOL)


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
def test_spectrum_matches_drift_matrix_definition(emb, reference, nu):
    actual = emb.spectrum(FREQ, nu=nu)

    assert_allclose(actual, reference.spectrum([THETA, GAMMA, OMEGA], FREQ, nu=nu), rtol=RTOL)


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
@pytest.mark.parametrize("method", ["kernel", "spectrum"])
def test_closed_forms_are_even_in_their_argument(emb, method, nu):
    f = getattr(emb, method)

    assert_array_equal(f(-TIME, nu=nu), f(TIME, nu=nu))


def test_from_harmonic_gives_theta_squared_of_two_weight_over_pi():
    actual = PronyCosineEmbedder.from_harmonic(np.pi / 2, OMEGA, GAMMA)

    assert actual == {"theta": 1.0, "gamma": GAMMA, "omega": OMEGA}


@pytest.mark.parametrize("params", [
    (-THETA, GAMMA, OMEGA),
    (THETA, -GAMMA, OMEGA),
    (THETA, GAMMA, -OMEGA),
], ids=["theta", "gamma", "omega"])
def test_default_mappers_reject_negative_params(params):
    with pytest.raises(ValueError, match="greater than"):
        PronyCosineEmbedder(*params)


def test_non_numeric_param_is_rejected():
    with pytest.raises(TypeError, match="must be a float"):
        PronyCosineEmbedder(THETA, GAMMA, "2.0")


def test_from_dict_forwards_extra_keys_to_constructor():
    mappers = 3 * [IdentityMapper()]

    emb = PronyCosineEmbedder.from_dict(
        {"theta": THETA, "gamma": GAMMA, "omega": -OMEGA, "mappers": mappers})

    assert_array_equal(emb.x, [THETA, GAMMA, -OMEGA])


def test_from_dict_leaves_its_input_unchanged():
    params = {"theta": THETA, "gamma": GAMMA, "omega": OMEGA}

    PronyCosineEmbedder.from_dict(params)

    assert params == {"theta": THETA, "gamma": GAMMA, "omega": OMEGA}
