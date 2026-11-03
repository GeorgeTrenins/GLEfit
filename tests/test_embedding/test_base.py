#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_base.py
@Time    :   2026/11/03 12:00:00
@Author  :   George Trenins
@Desc    :   Test the parameter handling and derivative dispatch of BaseEmbedder, through PronyEmbedder
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from glefit.embedding import PronyEmbedder
from glefit.mappers import IdentityMapper, LowerBoundMapper


RTOL = 1.0e-12 
THETA, GAMMA = 2.0, 0.5
TIME = np.array([0.0, 0.3, 1.7, 5.0])
FREQ = np.array([0.0, 0.9, 2.5, 10.0])
PARAM_NAMES = ["conventional_params", "primitive_params", "x"]


@pytest.fixture
def emb():
    return PronyEmbedder(THETA, GAMMA)


@pytest.fixture
def mixed_emb():
    """θ unbounded and γ > 0, so that dp/dx and d²p/dx² differ between parameters."""
    return PronyEmbedder(THETA, GAMMA, mappers=[IdentityMapper(), LowerBoundMapper()])


@pytest.fixture
def mapped_reference(sympy_reference):
    # mixed_emb: θ = x₀, γ = exp(x₁)
    x = sp.symbols("x0 x1", real=True)
    return sympy_reference(sp.Matrix([x[0]]), sp.Matrix([[sp.exp(x[1])]]), x)


def test_conventional_params_setter_updates_primitive_params_and_x(emb):
    emb.conventional_params = [3.0, 0.25]

    assert_array_equal(emb.primitive_params, [3.0, 0.25])
    assert_allclose(emb.x, np.log([3.0, 0.25]), rtol=RTOL)


def test_primitive_params_setter_updates_x(emb):
    emb.primitive_params = [3.0, 0.25]

    assert_allclose(emb.x, np.log([3.0, 0.25]), rtol=RTOL)


def test_x_setter_updates_primitive_and_conventional_params(emb):
    emb.x = np.log([3.0, 0.25])

    assert_allclose(emb.primitive_params, [3.0, 0.25], rtol=RTOL)
    assert_allclose(emb.conventional_params, [3.0, 0.25], rtol=RTOL)


@pytest.mark.parametrize("name", PARAM_NAMES)
def test_param_setter_rejects_wrong_shape(emb, name):
    with pytest.raises(ValueError, match="must have shape"):
        setattr(emb, name, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("name, expected", [
    ("conventional_params", [THETA, GAMMA]),
    ("primitive_params", [THETA, GAMMA]),
    ("x", np.log([THETA, GAMMA])),
], ids=PARAM_NAMES)
def test_param_getter_returns_a_copy(emb, name, expected):
    getattr(emb, name)[:] = 99.0

    assert_allclose(getattr(emb, name), expected, rtol=RTOL)


def test_mapper_count_must_match_ndof():
    with pytest.raises(ValueError, match="number of constraint mappers"):
        PronyEmbedder(THETA, GAMMA, mappers=[IdentityMapper()])


def test_identity_mappers_make_x_equal_to_primitive_params():
    emb = PronyEmbedder(-THETA, GAMMA, mappers=[IdentityMapper(), IdentityMapper()])

    assert_array_equal(emb.x, [-THETA, GAMMA])


def test_grad_A_is_drift_matrix_derivative_wrt_x(emb):
    # Ap = [[0, θ], [-θ, γ]] and p = exp(x), so dAp/dx = p · dAp/dp
    expected = [[[0.0, THETA], [-THETA, 0.0]],
                [[0.0, 0.0], [0.0, GAMMA]]]

    assert_allclose(emb.grad_A, expected, rtol=RTOL)


def test_drift_matrix_x_grad_uses_its_argument_not_stored_params(emb):
    expected = [[[0.0, 3.0], [-3.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.25]]]

    actual = emb.drift_matrix_x_grad(np.log([3.0, 0.25]))

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("nu", [1, 2], ids=["gradient", "hessian"])
def test_kernel_derivatives_wrt_x_match_sympy(mixed_emb, mapped_reference, nu):
    actual = mixed_emb.kernel(TIME, nu=nu, mapped=True)

    assert_allclose(actual, mapped_reference.kernel(mixed_emb.x, TIME, nu=nu), rtol=RTOL)


@pytest.mark.parametrize("nu", [1, 2], ids=["gradient", "hessian"])
def test_spectrum_derivatives_wrt_x_match_sympy(mixed_emb, mapped_reference, nu):
    actual = mixed_emb.spectrum(FREQ, nu=nu, mapped=True)

    assert_allclose(actual, mapped_reference.spectrum(mixed_emb.x, FREQ, nu=nu), rtol=RTOL)


@pytest.mark.parametrize("method", ["kernel", "spectrum"])
@pytest.mark.parametrize("nu, error", [(3, ValueError), (1.0, TypeError)], ids=["nu=3", "float-nu"])
def test_invalid_nu_is_rejected(emb, method, nu, error):
    with pytest.raises(error, match="nu"):
        getattr(emb, method)(TIME, nu=nu)


@pytest.mark.parametrize("method", ["kernel", "spectrum"])
def test_two_dimensional_grid_is_rejected(emb, method):
    with pytest.raises(ValueError, match="1D array"):
        getattr(emb, method)(np.ones((2, 2)))


@pytest.mark.parametrize("method", ["kernel", "spectrum"])
@pytest.mark.parametrize("nu, shape", [(0, (1,)), (1, (2, 1)), (2, (2, 2, 1))],
                         ids=["value", "gradient", "hessian"])
def test_scalar_grid_point_keeps_a_length_one_grid_axis(emb, method, nu, shape):
    assert getattr(emb, method)(1.0, nu=nu).shape == shape


def test_spectral_density_is_frequency_times_spectrum(emb):
    # Λ(2) = θ²γ / (γ² + 4) = 2 / 4.25
    assert_allclose(emb.spectral_density(2.0), [2.0 * 2.0 / 4.25], rtol=RTOL)
