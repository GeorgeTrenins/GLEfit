#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_base.py
@Time    :   2026/11/03 12:00:00
@Author  :   George Trenins
@Desc    :   Test the parameter handling and derivative dispatch of
             BaseEmbedder, mostly through PronyEmbedder
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from glefit.embedding import (MultiEmbedder, PronyCosineEmbedder,
                              PronyEmbedder, TwoAuxEmbedder)
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
    return sympy_reference.from_drift_matrix(sp.Matrix([x[0]]), sp.Matrix([[sp.exp(x[1])]]), x)


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


@pytest.mark.parametrize("name, value, primitive", [
    ("conventional_params", [3, 1], [3.0, 1.0]),
    ("primitive_params", [3, 1], [3.0, 1.0]),
    ("x", [1, 0], [np.e, 1.0]),
], ids=PARAM_NAMES)
def test_param_setter_does_not_truncate_integer_input(
        emb, name, value, primitive):
    # default mappers: p = exp(x)
    setattr(emb, name, value)

    assert_allclose(emb.primitive_params, primitive, rtol=RTOL)
    assert_allclose(emb.x, np.log(primitive), rtol=RTOL)


@pytest.mark.parametrize("method", ["jac_px", "hess_px"])
def test_mapper_derivatives_do_not_truncate_integer_input(emb, method):
    # default mappers: p = exp(x), so that dp/dx = d²p/dx² = exp(x)
    actual = getattr(emb, method)(np.array([1, 0]))

    assert_allclose(actual, [np.e, 1.0], rtol=RTOL)


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


@pytest.mark.parametrize("theta, gamma", [(0.0, GAMMA), (THETA, 0.0)],
                         ids=["theta-zero", "gamma-zero"])
def test_parameter_on_its_bound_is_rejected(theta, gamma):
    # the default mappers need θ, γ > 0: zero would give x = -inf
    with pytest.raises(ValueError, match="greater than"):
        PronyEmbedder(theta, gamma)


def test_mapper_count_must_match_ndof():
    with pytest.raises(ValueError, match="number of constraint mappers"):
        PronyEmbedder(THETA, GAMMA, mappers=[IdentityMapper()])


def test_custom_mappers_replace_the_defaults():
    emb = PronyEmbedder(-THETA, GAMMA, mappers=[IdentityMapper(), IdentityMapper()])

    assert_array_equal(emb.x, [-THETA, GAMMA])


def test_grad_A_is_drift_matrix_derivative_wrt_x(emb):
    # Ap = [[0, θ], [-θ, γ]] and p = exp(x), so dAp/dx = p · dAp/dp
    expected = [[[0.0, THETA], [-THETA, 0.0]],
                [[0.0, 0.0], [0.0, GAMMA]]]

    assert_allclose(emb.grad_A, expected, rtol=RTOL)


def test_drift_matrix_x_grad_applies_mapper_jacobian_at_its_argument(emb):
    # dAp/dp is constant for Prony, so only dp/dx depends on the argument;
    # test_gen2x2 checks that dAp/dp is also evaluated at the argument
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


EMBEDDERS = {
    "prony": lambda: PronyEmbedder(THETA, GAMMA),
    "pronycos": lambda: PronyCosineEmbedder(0.9, 0.7, 2.1),
    "gen2x2": lambda: TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0),
    "multi": lambda: MultiEmbedder([
        PronyEmbedder(0.5, 2.0),
        TwoAuxEmbedder([1.0, -0.5], 1.0, 0.2, 0.7)]),
}


@pytest.mark.parametrize("method, argument", [
    ("to_primitive", "conventional_params"),
    ("to_conventional", "primitive_params"),
    ("compute_drift_matrix", "primitive_params"),
], ids=["to_primitive", "to_conventional", "compute_drift_matrix"])
@pytest.mark.parametrize("make", list(EMBEDDERS.values()),
                         ids=list(EMBEDDERS))
def test_list_input_gives_the_same_result_as_an_array(make, method, argument):
    emb = make()
    values = getattr(emb, argument)
    expected = getattr(emb, method)(values)

    actual = getattr(emb, method)(values.tolist())

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ["primitive_params", "conventional_params"])
def test_value_rejected_by_a_mapper_leaves_the_embedder_unchanged(emb, name):
    # a new θ, but γ < 0, which the lower-bound mapper of γ rejects
    p_before, x_before = emb.primitive_params, emb.x

    with pytest.raises(ValueError, match="greater than"):
        setattr(emb, name, [3.0, -0.5])

    assert_array_equal(emb.primitive_params, p_before)
    assert_array_equal(emb.x, x_before)


@pytest.mark.parametrize("make, values, label", [
    (EMBEDDERS["prony"], [2.0, -0.5], "PronyEmbedder parameter γ"),
    (EMBEDDERS["pronycos"], [0.9, 0.7, -2.0],
     "PronyCosineEmbedder parameter ω"),
    (EMBEDDERS["gen2x2"], [1.0, 1.5, 0.0, 0.3], "TwoAuxEmbedder parameter Ω"),
], ids=["prony", "pronycos", "gen2x2"])
def test_mapper_rejection_names_the_parameter(make, values, label):
    emb = make()

    with pytest.raises(ValueError, match=f"{label}: Invalid parameter value"):
        emb.primitive_params = values


# each embedder class with valid constructor arguments
CONSTRUCTORS = {
    "prony": (PronyEmbedder, lambda: (THETA, GAMMA)),
    "pronycos": (PronyCosineEmbedder, lambda: (0.9, 0.7, 2.1)),
    "gen2x2": (TwoAuxEmbedder, lambda: ([1.0, 2.0], 1.5, 0.25, 3.0)),
    "multi": (MultiEmbedder, lambda: ([PronyEmbedder(0.5, 2.0)],)),
}


@pytest.mark.parametrize("cls, args", CONSTRUCTORS.values(),
                         ids=list(CONSTRUCTORS))
def test_constructor_rejects_an_unknown_keyword(cls, args):
    # a misspelling of `mappers`
    with pytest.raises(TypeError, match=r"\['mapper'\]"):
        cls(*args(), mapper=[])


@pytest.mark.parametrize("cls, args", CONSTRUCTORS.values(),
                         ids=list(CONSTRUCTORS))
def test_constructor_rejects_an_extra_positional_argument(cls, args):
    with pytest.raises(TypeError, match="positional"):
        cls(*args(), 1.0)


# NumPy scalars as constructor arguments, and the conventional parameters
# they give. gen2x2 is in the maximal gauge (δ = γ), so it reads back
# unchanged.
NUMPY_SCALARS = {
    "prony-float32": (
        PronyEmbedder, (np.float32(0.5), np.float32(2.0)), [0.5, 2.0]),
    "prony-int64": (
        PronyEmbedder, (np.int64(1), np.int64(3)), [1.0, 3.0]),
    "pronycos-float32": (
        PronyCosineEmbedder,
        (np.float32(0.5), np.float32(0.25), np.float32(2.0)),
        [0.5, 0.25, 2.0]),
    "pronycos-int64": (
        PronyCosineEmbedder, (np.int64(1), np.int64(2), np.int64(3)),
        [1.0, 2.0, 3.0]),
    "gen2x2-float32": (
        TwoAuxEmbedder,
        ([1.0, 2.0], np.float32(2.0), np.float32(2.0), np.float32(1.5)),
        [1.0, 2.0, 2.0, 2.0, 1.5]),
    "gen2x2-int64": (
        TwoAuxEmbedder,
        ([1.0, 2.0], np.int64(2), np.int64(2), np.int64(1)),
        [1.0, 2.0, 2.0, 2.0, 1.0]),
}


@pytest.mark.parametrize("cls, args, expected", NUMPY_SCALARS.values(),
                         ids=list(NUMPY_SCALARS))
def test_numpy_scalar_params_are_accepted(cls, args, expected):
    actual = cls(*args).conventional_params

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("cls, args, _expected", NUMPY_SCALARS.values(),
                         ids=list(NUMPY_SCALARS))
def test_numpy_scalar_params_are_stored_as_double(cls, args, _expected):
    emb = cls(*args)

    dtypes = (emb.primitive_params.dtype, emb.x.dtype)

    assert dtypes == (np.float64, np.float64)
