#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_multi.py
@Time    :   2026/11/04 10:00:00
@Author  :   George Trenins
@Desc    :   Test the direct sum of embedders
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from glefit.embedding import (MultiEmbedder, PronyEmbedder, 
                              PronyCosineEmbedder, TwoAuxEmbedder)


RTOL = 1.0e-12
TIME = np.array([0.0, 0.3, 1.1, 4.0])
FREQ = np.array([0.0, 0.9, 2.0, 10.0])
GRIDS = [("kernel", TIME), ("spectrum", FREQ)]
SETTER_CASES = {
    "conventional_params": [
        [2.0, 1.0], [1.0, 0.0, 2.5, 0.3, 1.0], [1.5, 0.5, 3.0]],  
    "primitive_params": [
        [2.0, 1.0], [1.1, 0.3, 0.6, -1.2], [1.5, 0.5, 3.0]],
    "x": [
        [0.1, -0.2], [0.2, 0.4, -0.5, -1.0], [0.3, 0.0, 0.5]],
}


@pytest.fixture
def components():
    # gen2x2 is given in standard form, so its conventional parameters
    # read back unchanged
    return [
        PronyEmbedder(1.3, 0.7),
        TwoAuxEmbedder([1.2, 0.0], 2.0, 0.5, 1.5),
        PronyCosineEmbedder(0.9, 0.4, 2.0),
    ]


@pytest.fixture
def emb(components):
    return MultiEmbedder(components)


def assert_close(actual, expected):
    """Within RTOL of the largest |expected|, since K(t) and its derivatives cross zero."""
    assert_allclose(actual, expected, rtol=RTOL, atol=RTOL * np.max(np.abs(expected)))


def test_sizes_add_up_over_components(emb):
    assert (len(emb), emb.nparam, emb.ndof) == (5, 10, 9)


def test_drift_matrix_places_component_blocks_on_the_diagonal():
    # Ap = [[0, θᵀ], [-θ, A]]
    expected = [[0.0, 1.3, 0.9, 0.0],
                [-1.3, 0.7, 0.0, 0.0],
                [-0.9, 0.0, 0.4, 2.0],
                [0.0, 0.0, -2.0, 0.4]]

    actual = MultiEmbedder([
        PronyEmbedder(1.3, 0.7), 
        PronyCosineEmbedder(0.9, 0.4, 2.0)]).drift_matrix

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("method, grid", GRIDS, ids=["kernel", "spectrum"])
def test_closed_form_matches_the_drift_matrix(emb, drift_matrix_reference, method, grid):
    expected, _ = getattr(drift_matrix_reference, method)(emb.drift_matrix, grid)

    actual = getattr(emb, method)(grid)

    assert_close(actual, expected)


@pytest.mark.parametrize("method, grid", GRIDS, ids=["kernel", "spectrum"])
def test_closed_form_gradient_matches_chain_rule_wrt_drift_matrix(
        emb, drift_matrix_reference, method, grid):
    p = emb.primitive_params
    _, expected = getattr(drift_matrix_reference, method)(
        emb.compute_drift_matrix(p), grid, emb.drift_matrix_param_grad(p))

    actual = getattr(emb, method)(grid, nu=1)

    assert_close(actual, expected)


def test_drift_matrix_and_its_gradient_use_their_argument_not_stored_params(
        emb, drift_matrix_reference):
    trial = MultiEmbedder([
        PronyEmbedder(2.0, 1.0),
        TwoAuxEmbedder([1.0, 0.0], 2.5, 0.3, 1.0),
        PronyCosineEmbedder(1.5, 0.5, 3.0),
    ])
    p = trial.primitive_params

    _, actual = drift_matrix_reference.kernel(
        emb.compute_drift_matrix(p), TIME, emb.drift_matrix_param_grad(p))

    assert_close(actual, trial.kernel(TIME, nu=1))


@pytest.mark.parametrize("method, grid", GRIDS, ids=["kernel", "spectrum"])
def test_mapped_gradient_applies_each_component_mapper_jacobian(emb, method, grid):
    p = emb.primitive_params
    # default mappers: p = exp(x), except α and Γ of gen2x2 (indices 3 and 5), which are unbounded
    dp_dx = np.array([p[0], p[1], p[2], 1.0, p[4], 1.0, p[6], p[7], p[8]])
    expected = dp_dx[:, None] * getattr(emb, method)(grid, nu=1)

    actual = getattr(emb, method)(grid, nu=1, mapped=True)

    assert_close(actual, expected)


def test_spectrum_hessian_is_block_diagonal_in_the_components(emb, components):
    prony, gen2x2, pronycos = components
    expected = np.zeros((9, 9, len(FREQ)))
    expected[0:2, 0:2] = prony.spectrum(FREQ, nu=2)
    expected[2:6, 2:6] = gen2x2.spectrum(FREQ, nu=2)
    expected[6:9, 6:9] = pronycos.spectrum(FREQ, nu=2)

    actual = emb.spectrum(FREQ, nu=2)

    assert_array_equal(actual, expected)


def test_kernel_hessian_is_block_diagonal_in_the_components():
    # gen2x2 has no kernel Hessian
    prony, pronycos = PronyEmbedder(1.3, 0.7), PronyCosineEmbedder(0.9, 0.4, 2.0)
    expected = np.zeros((5, 5, len(TIME)))
    expected[0:2, 0:2] = prony.kernel(TIME, nu=2)
    expected[2:5, 2:5] = pronycos.kernel(TIME, nu=2)

    actual = MultiEmbedder([prony, pronycos]).kernel(TIME, nu=2)

    assert_array_equal(actual, expected)


def test_conventional_params_concatenate_the_components(emb):
    expected = [1.3, 0.7, 1.2, 0.0, 2.0, 0.5, 1.5, 0.9, 0.4, 2.0]

    assert_allclose(emb.conventional_params, expected, rtol=RTOL)


@pytest.mark.parametrize("name, values", SETTER_CASES.items(), ids=list(SETTER_CASES))
def test_param_setter_updates_each_component_by_its_slice(emb, components, name, values):
    setattr(emb, name, np.concatenate(values))

    assert_allclose(getattr(components[0], name), values[0], rtol=RTOL)
    assert_allclose(getattr(components[1], name), values[1], rtol=RTOL)
    assert_allclose(getattr(components[2], name), values[2], rtol=RTOL)


def test_from_dict_builds_the_components_in_order():
    # gen2x2 reads back in standard form
    params = {"embedders": [
        {"type": "prony", "parameters": {"theta": 1.0, "gamma": 0.5}},
        {"type": "gen2x2", "parameters": {
            "theta": [1.0, 2.0], "gamma": 1.5, "delta": 0.25, "Omega": 3.0}},
    ]}

    emb = MultiEmbedder.from_dict(params)

    assert_allclose(
        emb.conventional_params,
        [1.0, 0.5, 0.0, np.sqrt(5.0), 1.5, 0.15, np.sqrt(8.96)],
        rtol=RTOL)


def test_from_dict_passes_keyword_arguments_to_the_components():
    # as in test_gen2x2: at σ = 50, λ = γ - |α| = 2.5 exactly; the default σ = 5 gives less
    params = {"embedders": [
        {"type": "gen2x2", "parameters": {
            "theta": [1.0, 0.0], "gamma": 3.0, "delta": 0.5, "Omega": 2.0, "sigma": 50.0}},
    ]}

    emb = MultiEmbedder.from_dict(params)

    assert emb.primitive_params[2] == 2.5
