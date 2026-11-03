#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_gen2x2.py
@Time    :   2026/11/03 17:00:00
@Author  :   George Trenins
@Desc    :   Test the general two-variable embedder
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from glefit.embedding import TwoAuxEmbedder


RTOL = 1.0e-12
SIGMA = 5.0  # TwoAuxEmbedder default
TIME = np.array([0.0, 0.3, 1.1, 4.0])
FREQ = np.array([0.0, 0.9, 2.0, 10.0])
# Primitive parameters [r, α, λ, Γ]. Γ > 0 is overdamped and Γ < 0 underdamped;
# TwoAuxEmbedder.to_conventional switches form at |α| = Γ.
CASES = {                              # [  r,       α,   λ,       Γ]
    "overdamped":                        [1.3,     0.4, 0.5,     1.1],
    "overdamped-alpha-zero":             [1.3,     0.0, 0.5,     1.1],
    "overdamped-large-alpha":            [1.3,     1.5, 0.5,     0.8],
    "overdamped-large-negative-alpha":   [1.3,    -1.5, 0.5,     0.8],
    "underdamped":                       [1.3,     0.6, 0.5,    -1.7],
    "underdamped-negative-alpha":        [1.3,    -0.6, 0.5,    -1.7],
    "underdamped-alpha-just-above-zero": [1.3,  1.0e-3, 0.5,    -1.7],
    "underdamped-alpha-just-below-zero": [1.3, -1.0e-3, 0.5,    -1.7],
    "alpha-just-above-Gamma":            [1.3,   1.101, 0.5,     1.1],
    "alpha-just-below-Gamma":            [1.3,   1.099, 0.5,     1.1],
    "Gamma-just-above-zero":             [1.3,     0.6, 0.5,  1.0e-3],
    "Gamma-just-below-zero":             [1.3,     0.6, 0.5, -1.0e-3],
    "beyond-softplus-threshold":         [1.3,     4.0, 0.5,    -4.0],
}


def _softmax(x1, x2):
    return sp.log(sp.exp(SIGMA * x1) + sp.exp(SIGMA * x2)) / SIGMA


def _softabs(x):
    return sp.log(sp.exp(-SIGMA * x) + sp.exp(SIGMA * x)) / SIGMA


@pytest.fixture(scope="module")
def reference(sympy_reference):
    # g = Γ|Γ| and γ = λ + Softmax(Γ, Softabs(α)), as in TwoAuxEmbedder
    r, lamda = sp.symbols("r lamda", positive=True)
    alpha, Gamma = sp.symbols("alpha Gamma", real=True)
    t, omega = sympy_reference.t, sympy_reference.omega
    gamma = lamda + _softmax(Gamma, _softabs(alpha))
    g = Gamma * sp.Abs(Gamma)
    omega_bar = sp.sqrt(-g)  # imaginary for g > 0, turning cos and sin into cosh and sinh
    kernel = r**2 * sp.exp(-gamma * t) * (
        sp.cos(omega_bar * t) - alpha / omega_bar * sp.sin(omega_bar * t))
    spectrum = r**2 * ((gamma + alpha) * omega**2 + (gamma - alpha) * (gamma**2 - g)) / (
        gamma**4 + 2 * gamma**2 * (omega**2 - g) + (omega**2 + g)**2)
    return sympy_reference(kernel, spectrum, (r, alpha, lamda, Gamma))


def embedder_at(primitive_params):
    emb = TwoAuxEmbedder([1.0, 1.0], 3.0, 0.0, 2.0)
    emb.primitive_params = np.array(primitive_params)
    return emb


def assert_close(actual, expected):
    """Within RTOL of the largest |expected|, since K(t) and its derivatives cross zero."""
    assert_allclose(actual, expected, rtol=RTOL, atol=RTOL * np.max(np.abs(expected)))


@pytest.mark.parametrize("nu", [0, 1], ids=["value", "gradient"])
@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_kernel_matches_closed_form_in_docs(reference, params, nu):
    emb = embedder_at(params)

    actual = emb.kernel(TIME, nu=nu)

    assert_close(actual, reference.kernel(params, TIME, nu=nu))


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_spectrum_matches_closed_form_in_docs(reference, params, nu):
    emb = embedder_at(params)

    actual = emb.spectrum(FREQ, nu=nu)

    assert_close(actual, reference.spectrum(params, FREQ, nu=nu))


def test_kernel_hessian_is_not_implemented():
    emb = embedder_at(CASES["underdamped"])

    with pytest.raises(NotImplementedError):
        emb.kernel(TIME, nu=2)


@pytest.mark.parametrize("method, nu", [
    ("kernel", 0), ("kernel", 1), ("spectrum", 0), ("spectrum", 1), ("spectrum", 2),
], ids=["kernel-value", "kernel-gradient", "spectrum-value", "spectrum-gradient", "spectrum-hessian"])
def test_closed_forms_are_even_in_their_argument(method, nu):
    f = getattr(embedder_at(CASES["underdamped"]), method)

    assert_array_equal(f(-TIME, nu=nu), f(TIME, nu=nu))
