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
TIME = np.array([0.0, 0.3, 1.1, 4.0])
FREQ = np.array([0.0, 0.9, 2.0, 10.0])
# Primitive parameters [r, α, g, y], where g = δ² - ζ² is overdamped for g > 0 and underdamped
# for g < 0, and y sets how far γ lies above its lower bound max(√g, |α|)
CASES = {                              # [  r,     α,       g,    y]
    "overdamped":                        [1.3,   0.4,    1.21, -0.5],
    "overdamped-alpha-zero":             [1.3,   0.0,    1.21, -0.5],
    "overdamped-large-alpha":            [1.3,   1.5,    0.64, -0.5],
    "overdamped-large-negative-alpha":   [1.3,  -1.5,    0.64, -0.5],
    "underdamped":                       [1.3,   0.6,   -2.89, -0.5],
    "underdamped-negative-alpha":        [1.3,  -0.6,   -2.89, -0.5],
    "underdamped-alpha-zero":            [1.3,   0.0,   -2.89, -0.5],
    "critical":                          [1.3,   0.6,     0.0, -0.5],
    "g-tiny-positive":                   [1.3,   0.6,  1.0e-6, -0.5],
    "g-tiny-negative":                   [1.3,   0.6, -1.0e-6, -0.5],
    # a single exponential, and where γ switches formula
    "g-equals-alpha-squared":            [1.3,   0.8,    0.64, -0.5],
    "gamma-just-above-alpha":            [1.3,   0.6,   -2.89, -8.0],
    "gamma-just-above-sqrt-g":           [1.3,   0.4,    1.21, -8.0],
    "large-y":                           [1.3,   0.6,   -2.89,  3.0],
}


@pytest.fixture(scope="module")
def reference(sympy_reference):
    # γ(α, g, y) from the docs. cosh(t√g) and sinhc(t√g) are written as ₀F₁ series in g,
    # which SymPy differentiates and mpmath evaluates for either sign of g and at g = 0
    r = sp.symbols("r", positive=True)
    alpha, g, y = sp.symbols("alpha g y", real=True)
    t, omega = sympy_reference.t, sympy_reference.omega
    gamma = sp.sqrt((g + alpha**2) / 2 + sp.sqrt(((g - alpha**2) / 2)**2 + sp.exp(2 * y)))
    cosh = sp.hyper([], [sp.S.Half], g * t**2 / 4)
    sinhc = sp.hyper([], [sp.Rational(3, 2)], g * t**2 / 4)
    kernel = r**2 * sp.exp(-gamma * t) * (cosh - alpha * t * sinhc)
    spectrum = r**2 * ((gamma + alpha) * omega**2 + (gamma - alpha) * (gamma**2 - g)) / (
        gamma**4 + 2 * gamma**2 * (omega**2 - g) + (omega**2 + g)**2)
    return sympy_reference(kernel, spectrum, (r, alpha, g, y))


def embedder_at(primitive_params):
    emb = TwoAuxEmbedder([1.0, 1.0], 3.0, 0.0, 2.0)
    emb.primitive_params = np.array(primitive_params)
    return emb


def assert_close(actual, expected, rtol=RTOL):
    """Within rtol of the largest |expected| in the same row (along the grid axis),
    since K(t) and its derivatives cross zero."""
    expected = np.asarray(expected)
    scale = np.max(np.abs(expected), axis=-1, keepdims=True)
    scale = np.where(scale > 0.0, scale, 1.0)
    assert_allclose(np.asarray(actual) / scale, expected / scale, rtol=rtol, atol=rtol)


@pytest.mark.parametrize("nu", [0, 1], ids=["value", "gradient"])
@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_kernel_matches_closed_form(reference, params, nu):
    emb = embedder_at(params)

    actual = emb.kernel(TIME, nu=nu)

    assert_close(actual, reference.kernel(params, TIME, nu=nu))


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_spectrum_matches_closed_form(reference, params, nu):
    emb = embedder_at(params)

    actual = emb.spectrum(FREQ, nu=nu)

    assert_close(actual, reference.spectrum(params, FREQ, nu=nu))


@pytest.mark.xfail(raises=NotImplementedError, strict=True,
                   reason="TwoAuxEmbedder.kernel_hess is not implemented")
def test_kernel_hessian_matches_closed_form(reference):
    params = CASES["underdamped"]

    actual = embedder_at(params).kernel(TIME, nu=2)

    assert_close(actual, reference.kernel(params, TIME, nu=2))


@pytest.mark.parametrize("nu", [0, 1], ids=["value", "gradient"])
def test_kernel_at_long_times_near_the_boundary_matches_closed_form(reference, nu):
    # cosh(t√g) alone overflows from t ≈ 645 on, and γ is so close to √g that K has
    # not decayed. Here ∂K/∂g is the difference of terms about 2000 times larger, so
    # the tolerance is 1e-11 rather than RTOL.
    params = CASES["gamma-just-above-sqrt-g"]
    time = np.array([600.0, 650.0, 700.0, 1000.0])

    actual = embedder_at(params).kernel(time, nu=nu)

    assert_close(actual, reference.kernel(params, time, nu=nu), rtol=1.0e-11)


@pytest.mark.parametrize("nu", [0, 1], ids=["value", "gradient"])
def test_kernel_of_integer_times_equals_kernel_of_float_times(nu):
    # g > 0, where the g-derivative fills its result on both sides of √g·t = 1
    emb = embedder_at(CASES["overdamped"])
    expected = emb.kernel(np.array([0.0, 1.0, 2.0, 4.0]), nu=nu)

    actual = emb.kernel(np.array([0, 1, 2, 4]), nu=nu)

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("method, nu", [
    ("kernel", 0), ("kernel", 1), ("spectrum", 0), ("spectrum", 1), ("spectrum", 2),
], ids=["kernel-value", "kernel-gradient", "spectrum-value", "spectrum-gradient", "spectrum-hessian"])
def test_closed_forms_are_even_in_their_argument(method, nu):
    f = getattr(embedder_at(CASES["underdamped"]), method)

    assert_array_equal(f(-TIME, nu=nu), f(TIME, nu=nu))


# The tests via the drift matrix below also check the left column; these add Ap[0, 0] = 0.
def test_drift_matrix_couples_theta_skew_symmetrically():
    Ap = embedder_at(CASES["overdamped"]).drift_matrix

    assert Ap[0, 0] == 0.0
    assert_array_equal(Ap[1:, 0], -Ap[0, 1:])


def test_drift_matrix_param_grad_couples_theta_skew_symmetrically():
    params = CASES["overdamped"]

    grad = embedder_at(params).drift_matrix_param_grad(np.array(params))

    assert_array_equal(grad[:, 0, 0], 0.0)
    assert_array_equal(grad[:, 1:, 0], -grad[:, 0, 1:])


def test_drift_matrix_leaves_s2_undamped_in_the_maximal_gauge():
    # δ = γ gives the diagonal [2γ, 0], so the symmetric part of A is positive semi-definite
    A = embedder_at(CASES["overdamped"]).drift_matrix[1:, 1:]

    assert A[1, 1] == 0.0
    assert A[0, 0] > 0.0


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_kernel_of_drift_matrix_matches_closed_form(reference, drift_matrix_reference, params):
    Ap = embedder_at(params).compute_drift_matrix(np.array(params))

    actual, _ = drift_matrix_reference.kernel(Ap, TIME)

    assert_close(actual, reference.kernel(params, TIME))


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_spectrum_of_drift_matrix_matches_closed_form(reference, drift_matrix_reference, params):
    Ap = embedder_at(params).compute_drift_matrix(np.array(params))

    actual, _ = drift_matrix_reference.spectrum(Ap, FREQ)

    assert_close(actual, reference.spectrum(params, FREQ))


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_kernel_gradient_via_drift_matrix_param_grad_matches_closed_form(
        reference, drift_matrix_reference, params):
    emb = embedder_at(params)
    p = np.array(params)

    _, actual = drift_matrix_reference.kernel(
        emb.compute_drift_matrix(p), TIME, emb.drift_matrix_param_grad(p))

    assert_close(actual, reference.kernel(params, TIME, nu=1))


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_spectrum_gradient_via_drift_matrix_param_grad_matches_closed_form(
        reference, drift_matrix_reference, params):
    emb = embedder_at(params)
    p = np.array(params)

    _, actual = drift_matrix_reference.spectrum(
        emb.compute_drift_matrix(p), FREQ, emb.drift_matrix_param_grad(p))

    assert_close(actual, reference.spectrum(params, FREQ, nu=1))


def test_drift_matrix_x_grad_uses_its_argument_not_stored_params(reference, drift_matrix_reference):
    emb = embedder_at(CASES["overdamped"])
    r, alpha, g, y = params = CASES["underdamped"]
    x = np.array([np.log(r), alpha, g, y])
    # default mappers: dp/dx = [r, 1, 1, 1]
    expected = np.array([r, 1.0, 1.0, 1.0])[:, None] * reference.kernel(params, TIME, nu=1)

    _, actual = drift_matrix_reference.kernel(
        emb.compute_drift_matrix(np.array(params)), TIME, emb.drift_matrix_x_grad(x))

    assert_close(actual, expected)


def test_default_mappers_take_log_of_r_only():
    emb = embedder_at(CASES["underdamped-negative-alpha"])

    assert_allclose(emb.x, [np.log(1.3), -0.6, -2.89, -0.5], rtol=RTOL)


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_to_primitive_inverts_to_conventional(params):
    emb = embedder_at(params)

    actual = emb.to_primitive(emb.to_conventional(np.array(params)))

    # g = δ² - ζ² cancels for |g| ≪ δ², so g comes back to ~ε·δ²/|g| only;
    # compare relative to the largest parameter
    assert_allclose(actual, params, rtol=RTOL, atol=RTOL * np.max(np.abs(params)))


def test_conventional_params_read_back_in_maximal_gauge():
    # θ = [1, 2], γ = 1.5, δ = 0.25, ζ = 3 gives r² = 5, α = -0.15 and g = -8.9375.
    # The maximal gauge has δ = γ, ζ = sqrt(γ² - g) and θ₁,₂ = r·sqrt((1 ± α/γ)/2)
    emb = TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0)

    assert_allclose(
        emb.conventional_params,
        [1.5, np.sqrt(2.75), 1.5, 1.5, np.sqrt(11.1875)],
        rtol=RTOL)


def test_maximal_gauge_keeps_the_kernel(drift_matrix_reference):
    Ap = np.array([[0.0, 1.0, 2.0], [-1.0, 1.75, 3.0], [-2.0, -3.0, 1.25]])
    expected, _ = drift_matrix_reference.kernel(Ap, TIME)

    actual = TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0).kernel(TIME)

    assert_close(actual, expected)


def test_weakly_damped_params_are_accepted_and_keep_the_kernel(drift_matrix_reference):
    # γ = 0.1 is far below the old lower bound Softmax(Γ, Softabs(α)) at σ = 5 (EMB-9)
    Ap = np.array([[0.0, 1.0, 0.0], [-1.0, 0.1, 2.0], [0.0, -2.0, 0.1]])
    expected, _ = drift_matrix_reference.kernel(Ap, TIME)

    actual = TwoAuxEmbedder([1.0, 0.0], 0.1, 0.0, 2.0).kernel(TIME)

    assert_close(actual, expected)


def test_from_matrix_keeps_the_kernel(drift_matrix_reference):
    theta, A = np.array([1.0, 0.5]), np.array([[1.5, 0.8], [0.2, 1.0]])
    expected, _ = drift_matrix_reference.kernel(
        np.block([[0.0, theta], [-theta[:, None], A]]), TIME)

    actual = TwoAuxEmbedder.from_matrix(theta, A).kernel(TIME)

    assert_close(actual, expected)


def test_from_matrix_rejects_indefinite_symmetric_part():
    with pytest.raises(ValueError, match="positive-definite"):
        TwoAuxEmbedder.from_matrix([1.0, 0.5], [[1.0, 0.0], [0.0, -0.5]])


def test_from_harmonic_puts_the_coupling_on_the_first_variable():
    # weight 2π gives θ₁² = 2·weight/π = 4
    actual = TwoAuxEmbedder.from_harmonic(2 * np.pi, 2.0, 0.5)

    assert actual == {
        "theta": [2.0, 0.0],
        "gamma": 0.5,
        "delta": 0.0,
        "zeta": 2.0
    }


def test_theta_must_have_two_components():
    with pytest.raises(ValueError, match="length-2"):
        TwoAuxEmbedder([1.0, 0.0, 0.0], 3.0, 0.0, 2.0)


def test_non_numeric_param_is_rejected():
    with pytest.raises(TypeError, match="must be a float"):
        TwoAuxEmbedder([1.0, 0.0], 3.0, "0.0", 2.0)


def test_inadmissible_params_are_rejected():
    # θ = [1, 0] and δ = 0.5 give α = 0.5 > γ = 0.1
    with pytest.raises(ValueError, match="admissible"):
        TwoAuxEmbedder([1.0, 0.0], 0.1, 0.5, 0.0)


def test_params_on_the_admissible_boundary_are_rejected():
    # γ = |α| = 0.5 needs y = -∞
    with pytest.raises(ValueError, match="admissible"):
        TwoAuxEmbedder([1.0, 0.0], 0.5, 0.5, 1.0)


@pytest.mark.parametrize("key", ["sigma", "threshold"])
def test_softmax_settings_are_rejected(key):
    params = {"theta": [1.0, 0.0], "gamma": 3.0, "delta": 0.5, "zeta": 2.0, key: 50.0}

    with pytest.raises(TypeError, match=f"no longer takes '{key}'"):
        TwoAuxEmbedder.from_dict(params)


def test_softmax_settings_are_rejected_by_the_constructor():
    with pytest.raises(TypeError, match="no longer takes 'sigma'"):
        TwoAuxEmbedder([1.0, 0.0], 3.0, 0.5, 2.0, sigma=50.0)


def test_old_key_omega_is_rejected_with_a_pointer_to_zeta():
    params = {"theta": [1.0, 0.0], "gamma": 3.0, "delta": 0.5, "Omega": 2.0}

    with pytest.raises(TypeError, match="no longer takes 'Omega'.*'zeta'"):
        TwoAuxEmbedder.from_dict(params)


def test_from_dict_leaves_its_input_unchanged():
    params = {
        "theta": [1.0, 0.0],
        "gamma": 3.0,
        "delta": 0.5,
        "zeta": 2.0
    }

    TwoAuxEmbedder.from_dict(params)
    assert params == {
        "theta": [1.0, 0.0],
        "gamma": 3.0,
        "delta": 0.5,
        "zeta": 2.0
    }
