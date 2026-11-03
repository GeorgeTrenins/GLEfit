#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_gen2x2.py
@Time    :   2026/11/03 17:00:00
@Author  :   George Trenins
@Desc    :   Test the general two-variable embedder
'''


from __future__ import print_function, division, absolute_import
import mpmath
import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_allclose, assert_array_equal
from glefit.embedding import TwoAuxEmbedder


RTOL = 1.0e-12
SIGMA = 5.0  # TwoAuxEmbedder default
TIME = np.array([0.0, 0.3, 1.1, 4.0])
FREQ = np.array([0.0, 0.9, 2.0, 10.0])
# Primitive parameters [r, α, λ, Γ]. Γ > 0 is overdamped and Γ < 0 underdamped
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


def _dual(X, dX):
    """Block matrix [[X, dX], [0, X]] in mpmath. Such matrices multiply by the product rule,
    and a matrix function f gives f([[X, dX], [0, X]]) = [[f(X), Df(X)[dX]], [0, f(X)]]
    (R. Mathias, SIAM J. Matrix Anal. Appl. 17, 1996)."""
    m, n = X.shape
    ans = mpmath.zeros(2 * m, 2 * n)
    for i in range(m):
        for j in range(n):
            ans[i, j] = ans[m + i, n + j] = mpmath.mpf(float(X[i, j]))
            ans[i, n + j] = mpmath.mpf(float(dX[i, j]))
    return ans


def _via_drift_matrix(f, Ap, grad_Ap, grid):
    """Evaluate a quantity of the drift matrix Ap = [[0, θᵀ], [-θ, A]] 
    on a grid, with its exact derivatives along each direction in grad_Ap, 
    at 30 significant digits.

    f(row, A, col, x): the quantity at grid point x in terms of θᵀ, A and θ. 
        It receives dual block matrices (see _dual) and returns 
        [[F, dF], [0, F]].
    Ap: drift matrix, shape (n+1, n+1). θᵀ is read from the top row and θ
        from the negated left column, as in the dynamics.
    grad_Ap: k directions dAp, e.g. from drift_matrix_param_grad, 
        shape (k, n+1, n+1).  None gives the value only.
    grid: times or frequencies, shape (ngrid,).

    Returns the values, shape (ngrid,), and the derivatives, 
        shape (k, ngrid), in the layout of kernel(t, nu=1).
    """
    if grad_Ap is None:
        grad_Ap = [np.zeros_like(Ap)]
    if not (np.all(np.isfinite(Ap)) and np.all(np.isfinite(grad_Ap))):
        raise ValueError("Drift matrix or its gradient is not finite; mpmath.expm would not return")
    with mpmath.workdps(30):
        value, grad = [], []
        for dAp in grad_Ap:
            row = _dual(Ap[:1, 1:], dAp[:1, 1:])
            col = _dual(-Ap[1:, :1], -dAp[1:, :1])
            A = _dual(Ap[1:, 1:], dAp[1:, 1:])
            blocks = [f(row, A, col, mpmath.mpf(float(x))) for x in grid]
            value = [float(b[0, 0]) for b in blocks]
            grad.append([float(b[0, 1]) for b in blocks])
    return np.array(value), np.array(grad)


def kernel_via_drift_matrix(Ap, grad_Ap=None, time=TIME):
    """K(t) = θᵀ exp(-tA) θ"""
    return _via_drift_matrix(
        lambda row, A, col, t: row * mpmath.expm(-t * A) * col, Ap, grad_Ap, time)


def spectrum_via_drift_matrix(Ap, grad_Ap=None, frequency=FREQ):
    """Λ(ω) = θᵀ A (A² + ω²)⁻¹ θ"""
    return _via_drift_matrix(
        (lambda row, A, col, w: 
            row * A * mpmath.inverse(A**2 + w**2 * mpmath.eye(A.rows)) * col),
        Ap, grad_Ap, frequency)


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


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_drift_matrix_couples_theta_skew_symmetrically(params):
    Ap = embedder_at(params).drift_matrix

    assert Ap[0, 0] == 0.0
    assert_array_equal(Ap[1:, 0], -Ap[0, 1:])


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_drift_matrix_param_grad_couples_theta_skew_symmetrically(params):
    grad = embedder_at(params).drift_matrix_param_grad(np.array(params))

    assert_array_equal(grad[:, 0, 0], 0.0)
    assert_array_equal(grad[:, 1:, 0], -grad[:, 0, 1:])


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_auxiliary_block_has_positive_definite_symmetric_part(params):
    A = embedder_at(params).drift_matrix[1:, 1:]

    assert np.all(np.linalg.eigvalsh(A + A.T) > 0.0)


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_kernel_of_drift_matrix_matches_closed_form(reference, params):
    Ap = embedder_at(params).compute_drift_matrix(np.array(params))

    actual, _ = kernel_via_drift_matrix(Ap)

    assert_close(actual, reference.kernel(params, TIME))


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_spectrum_of_drift_matrix_matches_closed_form(reference, params):
    Ap = embedder_at(params).compute_drift_matrix(np.array(params))

    actual, _ = spectrum_via_drift_matrix(Ap)

    assert_close(actual, reference.spectrum(params, FREQ))


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_kernel_gradient_via_drift_matrix_param_grad_matches_closed_form(reference, params):
    emb = embedder_at(params)
    p = np.array(params)

    _, actual = kernel_via_drift_matrix(emb.compute_drift_matrix(p), emb.drift_matrix_param_grad(p))

    assert_close(actual, reference.kernel(params, TIME, nu=1))


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_spectrum_gradient_via_drift_matrix_param_grad_matches_closed_form(reference, params):
    emb = embedder_at(params)
    p = np.array(params)

    _, actual = spectrum_via_drift_matrix(emb.compute_drift_matrix(p), emb.drift_matrix_param_grad(p))

    assert_close(actual, reference.spectrum(params, FREQ, nu=1))


def test_drift_matrix_x_grad_uses_its_argument_not_stored_params(reference):
    emb = embedder_at(CASES["overdamped"])
    r, alpha, lamda, Gamma = params = CASES["underdamped"]
    x = np.array([np.log(r), alpha, np.log(lamda), Gamma])
    # default mappers: dp/dx = [r, 1, λ, 1]
    expected = np.array([r, 1.0, lamda, 1.0])[:, None] * reference.kernel(params, TIME, nu=1)

    _, actual = kernel_via_drift_matrix(
        emb.compute_drift_matrix(np.array(params)), emb.drift_matrix_x_grad(x))

    assert_close(actual, expected)


def test_default_mappers_bound_r_and_lambda_below_by_zero():
    emb = embedder_at(CASES["underdamped-negative-alpha"])

    assert_allclose(emb.x, [np.log(1.3), -0.6, np.log(0.5), -1.7], rtol=RTOL)


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_to_primitive_inverts_to_conventional(params):
    emb = embedder_at(params)

    actual = emb.to_primitive(emb.to_conventional(np.array(params)))

    assert_close(actual, params)


def test_conventional_params_read_back_in_standard_form():
    # θ = [1, 2], γ = 1.5, δ = 0.25, Ω = 3 gives 
    # r² = 5, α = -0.15 and g = δ² - Ω² < 0,
    # so the standard form has θ = [0, r], δ = |α| and Ω = sqrt(α² - g)
    emb = TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0)

    assert_allclose(
        emb.conventional_params, 
        [0.0, np.sqrt(5.0), 1.5, 0.15, np.sqrt(8.96)], 
        rtol=RTOL)


def test_standard_form_keeps_the_kernel():
    Ap = np.array([[0.0, 1.0, 2.0], [-1.0, 1.75, 3.0], [-2.0, -3.0, 1.25]])
    expected, _ = kernel_via_drift_matrix(Ap)

    actual = TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0).kernel(TIME)

    assert_close(actual, expected)


def test_from_matrix_keeps_the_kernel():
    theta, A = np.array([1.0, 0.5]), np.array([[1.5, 0.8], [0.2, 1.0]])
    expected, _ = kernel_via_drift_matrix(np.block([[0.0, theta], [-theta[:, None], A]]))

    actual = TwoAuxEmbedder.from_matrix(theta, A).kernel(TIME)

    assert_close(actual, expected)


def test_from_matrix_rejects_indefinite_symmetric_part():
    with pytest.raises(ValueError, match="positive-definite"):
        TwoAuxEmbedder.from_matrix([1.0, 0.5], [[1.0, 0.0], [0.0, -0.5]])


def test_from_harmonic_puts_the_coupling_on_the_first_variable():
    actual = TwoAuxEmbedder.from_harmonic(np.pi / 2, 2.0, 0.5)

    assert actual == {
        "theta": [1.0, 0.0], 
        "gamma": 0.5, 
        "delta": 0.0, 
        "Omega": 2.0
    }


def test_theta_must_have_two_components():
    with pytest.raises(ValueError, match="length-2"):
        TwoAuxEmbedder([1.0, 0.0, 0.0], 3.0, 0.0, 2.0)


def test_non_numeric_param_is_rejected():
    with pytest.raises(TypeError, match="must be a float"):
        TwoAuxEmbedder([1.0, 0.0], 3.0, "0.0", 2.0)


def test_unstable_params_are_rejected():
    # γ = 0.1 < |δ| = 0.5
    with pytest.raises(ValueError, match="greater than"):
        TwoAuxEmbedder([1.0, 0.0], 0.1, 0.5, 0.0)


def test_from_dict_forwards_sigma():
    # θ = [1, 0] gives α = δ = 0.5. At σ = 50, 
    # Softmax and Softabs are exactly max and abs
    # here, so λ = γ - |α| exactly; 
    # the default σ = 5 would give a smaller λ.
    params = {
        "theta": [1.0, 0.0], 
        "gamma": 3.0, 
        "delta": 0.5, 
        "Omega": 2.0, 
        "sigma": 50.0
    }

    emb = TwoAuxEmbedder.from_dict(params)
    assert emb.primitive_params[2] == 2.5


def test_from_dict_leaves_its_input_unchanged():
    params = {
        "theta": [1.0, 0.0], 
        "gamma": 3.0, 
        "delta": 0.5, 
        "Omega": 2.0
    }

    TwoAuxEmbedder.from_dict(params)
    assert params == {
        "theta": [1.0, 0.0], 
        "gamma": 3.0, 
        "delta": 0.5,
        "Omega": 2.0
    }
