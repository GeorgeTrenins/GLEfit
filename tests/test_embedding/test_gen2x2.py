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
# Primitive parameters [r, γ, Ω, φ]. G = γ² - Ω² > 0 is overdamped and G < 0
# underdamped. For G > 0 the kernel changes form at √G·t = 1, which TIME
# crosses in the overdamped cases.
CASES = {                         #  r,    γ,    Ω,                  φ
    "overdamped":                  [ 1.3,  1.0,  0.8,                0.4],
    "underdamped":                 [ 1.3,  0.5,  1.8,                0.4],
    "critical":                    [ 1.3,  1.2,  1.2,                0.4],
    "G-tiny-positive":             [ 1.3,  1.2,  1.2 * (1 - 1.0e-6), 0.4],
    "G-tiny-negative":             [ 1.3,  1.2,  1.2 * (1 + 1.0e-6), 0.4],
    # zero static friction, shifted Drude-Lorentz and Brownian oscillator
    "phi-zero":                    [ 1.3,  1.0,  0.8,                0.0],
    "phi-quarter-pi":              [ 1.3,  0.5,  1.8,                np.pi/4],
    "phi-half-pi":                 [ 1.3,  1.0,  0.8,                np.pi/2],
    # Debye limit Ω = γ|sin 2φ|: one of the two exponentials vanishes
    "debye-below-quarter-pi":      [ 1.3,  1.0,  abs(np.sin(0.6)),   0.3],
    "debye-above-quarter-pi":      [ 1.3,  1.0,  abs(np.sin(2.2)),   1.1],
    # negative r and Ω, and φ outside [0, π/2]: K depends on r², Ω² and
    # cos 2φ only, so any real r, Ω and φ are valid
    "negative-overdamped":         [-1.3,  1.0, -0.8,                2.0],
    "negative-underdamped":        [-1.3,  0.5, -1.8,               -0.7],
}
CANONICAL_CASES = {name: params for name, params in CASES.items()
                   if not name.startswith("negative")}
# At Ω = 0, Λ(0) is 0/0, so this case is tested on the kernel and at ω > 0
OMEGA_ZERO = [1.3, 1.0, 0.0, 0.4]
KERNEL_CASES = {**CASES, "Omega-zero": OMEGA_ZERO}


@pytest.fixture(scope="module")
def reference(sympy_reference):
    # cosh(t√G) and sinhc(t√G) are written as ₀F₁ series in G = γ² - Ω², 
    # which SymPy differentiates and mpmath evaluates for either sign of G
    r, gamma, Omega, phi = sp.symbols("r gamma Omega phi", real=True)
    t, omega = sympy_reference.t, sympy_reference.omega
    G = gamma**2 - Omega**2
    cosh = sp.hyper([], [sp.S.Half], G * t**2 / 4)
    sinhc = sp.hyper([], [sp.Rational(3, 2)], G * t**2 / 4)
    kernel = r**2 * sp.exp(-gamma * t) * (
        cosh - gamma * t * sp.cos(2 * phi) * sinhc)
    spectrum = 2 * gamma * r**2 * (
        omega**2 * sp.cos(phi)**2 + Omega**2 * sp.sin(phi)**2) / (
        (omega**2 - Omega**2)**2 + 4 * gamma**2 * omega**2)
    return sympy_reference(kernel, spectrum, (r, gamma, Omega, phi))


def embedder_at(primitive_params):
    emb = TwoAuxEmbedder([1.0, 1.0], 3.0, 0.0, 2.0)
    emb.primitive_params = np.array(primitive_params)
    return emb


def assert_close(actual, expected, rtol=RTOL):
    """Within rtol of the largest |expected| in the same row 
    (along the grid axis), since K(t) and its derivatives cross zero."""
    expected = np.asarray(expected)
    scale = np.max(np.abs(expected), axis=-1, keepdims=True)
    scale = np.where(scale > 0.0, scale, 1.0)
    assert_allclose(np.asarray(actual) / scale, expected / scale,
                    rtol=rtol, atol=rtol)


@pytest.mark.parametrize("nu", [0, 1], ids=["value", "gradient"])
@pytest.mark.parametrize("params", list(KERNEL_CASES.values()),
                         ids=list(KERNEL_CASES))
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


@pytest.mark.parametrize("nu", [0, 1, 2], ids=["value", "gradient", "hessian"])
def test_spectrum_at_omega_zero_matches_closed_form_away_from_zero_frequency(
        reference, nu):
    emb = embedder_at(OMEGA_ZERO)

    actual = emb.spectrum(FREQ[1:], nu=nu)

    assert_close(actual, reference.spectrum(OMEGA_ZERO, FREQ[1:], nu=nu))


@pytest.mark.xfail(raises=NotImplementedError, strict=True,
                   reason="TwoAuxEmbedder.kernel_hess is not implemented")
def test_kernel_hessian_matches_closed_form(reference):
    params = CASES["underdamped"]

    actual = embedder_at(params).kernel(TIME, nu=2)

    assert_close(actual, reference.kernel(params, TIME, nu=2))


@pytest.mark.parametrize("nu", [0, 1], ids=["value", "gradient"])
def test_kernel_at_long_times_with_omega_much_below_gamma_matches_closed_form(
        reference, nu):
    # The slow rate Ω²/(γ + √G) ≈ 7.5e-7 keeps K alive while √G·t reaches
    # 1.5e6. Written as γ - √G, it would lose about six digits, and through
    # G = γ² - Ω², ∂K/∂γ would be a difference of terms about (γ/Ω)² = 1e6
    # times larger.
    params = [1.3, 1.5, 0.0015, 0.4]
    time = np.array([10.0, 1.0e3, 1.0e5, 1.0e6])

    actual = embedder_at(params).kernel(time, nu=nu)

    assert_close(actual, reference.kernel(params, time, nu=nu))


@pytest.mark.parametrize("phi", [0.3, 1.1],
                         ids=["phi-below-quarter-pi", "phi-above-quarter-pi"])
def test_kernel_in_the_debye_limit_is_a_single_exponential(phi):
    # Ω = γ|sin 2φ| gives K(t) = r² exp(-λ|t|) with λ = 2γ cos²φ. Below π/4
    # the slower exponential vanishes, so at t = 30 any rounding left in its
    # amplitude would dominate: at φ = 0.3, (1 - γ|cos 2φ|/√G)/2 gives -1e-16
    # instead of 0.
    r, gamma = 1.3, 1.0
    lamda = 2 * gamma * np.cos(phi)**2
    time = np.array([0.0, 0.3, 1.1, 4.0, 30.0])
    emb = embedder_at([r, gamma, gamma * abs(np.sin(2 * phi)), phi])

    actual = emb.kernel(time)

    assert_allclose(actual, r**2 * np.exp(-lamda * time), rtol=RTOL)


@pytest.mark.parametrize("phi", [0.3, 1.1],
                         ids=["phi-below-quarter-pi", "phi-above-quarter-pi"])
def test_spectrum_in_the_debye_limit_is_a_single_lorentzian(phi):
    # Λ(ω) = r² λ/(λ² + ω²), the cosine transform of r² exp(-λ|t|)
    r, gamma = 1.3, 1.0
    lamda = 2 * gamma * np.cos(phi)**2
    emb = embedder_at([r, gamma, gamma * abs(np.sin(2 * phi)), phi])

    actual = emb.spectrum(FREQ)

    assert_allclose(actual, r**2 * lamda / (lamda**2 + FREQ**2), rtol=RTOL)


def test_kernel_at_omega_zero_tends_to_a_constant():
    # K(t) = r²[sin²φ + cos²φ exp(-2γ|t|)]: s₂ is undamped and decoupled
    # from s₁
    r, gamma, Omega, phi = OMEGA_ZERO

    actual = embedder_at(OMEGA_ZERO).kernel([40.0])

    assert_allclose(actual, [r**2 * np.sin(phi)**2], rtol=RTOL)


@pytest.mark.parametrize("nu", [0, 1], ids=["value", "gradient"])
def test_kernel_of_integer_times_equals_kernel_of_float_times(nu):
    # √G = 0.6, so the kernel changes form between t = 1 and t = 2
    emb = embedder_at(CASES["overdamped"])
    expected = emb.kernel(np.array([0.0, 1.0, 2.0, 4.0]), nu=nu)

    actual = emb.kernel(np.array([0, 1, 2, 4]), nu=nu)

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("method, nu", [
    ("kernel", 0), ("kernel", 1),
    ("spectrum", 0), ("spectrum", 1), ("spectrum", 2),
], ids=["kernel-value", "kernel-gradient",
        "spectrum-value", "spectrum-gradient", "spectrum-hessian"])
def test_closed_forms_are_even_in_their_argument(method, nu):
    f = getattr(embedder_at(CASES["overdamped"]), method)

    assert_array_equal(f(-TIME, nu=nu), f(TIME, nu=nu))


# The tests via the drift matrix below also check the left column; these add
# Ap[0, 0] = 0.
def test_drift_matrix_couples_theta_skew_symmetrically():
    Ap = embedder_at(CASES["overdamped"]).drift_matrix

    assert Ap[0, 0] == 0.0
    assert_array_equal(Ap[1:, 0], -Ap[0, 1:])


def test_drift_matrix_param_grad_couples_theta_skew_symmetrically():
    params = CASES["overdamped"]

    grad = embedder_at(params).drift_matrix_param_grad(np.array(params))

    assert_array_equal(grad[:, 0, 0], 0.0)
    assert_array_equal(grad[:, 1:, 0], -grad[:, 0, 1:])


def test_drift_matrix_is_in_the_maximal_gauge():
    # θ = r(cos φ, sin φ) and A = [[2γ, Ω], [-Ω, 0]]: s₂ is undamped
    expected = [[0.0, 1.3, 0.0],
                [-1.3, 3.0, 0.8],
                [0.0, -0.8, 0.0]]

    actual = embedder_at([1.3, 1.5, 0.8, 0.0]).drift_matrix

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("params", list(KERNEL_CASES.values()),
                         ids=list(KERNEL_CASES))
def test_kernel_of_drift_matrix_matches_closed_form(
        reference, drift_matrix_reference, params):
    Ap = embedder_at(params).compute_drift_matrix(np.array(params))

    actual, _ = drift_matrix_reference.kernel(Ap, TIME)

    assert_close(actual, reference.kernel(params, TIME))


@pytest.mark.parametrize("params", list(CASES.values()), ids=list(CASES))
def test_spectrum_of_drift_matrix_matches_closed_form(
        reference, drift_matrix_reference, params):
    Ap = embedder_at(params).compute_drift_matrix(np.array(params))

    actual, _ = drift_matrix_reference.spectrum(Ap, FREQ)

    assert_close(actual, reference.spectrum(params, FREQ))


@pytest.mark.parametrize("params", list(KERNEL_CASES.values()),
                         ids=list(KERNEL_CASES))
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


def test_drift_matrix_x_grad_uses_its_argument_not_stored_params(
        reference, drift_matrix_reference):
    emb = embedder_at(CASES["overdamped"])
    r, gamma, Omega, phi = params = CASES["underdamped"]
    x = np.array([r, np.log(gamma), Omega, phi])
    # default mappers: dp/dx = [1, γ, 1, 1]
    expected = (np.array([1.0, gamma, 1.0, 1.0])[:, None]
                * reference.kernel(params, TIME, nu=1))

    _, actual = drift_matrix_reference.kernel(
        emb.compute_drift_matrix(np.array(params)), TIME,
        emb.drift_matrix_x_grad(x))

    assert_close(actual, expected)


def test_default_mappers_take_log_of_gamma_only():
    emb = embedder_at([-1.3, 1.5, -0.8, 2.0])

    assert_allclose(emb.x, [-1.3, np.log(1.5), -0.8, 2.0], rtol=RTOL)


def test_conversion_to_primitive():
    # θ = [1, 2], γ = 1.5, δ = 0.25, ζ = 3 give r² = θ₁² + θ₂² = 5,
    # Ω² = γ² - δ² + ζ² = 11.1875 and cos 2φ = δ(θ₁² - θ₂²)/(γr²) = -0.1
    emb = TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0)

    assert_allclose(
        emb.primitive_params,
        [np.sqrt(5.0), 1.5, np.sqrt(11.1875), np.arccos(-0.1) / 2],
        rtol=RTOL)


def test_conventional_params_read_back_in_maximal_gauge():
    # The parameters of test_conversion_to_primitive. The
    # maximal gauge has θ = r(cos φ, sin φ), δ = γ and ζ = Ω, where
    # r² cos²φ = r²(1 + cos 2φ)/2 = 2.25
    emb = TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0)

    assert_allclose(
        emb.conventional_params,
        [1.5, np.sqrt(2.75), 1.5, 1.5, np.sqrt(11.1875)],
        rtol=RTOL)


@pytest.mark.parametrize("params", list(CANONICAL_CASES.values()),
                         ids=list(CANONICAL_CASES))
def test_to_primitive_inverts_to_conventional(params):
    emb = embedder_at(params)

    actual = emb.to_primitive(emb.to_conventional(np.array(params)))

    assert_allclose(actual, params, rtol=RTOL, atol=RTOL)


def test_to_primitive_returns_the_canonical_point_with_the_same_kernel():
    # r → |r|, Ω → |Ω| and φ → φ' in [0, π/2] with cos 2φ' = cos 2φ,
    # here φ' = π - 2
    emb = embedder_at([-1.3, 1.0, -0.8, 2.0])

    actual = emb.to_primitive(emb.conventional_params)

    assert_allclose(actual, [1.3, 1.0, 0.8, np.pi - 2.0], rtol=RTOL)


def test_maximal_gauge_keeps_the_kernel(drift_matrix_reference):
    Ap = np.array([[0.0, 1.0, 2.0], [-1.0, 1.75, 3.0], [-2.0, -3.0, 1.25]])
    expected, _ = drift_matrix_reference.kernel(Ap, TIME)

    actual = TwoAuxEmbedder([1.0, 2.0], 1.5, 0.25, 3.0).kernel(TIME)

    assert_close(actual, expected)


def test_params_on_the_stability_boundary_are_accepted_and_keep_the_kernel(
        drift_matrix_reference):
    # γ = -δ leaves s₁ undamped, which gives a Brownian oscillator (φ = π/2)
    Ap = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 1.5], [0.0, -1.5, 2.0]])
    expected, _ = drift_matrix_reference.kernel(Ap, TIME)

    actual = TwoAuxEmbedder([1.0, 0.0], 1.0, -1.0, 1.5).kernel(TIME)

    assert_close(actual, expected)


def test_weakly_damped_params_are_accepted_and_keep_the_kernel(
        drift_matrix_reference):
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


@pytest.mark.parametrize("theta, gamma, delta", [
    ([1.0, 0.0], 0.1, 0.5),
    ([1.0, 0.0], 0.1, -0.5),
    ([0.0, 0.0], 1.0, 0.5),
    ([1.0, 0.0], 0.0, 0.0),
    ([np.nan, 0.0], 1.0, 0.5),
], ids=["gamma-below-delta", "gamma-below-minus-delta", "theta-zero",
        "gamma-zero", "not-finite"])
def test_inadmissible_params_are_rejected(theta, gamma, delta):
    with pytest.raises(ValueError, match="admissible"):
        TwoAuxEmbedder(theta, gamma, delta, 1.0)


@pytest.mark.parametrize("key", ["sigma", "threshold"])
def test_softmax_settings_are_rejected(key):
    params = {"theta": [1.0, 0.0], "gamma": 3.0, "delta": 0.5, "zeta": 2.0,
              key: 50.0}

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
