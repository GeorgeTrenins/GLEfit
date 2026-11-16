#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_spectral.py
@Time    :   2026/11/16 12:00:00
@Author  :   George Trenins
@Desc    :   Test the model spectra and memory kernels
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import IntegrationWarning
from glefit.embedding.prony import PronyEmbedder
from glefit.spectral import Debye, ExpOhmic


# η ≠ 1 and ωc ≠ 1, so that a missing or inverted factor shows
ETA = 0.7
OMEGA_CUT = 2.5
NM_OMEGA = 3.0
RTOL = 1.0e-13
# quad's default tolerances give nmK to about 2e-11
NM_KERNEL_ATOL = 1.0e-9
MODELS = [Debye, ExpOhmic]
MODEL_IDS = ["debye", "expohmic"]
EVALUATIONS = [
    lambda bath, x: bath.Lambda(x),
    lambda bath, x: bath.K(x),
    lambda bath, x: bath.nmLambda(NM_OMEGA, x),
    lambda bath, x: bath.nmK(NM_OMEGA, x),
]
EVALUATION_IDS = ["Lambda", "K", "nmLambda", "nmK"]


@pytest.mark.parametrize("model, expected", [
    # η ωc² / (ω² + ωc²)
    (Debye, [0.7, 0.35, 0.10355029585798817]),
    # η exp(-|ω|/ωc)
    (ExpOhmic, [0.7, 0.2575156088200096, 0.06350256730258876]),
], ids=MODEL_IDS)
def test_lambda_matches_the_closed_form(model, expected):
    omega = np.array([0.0, OMEGA_CUT, 6.0])

    actual = model(ETA, OMEGA_CUT).Lambda(omega)

    assert_allclose(actual, expected, rtol=RTOL)


# (2/π) ∫₀^∞ Λ(ω) cos(ωt) dω, with mpmath at 30 digits
@pytest.mark.parametrize("model, expected", [
    (Debye, [1.75, 0.6437890220500241, 0.0009678976477587088]),
    (ExpOhmic, [1.1140846016432673, 0.5570423008216336,
                0.019459993041803795]),
], ids=MODEL_IDS)
def test_kernel_is_the_cosine_transform_of_lambda(model, expected):
    t = np.array([0.0, 0.4, 3.0])

    actual = model(ETA, OMEGA_CUT).K(t)

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("evaluate", EVALUATIONS, ids=EVALUATION_IDS)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_negative_argument_gives_the_same_value(model, evaluate):
    # 3.1 and 6.0 lie above ωₙ, where nmLambda is not zero
    x = np.array([0.4, 3.1, 6.0])
    bath = model(ETA, OMEGA_CUT)
    expected = evaluate(bath, x)

    actual = evaluate(bath, -x)

    assert_allclose(actual, expected, rtol=RTOL)


# θ(x) √x Λ(ω√x) with x = 1 - ωₙ²/ω², with mpmath at 30 digits:
# zero at and below ωₙ
@pytest.mark.parametrize("model, expected", [
    (Debye, [0.0, 0.0, 0.16067827674264842, 0.11395071102426824]),
    (ExpOhmic, [0.0, 0.0, 0.12903969569715906, 0.07585156434200468]),
], ids=MODEL_IDS)
def test_nm_lambda_matches_the_transformed_spectrum(model, expected):
    omega = np.array([2.9, NM_OMEGA, 3.1, 6.0])

    actual = model(ETA, OMEGA_CUT).nmLambda(NM_OMEGA, omega)

    assert_allclose(actual, expected, rtol=RTOL, atol=0.0)


# (2/π) ∫₀^∞ Λ(u) u²/(u² + ωₙ²) cos(t√(u² + ωₙ²)) du, with mpmath at
# 30 digits; at t = 0, the Debye value is η ωc²/(ωc + ωₙ)
@pytest.mark.parametrize("model, expected", [
    (Debye, [0.7954545454545454, -0.18737358431293366,
             -0.007117004849324067]),
    (ExpOhmic, [0.36600113125987366, -0.12085373339234454,
                -0.0068643627371438014]),
], ids=MODEL_IDS)
def test_nm_kernel_matches_the_substituted_integral(model, expected):
    t = np.array([0.0, 0.4, 3.0])

    actual = model(ETA, OMEGA_CUT).nmK(NM_OMEGA, t)

    assert_allclose(actual, expected, rtol=0.0, atol=NM_KERNEL_ATOL)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_nm_lambda_at_zero_nm_omega_equals_lambda(model):
    omega = np.array([0.0, 0.4, 3.1, 6.0])
    bath = model(ETA, OMEGA_CUT)

    actual = bath.nmLambda(0.0, omega)

    assert_allclose(actual, bath.Lambda(omega), rtol=RTOL)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_nm_kernel_at_zero_nm_omega_equals_kernel(model):
    t = np.array([0.0, 0.4, 3.0])
    bath = model(ETA, OMEGA_CUT)

    actual = bath.nmK(0.0, t)

    assert_allclose(actual, bath.K(t), rtol=0.0, atol=NM_KERNEL_ATOL)


def test_nm_kernel_passes_keyword_arguments_to_quad():
    # with quad's default tolerances, the error is 1e-11 to 5e-11, and
    # epsabs below 1e-12 raises IntegrationWarning
    t = np.array([0.4, 3.0, 10.0])
    bath = Debye(ETA, OMEGA_CUT)

    actual = bath.nmK(0.0, t, epsabs=1.0e-12)

    assert_allclose(actual, bath.K(t), rtol=0.0, atol=1.0e-14)


def test_nm_kernel_at_zero_time_passes_limit_to_quad():
    # at t = 0, quad's defaults are exact to 1e-14, so no tolerance shows
    # whether the arguments arrive, but limit=1 makes quad warn
    bath = Debye(ETA, OMEGA_CUT)

    with pytest.warns(IntegrationWarning):
        bath.nmK(NM_OMEGA, 0.0, limit=1)


@pytest.mark.parametrize("prony, debye", [
    (lambda emb, x: emb.kernel(x), lambda bath, x: bath.K(x)),
    (lambda emb, x: emb.spectrum(x), lambda bath, x: bath.Lambda(x)),
], ids=["kernel", "spectrum"])
def test_debye_matches_its_exact_prony_embedding(prony, debye):
    x = np.array([0.0, 0.4, 3.0, 7.0])
    emb = PronyEmbedder(theta=np.sqrt(ETA * OMEGA_CUT), gamma=OMEGA_CUT)

    actual = debye(Debye(ETA, OMEGA_CUT), x)

    assert_allclose(actual, prony(emb, x), rtol=RTOL)


@pytest.mark.parametrize("method", ["nmLambda", "nmK"])
def test_negative_nm_omega_raises_value_error(method):
    bath = Debye(ETA, OMEGA_CUT)

    with pytest.raises(ValueError, match="normal-mode frequency"):
        getattr(bath, method)(-1.0, 1.0)


@pytest.mark.parametrize("method", ["nmLambda", "nmK"])
def test_array_nm_omega_raises_type_error(method):
    bath = Debye(ETA, OMEGA_CUT)

    with pytest.raises(TypeError, match="normal-mode frequency"):
        getattr(bath, method)(np.array([1.0, 2.0]), 1.0)
