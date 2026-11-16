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
from glefit.embedding.gen2x2 import TwoAuxEmbedder
from glefit.embedding.prony import PronyEmbedder
from glefit.spectral import BrownianOscillator, Debye, ExpOhmic


# η ≠ 1, ωc ≠ 1 and ω₀ ≠ 1, so that a missing or inverted factor shows
ETA = 0.7
OMEGA_CUT = 2.5
OMEGA_0 = 1.7
NM_OMEGA = 3.0
RTOL = 1.0e-13
# quad's default tolerances give nmK to about 2e-11
NM_KERNEL_ATOL = 1.0e-9
# each model, and the arguments of its constructor in signature order; the
# Brownian oscillator is under-, critically and overdamped, γ <, = and > 2ω₀
MODELS = {
    "debye": (Debye, {"eta": ETA, "omega_cut": OMEGA_CUT}),
    "expohmic": (ExpOhmic, {"eta": ETA, "omega_cut": OMEGA_CUT}),
    "brownian-under": (BrownianOscillator,
                       {"eta": ETA, "omega_0": OMEGA_0, "gamma": 0.9}),
    "brownian-critical": (BrownianOscillator,
                          {"eta": ETA, "omega_0": OMEGA_0,
                           "gamma": 2 * OMEGA_0}),
    "brownian-over": (BrownianOscillator,
                      {"eta": ETA, "omega_0": OMEGA_0, "gamma": 9.0}),
}
MODEL_IDS = list(MODELS)
BROWNIAN_IDS = ["brownian-under", "brownian-critical", "brownian-over"]
EVALUATIONS = [
    lambda bath, x: bath.Lambda(x),
    lambda bath, x: bath.K(x),
    lambda bath, x: bath.nmLambda(NM_OMEGA, x),
    lambda bath, x: bath.nmK(NM_OMEGA, x),
]
EVALUATION_IDS = ["Lambda", "K", "nmLambda", "nmK"]


def make_model(name, **changes):
    """The model `name` of MODELS, with some of its arguments changed,
    passed by position, so that their order is tested too."""
    model, arguments = MODELS[name]
    return model(*{**arguments, **changes}.values())


@pytest.mark.parametrize("name, expected", [
    # η ωc² / (ω² + ωc²)
    ("debye", [0.7, 0.35, 0.10355029585798817]),
    # η exp(-|ω|/ωc)
    ("expohmic", [0.7, 0.2575156088200096, 0.06350256730258876]),
    # η ω₀⁴ / ((ω² - ω₀²)² + γ²ω²)
    ("brownian-under", [0.7, 0.3575363408981109, 0.005194866931554556]),
    ("brownian-critical", [0.7, 0.06998441457703891, 0.00386560824780167]),
    ("brownian-over", [0.7, 0.011296662129815766, 0.0014571469367693182]),
], ids=MODEL_IDS)
def test_lambda_matches_the_closed_form(name, expected):
    omega = np.array([0.0, OMEGA_CUT, 6.0])

    actual = make_model(name).Lambda(omega)

    assert_allclose(actual, expected, rtol=RTOL)


# (2/π) ∫₀^∞ Λ(ω) cos(ωt) dω, with mpmath at 30 digits
@pytest.mark.parametrize("name, expected", [
    ("debye", [1.75, 0.6437890220500241, 0.0009678976477587088]),
    ("expohmic", [1.1140846016432673, 0.5570423008216336,
                  0.019459993041803795]),
    ("brownian-under", [2.247777777777778, 1.8023437536738574,
                        -0.03756674541069539]),
    ("brownian-critical", [0.595, 0.5064143455686434, 0.022128141659539]),
    ("brownian-over", [0.22477777777777777, 0.20429993533728139,
                       0.08596580342889688]),
], ids=MODEL_IDS)
def test_kernel_is_the_cosine_transform_of_lambda(name, expected):
    t = np.array([0.0, 0.4, 3.0])

    actual = make_model(name).K(t)

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("evaluate", EVALUATIONS, ids=EVALUATION_IDS)
@pytest.mark.parametrize("name", MODEL_IDS)
def test_negative_argument_gives_the_same_value(name, evaluate):
    # 3.1 and 6.0 lie above ωₙ, where nmLambda is not zero
    x = np.array([0.4, 3.1, 6.0])
    bath = make_model(name)
    expected = evaluate(bath, x)

    actual = evaluate(bath, -x)

    assert_allclose(actual, expected, rtol=RTOL)


# θ(x) √x Λ(ω√x) with x = 1 - ωₙ²/ω², with mpmath at 30 digits:
# zero at and below ωₙ
@pytest.mark.parametrize("name, expected", [
    ("debye", [0.0, 0.0, 0.16067827674264842, 0.11395071102426824]),
    ("expohmic", [0.0, 0.0, 0.12903969569715906, 0.07585156434200468]),
    ("brownian-under", [0.0, 0.0, 0.2587580739949168,
                        0.008394412617211206]),
    ("brownian-critical", [0.0, 0.0, 0.12024329275233174,
                           0.005667252035721933]),
    ("brownian-over", [0.0, 0.0, 0.026973512064372217,
                       0.001828994686819215]),
], ids=MODEL_IDS)
def test_nm_lambda_matches_the_transformed_spectrum(name, expected):
    omega = np.array([2.9, NM_OMEGA, 3.1, 6.0])

    actual = make_model(name).nmLambda(NM_OMEGA, omega)

    assert_allclose(actual, expected, rtol=RTOL, atol=0.0)


# as above, at ωₙ = 1 and ω = 0, 2, 3
@pytest.mark.parametrize("name, expected", [
    ("debye", [0.0, 0.4096066098980453, 0.2894589162751949]),
    ("expohmic", [0.0, 0.3032079809420304, 0.2128990196866008]),
    ("brownian-under", [0.0, 2.073294108539211, 0.1691239526553455]),
    ("brownian-critical", [0.0, 0.1459465279548833, 0.04647952753461897]),
    ("brownian-over", [0.0, 0.020835141717073378, 0.008176837023602284]),
], ids=MODEL_IDS)
def test_nm_lambda_accepts_integer_frequencies(name, expected):
    omega = np.array([0, 2, 3])

    actual = make_model(name).nmLambda(1.0, omega)

    assert_allclose(actual, expected, rtol=RTOL, atol=0.0)


# (2/π) ∫₀^∞ Λ(u) u²/(u² + ωₙ²) cos(t√(u² + ωₙ²)) du, with mpmath at
# 30 digits; at t = 0, the Debye value is η ωc²/(ωc + ωₙ)
@pytest.mark.parametrize("name, expected", [
    ("debye", [0.7954545454545454, -0.18737358431293366,
               -0.007117004849324067]),
    ("expohmic", [0.36600113125987366, -0.12085373339234454,
                  -0.0068643627371438014]),
    ("brownian-under", [0.4452417942274008, 0.051676654640324594,
                        -0.15694029681203195]),
    ("brownian-critical", [0.07784291534631055, -0.0002026381787078442,
                           -0.0199937462288828]),
    ("brownian-over", [0.016703722750778548, -0.0012868013019111208,
                       -0.00429938707092122]),
], ids=MODEL_IDS)
def test_nm_kernel_matches_the_substituted_integral(name, expected):
    t = np.array([0.0, 0.4, 3.0])

    actual = make_model(name).nmK(NM_OMEGA, t)

    assert_allclose(actual, expected, rtol=0.0, atol=NM_KERNEL_ATOL)


@pytest.mark.parametrize("name", MODEL_IDS)
def test_nm_lambda_at_zero_nm_omega_equals_lambda(name):
    omega = np.array([0.0, 0.4, 3.1, 6.0])
    bath = make_model(name)

    actual = bath.nmLambda(0.0, omega)

    assert_allclose(actual, bath.Lambda(omega), rtol=RTOL)


@pytest.mark.parametrize("name", MODEL_IDS)
def test_nm_kernel_at_zero_nm_omega_equals_kernel(name):
    t = np.array([0.0, 0.4, 3.0])
    bath = make_model(name)

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


@pytest.mark.parametrize("gen2x2, brownian", [
    (lambda emb, x: emb.kernel(x), lambda bath, x: bath.K(x)),
    (lambda emb, x: emb.spectrum(x), lambda bath, x: bath.Lambda(x)),
], ids=["kernel", "spectrum"])
@pytest.mark.parametrize("name", BROWNIAN_IDS)
def test_brownian_matches_its_exact_gen2x2_embedding(name, gen2x2, brownian):
    x = np.array([0.0, 0.4, 3.0, 7.0])
    eta, omega_0, gamma = MODELS[name][1].values()
    emb = TwoAuxEmbedder(theta=[omega_0 * np.sqrt(eta / gamma), 0.0],
                         gamma=gamma / 2, delta=-gamma / 2, zeta=omega_0)

    actual = brownian(make_model(name), x)

    assert_allclose(actual, gen2x2(emb, x), rtol=RTOL)


# K in closed form, with mpmath at 50 digits. With ω̃² = γ²/4 - ω₀² > 0,
# 1 - exp(-2ω̃t) is small just above critical damping, and for γ ≫ ω₀,
# γ/2 - ω̃ cancels and cosh(ω̃t) overflows at long times.
@pytest.mark.parametrize("gamma, t, expected", [
    (2 * OMEGA_0 * (1 + 1.0e-12), [0.4, 3.0, 10.0],
     [0.5064143455681684, 0.02212814165967729, 4.433873297217927e-07]),
    (1.0e3, [0.4, 30.0, 300.0],
     [0.0020206685961984047, 0.0018549990754360313,
      0.0008500857455754076]),
], ids=["near-critical", "strongly-overdamped"])
def test_brownian_kernel_is_accurate_when_overdamped(gamma, t, expected):
    bath = BrownianOscillator(ETA, OMEGA_0, gamma)

    actual = bath.K(np.array(t))

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("nm_omega", [
    np.float32(NM_OMEGA), np.int64(NM_OMEGA), int(NM_OMEGA),
], ids=["float32", "int64", "int"])
@pytest.mark.parametrize("method", ["nmLambda", "nmK"])
def test_nm_omega_may_be_any_real_number(method, nm_omega):
    x = np.array([0.4, 3.1, 6.0])
    bath = Debye(ETA, OMEGA_CUT)
    expected = getattr(bath, method)(NM_OMEGA, x)

    actual = getattr(bath, method)(nm_omega, x)

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("nm_omega", [-1.0, np.nan], ids=["negative", "nan"])
@pytest.mark.parametrize("method", ["nmLambda", "nmK"])
def test_negative_or_nan_nm_omega_raises_value_error(method, nm_omega):
    bath = Debye(ETA, OMEGA_CUT)

    with pytest.raises(ValueError, match="must be non-negative"):
        getattr(bath, method)(nm_omega, 1.0)


@pytest.mark.parametrize("method", ["nmLambda", "nmK"])
def test_array_nm_omega_raises_type_error(method):
    bath = Debye(ETA, OMEGA_CUT)

    with pytest.raises(TypeError, match="normal-mode frequency"):
        getattr(bath, method)(np.array([1.0, 2.0]), 1.0)


@pytest.mark.parametrize("call", [
    lambda model, arguments: model(*arguments.values(), 1.0),
    lambda model, arguments: model(**arguments, omgea=5.0),
], ids=["positional", "keyword"])
@pytest.mark.parametrize("name", MODEL_IDS)
def test_constructor_rejects_extra_arguments(name, call):
    model, arguments = MODELS[name]

    with pytest.raises(TypeError, match="argument"):
        call(model, arguments)


@pytest.mark.parametrize("eta", [-0.1, np.nan], ids=["negative", "nan"])
@pytest.mark.parametrize("name", MODEL_IDS)
def test_constructor_rejects_invalid_eta(name, eta):
    with pytest.raises(ValueError, match="eta must be non-negative"):
        make_model(name, eta=eta)


@pytest.mark.parametrize("name", MODEL_IDS)
def test_constructor_accepts_zero_eta(name):
    # a bath without friction
    omega = np.array([0.0, 1.0])

    actual = make_model(name, eta=0.0).Lambda(omega)

    assert_allclose(actual, [0.0, 0.0], atol=0.0)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan],
                         ids=["zero", "negative", "nan"])
@pytest.mark.parametrize("name, parameter", [
    ("debye", "omega_cut"),
    ("expohmic", "omega_cut"),
    ("brownian-under", "omega_0"),
    ("brownian-under", "gamma"),
], ids=["debye-omega_cut", "expohmic-omega_cut", "brownian-omega_0",
        "brownian-gamma"])
def test_constructor_rejects_parameter_not_positive(name, parameter, value):
    with pytest.raises(ValueError, match=f"{parameter} must be positive"):
        make_model(name, **{parameter: value})
