#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_bath.py
@Time    :   2026/11/16 15:00:00
@Author  :   George Trenins
@Desc    :   Test the discretisation of a bath spectrum
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.utils.bath import WAM_discretization


ETA = 0.7
OMEGA_CUT = 2.5
NBATH = 5
WMAX = 10.0
# the spline and the bisection (xtol = 2e-12) give errors of 1e-12 and 1e-11
WEIGHT_RTOL = 1.0e-11
FREQUENCY_ATOL = 1.0e-10


def debye_spectrum(omega):
    """Λ(ω) = η ωc² / (ω² + ωc²)"""
    return ETA * OMEGA_CUT**2 / (omega**2 + OMEGA_CUT**2)


def debye_cumulative(omega):
    """C(ω) = ∫₀^ω Λ = η ωc arctan(ω/ωc)"""
    return ETA * OMEGA_CUT * np.arctan(omega / OMEGA_CUT)


def debye_frequency(cumulative):
    """The ω at which C(ω) takes the given value"""
    return OMEGA_CUT * np.tan(cumulative / (ETA * OMEGA_CUT))


@pytest.fixture
def grid():
    return np.linspace(0.0, 40.0, 4001)


def test_wam_weight_matches_the_debye_closed_form(grid):
    expected = debye_cumulative(WMAX) / NBATH

    weight, _ = WAM_discretization(
        grid, debye_spectrum(grid), NBATH, wmax=WMAX)

    assert_allclose(weight, expected, rtol=WEIGHT_RTOL)


def test_wam_frequencies_match_the_debye_closed_form(grid):
    # C(ωⱼ) = (j + ½) C(ω_max)/n
    weight = debye_cumulative(WMAX) / NBATH
    expected = debye_frequency((np.arange(NBATH) + 0.5) * weight)

    _, freqs = WAM_discretization(
        grid, debye_spectrum(grid), NBATH, wmax=WMAX)

    assert_allclose(freqs, expected, rtol=0.0, atol=FREQUENCY_ATOL)


@pytest.mark.parametrize("kwargs", [{}, {"wmax": 100.0}],
                         ids=["default", "beyond-grid"])
def test_wam_caps_wmax_at_the_last_grid_point(grid, kwargs):
    expected = debye_cumulative(grid[-1]) / NBATH

    weight, _ = WAM_discretization(
        grid, debye_spectrum(grid), NBATH, **kwargs)

    assert_allclose(weight, expected, rtol=WEIGHT_RTOL)


def test_wam_integrates_from_the_first_grid_point():
    # C is integrated from 0.5, not from 0
    grid = np.linspace(0.5, 40.0, 3951)
    start = debye_cumulative(grid[0])
    weight = (debye_cumulative(WMAX) - start) / NBATH
    expected = debye_frequency(start + (np.arange(NBATH) + 0.5) * weight)

    _, freqs = WAM_discretization(
        grid, debye_spectrum(grid), NBATH, wmax=WMAX)

    assert_allclose(freqs, expected, rtol=0.0, atol=FREQUENCY_ATOL)


def test_wam_returns_a_float_and_nbath_frequencies(grid):
    weight, freqs = WAM_discretization(
        grid, debye_spectrum(grid), NBATH, wmax=WMAX)

    assert isinstance(weight, float)
    assert freqs.shape == (NBATH,)
