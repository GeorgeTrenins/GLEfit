#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_inputs.py
@Time    :   2026/11/08 15:00:00
@Author  :   George Trenins
@Desc    :   Test how merit functions are built: inputs, from_dict, weights
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.config.config_handler import ConfigError
from glefit.embedding import PronyEmbedder
from glefit.merit import MemoryKernel, MemorySpectrum


RTOL = 1.0e-12
# serves as a time grid and as a frequency grid
GRID = np.array([0.0, 0.4, 1.3, 3.0])
TARGET = np.array([0.3, -0.2, 0.1, 0.05])
# non-uniform, and a list, as a config file gives it
WEIGHT = [0.5, 2.0, 1.0, 3.0]
DATASETS = {"curve": (GRID, TARGET)}
PROPERTIES = {"kernel": MemoryKernel, "spectrum": MemorySpectrum}
BAD_ENTRIES = {
    "no-data": ({"metric": "squared"}, "'data'"),
    "no-metric": ({"data": "curve"}, "'metric'"),
    "unknown-dataset": ({"data": "nonexistent", "metric": "squared"},
                        "nonexistent"),
}
WRONG_SHAPES = {
    "wrong-length": [1.0, 2.0],
    # broadcasts against the grid, so only the shape check rejects it
    "2-d": np.ones((1, len(GRID))),
    "one-entry": [2.0],
}
BAD_WEIGHTS = {
    "negative-entry": ([0.5, -1.0, 1.0, 1.0], "non-negative"),
    "nan-entry": ([0.5, np.nan, 1.0, 1.0], "finite"),
    "infinite": (np.inf, "finite"),
    "all-zero": (0.0, "positive"),
}
BAD_INPUTS = {
    "1-point-target-4-point-grid": (GRID, TARGET[:1], "shape of grid"),
    "target-too-short": (GRID, TARGET[:3], "shape of grid"),
    "column-target": (GRID, TARGET.reshape(4, 1), "shape of grid"),
    "2-d": (GRID.reshape(2, 2), TARGET.reshape(2, 2), "non-empty 1-D"),
    "scalar": (1.0, 0.5, "non-empty 1-D"),
    "empty": ([], [], "non-empty 1-D"),
    "unsorted": ([0.0, 1.3, 0.4, 3.0], TARGET, "strictly increasing"),
    "repeated-point": ([0.0, 0.4, 0.4, 3.0], TARGET, "strictly increasing"),
}
INT_GRID = np.array([0, 1, 2, 4])
INT_TARGET = np.array([1, 0, 0, 0])
EXTRA_ARGUMENTS = {
    "misspelled-keyword": ((), {"wieght": 2.0}, "wieght"),
    "extra-positional": ((2.0,), {}, "positional"),
}


@pytest.fixture(params=list(PROPERTIES))
def kind(request):
    return request.param


@pytest.fixture
def emb():
    return PronyEmbedder(0.8, 1.3)


def squared_deviations(emb, kind):
    """(h - y)² at each grid point, with h from the closed form"""
    return (getattr(emb, kind)(GRID) - TARGET)**2


def test_from_dict_passes_grid_target_and_weight(kind, emb):
    parameters = {"data": "curve", "metric": "squared", "weight": WEIGHT}
    expected = np.dot(WEIGHT, squared_deviations(emb, kind))

    prop = PROPERTIES[kind].from_dict(parameters, DATASETS, emb)

    assert_allclose(prop.distance, expected, rtol=RTOL)


def test_from_dict_default_weight(kind, emb):
    # from_dict and the constructor each supply a default for weight
    parameters = {"data": "curve", "metric": "squared"}
    expected = PROPERTIES[kind](GRID, TARGET, emb, "squared").distance

    prop = PROPERTIES[kind].from_dict(parameters, DATASETS, emb)

    assert_allclose(prop.distance, expected, rtol=RTOL)


@pytest.mark.parametrize("parameters, message", list(BAD_ENTRIES.values()),
                         ids=list(BAD_ENTRIES))
def test_from_dict_bad_entry_raises_config_error(parameters, message, emb):
    with pytest.raises(ConfigError, match=message):
        MemoryKernel.from_dict(parameters, DATASETS, emb)


def test_scalar_weight_scales_every_deviation(kind, emb):
    expected = 2.5 * np.sum(squared_deviations(emb, kind))

    prop = PROPERTIES[kind](GRID, TARGET, emb, "squared", weight=2.5)

    assert_allclose(prop.distance, expected, rtol=RTOL)


@pytest.mark.parametrize("weight", list(WRONG_SHAPES.values()),
                         ids=list(WRONG_SHAPES))
def test_weight_of_wrong_shape_raises_value_error(weight, emb):
    with pytest.raises(ValueError, match="[Ww]eight"):
        MemoryKernel(GRID, TARGET, emb, "squared", weight=weight)


@pytest.mark.parametrize("weight, message", list(BAD_WEIGHTS.values()),
                         ids=list(BAD_WEIGHTS))
def test_bad_weight_value_raises_value_error(weight, message, emb):
    with pytest.raises(ValueError, match=message):
        MemoryKernel(GRID, TARGET, emb, "squared", weight=weight)


def test_weight_none_gives_the_default(emb):
    expected = MemoryKernel(GRID, TARGET, emb, "squared").distance

    prop = MemoryKernel(GRID, TARGET, emb, "squared", weight=None)

    assert_allclose(prop.distance, expected, rtol=RTOL)


def test_zero_weight_excludes_a_point(emb):
    expected = squared_deviations(emb, "kernel")[1]

    prop = MemoryKernel(GRID, TARGET, emb, "squared",
                        weight=[0.0, 1.0, 0.0, 0.0])

    assert_allclose(prop.distance, expected, rtol=RTOL)


def test_constructor_copies_the_weight(emb):
    weight = np.array(WEIGHT)
    prop = MemoryKernel(GRID, TARGET, emb, "squared", weight=weight)
    expected = np.dot(WEIGHT, squared_deviations(emb, "kernel"))

    weight[:] = 0.5

    assert_allclose(prop.distance, expected, rtol=RTOL)


# The tests below pass weight=1.0, so that they do not depend on the
# default weights.

@pytest.mark.parametrize("edited", ["grid", "target"])
def test_constructor_copies_grid_and_target(edited, emb):
    inputs = {"grid": GRID.copy(), "target": TARGET.copy()}
    prop = MemoryKernel(inputs["grid"], inputs["target"], emb, "squared",
                        weight=1.0)
    expected = np.sum(squared_deviations(emb, "kernel"))

    inputs[edited][:] = 0.5

    assert_allclose(prop.distance, expected, rtol=RTOL)


@pytest.mark.parametrize("grid, target, message", list(BAD_INPUTS.values()),
                         ids=list(BAD_INPUTS))
def test_bad_grid_or_target_raises_value_error(grid, target, message, emb):
    with pytest.raises(ValueError, match=message):
        MemoryKernel(grid, target, emb, "squared", weight=1.0)


def test_one_point_grid_is_accepted(kind, emb):
    grid, target = GRID[1:2], TARGET[1:2]
    expected = np.sum((getattr(emb, kind)(grid) - target)**2)

    prop = PROPERTIES[kind](grid, target, emb, "squared", weight=1.0)

    assert_allclose(prop.distance, expected, rtol=RTOL)


def test_integer_grid_and_target_are_accepted(kind, emb):
    f = getattr(emb, kind)
    h, g = f(INT_GRID), f(INT_GRID, nu=1, mapped=True)
    expected = 2 * g @ (h - INT_TARGET)
    prop = PROPERTIES[kind](INT_GRID, INT_TARGET, emb, "squared", weight=1.0)

    actual = prop.gradient()

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("args, kwargs, message",
                         list(EXTRA_ARGUMENTS.values()),
                         ids=list(EXTRA_ARGUMENTS))
def test_extra_argument_raises_type_error(args, kwargs, message, emb):
    with pytest.raises(TypeError, match=message):
        MemoryKernel(GRID, TARGET, emb, "squared", 1.0, *args, **kwargs)
