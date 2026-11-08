#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_inputs.py
@Time    :   2026/11/08 15:00:00
@Author  :   George Trenins
@Desc    :   Test how merit functions are built: from_dict and the weights
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
