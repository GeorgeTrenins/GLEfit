#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_config.py
@Time    :   2026/11/10 18:00:00
@Author  :   George Trenins
@Desc    :   Test the checks of a configuration
'''


from __future__ import print_function, division, absolute_import
import copy
import re
import pytest
from glefit.config.config_handler import ConfigError, ConfigHandler


CONFIG = {
    "data": {"kernel": {"source": "inline", "grid": [0.0, 1.0, 2.0],
                        "target": [1.0, 0.5, 0.25]}},
    "merit_function": {"type": "kernel",
                       "parameters": {"data": "kernel", "metric": "squared"}},
    "embedder": {"type": "prony", "parameters": {"theta": 1.0, "gamma": 0.7}},
    "optimization": {"type": "EF", "parameters": {"trajfile": "traj.out"},
                     "options": {"max_iter": 5, "gtol": 1e-3}},
}


def without(path):
    """A handler for CONFIG without the key at path."""
    config = copy.deepcopy(CONFIG)
    *parents, key = path
    section = config
    for parent in parents:
        section = section[parent]
    del section[key]
    return ConfigHandler.from_dict(config)


def with_options(options):
    """A handler for CONFIG with the given optimization options."""
    config = copy.deepcopy(CONFIG)
    config["optimization"]["options"] = options
    return ConfigHandler.from_dict(config)


def test_validate_accepts_a_complete_config():
    handler = ConfigHandler.from_dict(CONFIG)

    handler.validate()


# every key that optimize.py reads, and the text that the error names it by
REQUIRED = [
    (("data",), "data"),
    (("merit_function",), "merit_function"),
    (("merit_function", "type"), "merit_function.type"),
    (("merit_function", "parameters"), "merit_function.parameters"),
    (("embedder",), "embedder"),
    (("embedder", "type"), "embedder.type"),
    (("embedder", "parameters"), "embedder.parameters"),
    (("optimization",), "optimization"),
    # the message names optimization.optimizer, an old name of the key
    (("optimization", "type"), "optimization"),
    (("optimization", "parameters"), "optimization.parameters"),
    (("optimization", "options"), "optimization.options"),
    (("optimization", "options", "max_iter"),
     "optimization.options.max_iter"),
]


@pytest.mark.parametrize("path, name", REQUIRED,
                         ids=[".".join(path) for path, _ in REQUIRED])
def test_validate_requires_key(path, name):
    handler = without(path)

    with pytest.raises(ConfigError, match=re.escape(name)):
        handler.validate()


def test_validate_requires_options_to_be_a_dictionary():
    handler = with_options(None)

    with pytest.raises(ConfigError, match="optimization.options"):
        handler.validate()


@pytest.mark.parametrize("max_iter", ["5", 2.5, -1, True, None],
                         ids=["str", "float", "negative", "bool", "none"])
def test_validate_rejects_bad_max_iter(max_iter):
    handler = with_options({"max_iter": max_iter})

    with pytest.raises(ConfigError, match="max_iter"):
        handler.validate()


@pytest.mark.parametrize("max_iter", [0, 5])
def test_validate_accepts_a_non_negative_max_iter(max_iter):
    handler = with_options({"max_iter": max_iter})

    handler.validate()
