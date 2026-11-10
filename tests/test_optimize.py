#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_optimize.py
@Time    :   2026/11/05 18:00:00
@Author  :   George Trenins
@Desc    :   Test the top-level optimization driver
'''


from __future__ import print_function, division, absolute_import
import argparse
import copy
import json
import re
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.config.config_handler import ConfigError, ConfigHandler
from glefit.embedding import TwoAuxEmbedder
from glefit.optimize import Optimization, main


CONFIG = {
    "data": {"kernel": {"source": "inline", "grid": [0.0, 1.0, 2.0],
                        "target": [1.0, 0.5, 0.25]}},
    "merit_function": {"type": "kernel",
                       "parameters": {"data": "kernel", "metric": "squared"}},
    "embedder": {"type": "prony", "parameters": {"theta": 1.0, "gamma": 0.7}},
    "optimization": {"type": "EF", "parameters": {"trajfile": "traj.out"},
                     "options": {"max_iter": 5, "gtol": 1e-3}},
}


def test_main_writes_the_propagation_matrix_on_convergence(
        tmp_path, monkeypatch):
    # gen2x2 with r = 2, γ = 1.25, Ω = 0.75 and cos 2φ = 0.6, given in the
    # maximal gauge and fitted to its own kernel, so that EF converges at
    # once. Its minimal gauge has independent modes: δ = √(γ² - Ω²) = 1,
    # ζ = 0 and θ₁,₂² = r²(1 ± γ cos(2φ)/δ)/2 = (3.5, 0.5). The file keeps
    # eight significant digits.
    params = {"theta": [np.sqrt(3.2), np.sqrt(0.8)], "gamma": 1.25,
              "delta": 1.25, "zeta": 0.75}
    grid = [0.0, 0.3, 1.1, 4.0]
    target = TwoAuxEmbedder.from_dict(params).kernel(grid).tolist()
    config = {
        "data": {
            "kernel": {"source": "inline", "grid": grid, "target": target}},
        "merit_function": {
            "type": "kernel",
            "parameters": {"data": "kernel", "metric": "squared"}},
        "embedder": {"type": "gen2x2", "parameters": params},
        "optimization": {
            "type": "EF",
            "parameters": {"trajfile": "traj.out"},
            "options": {"max_iter": 5, "gtol": 1.0e-8}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)

    main(argparse.Namespace(config=str(tmp_path / "config.json")))

    written = json.loads((tmp_path / "optimized_Ap_matrix.json").read_text())
    assert_allclose(
        written["Ap"],
        [[0.0, np.sqrt(3.5), np.sqrt(0.5)],
         [-np.sqrt(3.5), 2.25, 0.0],
         [-np.sqrt(0.5), 0.0, 0.25]],
        rtol=1.0e-7, atol=1.0e-7)


# each section, with one of its known types, and a type that is unknown
# or cannot be a registry key at all
@pytest.mark.parametrize("section, known", [
    ("embedder", "prony"),
    ("merit_function", "kernel"),
    ("optimization", "EF"),
], ids=["embedder", "merit_function", "optimization"])
@pytest.mark.parametrize("bad", ["pronny", ["EF"]], ids=["unknown", "list"])
def test_from_config_rejects_an_unknown_type(
        section, known, bad, tmp_path, monkeypatch):
    config = copy.deepcopy(CONFIG)
    config[section]["type"] = bad
    handler = ConfigHandler.from_dict(config)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match=re.escape(
            f"{section}.type {bad!r}") + f".*'{known}'"):
        Optimization.from_config(handler)
