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
import pickle
import re
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.analysis.plot_objective import parse_trajectory_file
from glefit.config.config_handler import ConfigError, ConfigHandler
from glefit.embedding import EMBEDDER_MAP, TwoAuxEmbedder
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

# gen2x2 with r = 2, γ = 1.25, Ω = 0.75 and cos 2φ = 0.6, given in the
# maximal gauge and fitted to its own kernel, so that EF converges at
# once. Its minimal gauge has independent modes: δ = √(γ² - Ω²) = 1,
# ζ = 0 and θ₁,₂² = r²(1 ± γ cos(2φ)/δ)/2 = (3.5, 0.5).
GEN2X2 = {"theta": [np.sqrt(3.2), np.sqrt(0.8)], "gamma": 1.25,
          "delta": 1.25, "zeta": 0.75}
GRID = [0.0, 0.3, 1.1, 4.0]


def converging_config():
    """A config that fits GEN2X2 to its own kernel on GRID."""
    target = TwoAuxEmbedder.from_dict(GEN2X2).kernel(GRID).tolist()
    return {
        "data": {
            "kernel": {"source": "inline", "grid": GRID, "target": target}},
        "merit_function": {
            "type": "kernel",
            "parameters": {"data": "kernel", "metric": "squared"}},
        "embedder": {"type": "gen2x2", "parameters": GEN2X2},
        "optimization": {
            "type": "EF",
            "parameters": {"trajfile": "traj.out"},
            "options": {"max_iter": 5, "gtol": 1.0e-8}},
    }


def with_optimization(type_, options, embedder=None):
    """CONFIG with the given optimizer, options and prony parameters."""
    config = copy.deepcopy(CONFIG)
    config["optimization"]["type"] = type_
    config["optimization"]["options"] = options
    if embedder is not None:
        config["embedder"]["parameters"] = embedder
    return config


@pytest.fixture
def run_main(tmp_path, monkeypatch):
    """Run main on a config, in tmp_path as the working directory."""
    monkeypatch.chdir(tmp_path)

    def run(config):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        main(argparse.Namespace(config=str(path)))

    return run


def read_pickle(tmp_path):
    with open(tmp_path / "embedder_parameters.pkl", "rb") as f:
        return pickle.load(f)


def test_main_writes_the_propagation_matrix_on_convergence(
        run_main, tmp_path):
    run_main(converging_config())

    # the file keeps eight significant digits
    written = json.loads((tmp_path / "optimized_Ap_matrix.json").read_text())
    assert_allclose(
        written["Ap"],
        [[0.0, np.sqrt(3.5), np.sqrt(0.5)],
         [-np.sqrt(3.5), 2.25, 0.0],
         [-np.sqrt(0.5), 0.0, 0.25]],
        rtol=1.0e-7, atol=1.0e-7)


def test_main_pickles_a_converged_fit(run_main, tmp_path):
    run_main(converging_config())

    pickled = read_pickle(tmp_path)
    rebuilt = EMBEDDER_MAP[pickled["type"]].from_dict(pickled["parameters"])
    assert pickled["success"] is True
    assert_allclose(rebuilt.kernel(GRID),
                    TwoAuxEmbedder.from_dict(GEN2X2).kernel(GRID),
                    rtol=1e-12)


def test_main_pickles_the_start_if_it_takes_no_step(run_main, tmp_path):
    run_main(with_optimization("EF", {"max_iter": 0, "gtol": 1e-12}))

    pickled = read_pickle(tmp_path)
    assert pickled == {"type": "prony",
                       "parameters": {"theta": 1.0, "gamma": 0.7},
                       "success": False}
    assert pickled["success"] is False


def test_main_pickles_the_parameters_after_the_last_step(
        run_main, tmp_path):
    run_main(with_optimization("EF", {"max_iter": 2, "gtol": 1e-12}))

    # the trajectory keeps 7 significant digits
    steps, params = parse_trajectory_file(tmp_path / "traj.out")
    parameters = read_pickle(tmp_path)["parameters"]
    assert steps == [0, 1, 2]
    assert_allclose([parameters["theta"], parameters["gamma"]],
                    params[-1], rtol=1e-6)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_main_pickles_the_embedder_if_the_fit_fails(run_main, tmp_path):
    # NR from γ = 100 steps so far that γ = e^x overflows, with warnings,
    # and the kernel cannot be evaluated
    config = with_optimization("NR", {"max_iter": 30, "gtol": 1e-8},
                               embedder={"theta": 0.1, "gamma": 100.0})

    with pytest.raises(ValueError):
        run_main(config)

    assert read_pickle(tmp_path)["success"] is False


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
