#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   make_target.py
@Time    :   2026/11/16 15:00:00
@Author  :   George Trenins
@Desc    :   Write the memory kernel or spectrum of a model spectral
             density to a CSV file that a fit config reads as its target,
             and optionally an initial guess for the embedder.
'''


from __future__ import print_function, division, absolute_import
import argparse
import json
import numbers
from pathlib import Path
import numpy as np
import numpy.typing as npt
import yaml
from glefit.config.config_handler import ConfigError, ConfigHandler
from glefit.embedding import TwoAuxEmbedder
from glefit.spectral import SPECTRAL_MAP
from glefit.spectral._base import BaseSpectralDensity
from glefit.utils.bath import WAM_discretization

KEYS = {"model", "property", "nm_omega", "grid", "output", "initial_guess"}
# property: (method, grid)
PROPERTIES = {
    "kernel": ("nmK", "time"),
    "spectrum": ("nmLambda", "frequency"),
}
# initial_guess.type: its keys
GUESS_KEYS = {
    "exact": {"type", "output"},
    "wam": {"type", "nbath", "grid", "wmax", "output"},
}


def debye_embedding(model) -> dict:
    """The prony embedder with the kernel η ωc exp(−ωc|t|) of Debye."""
    return {"type": "prony", "parameters": {
        "theta": float(np.sqrt(model.eta * model.omega_cut)),
        "gamma": float(model.omega_cut)}}


# model.type: its exact embedding, as an `embedder` section
EXACT_EMBEDDINGS = {"debye": debye_embedding}


def read_model(section) -> BaseSpectralDensity:
    """The model that the `model` section of the config defines. Its
    constructor receives `parameters` as keyword arguments and checks them.

    Raises:
        ConfigError: if the section has no known `type`, or `parameters`
            is not a dictionary
    """
    if not isinstance(section, dict):
        raise ConfigError(f"model must be a dictionary, got {section!r}")
    name = section.get("type")
    if not isinstance(name, str) or name not in SPECTRAL_MAP:
        raise ConfigError(
            f"model.type {name!r} is unknown; "
            f"the known types are {sorted(SPECTRAL_MAP)}")
    parameters = section.get("parameters")
    if not isinstance(parameters, dict):
        raise ConfigError(
            f"model.parameters must be a dictionary, got {parameters!r}")
    return SPECTRAL_MAP[name](**parameters)


def read_grid(
    section, config_dir: Path, name: str = "grid"
) -> npt.NDArray[np.floating]:
    """The grid that the section `name` of the config defines, either
    `linspace: [start, stop, num]` or the `path` of a one-column CSV file,
    relative to config_dir, in which `#` starts a comment.

    Raises:
        ConfigError: unless the section has exactly one of these keys, with
            a valid value
    """
    if not isinstance(section, dict) or \
            list(section) not in (["linspace"], ["path"]):
        raise ConfigError(
            f"{name} must have exactly one key, linspace or path, "
            f"got {section!r}")
    if "linspace" in section:
        values = section["linspace"]
        if (not isinstance(values, list) or len(values) != 3
                or not all(isinstance(v, numbers.Real)
                           and not isinstance(v, bool) for v in values)
                or not isinstance(values[2], int) or values[2] < 1):
            raise ConfigError(
                f"{name}.linspace must be [start, stop, num], with num a "
                f"positive integer, got {values!r}")
        return np.linspace(*values)
    if not isinstance(section["path"], str):
        raise ConfigError(
            f"{name}.path must be a string, got {section['path']!r}")
    # an absolute path replaces config_dir
    path = config_dir / section["path"]
    try:
        grid = np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as e:
        raise ConfigError(f"{name}.path: cannot read {path}: {e}") from None
    if grid.shape[1] != 1:
        raise ConfigError(
            f"{name}.path: {path} must have one column, "
            f"it has {grid.shape[1]}")
    return grid[:, 0]


def wam_embedding(
    section: dict, model: BaseSpectralDensity, nm_omega, config_dir: Path
) -> dict:
    """A `multi` of `gen2x2` embedders, one for each mode of the WAM
    discretization of Λₙ on the grid of the section. The mode ωⱼ has the
    damping rate γⱼ = ωⱼ − ωⱼ₋₁, with γ₁ = ω₂ − ω₁, or ω₁ for one mode.

    Raises:
        ConfigError: if `nbath`, `wmax` or `grid` is malformed
    """
    nbath = section.get("nbath")
    if isinstance(nbath, bool) or not isinstance(nbath, int) or nbath < 1:
        raise ConfigError(
            "initial_guess.nbath must be a positive integer, "
            f"got {nbath!r}")
    wmax = section.get("wmax", np.inf)
    if isinstance(wmax, bool) or not isinstance(wmax, numbers.Real):
        raise ConfigError(
            f"initial_guess.wmax must be a real number, got {wmax!r}")
    grid = read_grid(section.get("grid"), config_dir, "initial_guess.grid")
    weight, freqs = WAM_discretization(
        grid, model.nmLambda(nm_omega, grid), nbath, wmax)
    gammas = np.diff(freqs, prepend=0.0)
    if nbath > 1:
        gammas[0] = gammas[1]
    return {"type": "multi", "parameters": {"embedders": [
        {"type": "gen2x2",
         "parameters": TwoAuxEmbedder.from_harmonic(weight, w, g)}
        for w, g in zip(freqs, gammas)]}}


def read_initial_guess(
    section, model: BaseSpectralDensity, model_type: str, nm_omega,
    config_dir: Path
) -> dict:
    """The `embedder` section of a fit config that the `initial_guess`
    section defines: the exact embedding of the model, which needs ωₙ = 0,
    or that of wam_embedding.

    Raises:
        ConfigError: if the section is malformed, or the exact embedding is
            requested for a model without one or for ωₙ ≠ 0
    """
    if not isinstance(section, dict):
        raise ConfigError(
            f"initial_guess must be a dictionary, got {section!r}")
    kind = section.get("type")
    if not isinstance(kind, str) or kind not in GUESS_KEYS:
        raise ConfigError(
            f"initial_guess.type {kind!r} is unknown; "
            f"the known types are {sorted(GUESS_KEYS)}")
    unknown = set(section) - GUESS_KEYS[kind]
    if unknown:
        raise ConfigError(
            f"unknown keys {sorted(unknown)} in initial_guess of type "
            f"{kind}; its keys are {sorted(GUESS_KEYS[kind])}")
    if kind == "wam":
        return wam_embedding(section, model, nm_omega, config_dir)
    if model_type not in EXACT_EMBEDDINGS:
        raise ConfigError(
            f"model.type {model_type!r} has no exact embedding; "
            f"the models with one are {sorted(EXACT_EMBEDDINGS)}")
    if nm_omega != 0:
        raise ConfigError(
            f"the exact embedding needs nm_omega = 0, got {nm_omega!r}")
    return EXACT_EMBEDDINGS[model_type](model)


def main(args: argparse.Namespace) -> None:
    """Write the target that the YAML or JSON file args.config defines:

        model:                         # required
          type: debye                  # a key of SPECTRAL_MAP
          parameters: {eta: 0.7, omega_cut: 2.5}  # to the constructor
        property: spectrum             # required: kernel or spectrum
        nm_omega: 3.0                  # optional, default 0
        grid:                          # required, one of
          linspace: [0.0, 40.0, 4001]  #   passed to np.linspace
          path: grid.csv               #   one column, relative to the config
        output: target.csv             # optional, default target.csv,
                                       # relative to the working directory
        initial_guess:                 # optional
          type: wam                    # exact or wam
          nbath: 11                    # wam: number of gen2x2 embedders
          grid: {linspace: [0.0, 2.0, 2001]}  # wam: as above, for Λₙ
          wmax: 1.0                    # wam, optional: limit of WAM
          output: embedder.yaml        # optional, default embedder.yaml

    The kernel is Kₙ(t) and the spectrum Λₙ(ω), for the normal-mode
    frequency ωₙ = nm_omega, without rescaling. The output has the columns
    grid and target, separated by commas, with 17 significant digits, so a
    fit config reads it with `columns: [0, 1]`. A `#` header records the
    inputs.

    The initial guess is written as the `embedder` section of a fit config,
    in YAML, after a `#` header. `exact` is the exact embedding of the
    model, for ωₙ = 0 only. `wam` is described in wam_embedding.

    Raises:
        ConfigError: if the file has an unknown key, or a section is
            missing or malformed. Nothing is written then.
    """
    handler = ConfigHandler(args.config)
    config = handler.config
    unknown = set(config) - KEYS
    if unknown:
        raise ConfigError(
            f"unknown keys {sorted(unknown)}; "
            f"the known keys are {sorted(KEYS)}")
    model = read_model(config.get("model"))
    prop = config.get("property")
    if not isinstance(prop, str) or prop not in PROPERTIES:
        raise ConfigError(
            f"property {prop!r} is unknown; "
            f"the known properties are {sorted(PROPERTIES)}")
    method, grid_name = PROPERTIES[prop]
    config_dir = handler.config_path.parent
    grid = read_grid(config.get("grid"), config_dir)
    nm_omega = config.get("nm_omega", 0.0)
    section = config["model"]
    guess = config.get("initial_guess")
    embedder = None if guess is None else read_initial_guess(
        guess, model, section["type"], nm_omega, config_dir)

    target = getattr(model, method)(nm_omega, grid)

    model_line = (
        f"model: {section['type']} {json.dumps(section['parameters'])}")
    header = "\n".join([
        model_line,
        f"property: {prop}",
        f"nm_omega: {nm_omega}",
        f"columns: {grid_name}, {prop}",
    ])
    np.savetxt(config.get("output", "target.csv"),
               np.column_stack([grid, target]),
               fmt="%.17g", delimiter=",", header=header)
    if embedder is not None:
        options = {k: v for k, v in guess.items() if k != "type"}
        guess_header = [
            model_line,
            f"nm_omega: {nm_omega}",
            f"initial_guess: {guess['type']} {json.dumps(options)}",
        ]
        Path(guess.get("output", "embedder.yaml")).write_text(
            "".join(f"# {line}\n" for line in guess_header)
            + yaml.safe_dump({"embedder": embedder}, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write the target data of a fit from a model.")
    parser.add_argument(
        'config', type=str, help="Path to config file (YAML/JSON)")
    main(parser.parse_args())
