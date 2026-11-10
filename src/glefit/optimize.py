#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   optimize.py
@Time    :   2026/01/13 10:32:37
@Author  :   George Trenins
@Desc    :   Top-level driver for GLE parameter optimization.
'''


from __future__ import print_function, division, absolute_import
import argparse
import pickle
from glefit.config.config_handler import ConfigError, ConfigHandler
from glefit.embedding import BaseEmbedder, EMBEDDER_MAP
from glefit.merit import BaseProperty, PROPERTY_MAP
from glefit.optimization import Optimizer, OPT_MAP

EMBEDDER_FILE = "embedder_parameters.pkl"
AP_FILE = "optimized_Ap_matrix.json"


def write_propagation_matrix(emb: BaseEmbedder) -> None:
    """Write emb.propagation_matrix(), the form of Ap best suited to
    simulations, to AP_FILE in the working directory, with 8 significant
    digits."""
    rows = ",\n".join(
        "          [" + ",".join(f"{x:14.7e}" for x in row) + "]"
        for row in emb.propagation_matrix().tolist())
    with open(AP_FILE, 'w') as f:
        f.write('{\n"Ap": [\n' + rows + '\n      ]\n}\n')


def write_embedder(emb: BaseEmbedder, success: bool) -> None:
    """Pickle emb.to_dict(), with `success`, to EMBEDDER_FILE in the
    working directory.

    `EMBEDDER_MAP[d["type"]].from_dict(d["parameters"])` rebuilds the
    embedder from the dictionary d that the file holds.
    """
    record = emb.to_dict()
    record["success"] = bool(success)
    with open(EMBEDDER_FILE, "wb") as f:
        pickle.dump(record, f)


def _registered(registry: dict, section: str, name) -> type:
    """The class that `registry` holds under `name`, the type of a config
    section.

    Raises:
        ConfigError: if `name` is not a type in `registry`.
    """
    if not isinstance(name, str) or name not in registry:
        raise ConfigError(
            f"{section}.type {name!r} is unknown; "
            f"the known types are {sorted(registry)}")
    return registry[name]


class Optimization:

    def __init__(
        self,
        embedder: BaseEmbedder,
        objective: BaseProperty,
        optimizer: Optimizer
    ) -> None:
        self.emb = embedder
        self.merit = objective
        self.opt = optimizer

    @classmethod
    def from_config(
        cls,
        handler: ConfigHandler
    ) -> "Optimization":
        # Instantiate the embedder
        embedder_config: dict = handler.get_embedder_config()
        EmbedderClass: BaseEmbedder = _registered(
            EMBEDDER_MAP, "embedder", embedder_config["type"])
        embedder: BaseEmbedder = EmbedderClass.from_dict(embedder_config["parameters"])
        
        # Get the data to define the objective function
        datasets: dict = handler.load_data()

        # Instantiate the objective function
        merit_config: dict = handler.get_merit_function_config()
        PropertyClass: BaseProperty = _registered(
            PROPERTY_MAP, "merit_function", merit_config["type"])
        merit: BaseProperty = PropertyClass.from_dict(merit_config["parameters"], datasets, embedder)

        # Instantiate the optimizer
        opt_config: dict = handler.get_optimization_config()
        OptClass: Optimizer = _registered(
            OPT_MAP, "optimization", opt_config["type"])
        # validate() requires optimization.parameters
        opt: Optimizer = OptClass(embedder, merit, **opt_config["parameters"])

        return cls(embedder, merit, opt)

    def run(
        self,
        steps: int,
        options: dict
    ) -> bool:
        """Run the optimizer, and return whether the fit converged."""
        return self.opt.run(steps=steps, options=options)


def main(args: argparse.Namespace):
    handler: ConfigHandler = ConfigHandler(args.config)
    handler.validate()
    optimization = Optimization.from_config(handler)
    opt_config: dict = handler.get_optimization_config()
    optimization_options = opt_config["options"]
    max_iter = optimization_options.pop("max_iter")
    success = False
    try:
        success = optimization.run(
            steps=max_iter, options=optimization_options)
    finally:
        # also if the run fails, to keep the parameters it stopped at
        write_embedder(optimization.emb, success)
    if success:
        print("YAY!")
        write_propagation_matrix(optimization.emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLE parameter optimization driver.")
    parser.add_argument('config', type=str, help="Path to config file (YAML/JSON)")

    args = parser.parse_args()
    main(args)
