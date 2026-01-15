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
from pathlib import Path
from glefit.config.config_handler import ConfigHandler
from glefit.config.data_io import load_data
from glefit.embedding import BaseEmbedder, EMBEDDER_MAP
from glefit.merit import BaseProperty, PROPERTY_MAP
from glefit.optimization import Optimizer, OPT_MAP


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
        embedder_class: str = embedder_config["type"]
        EmbedderClass: BaseEmbedder = EMBEDDER_MAP[embedder_class]
        embedder: BaseEmbedder = EmbedderClass.from_dict(embedder_config["parameters"])
        
        # Get the data to define the objective function
        datasets: dict = handler.load_data()

        # Instantiate the objective function
        merit_config: dict = handler.get_merit_function_config()
        merit_class: str = merit_config["type"]
        PropertyClass: BaseProperty = PROPERTY_MAP[merit_class]
        merit: BaseProperty = PropertyClass.from_dict(merit_config["parameters"], datasets, embedder)

        # Instantiate the optimizer
        opt_config: dict = handler.get_optimization_config()
        opt_class: str = opt_config["type"]
        OptClass: Optimizer = OPT_MAP[opt_class]
        opt_parameters = opt_config.get("parameters", {})
        opt: Optimizer = OptClass(embedder, merit, **opt_parameters)

        return cls(embedder, merit, opt)
        

    def run(
        self,
        steps: int,
        options: dict
    ) -> bool:
        self.opt.run(steps=steps, options=options)
        return self.opt.converged()


def main(args: argparse.Namespace):
    handler: ConfigHandler = ConfigHandler(args.config)
    handler.validate()
    optimization = Optimization.from_config(handler)
    opt_config: dict = handler.get_optimization_config()
    optimization_options = opt_config["options"]
    max_iter = optimization_options.pop("max_iter")
    success = optimization.run(steps=max_iter, options=optimization_options)
    if success:
        print("YAY!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLE parameter optimization driver.")
    parser.add_argument('config', type=str, help="Path to config file (YAML/JSON)")

    args = parser.parse_args()
    main(args)
