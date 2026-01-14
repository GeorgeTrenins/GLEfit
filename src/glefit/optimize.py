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

parser = argparse.ArgumentParser(description="GLE parameter optimization driver.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--config', type=str, help="Path to config file (YAML/JSON)")
group.add_argument('--chk', type=str, help="Path to checkpoint file JSON")

def main(args: argparse.Namespace):
    

    if args.config is not None:
        handler: ConfigHandler = ConfigHandler(args.config)
        optimization_state = None
    else:
        handler: ConfigHandler
        optimization_state: dict
        handler, optimization_state = ConfigHandler.load_checkpoint(args.chk)
    handler.validate()

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
    optimization_options = opt_config["options"]
    max_iter = optimization_options.pop("max_iter")
    opt.run(steps=max_iter, options=optimization_options)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
