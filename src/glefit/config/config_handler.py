#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Configuration management for GLE parameter optimization.

Handles loading YAML configs, validating structure, serializing/deserializing
checkpoints, and tracking optimization state.
"""

from __future__ import print_function, division, absolute_import
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import json
import copy
import numbers
import yaml
from .data_io import load_data, DataLoadError

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid or loading fails."""
    pass


class ConfigHandler:
    """Load, validate, and manage optimization configurations.
    
    Handles YAML input files, validation, and checkpoint serialization/deserialization.
    Converts relative paths to absolute paths relative to config file directory.
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        """Initialize configuration handler.
        
        Args:
            config_path: Path to YAML or JSON configuration file
            
        Raises:
            ConfigError: If file not found or parsing fails
        """
        self.config_path = Path(config_path).resolve()
        
        if not self.config_path.exists():
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        
        # any suffix but .json is read as YAML, which reads most JSON too
        if self.config_path.suffix == '.json':
            self._load_json()
        else:
            self._load_yaml()

        if self.config is None:
            raise ConfigError("Configuration file is empty")
        if not isinstance(self.config, dict):
            raise ConfigError(
                "The configuration must be a dictionary; got "
                f"{type(self.config).__name__}")

        # Resolve all relative paths to absolute paths
        self._resolve_paths()
        
        logger.info(f"Loaded configuration from {self.config_path}")

    def _load_yaml(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML config: {e}")
        except (IOError, OSError) as e:
            raise ConfigError(f"I/O error reading config: {e}")

    def _load_json(self) -> None:
        """Load configuration from JSON file (e.g., checkpoint)."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse JSON config: {e}")
        except (IOError, OSError) as e:
            raise ConfigError(f"I/O error reading config: {e}")

    def _resolve_paths(self) -> None:
        """Make the relative path of each external dataset absolute,
        against the config file's directory, in place.

        This runs before validate(), so it skips entries of the wrong type
        and leaves them for validate() and load_data() to report.
        """
        config_dir: Path = self.config_path.parent
        data = self.config.get('data')
        if not isinstance(data, dict):
            return
        for data_config in data.values():
            if (isinstance(data_config, dict)
                    and data_config.get('source') == 'external'
                    and isinstance(data_config.get('path'), str)):
                path = Path(data_config['path'])
                if not path.is_absolute():
                    data_config['path'] = str((config_dir / path).resolve())

    def validate(self) -> None:
        """Validate configuration structure and content.

        Raises:
            ConfigError: If validation fails
        """
        required_keys = {'data', 'merit_function', 'embedder', 'optimization'}
        missing = required_keys - set(self.config.keys())
        if missing:
            raise ConfigError(
                f"Configuration missing required keys: {missing}. "
                f"Required keys: {required_keys}"
            )

        # Validate data section
        if not isinstance(self.config.get('data'), dict):
            raise ConfigError("'data' must be a dictionary")
        if len(self.config['data']) == 0:
            raise ConfigError("'data' must contain at least one named dataset")
        
        # Validate each dataset
        for data_name, data_config in self.config['data'].items():
            if not isinstance(data_config, dict):
                raise ConfigError(f"Data '{data_name}' must be a dictionary")
        
        # Validate merit function
        merit = self.config.get('merit_function')
        if not isinstance(merit, dict):
            raise ConfigError("'merit_function' must be a dictionary")
        if 'type' not in merit:
            raise ConfigError("'merit_function.type' is required")
        if 'parameters' not in merit:
            raise ConfigError("'merit_function.parameters' is required")
        if not isinstance(merit['parameters'], dict):
            raise ConfigError(
                "'merit_function.parameters' must be a dictionary")
        
        # Validate embedder
        if not isinstance(self.config.get('embedder'), dict):
            raise ConfigError("'embedder' must be a dictionary")
        if 'type' not in self.config['embedder']:
            raise ConfigError("'embedder.type' is required")
        if 'parameters' not in self.config['embedder']:
            raise ConfigError("'embedder.parameters' is required")
        if not isinstance(self.config['embedder']['parameters'], dict):
            raise ConfigError("'embedder.parameters' must be a dictionary")

        # Validate optimization
        if not isinstance(self.config.get('optimization'), dict):
            raise ConfigError("'optimization' must be a dictionary")
        if 'type' not in self.config['optimization']:
            raise ConfigError("'optimization.type' is required")
        if 'parameters' not in self.config['optimization']:
            raise ConfigError("'optimization.parameters' is required")
        if not isinstance(self.config['optimization']['parameters'], dict):
            raise ConfigError(
                "'optimization.parameters' must be a dictionary")
        if 'options' not in self.config['optimization']:
            raise ConfigError("'optimization.options' is required")
        options = self.config['optimization']['options']
        if not isinstance(options, dict):
            raise ConfigError("'optimization.options' must be a dictionary")
        if 'max_iter' not in options:
            raise ConfigError("'optimization.options.max_iter' is required")
        max_iter = options['max_iter']
        # bool is an Integral, but true is no number of steps
        if (isinstance(max_iter, bool)
                or not isinstance(max_iter, numbers.Integral)
                or max_iter < 0):
            raise ConfigError(
                "'optimization.options.max_iter' must be a non-negative "
                f"integer, got {max_iter!r}")

        logger.info("Configuration validation passed")

    def load_data(self) -> Dict[str, tuple]:
        """Load all datasets specified in configuration.
        
        Returns:
            Dictionary mapping dataset names to (grid, target) tuples
            
        Raises:
            ConfigError: If any dataset fails to load
        """
        datasets = {}
        
        for data_name, data_config in self.config['data'].items():
            try:
                grid, target = load_data(data_config, data_name)
                datasets[data_name] = (grid, target)
                logger.info(f"Successfully loaded dataset '{data_name}'")
            except DataLoadError as e:
                raise ConfigError(f"Failed to load data '{data_name}': {e}")
        
        return datasets

    def get_merit_function_config(self) -> Dict[str, Any]:
        """Get merit function configuration.
        
        Returns:
            Merit function configuration dictionary
        """
        return copy.deepcopy(self.config['merit_function'])

    def get_embedder_config(self) -> Dict[str, Any]:
        """Get embedder configuration.
        
        Returns:
            Embedder configuration dictionary
        """
        return copy.deepcopy(self.config['embedder'])

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration.
        
        Returns:
            Optimization configuration dictionary
        """
        return copy.deepcopy(self.config['optimization'])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for serialization).
        
        Returns:
            Configuration as dictionary
        """
        return copy.deepcopy(self.config)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any], config_path: Optional[Path] = None) -> 'ConfigHandler':
        """Create ConfigHandler from dictionary (e.g., from checkpoint).
        
        Args:
            config_dict: Configuration dictionary
            config_path: Optional path to associate with this config (for relative path resolution)
            
        Returns:
            ConfigHandler instance
            
        """
        handler = ConfigHandler.__new__(ConfigHandler)
        handler.config = copy.deepcopy(config_dict)
        handler.config_path = Path(config_path).resolve() if config_path else None
        return handler
