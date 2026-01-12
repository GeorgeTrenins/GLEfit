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
from datetime import datetime

import yaml
import numpy as np

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
        
        # Determine file type and load
        if self.config_path.suffix in ['.yaml', '.yml']:
            self._load_yaml()
        elif self.config_path.suffix == '.json':
            self._load_json()
        else:
            # Try YAML first, then JSON
            try:
                self._load_yaml()
            except yaml.YAMLError:
                try:
                    self._load_json()
                except json.JSONDecodeError as e:
                    raise ConfigError(f"Failed to parse config as YAML or JSON: {e}")
        
        if self.config is None:
            raise ConfigError("Configuration file is empty")
        
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
        """Convert relative paths to absolute paths relative to config file directory.
        
        Modifies config dict in-place to store absolute paths for:
        - project.output_dir
        - data[*].path (for external sources)
        - restart.restart_file
        """
        config_dir: Path = self.config_path.parent
        
        # Resolve project output directory
        if 'project' in self.config and 'output_dir' in self.config['project']:
            output_dir: Path = Path(self.config['project']['output_dir'])
            if not output_dir.is_absolute():
                self.config['project']['output_dir'] = str((config_dir / output_dir).resolve())
        
        # Resolve data file paths
        if 'data' in self.config:
            for _, data_config in self.config['data'].items():
                if isinstance(data_config, dict) and data_config.get('source') == 'external':
                    if 'path' in data_config:
                        path: Path = Path(data_config['path'])
                        if not path.is_absolute():
                            data_config['path'] = str((config_dir / path).resolve())
        
        # Resolve restart file path
        if 'restart' in self.config and self.config['restart'].get('enabled'):
            if 'restart_file' in self.config['restart']:
                path = Path(self.config['restart']['restart_file'])
                if not path.is_absolute():
                    self.config['restart']['restart_file'] = str((config_dir / path).resolve())

    @staticmethod
    def _validate_config_structure(config: Dict[str, Any]) -> None:
        """Validate that required top-level keys are present.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ConfigError: If required keys are missing
        """
        required_keys = {'project', 'data', 'merit_function', 'embedder', 'optimization'}
        missing = required_keys - set(config.keys())
        
        if missing:
            raise ConfigError(
                f"Configuration missing required keys: {missing}. "
                f"Required keys: {required_keys}"
            )

    def validate(self) -> None:
        """Validate configuration structure and content.
        
        Raises:
            ConfigError: If validation fails
        """
        self._validate_config_structure(self.config)
        self.config: dict

        # Validate project section
        if not isinstance(self.config.get('project'), dict):
            raise ConfigError("'project' must be a dictionary")
        if 'name' not in self.config['project']:
            raise ConfigError("'project.name' is required")
        if 'output_dir' not in self.config['project']:
            raise ConfigError("'project.output_dir' is required")
        
        # Validate data section
        if not isinstance(self.config.get('data'), dict):
            raise ConfigError("'data' must be a dictionary")
        if len(self.config['data']) == 0:
            raise ConfigError("'data' must contain at least one named dataset")
        
        # Validate each dataset
        for data_name, data_config in self.config['data'].items():
            if not isinstance(data_config, dict):
                raise ConfigError(f"Data '{data_name}' must be a dictionary")
            if 'source' not in data_config:
                raise ConfigError(f"Data '{data_name}' missing 'source' key")
        
        # Validate merit function
        merit = self.config.get('merit_function')
        if not isinstance(merit, dict):
            raise ConfigError("'merit_function' must be a dictionary")
        #TODO: modify later for the case of composite merit functions
        if 'property' not in merit:
            raise ConfigError("'merit_function.property' is required")
        if 'data_ref' not in merit:
            raise ConfigError("'merit_function.data_ref' is required")
        
        # Validate data reference exists
        if merit['data_ref'] not in self.config['data']:
            raise ConfigError(
                f"Merit function references non-existent data '{merit['data_ref']}'. "
                f"Available datasets: {list(self.config['data'].keys())}"
            )
        
        # Validate embedder
        if not isinstance(self.config.get('embedder'), dict):
            raise ConfigError("'embedder' must be a dictionary")
        if 'type' not in self.config['embedder']:
            raise ConfigError("'embedder.type' is required")
        
        # Validate optimization
        if not isinstance(self.config.get('optimization'), dict):
            raise ConfigError("'optimization' must be a dictionary")
        if 'optimizer' not in self.config['optimization']:
            raise ConfigError("'optimization.optimizer' is required")
        if 'max_steps' not in self.config['optimization']:
            raise ConfigError("'optimization.max_steps' is required")
        
        # Validate restart section if present
        if 'restart' in self.config:
            restart = self.config['restart']
            if not isinstance(restart, dict):
                raise ConfigError("'restart' must be a dictionary")
            if 'enabled' in restart and not isinstance(restart['enabled'], bool):
                raise ConfigError("'restart.enabled' must be a boolean")
            
            # If restart enabled, validate restart_file exists
            if restart.get('enabled', False):
                if 'restart_file' not in restart:
                    raise ConfigError("'restart.restart_file' is required when restart is enabled")
                restart_path = Path(restart['restart_file'])
                if not restart_path.exists():
                    raise ConfigError(f"Restart file not found: {restart_path}")
        
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

    def get_project_config(self) -> Dict[str, Any]:
        """Get project metadata.
        
        Returns:
            Project configuration dictionary
        """
        return copy.deepcopy(self.config['project'])

    def get_restart_config(self) -> Optional[Dict[str, Any]]:
        """Get restart configuration if present.
        
        Returns:
            Restart configuration dictionary or None if not configured
        """
        restart = self.config.get('restart')
        if restart is None:
            return None
        return copy.deepcopy(restart)

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

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        optimization_state: Dict[str, Any]
    ) -> None:
        """Save configuration and optimization state as JSON checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            optimization_state: Dict with required keys:
                - completed_steps: Number of steps completed
                - max_steps: Maximum steps configured
                - timestamp: ISO 8601 timestamp
                - converged: Whether optimization converged
                - merit_value: Current merit function value
                - gradient_norm: Current gradient norm
                - gradient_max_norm: Current gradient max-norm
                
        Raises:
            IOError: If writing fails
            ValueError: If optimization_state missing required keys
        """
        filepath = Path(filepath)
        
        # Validate optimization_state
        #TODO: add the missing keys for embedder state
        required_keys = {'completed_steps', 'max_steps', 'timestamp', 'converged', 
                        'merit_value', 'gradient_norm', 'gradient_max_norm'}
        missing = required_keys - set(optimization_state.keys())
        if missing:
            raise ValueError(f"optimization_state missing required keys: {missing}")
        
        checkpoint = copy.deepcopy(self.config)
        checkpoint['optimization_state'] = optimization_state
        
        try:
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=_json_encoder)
            logger.info(f"Saved checkpoint to {filepath}")
        except (IOError, OSError, TypeError) as e:
            raise IOError(f"Failed to save checkpoint: {e}")

    @staticmethod
    def load_checkpoint(filepath: Union[str, Path]) -> tuple:
        """Load configuration and optimization state from JSON checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Tuple of (ConfigHandler, optimization_state_dict)
            
        Raises:
            IOError: If loading fails
            ConfigError: If configuration is invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise IOError(f"Checkpoint file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
        except (IOError, OSError, json.JSONDecodeError) as e:
            raise IOError(f"Failed to load checkpoint: {e}")
        
        # Extract optimization state (must be present in checkpoints)
        if 'optimization_state' not in checkpoint:
            raise ConfigError("Checkpoint missing 'optimization_state' section")
        
        optimization_state = checkpoint.pop('optimization_state')
        
        # Reconstruct ConfigHandler from checkpoint config
        try:
            handler = ConfigHandler.from_dict(checkpoint, config_path=filepath)
            logger.info(f"Loaded checkpoint from {filepath}")
            return handler, optimization_state
        except ConfigError as e:
            raise ConfigError(f"Checkpoint contains invalid configuration: {e}")


def _json_encoder(obj: Any) -> Any:
    """Custom JSON encoder for numpy types and other non-standard objects.
    
    Args:
        obj: Object to encode
        
    Returns:
        JSON-serializable representation
        
    Raises:
        TypeError: If object cannot be serialized
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")