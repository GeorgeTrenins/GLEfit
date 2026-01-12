#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Tests for configuration handler functionality."""

from __future__ import print_function, division, absolute_import
import pytest
import numpy as np
from pathlib import Path
from glefit.config.config_handler import ConfigHandler, ConfigError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def minimal_valid_config():
    """Return minimal valid configuration as dictionary."""
    return {
        'project': {
            'name': 'test_project',
            'output_dir': './output'
        },
        'data': {
            'test_data': {
                'source': 'inline',
                'grid': [0.0, 0.5, 1.0],
                'target': [1.0, 0.5, 0.1]
            }
        },
        'merit_function': {
            'property': 'MemoryKernel',
            'data_ref': 'test_data',
            'weight': 1.0,
            'metric': 'squared'
        },
        'embedder': {
            'type': 'PronyEmbedder',
            'parameters': {
                'theta': 1.0,
                'gamma': 0.5
            }
        },
        'optimization': {
            'optimizer': 'NewtonRaphson',
            'max_steps': 100,
            'options': {
                'gtol': 0.0001
            }
        }
    }


# ============================================================================
# GROUP 1: BASIC LOADING TESTS
# ============================================================================

class TestConfigLoading:
    """Test configuration file loading and parsing."""

    def test_load_yaml_file(self, fixtures_dir):
        """Load valid YAML configuration."""
        yaml_path = fixtures_dir / "minimal_valid.yaml"
        handler = ConfigHandler(yaml_path)
        assert handler.config is not None
        assert handler.config_path.exists()
        assert handler.config_path == yaml_path.resolve()
        assert handler.config['project']['name'] == 'test_project'

    def test_load_json_file(self, fixtures_dir):
        """Load valid JSON configuration (checkpoint format)."""
        json_path = fixtures_dir / "minimal_valid.json"
        handler = ConfigHandler(json_path)
        assert handler.config is not None
        assert handler.config_path.exists()
        assert handler.config_path == json_path.resolve()
        assert handler.config['project']['name'] == 'test_project'

    def test_load_without_extension(self, fixtures_dir):
        """Load file without .yaml/.json extension (tries both parsers)."""
        no_ext_path = fixtures_dir / "no_extension"
        handler = ConfigHandler(no_ext_path)
        assert handler.config is not None
        assert handler.config['project']['name'] == 'test_project'

    def test_file_not_found(self):
        """Raise ConfigError for missing file."""
        with pytest.raises(ConfigError, match="not found"):
            ConfigHandler('/nonexistent/path/config.yaml')

    def test_empty_file(self, fixtures_dir):
        """Raise ConfigError for empty file."""
        empty_path = fixtures_dir / "empty.yaml"
        with pytest.raises(ConfigError, match="empty"):
            ConfigHandler(empty_path)

    def test_invalid_yaml_syntax(self, fixtures_dir):
        """Raise ConfigError for malformed YAML."""
        invalid_yaml = fixtures_dir / "invalid_yaml.yaml"
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            ConfigHandler(invalid_yaml)

    def test_invalid_json_syntax(self, fixtures_dir):
        """Raise ConfigError for malformed JSON."""
        invalid_json = fixtures_dir / "invalid_json.json"
        with pytest.raises(ConfigError, match="Failed to parse JSON"):
            ConfigHandler(invalid_json)


# ============================================================================
# GROUP 2: PATH RESOLUTION TESTS
# ============================================================================

class TestPathResolution:
    """Test path resolution relative to config file directory."""

    def test_resolve_relative_output_dir(self, fixtures_dir):
        """Convert relative project.output_dir to absolute."""
        config_path = fixtures_dir / "relative_paths.yaml"
        handler = ConfigHandler(config_path)
        
        # Check that output_dir was resolved to absolute path
        output_dir = Path(handler.config['project']['output_dir'])
        assert output_dir.is_absolute()
        
        # Check it's resolved relative to fixtures_dir
        expected = (fixtures_dir / "output").resolve()
        assert output_dir == expected

    def test_preserve_absolute_output_dir(self, fixtures_dir):
        """Preserve absolute project.output_dir."""
        config_path = fixtures_dir / "absolute_paths.yaml"
        handler = ConfigHandler(config_path)
        
        # Check that absolute path is preserved
        output_dir = handler.config['project']['output_dir']
        assert output_dir == "/tmp/absolute_output"

    def test_resolve_relative_data_path(self, fixtures_dir):
        """Convert relative data file paths to absolute."""
        config_path = fixtures_dir / "relative_paths.yaml"
        handler = ConfigHandler(config_path)
        
        # Check that data path was resolved to absolute
        data_path = Path(handler.config['data']['kernel_data']['path'])
        assert data_path.is_absolute()
        
        # Check it's resolved relative to fixtures_dir
        expected = (fixtures_dir / "test_data.csv").resolve()
        assert data_path == expected

    def test_preserve_absolute_data_path(self, fixtures_dir):
        """Preserve absolute data file paths."""
        config_path = fixtures_dir / "absolute_paths.yaml"
        handler = ConfigHandler(config_path)
        
        # Check that absolute path is preserved
        data_path = handler.config['data']['kernel_data']['path']
        assert data_path == "/tmp/absolute_data.csv"

    def test_resolve_relative_restart_file(self, fixtures_dir):
        """Convert relative restart file path to absolute."""
        config_path = fixtures_dir / "with_restart.yaml"
        handler = ConfigHandler(config_path)
        
        # Check that restart_file was resolved to absolute
        restart_file = Path(handler.config['restart']['restart_file'])
        assert restart_file.is_absolute()
        
        # Check it's resolved relative to fixtures_dir
        expected = (fixtures_dir / "minimal_valid.json").resolve()
        assert restart_file == expected

    def test_inline_data_no_path_resolution(self, fixtures_dir):
        """Inline data sources don't trigger path resolution."""
        config_path = fixtures_dir / "inline_data.yaml"
        handler = ConfigHandler(config_path)
        
        # Check that inline data doesn't have a 'path' key
        assert 'path' not in handler.config['data']['kernel_data']
        
        # Check that inline data has grid and target
        assert 'grid' in handler.config['data']['kernel_data']
        assert 'target' in handler.config['data']['kernel_data']


# ============================================================================
# GROUP 3: VALIDATION TESTS
# ============================================================================

class TestConfigValidation:
    """Test configuration validation logic."""

    def test_valid_minimal_config(self, fixtures_dir):
        """Accept minimal valid configuration."""
        config_path = fixtures_dir / "minimal_valid.yaml"
        handler = ConfigHandler(config_path)
        # Should not raise any errors
        handler.validate()

    def test_missing_project_key(self, fixtures_dir):
        """Reject config missing 'project' key."""
        config_path = fixtures_dir / "missing_project.yaml"
        handler = ConfigHandler(config_path)
        with pytest.raises(ConfigError, match="missing required keys"):
            handler.validate()

    def test_missing_data_key(self, fixtures_dir):
        """Reject config missing 'data' key."""
        config_path = fixtures_dir / "missing_data.yaml"
        handler = ConfigHandler(config_path)
        with pytest.raises(ConfigError, match="missing required keys"):
            handler.validate()

    def test_missing_merit_function_key(self, fixtures_dir):
        """Reject config missing 'merit_function' key."""
        config_path = fixtures_dir / "missing_merit_function.yaml"
        handler = ConfigHandler(config_path)
        with pytest.raises(ConfigError, match="missing required keys"):
            handler.validate()

    def test_missing_embedder_key(self, fixtures_dir):
        """Reject config missing 'embedder' key."""
        config_path = fixtures_dir / "missing_embedder.yaml"
        handler = ConfigHandler(config_path)
        with pytest.raises(ConfigError, match="missing required keys"):
            handler.validate()

    def test_missing_optimization_key(self, fixtures_dir):
        """Reject config missing 'optimization' key."""
        config_path = fixtures_dir / "missing_optimization.yaml"
        handler = ConfigHandler(config_path)
        with pytest.raises(ConfigError, match="missing required keys"):
            handler.validate()

    def test_invalid_project_type(self, fixtures_dir):
        """Reject invalid project section (not a dict)."""
        config_path = fixtures_dir / "invalid_project_type.yaml"
        with pytest.raises(ConfigError, match="'project' must be a dictionary"):
            ConfigHandler(config_path).validate()

    def test_missing_project_name(self, fixtures_dir):
        """Reject config missing project.name."""
        config_path = fixtures_dir / "missing_project_name.yaml"
        with pytest.raises(ConfigError, match="'project.name' is required"):
            ConfigHandler(config_path).validate()

    def test_missing_project_output_dir(self, fixtures_dir):
        """Reject config missing project.output_dir."""
        config_path = fixtures_dir / "missing_project_output_dir.yaml"
        with pytest.raises(ConfigError, match="'project.output_dir' is required"):
            ConfigHandler(config_path).validate()

    def test_empty_data_section(self, fixtures_dir):
        """Reject config with empty data dict."""
        config_path = fixtures_dir / "empty_data.yaml"
        with pytest.raises(ConfigError, match="must contain at least one"):
            ConfigHandler(config_path).validate()

    def test_missing_data_source(self, fixtures_dir):
        """Reject dataset missing 'source' key."""
        config_path = fixtures_dir / "missing_data_source.yaml"
        with pytest.raises(ConfigError, match="missing 'source' key"):
            ConfigHandler(config_path).validate()

    def test_invalid_merit_function_data_ref(self, fixtures_dir):
        """Reject merit function referencing non-existent dataset."""
        config_path = fixtures_dir / "invalid_data_ref.yaml"
        with pytest.raises(ConfigError, match="non-existent data"):
            ConfigHandler(config_path).validate()

    def test_missing_merit_property(self, fixtures_dir):
        """Reject merit function missing 'property' key."""
        config_path = fixtures_dir / "missing_merit_property.yaml"
        with pytest.raises(ConfigError, match="'merit_function.property' is required"):
            ConfigHandler(config_path).validate()

    def test_missing_merit_data_ref(self, fixtures_dir):
        """Reject merit function missing 'data_ref' key."""
        config_path = fixtures_dir / "missing_merit_data_ref.yaml"
        with pytest.raises(ConfigError, match="'merit_function.data_ref' is required"):
            ConfigHandler(config_path).validate()

    def test_missing_embedder_type(self, fixtures_dir):
        """Reject embedder missing 'type' key."""
        config_path = fixtures_dir / "missing_embedder_type.yaml"
        with pytest.raises(ConfigError, match="'embedder.type' is required"):
            ConfigHandler(config_path).validate()

    def test_missing_optimizer(self, fixtures_dir):
        """Reject optimization missing 'optimizer' key."""
        config_path = fixtures_dir / "missing_optimizer.yaml"
        with pytest.raises(ConfigError, match="'optimization.optimizer' is required"):
            ConfigHandler(config_path).validate()

    def test_missing_max_steps(self, fixtures_dir):
        """Reject optimization missing 'max_steps' key."""
        config_path = fixtures_dir / "missing_max_steps.yaml"
        with pytest.raises(ConfigError, match="'optimization.max_steps' is required"):
            ConfigHandler(config_path).validate()

    def test_invalid_restart_enabled(self, fixtures_dir):
        """Reject restart.enabled that is not boolean."""
        config_path = fixtures_dir / "invalid_restart_enabled.yaml"
        with pytest.raises(ConfigError, match="'restart.enabled' must be a boolean"):
            ConfigHandler(config_path).validate()

    def test_restart_enabled_missing_file(self, fixtures_dir):
        """Reject restart.enabled=true without restart_file."""
        config_path = fixtures_dir / "missing_restart_file.yaml"
        with pytest.raises(ConfigError, match="'restart.restart_file' is required"):
            ConfigHandler(config_path).validate()

    def test_restart_enabled_file_not_found(self, fixtures_dir):
        """Reject restart.enabled=true with non-existent restart_file."""
        config_path = fixtures_dir / "invalid_restart_file.yaml"
        with pytest.raises(ConfigError, match="Restart file not found"):
            ConfigHandler(config_path).validate()


# ============================================================================
# GROUP 4: DATA LOADING TESTS
# ============================================================================

class TestDataLoading:
    """Test loading datasets via ConfigHandler.load_data()."""

    def test_load_single_external_csv(self, fixtures_dir):
        """Load single external CSV dataset."""
        config_path = fixtures_dir / "minimal_valid.yaml"
        handler = ConfigHandler(config_path)
        handler.validate()
        datasets = handler.load_data()
        assert 'test_data' in datasets
        grid, target = datasets['test_data']
        assert len(grid) == 3
        assert np.allclose(grid, [0.0, 0.5, 1.0])
        assert np.allclose(target, [1.0, 0.5, 0.1])

    def test_load_multiple_datasets(self, fixtures_dir):
        """Load multiple datasets from config."""
        # Create a config with two datasets, one external, one inline
        config = {
            'project': {'name': 'multi', 'output_dir': './output'},
            'data': {
                'csv_data': {
                    'source': 'external',
                    'path': str(fixtures_dir / 'test_data.csv'),
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ','
                },
                'inline_data': {
                    'source': 'inline',
                    'grid': [0.0, 0.5, 1.0],
                    'target': [1.0, 0.5, 0.1]
                }
            },
            'merit_function': {
                'property': 'MemoryKernel',
                'data_ref': 'csv_data',
                'weight': 1.0,
                'metric': 'squared'
            },
            'embedder': {'type': 'PronyEmbedder', 'parameters': {'theta': 1.0, 'gamma': 0.5}},
            'optimization': {'optimizer': 'NewtonRaphson', 'max_steps': 100, 'options': {'gtol': 0.0001}}
        }
        handler = ConfigHandler.from_dict(config)
        handler.validate()
        datasets = handler.load_data()
        assert 'csv_data' in datasets
        assert 'inline_data' in datasets
        grid_csv, target_csv = datasets['csv_data']
        grid_inline, target_inline = datasets['inline_data']
        assert np.allclose(grid_csv, [0.0, 0.5, 1.0])
        assert np.allclose(target_csv, [1.0, 0.5, 0.1])
        assert np.allclose(grid_inline, [0.0, 0.5, 1.0])
        assert np.allclose(target_inline, [1.0, 0.5, 0.1])

    def test_load_inline_dataset(self, fixtures_dir):
        """Load inline dataset from config."""
        config_path = fixtures_dir / "inline_data.yaml"
        handler = ConfigHandler(config_path)
        handler.validate()
        datasets = handler.load_data()
        assert 'kernel_data' in datasets
        grid, target = datasets['kernel_data']
        assert np.allclose(grid, [0.0, 0.5, 1.0])
        assert np.allclose(target, [1.0, 0.5, 0.1])

    def test_load_data_failure_propagates(self, fixtures_dir):
        """Raise ConfigError if data loading fails (bad file path)."""
        # Use a config with a non-existent CSV file
        config = {
            'project': {'name': 'fail', 'output_dir': './output'},
            'data': {
                'bad_data': {
                    'source': 'external',
                    'path': str(fixtures_dir / 'nonexistent.csv'),
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ','
                }
            },
            'merit_function': {
                'property': 'MemoryKernel',
                'data_ref': 'bad_data',
                'weight': 1.0,
                'metric': 'squared'
            },
            'embedder': {'type': 'PronyEmbedder', 'parameters': {'theta': 1.0, 'gamma': 0.5}},
            'optimization': {'optimizer': 'NewtonRaphson', 'max_steps': 100, 'options': {'gtol': 0.0001}}
        }
        handler = ConfigHandler.from_dict(config)
        handler.validate()
        with pytest.raises(ConfigError, match="Failed to load data"):
            handler.load_data()


# ============================================================================
# GROUP 5: FROM_DICT AND TO_DICT TESTS
# ============================================================================

class TestConfigFromDict:
    """Test ConfigHandler.from_dict() and to_dict() methods."""

    def test_from_dict_valid(self, minimal_valid_config):
        """from_dict() creates handler from valid dictionary."""
        handler = ConfigHandler.from_dict(minimal_valid_config)
        handler.validate()
        assert handler.config['project']['name'] == minimal_valid_config['project']['name']

    def test_from_dict_invalid(self, minimal_valid_config):
        """from_dict() raises ConfigError for invalid dictionary."""
        bad_config = dict(minimal_valid_config)
        bad_config.pop('project')
        handler = ConfigHandler.from_dict(bad_config)
        with pytest.raises(ConfigError, match="missing required keys"):
            handler.validate()

    def test_from_dict_config_path(self, minimal_valid_config, fixtures_dir):
        """from_dict() sets config_path if provided."""
        config_path = fixtures_dir / "minimal_valid.yaml"
        handler = ConfigHandler.from_dict(minimal_valid_config, config_path=config_path)
        assert handler.config_path == config_path.resolve()

    def test_to_dict_returns_deep_copy(self, minimal_valid_config):
        """to_dict() returns a deep copy (modifying it does not affect handler)."""
        handler = ConfigHandler.from_dict(minimal_valid_config)
        config_copy = handler.to_dict()
        config_copy['project']['name'] = "changed"
        assert handler.config['project']['name'] != "changed"


# ============================================================================
# GROUP 6: CHECKPOINT SAVE/LOAD TESTS
# ============================================================================

import json
from datetime import datetime

class TestCheckpointIO:
    """Test checkpoint saving and loading functionality."""

    def test_save_and_load_checkpoint_round_trip(self, tmp_path, fixtures_dir):
        """Save then load checkpoint preserves config and optimization_state."""
        config_path = fixtures_dir / "minimal_valid.yaml"
        handler = ConfigHandler(config_path)
        handler.validate()
        optimization_state = {
            'completed_steps': 5,
            'max_steps': 100,
            'timestamp': datetime.now().isoformat() + "Z",
            'converged': False,
            'merit_value': 0.123,
            'gradient_norm': 0.01,
            'gradient_max_norm': 0.02
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        handler.save_checkpoint(checkpoint_path, optimization_state)
        assert checkpoint_path.exists()

        # Now load the checkpoint
        loaded_handler, loaded_state = ConfigHandler.load_checkpoint(checkpoint_path)
        loaded_handler.validate()
        assert loaded_state == optimization_state
        # Config sections should match
        assert loaded_handler.config['project']['name'] == handler.config['project']['name']

    def test_checkpoint_requires_optimization_state(self, tmp_path, minimal_valid_config):
        """Raise ConfigError if checkpoint missing optimization_state."""
        checkpoint_path = tmp_path / "bad_checkpoint.json"
        # Save config without optimization_state
        with open(checkpoint_path, 'w') as f:
            json.dump(minimal_valid_config, f)
        with pytest.raises(ConfigError, match="missing 'optimization_state'"):
            ConfigHandler.load_checkpoint(checkpoint_path)

    def test_checkpoint_file_not_found(self, tmp_path):
        """Raise IOError for missing checkpoint file."""
        missing_path = tmp_path / "does_not_exist.json"
        with pytest.raises(IOError, match="Checkpoint file not found"):
            ConfigHandler.load_checkpoint(missing_path)

    def test_checkpoint_invalid_json(self, tmp_path):
        """Raise IOError for malformed JSON checkpoint."""
        bad_path = tmp_path / "bad.json"
        with open(bad_path, 'w') as f:
            f.write("{invalid json")
        with pytest.raises(IOError, match="Failed to load checkpoint"):
            ConfigHandler.load_checkpoint(bad_path)

    def test_checkpoint_absolute_paths(self, tmp_path, fixtures_dir):
        """Checkpoint stores absolute paths for data files."""
        config_path = fixtures_dir / "relative_paths.yaml"
        handler = ConfigHandler(config_path)
        handler.validate()
        optimization_state = {
            'completed_steps': 0,
            'max_steps': 100,
            'timestamp': datetime.now().isoformat() + "Z",
            'converged': False,
            'merit_value': None,
            'gradient_norm': None,
            'gradient_max_norm': None
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        handler.save_checkpoint(checkpoint_path, optimization_state)
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        # All paths should be absolute
        assert Path(checkpoint['project']['output_dir']).is_absolute()
        assert Path(checkpoint['data']['kernel_data']['path']).is_absolute()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])