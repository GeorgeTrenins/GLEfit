#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Tests for data loading functionality."""

from __future__ import print_function, division, absolute_import
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from glefit.config.data_io import (
    load_data, DataLoadError
)


class TestLoadExternalCSV:
    """Test external CSV loading via numpy.loadtxt."""

    def test_simple_csv(self):
        """Load simple comma-separated CSV."""
        ref_grid = [0.0, 0.5, 1.0]
        ref_target = [1.0, 0.5, 0.1]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for x, y in np.c_[ref_grid, ref_target]:
                f.write(f"{x:3.1f},{y:3.1f}\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ','
                }
                grid, target = load_data(config, 'test_data')
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)

    def test_tab_delimiter(self):
        """Load tab-separated data."""
        ref_grid = [0.0, 0.5]
        ref_target = [1.0, 0.5]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for x, y in np.c_[ref_grid, ref_target]:
                f.write(f"{x:3.1f}\t{y:3.1f}\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': '\t'
                }
                grid, target = load_data(config, 'test_data')
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)

    def test_semicolon_delimiter(self):
        """Load semicolon-separated data."""
        ref_grid = [0.0, 0.5]
        ref_target = [1.0, 0.5]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for x, y in np.c_[ref_grid, ref_target]:
                f.write(f"{x:3.1f};{y:3.1f}\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ';'
                }
                grid, target = load_data(config, 'test_data')
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)

    def test_auto_skip_comments(self):
        """numpy.loadtxt auto-skips lines starting with #."""
        ref_grid = [0.0, 0.5]
        ref_target = [1.0, 0.5]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("# This is a comment\n")
            for x, y in np.c_[ref_grid, ref_target]:
                f.write(f"{x:3.1f},{y:3.1f}\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ','
                }
                grid, target = load_data(config, 'test_data')
                assert len(grid) == 2
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)

    def test_column_selection(self):
        """Extract correct columns from multi-column CSV."""
        ref_grid = [0.0, 0.5]
        ref_ignore = [10.0, 20.0]  # Middle column to skip
        ref_target = [1.0, 0.5]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for x, skip, y in np.c_[ref_grid, ref_ignore, ref_target]:
                f.write(f"{x:3.1f},{skip:5.1f},{y:3.1f}\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 2],  # Skip column 1
                    'delimiter': ','
                }
                grid, target = load_data(config, 'test_data')
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)

    def test_reversed_columns(self):
        """Allow grid and target in either order."""
        ref_grid = [0.0, 0.5]
        ref_target = [1.0, 0.5]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write in reversed order: target, grid
            for y, x in np.c_[ref_target, ref_grid]:
                f.write(f"{y:3.1f},{x:3.1f}\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [1, 0],  # grid col=1, target col=0
                    'delimiter': ','
                }
                grid, target = load_data(config, 'test_data')
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)

    def test_out_of_bounds_columns(self):
        """Raise error if columns exceed CSV width."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.0,1.0\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 5],  # Column 5 doesn't exist
                    'delimiter': ','
                }
                with pytest.raises(DataLoadError, match="out of bounds"):
                    load_data(config, 'test_data')
            finally:
                os.unlink(f.name)

    def test_non_numeric_data(self):
        """Raise error for non-numeric values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            ref_grid = [0.0, 0.5]
            ref_target = [1.0, "abc"]  # Second value is non-numeric
            for x, y in np.c_[ref_grid, ref_target]:
                f.write(f"{float(x):3.1f},{y}\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ','
                }
                with pytest.raises(DataLoadError, match="non-numeric"):
                    load_data(config, 'test_data')
            finally:
                os.unlink(f.name)

    def test_empty_file(self):
        """Raise error for empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ','
                }
                with pytest.raises(DataLoadError):
                    load_data(config, 'test_data')
            finally:
                os.unlink(f.name)

    def test_file_not_found(self):
        """Raise error for missing file."""
        config = {
            'source': 'external',
            'path': '/nonexistent/path/to/file.csv',
            'format': 'csv',
            'columns': [0, 1],
            'delimiter': ','
        }
        with pytest.raises(DataLoadError, match="not found"):
            load_data(config, 'test_data')

    def test_missing_path(self):
        """Raise error if path not specified."""
        config = {
            'source': 'external',
            'format': 'csv',
            'columns': [0, 1]
            # 'path' key missing
        }
        with pytest.raises(DataLoadError, match="path.*required"):
            load_data(config, 'test_data')

    def test_missing_columns(self):
        """Raise error if columns not specified."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.0,1.0\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'delimiter': ','
                    # 'columns' key missing
                }
                with pytest.raises(DataLoadError, match="columns.*required"):
                    load_data(config, 'test_data')
            finally:
                os.unlink(f.name)

    def test_invalid_column_indices_non_integer(self):
        """Raise error for non-integer column indices."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.0,1.0\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0.5, 1],  # Non-integer
                    'delimiter': ','
                }
                with pytest.raises(DataLoadError, match="Invalid column"):
                    load_data(config, 'test_data')
            finally:
                os.unlink(f.name)

    def test_invalid_column_indices_same(self):
        """Raise error if grid and target columns are identical."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.0,1.0\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 0],  # Same column
                    'delimiter': ','
                }
                with pytest.raises(DataLoadError, match="different"):
                    load_data(config, 'test_data')
            finally:
                os.unlink(f.name)

    def test_invalid_column_indices_negative(self):
        """Raise error for negative column indices."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.0,1.0\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [-1, 1],  # Negative index
                    'delimiter': ','
                }
                with pytest.raises(DataLoadError, match="non-negative"):
                    load_data(config, 'test_data')
            finally:
                os.unlink(f.name)

    def test_unsupported_format(self):
        """Raise error for non-CSV format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.0,1.0\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'hdf5',  # Not supported
                    'columns': [0, 1]
                }
                with pytest.raises(DataLoadError, match="Only CSV"):
                    load_data(config, 'test_data')
            finally:
                os.unlink(f.name)

    def test_single_row_csv(self):
        """Handle CSV with single row (numpy.loadtxt returns 1D array)."""
        ref_grid = [0.5]
        ref_target = [0.7]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.5,0.7\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ','
                }
                grid, target = load_data(config, 'test_data')
                assert grid.shape == (1,)
                assert target.shape == (1,)
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)

    def test_whitespace_tolerance(self):
        """Handle extra whitespace (numpy.loadtxt is tolerant)."""
        ref_grid = [0.0, 0.5]
        ref_target = [1.0, 0.5]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for x, y in np.c_[ref_grid, ref_target]:
                f.write(f"  {x:3.1f}  ,  {y:3.1f}  \n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1],
                    'delimiter': ','
                }
                grid, target = load_data(config, 'test_data')
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)

    def test_default_delimiter(self):
        """Use comma as default delimiter if not specified."""
        ref_grid = [0.0, 0.5]
        ref_target = [1.0, 0.5]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for x, y in np.c_[ref_grid, ref_target]:
                f.write(f"{x:3.1f},{y:3.1f}\n")
            f.flush()
            try:
                config = {
                    'source': 'external',
                    'path': f.name,
                    'format': 'csv',
                    'columns': [0, 1]
                    # delimiter not specified, should default to ','
                }
                grid, target = load_data(config, 'test_data')
                np.testing.assert_allclose(grid, ref_grid)
                np.testing.assert_allclose(target, ref_target)
            finally:
                os.unlink(f.name)


class TestLoadInline:
    """Test inline data loading."""

    def test_inline_valid(self):
        """Load valid inline arrays."""
        ref_grid = [0.0, 0.5, 1.0]
        ref_target = [1.0, 0.5, 0.1]
        config = {
            'source': 'inline',
            'grid': ref_grid,
            'target': ref_target
        }
        grid, target = load_data(config, 'test_data')
        np.testing.assert_allclose(grid, ref_grid)
        np.testing.assert_allclose(target, ref_target)

    def test_inline_numpy_arrays(self):
        """Load inline numpy arrays."""
        ref_grid = np.array([0.0, 0.5, 1.0])
        ref_target = np.array([1.0, 0.5, 0.1])
        config = {
            'source': 'inline',
            'grid': ref_grid,
            'target': ref_target
        }
        grid, target = load_data(config, 'test_data')
        np.testing.assert_allclose(grid, ref_grid)
        np.testing.assert_allclose(target, ref_target)

    def test_inline_scalar_converted_to_array(self):
        """Convert scalar inputs to arrays."""
        config = {
            'source': 'inline',
            'grid': 0.5,
            'target': 1.0
        }
        grid, target = load_data(config, 'test_data')
        assert grid.shape == (1,)
        assert target.shape == (1,)
        np.testing.assert_allclose(grid, [0.5])
        np.testing.assert_allclose(target, [1.0])

    def test_inline_missing_grid(self):
        """Raise error if grid missing."""
        config = {
            'source': 'inline',
            'target': [1.0, 0.5]
            # 'grid' key missing
        }
        with pytest.raises(DataLoadError, match="grid"):
            load_data(config, 'test_data')

    def test_inline_missing_target(self):
        """Raise error if target missing."""
        config = {
            'source': 'inline',
            'grid': [0.0, 0.5]
            # 'target' key missing
        }
        with pytest.raises(DataLoadError, match="target"):
            load_data(config, 'test_data')

    def test_inline_mismatched_lengths(self):
        """Raise error if grid/target lengths differ."""
        config = {
            'source': 'inline',
            'grid': [0.0, 0.5, 1.0],
            'target': [1.0, 0.5]  # Different length
        }
        with pytest.raises(DataLoadError, match="different lengths"):
            load_data(config, 'test_data')

    def test_inline_non_1d_arrays(self):
        """Raise error for non-1D arrays."""
        config = {
            'source': 'inline',
            'grid': [[0.0, 0.5]],  # 2D array
            'target': [1.0, 0.5]
        }
        with pytest.raises(DataLoadError, match="1D"):
            load_data(config, 'test_data')

    def test_inline_non_numeric(self):
        """Raise error for non-numeric data."""
        config = {
            'source': 'inline',
            'grid': ['a', 'b'],  # Non-numeric
            'target': [1.0, 2.0]
        }
        with pytest.raises(DataLoadError, match="Failed to convert"):
            load_data(config, 'test_data')


class TestLoadDataGeneral:
    """General tests for load_data function."""

    def test_unknown_source(self):
        """Raise error for unknown source type."""
        config = {'source': 'unknown_source_type'}
        with pytest.raises(DataLoadError, match="Unknown data source"):
            load_data(config, 'test_data')

    def test_missing_source(self):
        """Raise error if source not specified."""
        config = {
            'grid': [1.0],
            'target': [2.0]
            # 'source' key missing
        }
        with pytest.raises(DataLoadError, match="Unknown data source"):
            load_data(config, 'test_data')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])