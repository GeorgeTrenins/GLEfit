#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   data_io.py
@Time    :   2026/01/12 11:38:01
@Author  :   George Trenins
@Desc    :   None

Data loading utilities for external arrays (CSV) or inline specifications.

Provides flexible loading of data from CSV files or inline (YAML-specified) arrays.
Uses numpy.loadtxt for robust CSV parsing with configurable delimiters and column selection.
'''

from __future__ import print_function, division, absolute_import
from pathlib import Path
from typing import Dict, Tuple
import logging
import numpy as np
import numpy.typing as npt
import warnings


logger = logging.getLogger(__name__)

class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


def load_data(
    config_dict: Dict,
    data_name: str
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Load data from config specification.
    
    Args:
        config_dict: Data configuration block from YAML
        data_name: Name of the dataset (for logging)
        
    Returns:
        Tuple of (grid, target) as 1D float arrays
        
    Raises:
        DataLoadError: If configuration is invalid or loading fails
    """
    source = config_dict.get('source')
    
    if source == 'external':
        return _load_external(config_dict, data_name)
    elif source == 'inline':
        return _load_inline(config_dict, data_name)
    else:
        raise DataLoadError(
            f"Unknown data source: {source}. Must be 'external' or 'inline'"
        )


def _load_external(
    config_dict: Dict,
    data_name: str
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Load data from external CSV file.
    
    Args:
        config_dict: Configuration dict with 'path', 'columns', and optional 'delimiter'
        data_name: Dataset name for logging
        
    Returns:
        Tuple of (grid, target) arrays
        
    Raises:
        DataLoadError: If path missing, file not found, or columns invalid
    """
    path = config_dict.get('path')
    if not path:
        raise DataLoadError(f"Data '{data_name}': 'path' required for external source")
    
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Data '{data_name}': File not found at {path}")
    
    fmt = config_dict.get('format', 'csv').lower()
    if fmt != 'csv':
        raise DataLoadError(f"Data '{data_name}': Only CSV format supported, got {fmt}")
    
    columns = config_dict.get('columns')
    if columns is None:
        raise DataLoadError(f"Data '{data_name}': 'columns' required for external source")
    
    try:
        grid_col, target_col = columns
        if not isinstance(grid_col, int) or not isinstance(target_col, int):
            raise ValueError("Column indices must be integers")
        if grid_col < 0 or target_col < 0:
            raise ValueError("Column indices must be non-negative")
        if grid_col == target_col:
            raise ValueError("Grid and target columns must be different")
    except (TypeError, ValueError) as e:
        raise DataLoadError(
            f"Data '{data_name}': Invalid column specification {columns}: {e}"
        )
    
    delimiter = config_dict.get('delimiter', ',')
    
    try:
        # Suppress the "input contained no data" warning from numpy.loadtxt
        # We handle empty data with explicit error checking below
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*loadtxt: input contained no data.*")
            data = np.loadtxt(path, delimiter=delimiter, dtype=float)
        # Handle case where file has only one row (returns 1D array)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Validate column indices
        if data.shape[1] <= max(grid_col, target_col):
            raise DataLoadError(
                f"Columns out of bounds; data '{data_name}' has {data.shape[1]} columns, "
                f"but columns {columns} were requested (0-indexed)"
            )
        grid = data[:, grid_col]
        target = data[:, target_col]
        logger.info(
            f"Loaded {len(grid)} data points from {path} (columns {grid_col}, {target_col})"
        )
        return grid, target
    except (OSError, IOError) as e:
        raise DataLoadError(f"Data '{data_name}': I/O error reading {path}: {e}")
    except ValueError as e:
        # numpy.loadtxt raises ValueError for non-numeric data
        raise DataLoadError(
            f"Data '{data_name}': Failed to parse CSV (non-numeric data?): {e}"
        )


def _load_inline(
    config_dict: Dict,
    data_name: str
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Load inline (YAML-specified) arrays.
    
    Args:
        config_dict: Configuration dict with 'grid' and 'target' keys
        data_name: Dataset name for logging
        
    Returns:
        Tuple of (grid, target) arrays
        
    Raises:
        DataLoadError: If grid/target missing, wrong shape, or mismatched lengths
    """
    grid = config_dict.get('grid')
    target = config_dict.get('target')
    
    if grid is None:
        raise DataLoadError(f"Data '{data_name}': 'grid' required for inline source")
    if target is None:
        raise DataLoadError(f"Data '{data_name}': 'target' required for inline source")
    
    try:
        grid = np.atleast_1d(np.asarray(grid, dtype=float))
        target = np.atleast_1d(np.asarray(target, dtype=float))
    except (TypeError, ValueError) as e:
        raise DataLoadError(f"Data '{data_name}': Failed to convert to arrays: {e}")
    
    if grid.ndim != 1 or target.ndim != 1:
        raise DataLoadError(
            f"Data '{data_name}': grid and target must be 1D, "
            f"got shapes {grid.shape}, {target.shape}"
        )
    
    if len(grid) != len(target):
        raise DataLoadError(
            f"Data '{data_name}': grid and target have different lengths "
            f"({len(grid)} vs {len(target)})"
        )
    
    logger.info(f"Loaded {len(grid)} inline data points for '{data_name}'")
    return grid, target