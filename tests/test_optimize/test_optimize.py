#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_optimize.py
@Time    :   2026/01/13 11:50:43
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
import pytest
from pathlib import Path
import argparse
from glefit import optimize

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"

# ============================================================================
# TESTS
# ============================================================================

def test_optimize(fixtures_dir):
    args = argparse.Namespace(
        config = str(fixtures_dir / "multi_embedder.yaml"),
        chk = None
    )
    optimize.main(args)

def test_optimize_2x2(fixtures_dir):
    args = argparse.Namespace(
        config = str(fixtures_dir / "gen2x2_embedder.yaml"),
        chk = None
    )
    optimize.main(args)

if __name__ == "__main__":
    test_optimize_2x2(Path(__file__).parent / "fixtures")
