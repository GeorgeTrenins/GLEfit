#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_intervals.py
@Time    :   2025/10/09 10:43:44
@Author  :   George Trenins
@Desc    :   Test the first and second derivatives of mapper functions imposing interval constraints on optimizable parameters.
'''


from __future__ import print_function, division, absolute_import
from glefit.mappers import UpperBoundMapper, LowerBoundMapper, IntervalMapper
import pytest
from numpy.testing import assert_allclose
# from glefit.utils.numderiv import 
