#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_mappers.py
@Time    :   2026/11/05 20:00:00
@Author  :   George Trenins
@Desc    :   Test the parameter mappers
'''


from __future__ import print_function, division, absolute_import
import pytest
from glefit.mappers import LowerBoundMapper, UpperBoundMapper, IntervalMapper


@pytest.mark.parametrize("mapper, p, match", [
    (LowerBoundMapper(1.0), 1.0, "greater than"),
    (UpperBoundMapper(1.0), 1.0, "smaller than"),
    (IntervalMapper(0.0, 1.0), 0.0, "greater than"),
    (IntervalMapper(0.0, 1.0), 1.0, "smaller than"),
], ids=["lower-bound", "upper-bound", "interval-lower", "interval-upper"])
def test_forward_rejects_p_on_its_bound(mapper, p, match):
    # the bound itself would map to x = ±inf
    with pytest.raises(ValueError, match=match):
        mapper.forward(p)
