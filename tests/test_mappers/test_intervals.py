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
from glefit.utils.numderiv import diff


def test_mapper_derivatives():
    """Verify the first and second derivatives of the conventional parameters with respect to the mapped variable"""
    x_lst = [-3.0, 0.0, 5.0]
    for Mapper in LowerBoundMapper, UpperBoundMapper, IntervalMapper:
        mapper = Mapper()
        for x in x_lst:
            ref = diff(mapper.inverse, x, nu=1, order=4)
            grad = mapper.grad(x)
            assert_allclose(grad, ref, rtol=1.0e-10, atol=1.0e-12, err_msg=f"Gradient method of {mapper.__class__.__name__} returned {grad} when {ref} was expected")
            ref = diff(mapper.inverse, x, nu=2, order=6)
            hess = mapper.hess(x)
            assert_allclose(hess, ref, rtol=1.0e-10, atol=1.0e-12, err_msg=f"Hessian method of {mapper.__class__.__name__} returned {grad} when {ref} was expected")


if __name__ == "__main__":
    pytest.main([__file__])