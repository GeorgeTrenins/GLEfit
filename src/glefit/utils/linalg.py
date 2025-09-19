#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   linalg.py
@Time    :   2025/09/19 11:48:35
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
from scipy.linalg import cho_factor, cho_solve


def symmat_inv_vec(M, v):
    """Compute the matrix product M^(-1) * v for a symmetric matrix M and a vector v
    """

    c, lower = cho_factor(M, overwrite_a=False, check_finite=False)
    return cho_solve((c, lower), v)