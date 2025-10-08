#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   rp.py
@Time    :   2025/10/08 16:49:14
@Author  :   George Trenins
@Desc    :   ring-polymer utilities
'''


from __future__ import print_function, division, absolute_import
from typing import Optional
import numpy as np

def nmfreq(beta: float, P: int, n: int, hbar: Optional[float] = 1.0) -> float:
    """n-th normal-mode frequency of a free ring-polymer

    Args:
        beta (float): reciprocal temp 1/(kB*T)
        P (int): number of ring-polymer beads
        n (int): index of normal mode
        hbar (float, optional): reduced Planck constant in the user's unit system. Defaults to 1.0.
    """
    omega_P = P / (beta*hbar)
    wn = 2*omega_P * np.abs(np.sin(np.pi * n/P))
    return wn