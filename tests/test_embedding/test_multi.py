#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_multi.py
@Time    :   2025/09/09 09:34:14
@Author  :   George Trenins
@Desc    :   Test direct sum of embedders
'''


from __future__ import print_function, division, absolute_import
import pytest
import numpy as np
from glefit.embedding import PronyEmbedder, MultiEmbedder


def test_multi_naux():
    """Test that MultiEmbedder returns correct number of auxiliary variables."""
    # Setup
    naux = 3
    theta = 1.0
    gamma = 0.5
    embs = [PronyEmbedder(theta, gamma) for n in range(naux)]
    multi_emb = MultiEmbedder(embs)
    
    # Verify length
    assert len(multi_emb) == naux, f"Expected {naux} auxiliary variables, got {len(multi_emb)}"

def test_multi_drift_matrix():
    # Setup
    thetas = np.asarray([1.0, 2.0, 3.0])
    gammas = np.asarray([0.5, 0.25, 0.1])
    Aref = np.zeros(2*(len(thetas,)))
    Aref[0,1:] = thetas
    Aref[1:,0] = thetas
    Adiag = np.einsum('ii->i', Aref[1:,1:])
    Adiag[:] = gammas
    embs = [PronyEmbedder(theta, gamma) for theta, gamma in zip(thetas, gammas)]
    multi_emb = MultiEmbedder(embs)
    
    # Verify the packing of combined A:
    # TODO




if __name__ == "__main__":
    pytest.main([__file__])