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
    """Test that MultiEmbedder correctly constructs combined drift matrix."""
    # Setup
    thetas = np.asarray([1.0, 2.0, 3.0])
    gammas = np.asarray([0.5, 0.25, 0.1])
    naux = len(thetas)
    
    # Build reference matrix
    Aref = np.zeros((naux + 1, naux + 1))
    Aref[0, 1:] = thetas  # coupling terms in first row
    Aref[1:, 0] = thetas  # coupling terms in first column
    np.fill_diagonal(Aref[1:, 1:], gammas)  # diagonal terms
    
    # Create MultiEmbedder
    embs = [PronyEmbedder(theta, gamma) for theta, gamma in zip(thetas, gammas)]
    multi_emb = MultiEmbedder(embs)
    
    # Verify the packing of combined drift matrix
    np.testing.assert_allclose(
        multi_emb.drift_matrix, Aref,
        rtol=1e-15,
        err_msg="Combined drift matrix does not match expected structure"
    )
    
def test_multi_drift_matrix_gradient():
    """Test that MultiEmbedder correctly constructs combined drift matrix gradient."""
    # Setup
    thetas = np.asarray([1.0, 2.0, 3.0])
    gammas = np.asarray([0.5, 0.25, 0.1])
    naux = len(thetas)
    nparam = 2 * naux  # 2 parameters (theta, gamma) per embedder
    
    # Build reference gradient array
    grad_ref = np.zeros((nparam, naux + 1, naux + 1))
    
    # Fill gradient blocks for each embedder
    for i in range(naux):
        # Parameter indices for current embedder
        theta_idx = 2*i      # derivative wrt theta
        gamma_idx = 2*i + 1  # derivative wrt gamma
        
        # Block indices for current embedder
        block_idx = i + 1
        
        # Derivative wrt theta: coupling terms
        grad_ref[theta_idx, 0, block_idx] = 1.0
        grad_ref[theta_idx, block_idx, 0] = 1.0
        
        # Derivative wrt gamma: diagonal term
        grad_ref[gamma_idx, block_idx, block_idx] = 1.0
    
    # Create MultiEmbedder
    embs = [PronyEmbedder(theta, gamma) for theta, gamma in zip(thetas, gammas)]
    multi_emb = MultiEmbedder(embs)
    
    # Verify the packing of combined gradient
    np.testing.assert_allclose(
        multi_emb._drift_matrix_param_grad(multi_emb.params), grad_ref,
        rtol=1e-15,
        err_msg="Combined drift matrix gradient does not match expected structure"
    )

def test_multi_drift_matrix_x_gradient():
    """Test that MultiEmbedder correctly constructs combined drift matrix gradient."""
    # Setup
    thetas = np.asarray([1.0, 2.0, 3.0])
    gammas = np.asarray([0.5, 0.25, 0.1])
    naux = len(thetas)
    nparam = 2 * naux  # 2 parameters (theta, gamma) per embedder
    
    # Build reference gradient array
    grad_ref = np.zeros((nparam, naux + 1, naux + 1))
    
    # Fill gradient blocks for each embedder
    for i in range(naux):
        # Parameter indices for current embedder
        theta_idx = 2*i      # derivative wrt theta
        gamma_idx = 2*i + 1  # derivative wrt gamma
        
        # Block indices for current embedder
        block_idx = i + 1
        
        # Derivative wrt theta: coupling terms
        grad_ref[theta_idx, 0, block_idx] = thetas[i]
        grad_ref[theta_idx, block_idx, 0] = thetas[i]
        
        # Derivative wrt gamma: diagonal term
        grad_ref[gamma_idx, block_idx, block_idx] = gammas[i]
    
    # Create MultiEmbedder
    embs = [PronyEmbedder(theta, gamma) for theta, gamma in zip(thetas, gammas)]
    multi_emb = MultiEmbedder(embs)
    
    # Verify the packing of combined gradient
    np.testing.assert_allclose(
        multi_emb.drift_matrix_gradient, grad_ref,
        rtol=1e-15,
        err_msg="Combined drift matrix gradient does not match expected structure"
    )


if __name__ == "__main__":
    pytest.main([__file__])
