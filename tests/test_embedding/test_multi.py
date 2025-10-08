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
from scipy.linalg import block_diag
from glefit.embedding import PronyEmbedder, PronyCosineEmbedder, MultiEmbedder


def test_multi_naux():
    """Test that MultiEmbedder returns correct number of auxiliary variables."""
    # Setup
    theta = 1.0
    gamma = 0.5
    omega = 2.5
    n_prony = 3
    n_cos = 1
    embs = [PronyEmbedder(theta, gamma) for n in range(n_prony)] + [PronyCosineEmbedder(theta, gamma, omega) for n in range(n_cos)]
    multi_emb = MultiEmbedder(embs)
    
    # Verify length
    naux = n_prony + 2*n_cos
    assert len(multi_emb) == naux, f"Expected {naux} auxiliary variables, got {len(multi_emb)}"

def test_multi_drift_matrix():
    """Test that MultiEmbedder correctly constructs combined drift matrix."""

    
    A_lst = []
    theta_lst = []
    embs = []
    # Build Prony A matrix and coupling vector
    prony_thetas = np.asarray([1.0, 2.0, 3.0])
    prony_gammas = np.asarray([0.5, 0.25, 0.1])
    theta_lst.append(prony_thetas)
    A_lst.extend(list([np.eye(1)*g for g in prony_gammas]))
    embs.extend([PronyEmbedder(theta, gamma) for theta, gamma in zip(prony_thetas, prony_gammas)])
    # Add oscillatory kernels
    osc_thetas = np.asarray([3.5, 4.0])
    osc_gammas = np.asarray([0.1, 2.0])
    osc_omegas = np.asarray([5.0, 10.0])
    theta_lst.extend(list([ [th, 0.0] for th in osc_thetas ]))
    A_lst.extend(
            list([ np.asarray([[g, -w], [w, g]]) 
            for g,w in zip(osc_gammas, osc_omegas)])
    )
    embs.extend(
        PronyCosineEmbedder(theta, gamma, omega) for
        theta, gamma, omega in zip(
            osc_thetas, osc_gammas, osc_omegas
        )
    )
    # Build multi-component matrix
    AMAT = block_diag(*A_lst)
    avec = np.concatenate(theta_lst)
    Aref = np.zeros(2*(len(AMAT)+1,))
    Aref[1:,1:] = AMAT
    Aref[0,1:] = Aref[1:,0] = avec
    
    # Create MultiEmbedder
    multi_emb = MultiEmbedder(embs)
    
    # Verify the packing of combined drift matrix
    np.testing.assert_allclose(
        multi_emb.drift_matrix, Aref,
        rtol=1e-15,
        err_msg="Combined drift matrix does not match expected structure"
    )
    
def test_multi_drift_matrix_gradient():
    """Test that MultiEmbedder correctly constructs combined drift matrix gradient."""
    # Setup - Prony embedders
    prony_thetas = np.asarray([1.0, 2.0, 3.0])
    prony_gammas = np.asarray([0.5, 0.25, 0.1])
    n_prony = len(prony_thetas)
    
    # Setup - Oscillatory embedders
    osc_thetas = np.asarray([3.5, 4.0])
    osc_gammas = np.asarray([0.1, 2.0])
    osc_omegas = np.asarray([5.0, 10.0])
    n_osc = len(osc_thetas)
    
    # Total number of auxiliary variables and parameters
    naux = n_prony + 2*n_osc  
    nparam = 2*n_prony + 3*n_osc  # (θ,γ) per Prony, (θ,γ,ω) per oscillatory
    
    grad_ref = np.zeros((nparam, naux + 1, naux + 1))
    
    # Fill gradient blocks for Prony embedders
    param_offset = 0
    block_offset = 1
    for i in range(n_prony):
        theta_idx = param_offset     
        gamma_idx = param_offset + 1 
        block_idx = block_offset + i
        grad_ref[theta_idx, 0, block_idx] = 1.0
        grad_ref[theta_idx, block_idx, 0] = 1.0
        grad_ref[gamma_idx, block_idx, block_idx] = 1.0
        param_offset += 2
    
    # Fill gradient blocks for oscillatory embedders
    block_offset += n_prony
    for i in range(n_osc):
        theta_idx = param_offset     
        gamma_idx = param_offset + 1 
        omega_idx = param_offset + 2 
        lb = block_offset + 2*i
        ub = lb + 2
        grad_ref[theta_idx, 0, lb] = 1.0
        grad_ref[theta_idx, lb, 0] = 1.0
        grad_ref[gamma_idx, lb:ub, lb:ub] = np.eye(2)
        grad_ref[omega_idx, lb, lb+1] = -1.0
        grad_ref[omega_idx, lb+1, lb] = 1.0
        param_offset += 3
    
    embs = ([PronyEmbedder(theta, gamma) 
             for theta, gamma in zip(prony_thetas, prony_gammas)] +
            [PronyCosineEmbedder(theta, gamma, omega) 
             for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)])
    multi_emb = MultiEmbedder(embs)
    
    np.testing.assert_allclose(
        multi_emb.drift_matrix_param_grad(multi_emb.params), grad_ref,
        rtol=1e-15,
        err_msg="Combined drift matrix gradient does not match expected structure"
    )

def test_multi_drift_matrix_x_gradient():
    """Test that MultiEmbedder correctly constructs combined drift matrix gradient."""
    # Setup - Prony embedders
    prony_thetas = np.asarray([1.0, 2.0, 3.0])
    prony_gammas = np.asarray([0.5, 0.25, 0.1])
    n_prony = len(prony_thetas)
    
    # Setup - Oscillatory embedders
    osc_thetas = np.asarray([3.5, 4.0])
    osc_gammas = np.asarray([0.1, 2.0])
    osc_omegas = np.asarray([5.0, 10.0])
    n_osc = len(osc_thetas)
    
    # Total number of auxiliary variables and parameters
    naux = n_prony + 2*n_osc
    # (θ,γ) per Prony, (θ,γ,ω) per oscillatory
    nparam = 2*n_prony + 3*n_osc  
    grad_ref = np.zeros((nparam, naux + 1, naux + 1))
    
    # Fill gradient blocks for Prony embedders
    param_offset = 0
    block_offset = 1
    for i in range(n_prony):
        theta_idx = param_offset    
        gamma_idx = param_offset + 1
        block_idx = block_offset + i
        grad_ref[theta_idx, 0, block_idx] = prony_thetas[i]
        grad_ref[theta_idx, block_idx, 0] = prony_thetas[i]
        grad_ref[gamma_idx, block_idx, block_idx] = prony_gammas[i]
        param_offset += 2
    
    # Fill gradient blocks for oscillatory embedders
    block_offset += n_prony
    for i in range(n_osc):
        theta_idx = param_offset    
        gamma_idx = param_offset + 1
        omega_idx = param_offset + 2
        lb = block_offset + 2*i
        ub = lb + 2
        grad_ref[theta_idx, 0, lb] = osc_thetas[i]
        grad_ref[theta_idx, lb, 0] = osc_thetas[i]
        grad_ref[gamma_idx, lb:ub, lb:ub] = osc_gammas[i] * np.eye(2)
        grad_ref[omega_idx, lb, lb+1] = -osc_omegas[i]
        grad_ref[omega_idx, lb+1, lb] = osc_omegas[i]
        param_offset += 3
    
    embs = ([PronyEmbedder(theta, gamma) 
             for theta, gamma in zip(prony_thetas, prony_gammas)] +
            [PronyCosineEmbedder(theta, gamma, omega) 
             for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)])
    multi_emb = MultiEmbedder(embs)
    
    np.testing.assert_allclose(
        multi_emb.drift_matrix_gradient, grad_ref,
        rtol=1e-15,
        err_msg="Combined drift matrix gradient does not match expected structure"
    )


if __name__ == "__main__":
    pytest.main([__file__])