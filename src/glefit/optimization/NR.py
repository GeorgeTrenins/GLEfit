#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   NR.py
@Time    :   2025/09/19 11:34:35
@Author  :   George Trenins
@Desc    :   Newton-Raphson optimization
'''


from __future__ import print_function, division, absolute_import
from ._base import Optimizer, DEFAULT_MAX_STEPS
from typing import Optional
from glefit.utils.linalg import mat_inv_vec
import numpy as np
import time

class NewtonRaphson(Optimizer):

    def run(self, steps: int = DEFAULT_MAX_STEPS, options: Optional[dict] = None):
        """Run the optimizer.

        This method will return whenever the gradient of the merit function drops below `gtol` for all optimizable parameters or when the number of steps exceeds `steps`.

        Args:
            steps (int, optional): Maximum number of steps to take. Defaults to DEFAULT_MAX_STEPS.
        
        Options:
            gtol (float): upper bound on the maximum norm of the gradient of the merit function w.r.t. optimizable parameters.
            max_step (float): maximum size of update step
        """
        self.gtol = options.get("gtol", 0.05)
        self.max_step = options.get("max_step")
        self._eigvals = None
        super().run(steps=steps, options=options)

    def initialize(self):
        self.update()
        
    def update(self):
        # compute distance value, gradient and hessian
        self._value, self._grad, self._hess = self.merit.all()

    def step(self):
        h = -mat_inv_vec(self._hess, self._grad) 
        h = self.rescale_step(h)
        return h
    
    def rescale_step(self, h):
        if self.max_step is None:
            return h
        else:
            max_component = np.max(np.abs(h))
            if max_component < self.max_step:
                return h
            else:
                return self.max_step/max_component * h
    
    def converged(self):
        return np.linalg.norm(self._grad, ord=np.inf) < self.gtol
    
    def get_eigvals(self):
        return np.linalg.eigvalsh(self._hess)
    
    def log(self, T=None):
        if T is None:
            T = time.localtime()
        name = self.__class__.__name__
        distance = self._value
        grad = self._grad
        gmax = np.linalg.norm(grad, ord=np.inf)
        if self.logfile is not None:
            args = (" " * len(name), "Step", "Time", "Value", "Gmax")
            msg = "{:s}  {:4s} {:8s} {:14s} {:14s}\n".format(*args)
            self.logfile.write(msg)
            args = (name, self.nsteps, T[3], T[4], T[5], distance, gmax)
            msg = "{:s}:  {: 3d} {:02d}:{:02d}:{:02d} {: 14.7e} {: 14.7e}\n".format(*args)
            self.logfile.write(msg)
            self.logfile.write("=== Gradient ===\n")
            self.logfile.write(np.array2string(
                grad,
                max_line_width=70,    
                separator=", ",       
                suppress_small=False, 
                threshold=1_000_000,  
                formatter={'float_kind':lambda x: f"{x: .6e}"}
            ))
            self.logfile.write("\n")
            self.logfile.write("===  End of gradient ===\n")
            self.logfile.write("=== Hessian eigenvalues ===\n")
            eigvals = self.get_eigvals()
            self.logfile.write(np.array2string(
                eigvals,
                max_line_width=70,    
                separator=", ",       
                suppress_small=False, 
                threshold=1_000_000,  
                formatter={'float_kind':lambda x: f"{x: .6e}"}
            ))
            self.logfile.write("\n")
            self.logfile.write("===  End of hessian eigenvalues ===\n")
            self.logfile.write("\n")
            self.logfile.flush()
        super().log(T=T)


class EigenvectorFollowing(NewtonRaphson):

    def get_eigvals(self):
        if self._eigvals is None:
            self._eigvals = np.linalg.eigvalsh(self._hess)
        return np.copy(self._eigvals)

    def step(self):
        self._eigvals, self._eigvecs = np.linalg.eigh(self._hess)
        nm_grad = self._grad @ self._eigvecs
        nm_step = -2 * nm_grad / (np.abs(self._eigvals) * (1 + np.sqrt(
            1 + 4 * (nm_grad / self._eigvals)**2
        )))
        h = self._eigvecs @ nm_step
        h = self.rescale_step(h)
        return h