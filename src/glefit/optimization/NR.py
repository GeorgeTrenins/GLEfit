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
from glefit.utils.linalg import symmat_inv_vec
import numpy as np
import time

class NewtonRaphson(Optimizer):

    def run(self, steps: int = DEFAULT_MAX_STEPS, options: Optional[dict] = None):
        """Run the optimizer.

        This method will return whenever the gradient of the merit function drops below `gtol` for all optimizable parameters or when the number of steps exceeds `steps`.

        Args:
            steps (int, optional): Maximum number of steps to take. Defaults to DEFAULT_MAX_STEPS.
        
        Options:
            gtol (float): upper bound on the maximum norm of the gradient of the merit function w.r.t.
            optimizable parameters.
        """
        self.gmax = options.get("gmax", 0.05)
        super().run(steps=steps, options=options)

    def initialize(self):
        self.update()
        
    def update(self):
        # compute distance value, gradient and hessian
        self._value, self._grad, self._hess = self.merit.all()

    def step(self):
        return -symmat_inv_vec(self._hess, self._grad)
    
    def converged(self):
        return np.linalg.norm(self._grad, ord=np.inf) < self.gmax
    
    def log(self, T=None):
        if T is None:
            T = time.localtime()
        name = self.__class__.__name__
        distance = self._value
        grad = self._grad
        hess = self._hess
        gmax = np.linalg.norm(grad, ord=np.inf)
        if self.logfile is not None:
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Time", "Value", "Gmax")
                msg = "{:s}  {:4s} {:8s} {:14s} {:14s}\n".format(*args)
                self.trajectory.write(msg)
            args = (name, self.nsteps, T[3], T[4], T[5], distance, gmax)
            msg = "{:s}:  {: 3d} {:02d}:{:02d}:{:02d} {: 14.7e} {: 14.7e}\n".format(*args)
            self.logfile.write(msg)
            self.logfile.write("=== Gradient ===\n")
            self.trajectory.write(np.array2string(
                grad,
                max_line_width=70,    
                separator=", ",       
                suppress_small=False, 
                threshold=1_000_000,  
                formatter={'float_kind':lambda x: f"{x: .6e}"}
            ))
            self.logfile.write("===  End of gradient ===\n")
            self.logfile.write("=== Hessian eigenvalues ===\n")
            eigvals = np.linalg.eigvalsh(hess)
            self.trajectory.write(np.array2string(
                eigvals,
                max_line_width=70,    
                separator=", ",       
                suppress_small=False, 
                threshold=1_000_000,  
                formatter={'float_kind':lambda x: f"{x: .6e}"}
            ))
            self.logfile.write("===  End of hessian eigenvalues ===\n")
            self.logfile.flush()
        super().log(T=T)
