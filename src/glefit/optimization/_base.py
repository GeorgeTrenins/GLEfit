#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/09/11 10:02:48
@Author  :   George Trenins
@Desc    :   Base class for auxiliary variable parameter optimization - based on ASE's Optimizer
'''

#TODO: implement restart method (abstract)

from __future__ import print_function, division, absolute_import
from glefit.embedding import BaseEmbedder
from glefit.merit import BaseProperty
from pathlib import Path
from typing import Optional, Union, IO
from contextlib import ExitStack
import functools
import os
import sys 
import numpy as np
import numpy.typing as npt
import time

MAX_ITERATIONS_DEFAULT = 1_000_000
DEFAULT_TRAJ_FILE = "traj.out"


class _DelExitStack(ExitStack):
    """A private helper class to hold the finalizer. When _DelExitStack is garbage-collected, __del__ calls .close(), which closes all registered resources."""
    def __del__(self):
        self.close()


class IOContext(object):

    @functools.cached_property
    def _exitstack(self):
        """Lazily creates the _DelExitStack the first time it's needed, then caches it."""
        return _DelExitStack()
    
    def __enter__(self):
        return self 
    
    def __exit__(self, *args): 
        self.close()

    def closelater(self, fd):
        """Enter a given context manager and registers its __exit__ to run on close."""
        return self._exitstack.enter_context(fd)

    def close(self):
        self._exitstack.close()

    def openfile(self, file, mode='w'):
        if hasattr(file, 'close'):
            # treated as user-managed and returned as-is. 
            # IOContext deliberately does not take responsibility for closing it.
            return file
        encoding = None if mode.endswith('b') else 'utf-8'
        if file is None:
            return self.closelater(open(os.devnull, mode=mode, encoding=encoding))
        if file == '-':
            return sys.stdout
        return self.closelater(open(file, mode=mode, encoding=encoding))


class Optimizer(IOContext):
    """Base class for all GLE auxiliary variable optimizers.
    
    Optimization Loop:
    1. Subclass computes step h via step() method
    2. Update embedder: emb.x = emb.x + h (transformed parameters)
    3. Embedder automatically maps: x → params (via _inverse_map)
    4. Recompute merit function and derivatives via update()
    5. Check convergence via converged()
    6. Log progress via log()
    
    Parameter Mapping:
    - Conventional params: User-facing embedder parameters
    - Transformed params (x): Constraint-mapped, used in optimization
    - The optimization always works in the x space (constraints built-in)
    
    Subclasses must implement:
    - converged(): convergence criterion
    - step(): compute parameter update direction
    
    Subclasses may override:
    - initialize(): set up initial state
    - update(): refresh quantities needed for next step
    - log(): record optimization progress
    
    Attributes:
        emb (BaseEmbedder): The embedder to optimize
        merit (BaseProperty): Objective function to minimize
        logfile: File handle for detailed logging (gradient, Hessian, etc.)
        trajectory: File handle for parameter trajectory output
        nsteps (int): Number of completed optimization steps
    """

    def __init__(
            self,
            emb: BaseEmbedder,
            merit_function: BaseProperty,
            logfile: Optional[Union[IO, str, Path]] = None,
            trajfile: Optional[Union[IO, str, Path]] = None,
            **kwargs
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        emb : BaseEmbedder
            The Markovian embedder to optimize
        merit_function: BaseProperty
            Target function to minimize
        logfile : file-like object, str, or pathlib.Path, optional
            File to log optimization progress. If None, logging is disabled.
            (default: None)
        trajfile: file-like object, str, or pathlib.Path, optional
            File to write optimizable parameter values over the course of the 
            optimization. If 'None', write to default location 'traj.out'.
            If '-', write to stdout.
        """
        if not isinstance(emb, BaseEmbedder):
            raise ValueError("Optimizer requires a BaseEmbedder instance.")
        self.emb = emb
        self.merit = merit_function
        self.logfile = self.openfile(file=logfile, mode='a')
        if trajfile is None:
            trajfile = DEFAULT_TRAJ_FILE
        self.trajectory = self.openfile(file=trajfile)
        self.nsteps = 0
        self.max_iter = 0 # set by run
        self.initialize()

    def run(self, steps: int = MAX_ITERATIONS_DEFAULT, options: Optional[dict] = None) -> bool:
        """Run the optimizer.

        This method will return whenever the convergence criteria are fulfilled
        or when the number of steps exceeds `steps`.

        Args:
            steps (int, optional): Maximum number of steps to take. Defaults to DEFAULT_MAX_STEPS.
            options (dict, optional): Optimizer-specific options for controlling convergence 
                (e.g., tolerance thresholds). Defaults to None.
        
        Returns:
            bool: True if converged, False if max steps reached.
        """
        # update the maximum number of steps 
        # TODO: this will become relevant if we implement restarts
        self.max_iter = self.nsteps + steps
        if self.nsteps == 0:
            self.log()
        # check if we are already converged
        is_converged = self.converged()
        if is_converged:
            # Already converged!
            return True
        # start the actual optimization
        while not is_converged and self.nsteps < self.max_iter:
            self.iteration()
            is_converged = self.converged()
            self.nsteps += 1
            self.log()
        return is_converged
    
    def initialize(self) -> None:
        """Compute any relevant quantities for judging initial convergence
        and making first optimization step
        """
        pass
    
    def update(self) -> None:
        """Update the relevant quantities for judging convergence / making 
        next step
        """
        pass

    def converged(self) -> bool:
        """Optimization convergence criterion"""
        raise NotImplementedError

    def iteration(self) -> None:
        """Perform a single optimization iteration.
        
        Steps:
        1. Compute step direction via step()
        2. Update embedder parameters
        3. Refresh cached quantities via update()
        """
        h = self.step()
        x = self.emb.x
        self.emb.x = x+h 
        self.update()

    def step(self) -> npt.NDArray[np.floating]:
        """Compute an update step for the optimizable params.
        
        Returns:
            Step vector h with same shape as embedder parameters.
            The step is applied as: x_new = x_old + h
        """
        pass

    def log(self, T: Optional[time.struct_time] = None) -> None:
        if T is None:
            T = time.localtime()
        name = self.__class__.__name__
        params = self.emb.params
        if self.trajectory is not None:
            args = (" " * len(name), "Step", "Time")
            msg = "%s  %4s %8s\n" % args
            self.trajectory.write(msg)
            args = (name, self.nsteps, T[3], T[4], T[5])
            msg = "%s:  %3d %02d:%02d:%02d\n" % args
            self.trajectory.write(msg)
            self.trajectory.write("=== Embedding parameters ===\n")
            self.trajectory.write(np.array2string(
                params,
                max_line_width=70,    # wrap lines to 70 chars
                separator=", ",       # separator
                suppress_small=False, # do not represent small numbers with 0
                threshold=1_000_000,  # basically, always print full array
                formatter={'float_kind':lambda x: f"{x: .6e}"}
            ))
            self.trajectory.write("\n")
            self.trajectory.write("==== End of parameters =====\n")
            self.trajectory.write("\n")
            self.trajectory.flush()


