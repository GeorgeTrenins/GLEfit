#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/09/11 10:02:48
@Author  :   George Trenins
@Desc    :   Base class for auxiliary variable parameter optimization - based on ASE's Optimizer
'''

from __future__ import print_function, division, absolute_import
from glefit.embedding import BaseEmbedder
from glefit.merit import BaseProperty
from pathlib import Path
from typing import Optional, Union, IO
from contextlib import ExitStack
import os
import sys
import numpy as np
import numpy.typing as npt
import time

MAX_ITERATIONS_DEFAULT = 1_000_000
DEFAULT_TRAJ_FILE = "traj.out"


def _format(values: npt.ArrayLike) -> str:
    """Format an array for the log and the trajectory: every entry, with 7
    significant digits, in lines of at most 70 characters."""
    return np.array2string(
        np.asarray(values),
        max_line_width=70,
        separator=", ",
        suppress_small=False,
        threshold=1_000_000,
        formatter={'float_kind': lambda x: f"{x: .6e}"})


class _DelExitStack(ExitStack):
    """An ExitStack that also closes its files when garbage-collected."""
    def __del__(self):
        self.close()


class Optimizer:
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
            If '-', write to stdout. (default: None)
        trajfile: file-like object, str, or pathlib.Path, optional
            File to write optimizable parameter values over the course of the 
            optimization. If 'None', write to default location 'traj.out'.
            If '-', write to stdout.
        """
        if not isinstance(emb, BaseEmbedder):
            raise ValueError("Optimizer requires a BaseEmbedder instance.")
        self.emb = emb
        self.merit = merit_function
        self._exitstack = _DelExitStack()
        self.logfile = self.openfile(file=logfile)
        if trajfile is None:
            trajfile = DEFAULT_TRAJ_FILE
        self.trajectory = self.openfile(file=trajfile)
        self.nsteps = 0
        self.initialize()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        """Close the files that the optimizer opened."""
        self._exitstack.close()

    def openfile(
            self,
            file: Optional[Union[IO, str, Path]],
            mode: str = 'w'
    ) -> IO:
        """Return `file` itself if it is a file object, which its caller
        closes; stdout for '-'; or the file at that path, os.devnull for
        None, which the optimizer closes when closed or garbage-collected.
        """
        if hasattr(file, 'close'):
            return file
        if file == '-':
            return sys.stdout
        if file is None:
            file = os.devnull
        return self._exitstack.enter_context(
            open(file, mode=mode, encoding='utf-8'))

    def run(self, steps: int = MAX_ITERATIONS_DEFAULT, options: Optional[dict] = None) -> bool:
        """Run the optimizer.

        This method will return whenever the convergence criteria are fulfilled
        or when the number of steps exceeds `steps`.

        Args:
            steps (int, optional): Maximum number of steps to take.
                Defaults to MAX_ITERATIONS_DEFAULT.
            options (dict, optional): Optimizer-specific options for controlling convergence 
                (e.g., tolerance thresholds). Defaults to None.
        
        Returns:
            bool: True if converged, False if max steps reached.
        """
        # a later call continues the count of steps, and does not log the
        # start again
        max_iter = self.nsteps + steps
        if self.nsteps == 0:
            self.log()
        # check if we are already converged
        is_converged = self.converged()
        if is_converged:
            # Already converged!
            return True
        # start the actual optimization
        while not is_converged and self.nsteps < max_iter:
            self.iteration()
            is_converged = self.converged()
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
        3. Count the step, so that update() knows its number
        4. Refresh cached quantities via update()
        """
        h = self.step()
        x = self.emb.x
        self.emb.x = x+h
        self.nsteps += 1
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
        args = (name, self.nsteps, T[3], T[4], T[5])
        msg = "%s | Step %3d, Time %02d:%02d:%02d\n" % args
        self.trajectory.write(msg)
        self.trajectory.write(_format(self.emb.conventional_params))
        self.trajectory.write("\n\n")
        self.trajectory.flush()


