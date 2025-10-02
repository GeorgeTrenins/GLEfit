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
from glefit.merit import BaseMerit
from pathlib import Path
from typing import Optional, Union, IO
from contextlib import ExitStack
import functools
import os
import sys 
import numpy as np
import time

DEFAULT_MAX_STEPS = 1_000_000
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
    """Base class for all GLE auxiliary variable optimizers."""

    defaults = {'maxstep': 0.2}

    def __init__(
            self,
            emb: BaseEmbedder,
            merit_function: BaseMerit,
            logfile: Optional[Union[IO, str, Path]] = None,
            trajfile: Optional[Union[IO, str, Path]] = None,
            **kwargs
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        emb : BaseEmbedder
            The Markovian embedder to optimize
        merit_function: BaseMerit
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
        self.max_steps = 0 # set by run
        self.maxstep = self.defaults['maxstep']
        self.initialize()

    def run(self, steps: int = DEFAULT_MAX_STEPS, options: Optional[dict] = None):
        """Run the optimizer.

        This method will return whenever the convergence criteria are fulfilled
        or when the number of steps exceeds `steps`.

        Args:
            steps (int, optional): Maximum number of steps to take. Defaults to DEFAULT_MAX_STEPS.
        
        Options:
            optimizer-specific options for controlling convergence
        """
        # update the maximum number of steps 
        # TODO: this will become relevant if we implement restarts
        self.max_steps = self.nsteps + steps
        if self.nsteps == 0:
            self.log()
        # check if we are already converged
        is_converged = self.converged()
        if is_converged:
            # Already converged!
            return True
        # start the actual optimization
        while not is_converged and self.nsteps < self.max_steps:
            self.iteration()
            is_converged = self.converged()
            self.nsteps += 1
            self.log()
        return is_converged
    
    def initialize(self):
        """Compute any relevant quantities for judging initial convergence
        and making first optimization step
        """
        pass
    
    def update(self):
        """Update the relevant quantities for judging convergence / making 
        next step
        """
        pass

    def converged(self):
        """Optimization convergence criterion"""
        raise NotImplementedError

    def iteration(self):
        h = self.step()
        x = self.emb.x
        self.emb.x = x+h
        self.update()

    def step(self):
        """Compute an update step for the optimizable params"""
        pass

    def log(self, T=None):
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


