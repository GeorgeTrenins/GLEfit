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
from pathlib import Path
from typing import Optional, Union, IO
from contextlib import ExitStack
import functools
import os
import sys 


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
            logfile: Optional[Union[IO, str, Path]] = None,
            **kwargs
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        emb : BaseEmbedder
            The embedder to optimize.
        logfile : file-like object, str, or pathlib.Path, optional
            File to log optimization progress. If None, logging is disabled.
            (default: None)
        """
        if not isinstance(emb, BaseEmbedder):
            raise ValueError("Optimizer requires a BaseEmbedder instance.")
        self.emb = emb
        self.logfile = self.openfile(file=logfile, mode='a')
        self.nsteps = 0
        self.max_steps = 0 # set by run or irun
        self.fmax = None   # set by run or irun
        self.maxstep = self.defaults['maxstep']