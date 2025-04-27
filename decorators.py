#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decorator utilities"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import functools
import inspect
import six

try:
    from numba.decorators import jit as optional_jit
except ImportError:
    try:
        from numba import jit as optional_jit
    except ImportError:
        def optional_jit(*args, **kwargs):
            '''Dummy decorator'''
            if len(args) == 1 and callable(args[0]):
                return args[0]
            else:
                return lambda func: func


def moved(moved_from, version, version_removed):
    '''This is a decorator which can be used to mark functions
    as moved/renamed. It will result in a warning being emitted when the function is used.'''
    def decorator(func, *args, **kwargs):
        '''Decorator for moved functions.'''
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            '''Wrapped function for moved functions.'''
            warnings.warn('{:s} has been moved in version {:s}.\n'
                          'It will be removed in version {:s}.\n'
                          'Please use {:s}.{:s} instead.'.format(
                              moved_from.__name__,
                              version, version_removed,
                              func.__module__, func.__name__),
                          category=DeprecationWarning,
                          stacklevel=2)

            return func(*args, **kwargs)

        # Deprecate the function, but don't worry about the return type
        # of optional_jit
        return wrapped
    return decorator


def deprecated(version, version_removed):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted when the function is used.'''
    def decorator(func, *args, **kwargs):
        '''Decorator for deprecated functions.'''
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            '''Wrapped function for deprecated functions.'''
            warnings.warn('Function {:s} is deprecated in version {:s}.\n'
                          'It will be removed in version {:s}.'
                          .format(func.__name__, version, version_removed),
                          category=DeprecationWarning,
                          stacklevel=2)

            return func(*args, **kwargs)

        # Deprecate the function, but don't worry about the return type
        # of optional_jit
        return wrapped
    return decorator


def vectorize(*, otypes=None):
    '''Decorator to make a function vectorize across multiple axes.
    This differs from np.vectorize, which only vectorizes across a single axis.
    Parameters
    ----------
    otypes : list of types, optional
        The output types. If not provided, they will be determined by
        inspection.
    Returns
    -------
    vectorized : callable
        Vectorized function
    '''
    def decorator(function):
        '''Decorator'''
        def wrapper(x, *args, **kwargs):
            '''Wrapper'''
            if not hasattr(x, 'shape'):
                return function(x, *args, **kwargs)

            input_shape = x.shape
            if len(input_shape):
                x_flat = x.reshape(-1)
                results = [function(xi, *args, **kwargs) for xi in x_flat]
                out_shape = input_shape
            else:
                results = [function(x, *args, **kwargs)]
                out_shape = tuple()

            if otypes is None:
                return np.array(results).reshape(out_shape)
            else:
                return np.array(results, dtype=otypes).reshape(out_shape)

            return wrapper
    return decorator
