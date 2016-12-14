# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tensor utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect
import re

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils


def _add_deprecated_function_notice_to_docstring(doc, date, instructions):
  """Adds a deprecation notice to a docstring for deprecated functions."""
  return decorator_utils.add_notice_to_docstring(
      doc, instructions,
      'DEPRECATED FUNCTION',
      '(deprecated)', [
          'THIS FUNCTION IS DEPRECATED. It will be removed after %s.' % date,
          'Instructions for updating:'])


def _add_deprecated_arg_notice_to_docstring(doc, date, instructions):
  """Adds a deprecation notice to a docstring for deprecated arguments."""
  return decorator_utils.add_notice_to_docstring(
      doc, instructions,
      'DEPRECATED FUNCTION ARGUMENTS',
      '(deprecated arguments)', [
          'SOME ARGUMENTS ARE DEPRECATED. '
          'They will be removed after %s.' % date,
          'Instructions for updating:'])


def _validate_deprecation_args(date, instructions):
  if not date:
    raise ValueError('Tell us what date this will be deprecated!')
  if not re.match(r'20\d\d-[01]\d-[0123]\d', date):
    raise ValueError('Date must be YYYY-MM-DD.')
  if not instructions:
    raise ValueError('Don\'t deprecate things without conversion instructions!')


def _call_location(level=2):
  """Returns call location given level up from current call."""
  stack = inspect.stack()
  # Check that stack has enough elements.
  if len(stack) > level:
    location = stack[level]
    return '%s:%d in %s.' % (location[1], location[2], location[3])
  return '<unknown>'


def deprecated(date, instructions):
  """Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String. The date the function is scheduled to be removed. Must be
      ISO 8601 (YYYY-MM-DD).
    instructions: String. Instructions on how to update code using the
      deprecated function.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not in ISO 8601 format, or instructions are empty.
  """
  _validate_deprecation_args(date, instructions)

  def deprecated_wrapper(func):
    """Deprecation wrapper."""
    decorator_utils.validate_callable(func, 'deprecated')
    @functools.wraps(func)
    def new_func(*args, **kwargs):
      logging.warning(
          'From %s: %s (from %s) is deprecated and will be removed '
          'after %s.\n'
          'Instructions for updating:\n%s',
          _call_location(), decorator_utils.get_qualified_name(func),
          func.__module__, date, instructions)
      return func(*args, **kwargs)
    new_func.__doc__ = _add_deprecated_function_notice_to_docstring(
        func.__doc__, date, instructions)
    return new_func
  return deprecated_wrapper


DeprecatedArgSpec = collections.namedtuple(
    'DeprecatedArgSpec', ['position', 'has_ok_value', 'ok_value'])


def deprecated_args(date, instructions, *deprecated_arg_names_or_tuples):
  """Decorator for marking specific function arguments as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument. It has the following format:

    Calling <function> (from <module>) with <arg> is deprecated and will be
    removed after <date>. Instructions for updating:
      <instructions>

  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String. The date the function is scheduled to be removed. Must be
      ISO 8601 (YYYY-MM-DD).
    instructions: String. Instructions on how to update code using the
      deprecated function.
    *deprecated_arg_names_or_tuples: String. or 2-Tuple(String,
      [ok_vals]).  The string is the deprecated argument name.
      Optionally, an ok-value may be provided.  If the user provided
      argument equals this value, the warning is suppressed.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not in ISO 8601 format, instructions are
      empty, the deprecated arguments are not present in the function
      signature, or the second element of a deprecated_tuple is not a
      list.
  """
  _validate_deprecation_args(date, instructions)
  if not deprecated_arg_names_or_tuples:
    raise ValueError('Specify which argument is deprecated.')

  def _get_arg_names_to_ok_vals():
    """Returns a dict mapping arg_name to DeprecatedArgSpec w/o position."""
    d = {}
    for name_or_tuple in deprecated_arg_names_or_tuples:
      if isinstance(name_or_tuple, tuple):
        d[name_or_tuple[0]] = DeprecatedArgSpec(-1, True, name_or_tuple[1])
      else:
        d[name_or_tuple] = DeprecatedArgSpec(-1, False, None)
    return d

  def _get_deprecated_positional_arguments(names_to_ok_vals, arg_spec):
    """Builds a dictionary from deprecated arguments to thier spec.

    Returned dict is keyed by argument name.
    Each value is a DeprecatedArgSpec with the following fields:
       position: The zero-based argument position of the argument
         within the signature.  None if the argument isn't found in
         the signature.
       ok_values:  Values of this argument for which warning will be
         suppressed.

    Args:
      names_to_ok_vals: dict from string arg_name to a list of values,
        possibly empty, which should not elicit a warning.
      arg_spec: Output from inspect.getargspec on the called function.

    Returns:
      Dictionary from arg_name to DeprecatedArgSpec.
    """
    arg_name_to_pos = dict(
        (name, pos) for (pos, name) in enumerate(arg_spec.args))
    deprecated_positional_args = {}
    for arg_name, spec in iter(names_to_ok_vals.items()):
      if arg_name in arg_name_to_pos:
        pos = arg_name_to_pos[arg_name]
        deprecated_positional_args[arg_name] = DeprecatedArgSpec(
            pos, spec.has_ok_value, spec.ok_value)
    return deprecated_positional_args

  def deprecated_wrapper(func):
    """Deprecation decorator."""
    decorator_utils.validate_callable(func, 'deprecated_args')
    deprecated_arg_names = _get_arg_names_to_ok_vals()

    arg_spec = inspect.getargspec(func)
    deprecated_positions = _get_deprecated_positional_arguments(
        deprecated_arg_names, arg_spec)

    is_varargs_deprecated = arg_spec.varargs in deprecated_arg_names
    is_kwargs_deprecated = arg_spec.keywords in deprecated_arg_names

    if (len(deprecated_positions) + is_varargs_deprecated + is_kwargs_deprecated
        != len(deprecated_arg_names_or_tuples)):
      known_args = arg_spec.args + [arg_spec.varargs, arg_spec.keywords]
      missing_args = [arg_name for arg_name in deprecated_arg_names
                      if arg_name not in known_args]
      raise ValueError('The following deprecated arguments are not present '
                       'in the function signature: %s. '
                       'Found next arguments: %s.' % (missing_args, known_args))

    @functools.wraps(func)
    def new_func(*args, **kwargs):
      """Deprecation wrapper."""
      invalid_args = []
      named_args = inspect.getcallargs(func, *args, **kwargs)
      for arg_name, spec in iter(deprecated_positions.items()):
        if (spec.position < len(args) and
            not (spec.has_ok_value and
                 named_args[arg_name] == spec.ok_value)):
          invalid_args.append(arg_name)
      if is_varargs_deprecated and len(args) > len(arg_spec.args):
        invalid_args.append(arg_spec.varargs)
      if is_kwargs_deprecated and kwargs:
        invalid_args.append(arg_spec.keywords)
      for arg_name in deprecated_arg_names:
        if (arg_name in kwargs and
            not (deprecated_positions[arg_name].has_ok_value and
                 (named_args[arg_name] ==
                  deprecated_positions[arg_name].ok_value))):
          invalid_args.append(arg_name)
      for arg_name in invalid_args:
        logging.warning(
            'From %s: calling %s (from %s) with %s is deprecated and will '
            'be removed after %s.\nInstructions for updating:\n%s',
            _call_location(), decorator_utils.get_qualified_name(func),
            func.__module__, arg_name, date, instructions)
      return func(*args, **kwargs)
    new_func.__doc__ = _add_deprecated_arg_notice_to_docstring(
        func.__doc__, date, instructions)
    return new_func
  return deprecated_wrapper


def deprecated_arg_values(date, instructions, **deprecated_kwargs):
  """Decorator for marking specific function argument values as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument values. It has the following format:

    Calling <function> (from <module>) with <arg>=<value> is deprecated and
    will be removed after <date>. Instructions for updating:
      <instructions>

  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String. The date the function is scheduled to be removed. Must be
      ISO 8601 (YYYY-MM-DD).
    instructions: String. Instructions on how to update code using the
      deprecated function.
    **deprecated_kwargs: The deprecated argument values.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not in ISO 8601 format, or instructions are empty.
  """
  _validate_deprecation_args(date, instructions)
  if not deprecated_kwargs:
    raise ValueError('Specify which argument values are deprecated.')

  def deprecated_wrapper(func):
    """Deprecation decorator."""
    decorator_utils.validate_callable(func, 'deprecated_arg_values')
    @functools.wraps(func)
    def new_func(*args, **kwargs):
      """Deprecation wrapper."""
      named_args = inspect.getcallargs(func, *args, **kwargs)
      for arg_name, arg_value in deprecated_kwargs.items():
        if arg_name in named_args and named_args[arg_name] == arg_value:
          logging.warning(
              'From %s: calling %s (from %s) with %s=%s is deprecated and will '
              'be removed after %s.\nInstructions for updating:\n%s',
              _call_location(), decorator_utils.get_qualified_name(func),
              func.__module__, arg_name, arg_value, date, instructions)
      return func(*args, **kwargs)
    new_func.__doc__ = _add_deprecated_arg_notice_to_docstring(
        func.__doc__, date, instructions)
    return new_func
  return deprecated_wrapper


def deprecated_argument_lookup(new_name, new_value, old_name, old_value):
  """Looks up deprecated argument name and ensures both are not used.

  Args:
    new_name: new name of argument
    new_value: value of new argument (or None if not used)
    old_name: old name of argument
    old_value: value of old argument (or None if not used)
  Returns:
    The effective argument that should be used.
  Raises:
    ValueError: if new_value and old_value are both non-null
  """
  if old_value is not None:
    if new_value is not None:
      raise ValueError("Cannot specify both '%s' and '%s'" %
                       (old_name, new_name))
    return old_value
  return new_value


def rewrite_argument_docstring(old_doc, old_argument, new_argument):
  return old_doc.replace('`%s`' % old_argument, '`%s`' % new_argument).replace(
      '%s:' % old_argument, '%s:' % new_argument)
