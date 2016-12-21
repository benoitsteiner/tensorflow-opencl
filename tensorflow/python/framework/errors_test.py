# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.errors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import test


class ErrorsTest(test.TestCase):

  def testUniqueClassForEachErrorCode(self):
    for error_code, exc_type in [
        (errors.CANCELLED, errors_impl.CancelledError),
        (errors.UNKNOWN, errors_impl.UnknownError),
        (errors.INVALID_ARGUMENT, errors_impl.InvalidArgumentError),
        (errors.DEADLINE_EXCEEDED, errors_impl.DeadlineExceededError),
        (errors.NOT_FOUND, errors_impl.NotFoundError),
        (errors.ALREADY_EXISTS, errors_impl.AlreadyExistsError),
        (errors.PERMISSION_DENIED, errors_impl.PermissionDeniedError),
        (errors.UNAUTHENTICATED, errors_impl.UnauthenticatedError),
        (errors.RESOURCE_EXHAUSTED, errors_impl.ResourceExhaustedError),
        (errors.FAILED_PRECONDITION, errors_impl.FailedPreconditionError),
        (errors.ABORTED, errors_impl.AbortedError),
        (errors.OUT_OF_RANGE, errors_impl.OutOfRangeError),
        (errors.UNIMPLEMENTED, errors_impl.UnimplementedError),
        (errors.INTERNAL, errors_impl.InternalError),
        (errors.UNAVAILABLE, errors_impl.UnavailableError),
        (errors.DATA_LOSS, errors_impl.DataLossError),
    ]:
      # pylint: disable=protected-access
      self.assertTrue(
          isinstance(
              errors_impl._make_specific_exception(None, None, None,
                                                   error_code), exc_type))
      # pylint: enable=protected-access

  def testKnownErrorClassForEachErrorCodeInProto(self):
    for error_code in error_codes_pb2.Code.values():
      # pylint: disable=line-too-long
      if error_code in (
          error_codes_pb2.OK, error_codes_pb2.
          DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_
      ):
        continue
      # pylint: enable=line-too-long
      with warnings.catch_warnings(record=True) as w:
        # pylint: disable=protected-access
        exc = errors_impl._make_specific_exception(None, None, None, error_code)
        # pylint: enable=protected-access
      self.assertEqual(0, len(w))  # No warning is raised.
      self.assertTrue(isinstance(exc, errors_impl.OpError))
      self.assertTrue(errors_impl.OpError in exc.__class__.__bases__)

  def testUnknownErrorCodeCausesWarning(self):
    with warnings.catch_warnings(record=True) as w:
      # pylint: disable=protected-access
      exc = errors_impl._make_specific_exception(None, None, None, 37)
      # pylint: enable=protected-access
    self.assertEqual(1, len(w))
    self.assertTrue("Unknown error code: 37" in str(w[0].message))
    self.assertTrue(isinstance(exc, errors_impl.OpError))


if __name__ == "__main__":
  test.main()
