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
"""Tests for numpy_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.learn.python.learn.learn_io import numpy_io
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


class NumpyIoTest(test.TestCase):

  def testNumpyInputFn(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    y = np.arange(-32, -28)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [0, 1])
      self.assertAllEqual(res[0]['b'], [32, 33])
      self.assertAllEqual(res[1], [-32, -31])

      session.run([features, target])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithDifferentDimensionsOfFeatures(self):
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    x = {'a': a, 'b': b}
    y = np.arange(-32, -30)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [[1, 2], [3, 4]])
      self.assertAllEqual(res[0]['b'], [5, 6])
      self.assertAllEqual(res[1], [-32, -31])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithXAsNonDict(self):
    x = np.arange(32, 36)
    y = np.arange(4)
    with self.test_session():
      with self.assertRaisesRegexp(TypeError, 'x must be dict'):
        failing_input_fn = numpy_io.numpy_input_fn(
            x, y, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testNumpyInputFnWithTargetKeyAlreadyInX(self):
    array = np.arange(32, 36)
    x = {'__target_key__': array}
    y = np.arange(4)

    with self.test_session():
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      input_fn()
      self.assertAllEqual(x['__target_key__'], array)
      self.assertAllEqual(x['__target_key___n'], y)

  def testNumpyInputFnWithMismatchLengthOfInputs(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    x_mismatch_length = {'a': np.arange(1), 'b': b}
    y_longer_length = np.arange(10)

    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, 'Length of tensors in x and y is mismatched.'):
        failing_input_fn = numpy_io.numpy_input_fn(
            x, y_longer_length, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

      with self.assertRaisesRegexp(
          ValueError, 'Length of tensors in x and y is mismatched.'):
        failing_input_fn = numpy_io.numpy_input_fn(
            x=x_mismatch_length,
            y=None,
            batch_size=2,
            shuffle=False,
            num_epochs=1)
        failing_input_fn()


if __name__ == '__main__':
  test.main()
