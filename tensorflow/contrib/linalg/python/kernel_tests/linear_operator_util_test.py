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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.linalg.python.ops import linear_operator_util


linalg = tf.contrib.linalg
tf.set_random_seed(23)


class AssertZeroImagPartTest(tf.test.TestCase):

  def test_real_tensor_doesnt_raise(self):
    x = tf.convert_to_tensor([0., 2, 3])
    with self.test_session():
      # Should not raise.
      linear_operator_util.assert_zero_imag_part(x, message="ABC123").run()

  def test_complex_tensor_with_imag_zero_doesnt_raise(self):
    x = tf.convert_to_tensor([1., 0, 3])
    y = tf.convert_to_tensor([0., 0, 0])
    z = tf.complex(x, y)
    with self.test_session():
      # Should not raise.
      linear_operator_util.assert_zero_imag_part(z, message="ABC123").run()

  def test_complex_tensor_with_nonzero_imag_raises(self):
    x = tf.convert_to_tensor([1., 2, 0])
    y = tf.convert_to_tensor([1., 2, 0])
    z = tf.complex(x, y)
    with self.test_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_zero_imag_part(z, message="ABC123").run()


class AssertNoEntriesWithModulusZeroTest(tf.test.TestCase):

  def test_nonzero_real_tensor_doesnt_raise(self):
    x = tf.convert_to_tensor([1., 2, 3])
    with self.test_session():
      # Should not raise.
      linear_operator_util.assert_no_entries_with_modulus_zero(
          x, message="ABC123").run()

  def test_nonzero_complex_tensor_doesnt_raise(self):
    x = tf.convert_to_tensor([1., 0, 3])
    y = tf.convert_to_tensor([1., 2, 0])
    z = tf.complex(x, y)
    with self.test_session():
      # Should not raise.
      linear_operator_util.assert_no_entries_with_modulus_zero(
          z, message="ABC123").run()

  def test_zero_real_tensor_raises(self):
    x = tf.convert_to_tensor([1., 0, 3])
    with self.test_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_no_entries_with_modulus_zero(
            x, message="ABC123").run()

  def test_zero_complex_tensor_raises(self):
    x = tf.convert_to_tensor([1., 2, 0])
    y = tf.convert_to_tensor([1., 2, 0])
    z = tf.complex(x, y)
    with self.test_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_no_entries_with_modulus_zero(
            z, message="ABC123").run()


if __name__ == "__main__":
  tf.test.main()
