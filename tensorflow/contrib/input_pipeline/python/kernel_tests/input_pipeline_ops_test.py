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
"""Tests for input_pipeline_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.input_pipeline.python.ops import input_pipeline_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as var_ops


class InputPipelineOpsTest(tf.test.TestCase):

  def testObtainNext(self):
    with self.test_session():
      var = state_ops.variable_op([1], tf.int64)
      tf.assign(var, [-1]).op.run()
      c = tf.constant(["a", "b"])
      sample1 = input_pipeline_ops.obtain_next(c, var)
      self.assertEqual(b"a", sample1.eval())
      self.assertEqual([0], var.eval())
      sample2 = input_pipeline_ops.obtain_next(c, var)
      self.assertEqual(b"b", sample2.eval())
      self.assertEqual([1], var.eval())
      sample3 = input_pipeline_ops.obtain_next(c, var)
      self.assertEqual(b"a", sample3.eval())
      self.assertEqual([0], var.eval())

  def testSeekNext(self):
    string_list = ["a", "b", "c"]
    with self.test_session() as session:
      elem = input_pipeline_ops.seek_next(string_list)
      session.run(tf.initialize_all_variables())
      self.assertEqual(b"a", session.run(elem))
      self.assertEqual(b"b", session.run(elem))
      self.assertEqual(b"c", session.run(elem))
      self.assertEqual(b"a", session.run(elem))


if __name__ == "__main__":
  tf.test.main()
