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
"""Tests for tf.contrib.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

# TODO(sguada) Expose tf.with_dependencies
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers.python.layers import layers as _layers
from tensorflow.python.ops import state_ops


class AvgPool2DTest(tf.test.TestCase):

  def testInvalidDataFormat(self):
    height, width = 3, 6
    images = np.random.uniform(size=(5, height, width, 3))
    with self.assertRaisesRegexp(
        ValueError, 'data_format has to be either NCHW or NHWC.'):
      tf.contrib.layers.avg_pool2d(images, [3, 3], data_format='CHWN')

  def testCreateAvgPool(self):
    height, width = 3, 6
    images = np.random.uniform(size=(5, height, width, 3))
    output = tf.contrib.layers.avg_pool2d(images, [3, 3])
    self.assertEqual(output.op.name, 'AvgPool2D/AvgPool')
    self.assertListEqual(output.get_shape().as_list(), [5, 1, 2, 3])

  def testCreateAvgPoolNCHW(self):
    height, width = 3, 6
    images = np.random.uniform(size=(5, 2, height, width))
    output = tf.contrib.layers.avg_pool2d(images, [3, 3], data_format='NCHW')
    self.assertEquals(output.op.name, 'AvgPool2D/AvgPool')
    self.assertListEqual(output.get_shape().as_list(), [5, 2, 1, 2])

  def testCollectOutputs(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.avg_pool2d(images, [3, 3],
                                          outputs_collections='outputs')
    output_collected = tf.get_collection('outputs')[0]
    self.assertEqual(output_collected.alias, 'AvgPool2D')
    self.assertEqual(output_collected, output)

  def testCreateSquareAvgPool(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.avg_pool2d(images, 3)
    self.assertEqual(output.op.name, 'AvgPool2D/AvgPool')
    self.assertListEqual(output.get_shape().as_list(), [5, 1, 2, 3])

  def testCreateAvgPoolWithScope(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.avg_pool2d(images, [3, 3], scope='pool1')
    self.assertEqual(output.op.name, 'pool1/AvgPool')

  def testCreateAvgPoolWithSamePadding(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.avg_pool2d(images, [3, 3], padding='SAME')
    self.assertListEqual(output.get_shape().as_list(), [5, 2, 3, 3])

  def testCreateAvgPoolWithSamePaddingNCHW(self):
    height, width = 3, 6
    images = tf.random_uniform((5, 3, height, width), seed=1)
    output = tf.contrib.layers.avg_pool2d(images, [3, 3], padding='SAME',
                                          data_format='NCHW')
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 2, 3])

  def testCreateAvgPoolStrideWithSamePadding(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.avg_pool2d(images, [3, 3], stride=1,
                                          padding='SAME')
    self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testGlobalAvgPool(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.avg_pool2d(images, images.get_shape()[1:3],
                                          stride=1)
    self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])


class PoolTest(tf.test.TestCase):

  def testCreatePool(self):
    height, width = 3, 3
    images = np.random.uniform(size=(5, height, width, 3))
    output = tf.contrib.layers.pool(images, [3, 3], pooling_type='AVG')
    self.assertEqual(output.op.name, 'avg_pool')
    self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCreatePoolNCHW(self):
    height, width = 3, 3
    images = np.random.uniform(size=(5, 3, height, width))
    output = tf.contrib.layers.pool(
        images, [3, 3], pooling_type='AVG', data_format='NCHW')
    self.assertEqual(output.op.name, 'avg_pool')
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 1, 1])

  def testCollectOutputs(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.pool(images, [3, 3],
                                    pooling_type='AVG',
                                    outputs_collections='outputs')
    output_collected = tf.get_collection('outputs')[0]
    self.assertEqual(output_collected.alias, 'avg_pool')
    self.assertEqual(output_collected, output)

  def testCreateSquareAvgPool(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.pool(images, 3, pooling_type='AVG')
    self.assertEqual(output.op.name, 'avg_pool')
    self.assertEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCreateMaxPoolWithScope(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.pool(
        images, [3, 3], pooling_type='MAX', scope='pool1')
    self.assertEqual(output.op.name, 'pool1')

  def testCreateMaxPoolWithSamePadding(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.pool(
        images, [3, 3], pooling_type='MAX', padding='SAME')
    self.assertEqual(output.get_shape().as_list(), [5, 3, 3, 3])

  def testCreateAvgPoolStrideWithSamePadding(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.pool(
        images, [3, 3], stride=1, padding='SAME', pooling_type='AVG')
    self.assertEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testGlobalAvgPool(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.pool(
        images, images.get_shape()[1:3], stride=1, pooling_type='AVG')
    self.assertEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testAvgPoolWithStride(self):
    height, width = 5, 8
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.pool(
        images, [2, 3], stride=[1, 2], pooling_type='AVG')
    self.assertEqual(output.get_shape().as_list(), [5, 4, 3, 3])

  def testAvgPoolWithDilation(self):
    height, width = 5, 8
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.pool(
        images, [2, 3], dilation_rate=[1, 2], pooling_type='AVG')
    self.assertEqual(output.get_shape().as_list(), [5, 4, 4, 3])

  def testAvgPoolWithDilationNCHW(self):
    height, width = 5, 8
    images = tf.random_uniform((5, 3, height, width), seed=1)
    output = tf.contrib.layers.pool(
        images, [2, 3], dilation_rate=[1, 2], pooling_type='AVG', data_format='NCHW')
    self.assertEqual(output.get_shape().as_list(), [5, 3, 4, 4])


class BiasAddTest(tf.test.TestCase):

  def testCreate(self):
    height, width = 3, 3
    with self.test_session():
      images = np.random.uniform(size=(5, height, width, 3))
      output = tf.contrib.layers.bias_add(images)
      self.assertEqual(output.op.name, 'BiasAdd/BiasAdd')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testCreateWithActivation(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.bias_add(images, activation_fn=tf.nn.relu)
      self.assertEqual(output.op.name, 'BiasAdd/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testCreateDimensions(self):
    dims = (2, 3, 4)
    shape = [5, 2, 3, 4]
    with self.test_session():
      for d in dims:
        input_shape = shape[:d]
        inputs = tf.random_uniform(input_shape, seed=1)
        output = tf.contrib.layers.bias_add(inputs)
        self.assertListEqual(output.get_shape().as_list(), input_shape)
        biases = tf.contrib.framework.get_variables_by_name('biases')[-1]
        self.assertListEqual(biases.get_shape().as_list(), [input_shape[-1]])


class ConvolutionTest(tf.test.TestCase):

  def testInvalidDataFormat(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      with self.assertRaisesRegexp(
          ValueError, 'data_format'):
        tf.contrib.layers.convolution2d(images, 32, 3, data_format='CHWN')

  def testCreateConv(self):
    height, width = 7, 9
    with self.test_session():
      images = np.random.uniform(size=(5, height, width, 4))
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3])
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])
      weights = tf.contrib.framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 3, 4, 32])
      biases = tf.contrib.framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateConvNCHW(self):
    height, width = 7, 9
    with self.test_session():
      images = np.random.uniform(size=(5, 4, height, width))
      output = tf.contrib.layers.convolution2d(
          images, 32, [3, 3], data_format='NCHW')
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 32, height, width])
      weights = tf.contrib.framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 3, 4, 32])
      biases = tf.contrib.framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateSquareConv(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, 3)
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateConvWithTensorShape(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32,
                                               images.get_shape()[1:3])
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateFullyConv(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      output = tf.contrib.layers.convolution2d(images, 64,
                                               images.get_shape()[1:3],
                                               padding='VALID')
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 64])
      biases = tf.contrib.framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [64])

  def testCreateVerticalConv(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 4), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 1])
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height, width, 32])
      weights = tf.contrib.framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 1, 4, 32])
      biases = tf.contrib.framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateHorizontalConv(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 4), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [1, 3])
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height, width, 32])
      weights = tf.contrib.framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [1, 3, 4, 32])

  def testCreateConvWithStride(self):
    height, width = 6, 8
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3], stride=2)
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height/2, width/2, 32])

  def testCreateConvCreatesWeightsAndBiasesVars(self):
    height, width = 7, 9
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with self.test_session():
      self.assertFalse(tf.contrib.framework.get_variables('conv1/weights'))
      self.assertFalse(tf.contrib.framework.get_variables('conv1/biases'))
      tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1')
      self.assertTrue(tf.contrib.framework.get_variables('conv1/weights'))
      self.assertTrue(tf.contrib.framework.get_variables('conv1/biases'))

  def testCreateConvWithScope(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                               scope='conv1')
      self.assertEqual(output.op.name, 'conv1/Relu')

  def testCreateConvWithCollection(self):
    height, width = 7, 9
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with tf.name_scope('fe'):
      conv = tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                             outputs_collections='outputs',
                                             scope='Conv')
    output_collected = tf.get_collection('outputs')[0]
    self.assertEqual(output_collected.alias, 'fe/Conv')
    self.assertEqual(output_collected, conv)

  def testCreateConvWithoutActivation(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                               activation_fn=None)
      self.assertEqual(output.op.name, 'Conv/BiasAdd')

  def testCreateConvValid(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                               padding='VALID')
      self.assertListEqual(output.get_shape().as_list(), [5, 5, 7, 32])

  def testCreateConvWithWD(self):
    height, width = 7, 9
    weight_decay = 0.01
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1)
      regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
      tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                      weights_regularizer=regularizer)
      l2_loss = tf.nn.l2_loss(
          tf.contrib.framework.get_variables_by_name('weights')[0])
      wd = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertEqual(wd.op.name,
                        'Conv/weights/Regularizer/l2_regularizer')
      sess.run(tf.global_variables_initializer())
      self.assertAlmostEqual(sess.run(wd), weight_decay * l2_loss.eval())

  def testCreateConvNoRegularizers(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.convolution2d(images, 32, [3, 3])
      self.assertEqual(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), [])

  def testReuseVars(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1')
      self.assertEqual(len(tf.contrib.framework.get_variables()), 2)
      tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1',
                                      reuse=True)
      self.assertEqual(len(tf.contrib.framework.get_variables()), 2)

  def testNonReuseVars(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.convolution2d(images, 32, [3, 3])
      self.assertEqual(len(tf.contrib.framework.get_variables()), 2)
      tf.contrib.layers.convolution2d(images, 32, [3, 3])
      self.assertEqual(len(tf.contrib.framework.get_variables()), 4)

  def testReuseConvWithWD(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      weight_decay = tf.contrib.layers.l2_regularizer(0.01)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.convolution2d],
          weights_regularizer=weight_decay):
        tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1')
        self.assertEqual(len(tf.contrib.framework.get_variables()), 2)
        self.assertEqual(
            len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)
        tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1',
                                        reuse=True)
        self.assertEqual(len(tf.contrib.framework.get_variables()), 2)
        self.assertEqual(
            len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)

  def testConvWithBatchNorm(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.convolution2d],
          normalizer_fn=tf.contrib.layers.batch_norm,
          normalizer_params={'decay': 0.9}):
        net = tf.contrib.layers.convolution2d(images, 32, [3, 3])
        net = tf.contrib.layers.convolution2d(net, 32, [3, 3])
      self.assertEqual(len(tf.contrib.framework.get_variables()), 8)
      self.assertEqual(
          len(tf.contrib.framework.get_variables('Conv/BatchNorm')), 3)
      self.assertEqual(
          len(tf.contrib.framework.get_variables('Conv_1/BatchNorm')), 3)

  def testReuseConvWithBatchNorm(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.convolution2d],
          normalizer_fn=tf.contrib.layers.batch_norm,
          normalizer_params={'decay': 0.9}):
        net = tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='Conv')
        net = tf.contrib.layers.convolution2d(net, 32, [3, 3], scope='Conv',
                                              reuse=True)
      self.assertEqual(len(tf.contrib.framework.get_variables()), 4)
      self.assertEqual(
          len(tf.contrib.framework.get_variables('Conv/BatchNorm')), 3)
      self.assertEqual(
          len(tf.contrib.framework.get_variables('Conv_1/BatchNorm')), 0)

  def testCreateConvCreatesWeightsAndBiasesVarsWithRateTwo(self):
    height, width = 7, 9
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with self.test_session():
      self.assertFalse(tf.contrib.framework.get_variables('conv1/weights'))
      self.assertFalse(tf.contrib.framework.get_variables('conv1/biases'))
      tf.contrib.layers.convolution2d(images, 32, [3, 3], rate=2, scope='conv1')
      self.assertTrue(tf.contrib.framework.get_variables('conv1/weights'))
      self.assertTrue(tf.contrib.framework.get_variables('conv1/biases'))

  def testOutputSizeWithRateTwoSamePadding(self):
    num_filters = 32
    input_size = [5, 10, 12, 3]
    expected_size = [5, 10, 12, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.convolution2d(images, num_filters,
                                             [3, 3], rate=2, padding='SAME')
    self.assertListEqual(list(output.get_shape().as_list()), expected_size)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithRateTwoValidPadding(self):
    num_filters = 32
    input_size = [5, 10, 12, 3]
    expected_size = [5, 6, 8, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.convolution2d(images, num_filters, [3, 3],
                                             rate=2, padding='VALID')
    self.assertListEqual(list(output.get_shape().as_list()), expected_size)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithRateTwoThreeValidPadding(self):
    num_filters = 32
    input_size = [5, 10, 12, 3]
    expected_size = [5, 6, 6, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.convolution2d(images, num_filters, [3, 3],
                                             rate=[2, 3], padding='VALID')
    self.assertListEqual(list(output.get_shape().as_list()), expected_size)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testDynamicOutputSizeWithRateOneValidPadding(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [None, None, None, num_filters]
    expected_size_dynamic = [5, 7, 9, num_filters]

    with self.test_session():
      images = tf.placeholder(np.float32, [None, None, None, input_size[3]])
      output = tf.contrib.layers.convolution2d(images, num_filters, [3, 3],
                                               rate=1, padding='VALID')
      tf.global_variables_initializer().run()
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), expected_size)
      eval_output = output.eval({images: np.zeros(input_size, np.float32)})
      self.assertListEqual(list(eval_output.shape), expected_size_dynamic)

  def testDynamicOutputSizeWithRateOneValidPaddingNCHW(self):
    if tf.test.is_gpu_available():
      num_filters = 32
      input_size = [5, 3, 9, 11]
      expected_size = [None, num_filters, None, None]
      expected_size_dynamic = [5, num_filters, 7, 9]

      with self.test_session(use_gpu=True):
        images = tf.placeholder(np.float32, [None, input_size[1], None, None])
        output = tf.contrib.layers.convolution2d(
            images,
            num_filters, [3, 3],
            rate=1,
            padding='VALID',
            data_format='NCHW')
        tf.global_variables_initializer().run()
        self.assertEqual(output.op.name, 'Conv/Relu')
        self.assertListEqual(output.get_shape().as_list(), expected_size)
        eval_output = output.eval({images: np.zeros(input_size, np.float32)})
        self.assertListEqual(list(eval_output.shape), expected_size_dynamic)

  def testDynamicOutputSizeWithRateTwoValidPadding(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [None, None, None, num_filters]
    expected_size_dynamic = [5, 5, 7, num_filters]

    with self.test_session():
      images = tf.placeholder(np.float32, [None, None, None, input_size[3]])
      output = tf.contrib.layers.convolution2d(images, num_filters, [3, 3],
                                               rate=2, padding='VALID')
      tf.global_variables_initializer().run()
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), expected_size)
      eval_output = output.eval({images: np.zeros(input_size, np.float32)})
      self.assertListEqual(list(eval_output.shape), expected_size_dynamic)

  def testWithScope(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [5, 5, 7, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.convolution2d(images, num_filters, [3, 3],
                                             rate=2, padding='VALID',
                                             scope='conv7')
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'conv7/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testWithScopeWithoutActivation(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [5, 5, 7, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.convolution2d(images, num_filters, [3, 3],
                                             rate=2, padding='VALID',
                                             activation_fn=None, scope='conv7')
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'conv7/BiasAdd')
      self.assertListEqual(list(output.eval().shape), expected_size)


class Convolution2dTransposeTests(tf.test.TestCase):

  def testInvalidDataFormat(self):
    height, width = 7, 9
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      with self.assertRaisesRegexp(
          ValueError, 'data_format has to be either NCHW or NHWC.'):
        tf.contrib.layers.convolution2d_transpose(
            images, 32, 3, data_format='CHWN')


  def testOutputSizeWithStrideOneSamePaddingNCHW(self):
    # `NCHW` data fomat is only supported for `GPU` device.
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 32
        input_size = [5, 3, 10, 12]
        expected_size = [5, num_filters, 10, 12]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [3, 3], stride=1,
            padding='SAME', data_format='NCHW')
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')

        sess.run(tf.global_variables_initializer())
        self.assertListEqual(list(output.eval().shape), expected_size)


  def testOutputSizeWithStrideOneValidPaddingNCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 32
        input_size = [5, 3, 10, 12]
        expected_size = [5, num_filters, 12, 14]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [3, 3], stride=1,
            padding='VALID', data_format='NCHW')
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')

        sess.run(tf.global_variables_initializer())
        self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStrideTwoValidPaddingNCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 32
        input_size = [5, 3, 9, 11]
        expected_size = [5, num_filters, 19, 23]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [3, 3], stride=[2, 2],
            padding='VALID', data_format='NCHW')
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
        self.assertListEqual(list(output.get_shape().as_list()), expected_size)

        sess.run(tf.global_variables_initializer())
        self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWith1x1StrideTwoSamePaddingNCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 1
        input_size = [1, 1, 1, 1]
        expected_size = [1, num_filters, 2, 2]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [2, 2], stride=[2, 2],
            padding='SAME', data_format='NCHW')
        self.assertListEqual(list(output.get_shape().as_list()), expected_size)

        sess.run(tf.global_variables_initializer())
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
        self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWith1x1StrideTwoValidPaddingNCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 1
        input_size = [1, 1, 1, 1]
        expected_size = [1, num_filters, 2, 2]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [2, 2], stride=[2, 2],
            padding='VALID', data_format='NCHW')
        sess.run(tf.global_variables_initializer())
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
        self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWith2x2StrideTwoSamePaddingNCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 1
        input_size = [1, 1, 2, 2]
        expected_size = [1, num_filters, 4, 4]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [2, 2], stride=[2, 2],
            padding='SAME', data_format='NCHW')
        sess.run(tf.global_variables_initializer())
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
        self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWith2x2StrideTwoValidPaddingNCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 1
        input_size = [1, 1, 2, 2]
        expected_size = [1, num_filters, 4, 4]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [2, 2], stride=[2, 2],
            padding='VALID', data_format='NCHW')
        sess.run(tf.global_variables_initializer())
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
        self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStride2x1NCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 1
        input_size = [1, 1, 3, 2]
        expected_size = [1, num_filters, 6, 5]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [2, 4], stride=[2, 1],
            padding='VALID', data_format='NCHW')
        sess.run(tf.global_variables_initializer())
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
        self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStride2x4NCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 1
        input_size = [1, 1, 3, 2]
        expected_size = [1, num_filters, 6, 8]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [2, 4], stride=[2, 4],
            padding='VALID', data_format='NCHW')
        sess.run(tf.global_variables_initializer())
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
        self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStride2x5NCHW(self):
    if tf.test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        num_filters = 1
        input_size = [1, 1, 3, 2]
        expected_size = [1, num_filters, 6, 10]

        images = tf.random_uniform(input_size, seed=1)
        output = tf.contrib.layers.conv2d_transpose(
            images, num_filters, [2, 4], stride=[2, 5],
            padding='VALID', data_format='NCHW')
        sess.run(tf.global_variables_initializer())
        self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
        self.assertListEqual(list(output.eval().shape), expected_size)


  def testOutputSizeWithStrideOneSamePadding(self):
    num_filters = 32
    input_size = [5, 10, 12, 3]
    expected_size = [5, 10, 12, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [3, 3], stride=1, padding='SAME')
    self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStrideOneValidPadding(self):
    num_filters = 32
    input_size = [5, 10, 12, 3]
    expected_size = [5, 12, 14, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [3, 3], stride=1, padding='VALID')
    self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStrideTwoValidPadding(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [5, 19, 23, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [3, 3], stride=[2, 2], padding='VALID')
    self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
    self.assertListEqual(list(output.get_shape().as_list()), expected_size)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWith1x1StrideTwoSamePadding(self):
    num_filters = 1
    input_size = [1, 1, 1, 1]
    expected_size = [1, 2, 2, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [2, 2], stride=[2, 2], padding='SAME')
    self.assertListEqual(list(output.get_shape().as_list()), expected_size)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWith1x1StrideTwoValidPadding(self):
    num_filters = 1
    input_size = [1, 1, 1, 1]
    expected_size = [1, 2, 2, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [2, 2], stride=[2, 2], padding='VALID')
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWith2x2StrideTwoSamePadding(self):
    num_filters = 1
    input_size = [1, 2, 2, 1]
    expected_size = [1, 4, 4, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [2, 2], stride=[2, 2], padding='SAME')
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWith2x2StrideTwoValidPadding(self):
    num_filters = 1
    input_size = [1, 2, 2, 1]
    expected_size = [1, 4, 4, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [2, 2], stride=[2, 2], padding='VALID')
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStride2x1(self):
    num_filters = 1
    input_size = [1, 3, 2, 1]
    expected_size = [1, 6, 5, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [2, 4], stride=[2, 1], padding='VALID')
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStride2x4(self):
    num_filters = 1
    input_size = [1, 3, 2, 1]
    expected_size = [1, 6, 8, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [2, 4], stride=[2, 4], padding='VALID')
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithStride2x5(self):
    num_filters = 1
    input_size = [1, 3, 2, 1]
    expected_size = [1, 6, 10, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [2, 4], stride=[2, 5], padding='VALID')
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeRandomSizesAndStridesValidPadding(self):
    np.random.seed(0)
    max_image_size = 10

    for _ in range(10):
      num_filters = 1
      input_size = [1, np.random.randint(1, max_image_size),
                    np.random.randint(1, max_image_size), 1]
      filter_size = [np.random.randint(1, input_size[1] + 1),
                     np.random.randint(1, input_size[2] + 1)]
      stride = [np.random.randint(1, 3), np.random.randint(1, 3)]

      tf.reset_default_graph()
      graph = tf.Graph()
      with graph.as_default():
        images = tf.random_uniform(input_size, seed=1)
        transpose = tf.contrib.layers.conv2d_transpose(
            images, num_filters, filter_size, stride=stride, padding='VALID')
        conv = tf.contrib.layers.conv2d(
            transpose, num_filters, filter_size, stride=stride, padding='VALID')

        with self.test_session(graph=graph) as sess:
          sess.run(tf.global_variables_initializer())
          self.assertListEqual(list(conv.eval().shape), input_size)

  def testDynamicOutputSizeWithStrideTwoValidPadding(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [None, None, None, num_filters]
    expected_size_dynamic = [5, 19, 23, num_filters]

    images = tf.placeholder(np.float32, [None, None, None, input_size[3]])
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [3, 3], stride=[2, 2], padding='VALID')
    self.assertListEqual(output.get_shape().as_list(), expected_size)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      eval_output = output.eval({images: np.zeros(input_size, np.float32)})
      self.assertListEqual(list(eval_output.shape), expected_size_dynamic)

  def testDynamicOutputSizeWithStrideTwoSamePadding(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [None, None, None, num_filters]
    expected_size_dynamic = [5, 18, 22, num_filters]

    with self.test_session():
      images = tf.placeholder(np.float32, [None, None, None, input_size[3]])
      output = tf.contrib.layers.conv2d_transpose(
          images, num_filters, [3, 3], stride=[2, 2], padding='SAME')
      tf.global_variables_initializer().run()
      self.assertEqual(output.op.name, 'Conv2d_transpose/Relu')
      self.assertListEqual(output.get_shape().as_list(), expected_size)
      eval_output = output.eval({images: np.zeros(input_size, np.float32)})
      self.assertListEqual(list(eval_output.shape), expected_size_dynamic)

  def testWithScope(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [5, 19, 23, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [3, 3], stride=2, padding='VALID', scope='conv7')
    self.assertEqual(output.op.name, 'conv7/Relu')

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testWithScopeWithoutActivation(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [5, 19, 23, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = tf.contrib.layers.conv2d_transpose(
        images, num_filters, [3, 3], stride=2, padding='VALID',
        activation_fn=None, scope='conv7')
    self.assertEqual(output.op.name, 'conv7/BiasAdd')

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testDeconvWithoutBiasesProducesConv2dTranspose(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [5, 19, 23, num_filters]
    stride = 2
    padding = 'VALID'

    with self.test_session() as sess:
      images = tf.random_uniform(input_size, seed=1)
      output_deconv = tf.contrib.layers.conv2d_transpose(
          images, num_filters, [3, 3], stride=stride, padding=padding,
          activation_fn=None, scope='conv7')

      weights = tf.contrib.framework.get_variables_by_name('conv7/weights')[0]
      output_conv2d_transpose = tf.nn.conv2d_transpose(
          images,
          weights,
          expected_size,
          [1, stride, stride, 1],
          padding=padding)

      sess.run(tf.global_variables_initializer())

      output_deconv, output_conv2d_transpose = sess.run(
          [output_deconv, output_conv2d_transpose])

      self.assertTrue(np.isclose(output_deconv,
                                 output_conv2d_transpose, 1e-5, 1e-5).all())


class ConvolutionInPlaneTest(tf.test.TestCase):

  def testHorzConvWithBlankImage(self):
    image = tf.ones((1, 10, 10, 1))
    horz_gradients = tf.contrib.layers.conv2d_in_plane(
        image,
        weights_initializer=tf.constant_initializer([1, -1]),
        kernel_size=[1, 2],
        padding='VALID',
        activation_fn=None)
    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      result = sess.run(horz_gradients)
      expected = np.zeros((1, 10, 9, 1))

      self.assertAllEqual(result, expected)

  def testHorzConvWithBlankImageAndPlaceholder(self):
    image = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    horz_gradients = tf.contrib.layers.conv2d_in_plane(
        image,
        weights_initializer=tf.constant_initializer([1, -1]),
        kernel_size=[1, 2],
        padding='VALID',
        activation_fn=None)
    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      result = sess.run(horz_gradients,
                        feed_dict={image: np.ones((1, 10, 10, 1))})
      expected = np.zeros((1, 10, 9, 1))

      self.assertAllEqual(result, expected)

  def testHorzConvWithRandomImageMultiBatch(self):
    np.random.seed(1)
    image = np.random.rand(5, 10, 10, 1)
    expected = image[:, :, 0:-1, :] - image[:, :, 1:, :]

    tf_image = tf.constant(image, dtype=tf.float32)
    horz_gradients = tf.contrib.layers.conv2d_in_plane(
        tf_image,
        weights_initializer=tf.constant_initializer([1, -1]),
        kernel_size=[1, 2],
        padding='VALID',
        activation_fn=None)
    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      result = sess.run(horz_gradients)

      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testHorzConvWithRandomImageMultiBatchMultiChannel(self):
    np.random.seed(1)
    image = np.random.rand(5, 10, 10, 7)
    expected = image[:, :, 0:-1, :] - image[:, :, 1:, :]

    tf_image = tf.constant(image, dtype=tf.float32)
    horz_gradients = tf.contrib.layers.conv2d_in_plane(
        tf_image,
        weights_initializer=tf.constant_initializer([1, -1]),
        kernel_size=[1, 2],
        padding='VALID',
        activation_fn=None)
    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      result = sess.run(horz_gradients)

      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testHorzConvWithVaryingImage(self):
    image = np.asmatrix(('1.0 2.0 3.0;'
                         '1.1 2.0 4.0;'
                         '-4.3 0.0 8.9'))

    expected = np.asmatrix(('-1.0 -1.0;'
                            '-0.9 -2.0;'
                            '-4.3 -8.9'))
    expected = np.reshape(np.asarray(expected), (1, 3, 2, 1))

    tf_image = tf.constant(image, shape=(1, 3, 3, 1), dtype=tf.float32)
    horz_gradients = tf.contrib.layers.conv2d_in_plane(
        tf_image,
        weights_initializer=tf.constant_initializer([1, -1]),
        kernel_size=[1, 2],
        padding='VALID',
        activation_fn=None)
    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      result = sess.run(horz_gradients)

      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testVertConvWithBlankImage(self):
    image = tf.ones((1, 10, 10, 1))
    vert_gradients = tf.contrib.layers.conv2d_in_plane(
        image,
        weights_initializer=tf.constant_initializer([1, -1]),
        kernel_size=[2, 1],
        padding='VALID',
        activation_fn=None)
    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      result = sess.run(vert_gradients)
      expected = np.zeros((1, 9, 10, 1))

      self.assertAllEqual(result, expected)

  def testVertConvWithVaryingImage(self):
    image = np.asmatrix(('1.0 2.0 3.0;'
                         '1.1 2.0 4.0;'
                         '-4.3 0.0 8.9'))

    expected = np.asmatrix(('-0.1 0.0 -1.0;'
                            ' 5.4 2.0 -4.9'))
    expected = np.reshape(np.asarray(expected), (1, 2, 3, 1))

    tf_image = tf.constant(image, shape=(1, 3, 3, 1), dtype=tf.float32)
    vert_gradients = tf.contrib.layers.conv2d_in_plane(
        tf_image,
        weights_initializer=tf.constant_initializer([1, -1]),
        kernel_size=[2, 1],
        padding='VALID',
        activation_fn=None)
    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      result = sess.run(vert_gradients)

      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)


class DropoutTest(tf.test.TestCase):

  def testCreateDropout(self):
    height, width = 3, 3
    with self.test_session():
      images = np.random.uniform(size=(5, height, width, 3))
      output = tf.contrib.layers.dropout(images)
      self.assertEqual(output.op.name, 'Dropout/dropout/mul')
      output.get_shape().assert_is_compatible_with(
          tf.convert_to_tensor(images).get_shape())

  def testCreateDropoutWithConstantTrue(self):
    height, width = 3, 3
    with self.test_session():
      is_training = tf.constant(True)
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.dropout(images, is_training=is_training)
      self.assertEqual(output.op.name, 'Dropout/dropout/mul')
      output.get_shape().assert_is_compatible_with(images.get_shape())

  def testCreateDropoutWithConstantFalse(self):
    height, width = 3, 3
    with self.test_session():
      is_training = tf.constant(False)
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.dropout(images, is_training=is_training)
      self.assertEqual(output.op.name, 'Dropout/Identity')
      output.get_shape().assert_is_compatible_with(images.get_shape())

  def testCreateDropoutWithPlaceholder(self):
    height, width = 3, 3
    with self.test_session():
      is_training = tf.placeholder(dtype=tf.bool, shape=[])
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.dropout(images, is_training=is_training)
      self.assertEqual(output.op.name, 'Dropout/cond/Merge')
      output.get_shape().assert_is_compatible_with(images.get_shape())

  def testCollectOutputs(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.dropout(images, outputs_collections='outputs')
      c_output = tf.get_collection('outputs')[0]
      self.assertEqual(c_output.alias, 'Dropout')
      self.assertEqual(c_output, output)

  def testDropout(self):
    height, width = 10, 10
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      num_elem_initial = tf.reduce_mean(tf.to_float(images > 0))
      output = tf.contrib.layers.dropout(images)
      num_elem = tf.reduce_mean(tf.to_float(output > 0))
      sess.run(tf.global_variables_initializer())
      num_elem, num_elem_initial = sess.run([num_elem, num_elem_initial])
      self.assertLess(num_elem, num_elem_initial/2 + 0.1)
      self.assertGreater(num_elem, num_elem_initial/2 - 0.1)

  def testCreateDropoutNoTraining(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      num_elem_initial = tf.reduce_mean(tf.to_float(images > 0))
      output = tf.contrib.layers.dropout(images, is_training=False)
      num_elem = tf.reduce_mean(tf.to_float(output > 0))
      sess.run(tf.global_variables_initializer())
      num_elem, num_elem_initial = sess.run([num_elem, num_elem_initial])
      self.assertEqual(num_elem, num_elem_initial)
      outputs, inputs = sess.run([output, images])
      self.assertAllClose(outputs, inputs)

  def testCreateFCFollowByDropout(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.fully_connected(images, 50)
      num_elem_initial = tf.reduce_mean(tf.to_float(output > 0))
      output = tf.contrib.layers.dropout(output)
      num_elem = tf.reduce_mean(tf.to_float(output > 0))
      sess.run(tf.global_variables_initializer())
      num_elem, num_elem_initial = sess.run([num_elem, num_elem_initial])
      self.assertLess(num_elem, num_elem_initial/2 + 0.1)
      self.assertGreater(num_elem, num_elem_initial/2 - 0.1)

  def testCreateFCWithDropout(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.fully_connected(
          images, 50, normalizer_fn=tf.contrib.layers.dropout)
      num_elem = tf.reduce_mean(tf.to_float(output > 0))
      sess.run(tf.global_variables_initializer())
      num_elem = sess.run(num_elem)
      self.assertLess(num_elem, 0.5)
      self.assertGreater(num_elem, 0.1)


class FlattenTest(tf.test.TestCase):

  def testInvalidRank(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32)
      inputs.set_shape(tf.TensorShape((5,)))
      with self.assertRaisesRegexp(
          ValueError, 'must have a least 2 dimensions'):
        tf.contrib.layers.flatten(inputs)

  def testUnknownLastDim(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32)
      inputs.set_shape(tf.TensorShape((5, None)))
      with self.assertRaisesRegexp(ValueError, '2nd dimension must be defined'):
        tf.contrib.layers.flatten(inputs)

  def testCollectOutputs(self):
    height, width = 3, 3
    with self.test_session():
      images = np.random.uniform(size=(5, height, width, 3))
      output = tf.contrib.layers.flatten(images, outputs_collections='outputs')
      c_output = tf.get_collection('outputs')[0]
      self.assertEqual(c_output.alias, 'Flatten')
      self.assertEqual(c_output, output)

  def testFlatten4D(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.flatten(images)
      self.assertEqual(output.get_shape().num_elements(),
                        images.get_shape().num_elements())
      self.assertEqual(output.get_shape()[0], images.get_shape()[0])

  def testFlatten3D(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width), seed=1, name='images')
      output = tf.contrib.layers.flatten(images)
      self.assertEqual(output.get_shape().num_elements(),
                        images.get_shape().num_elements())
      self.assertEqual(output.get_shape()[0], images.get_shape()[0])

  def testFlattenBatchSize(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      inputs = tf.placeholder(tf.int32, (None, height, width, 3))
      output = tf.contrib.layers.flatten(inputs)
      self.assertEqual(output.get_shape().as_list(),
                        [None, height * width * 3])
      output = sess.run(output, {inputs: images.eval()})
      self.assertEqual(output.size,
                        images.get_shape().num_elements())
      self.assertEqual(output.shape[0], images.get_shape()[0])


def _sparsify(array, threshold=0.5):
  array[array < threshold] = 0
  non_zero = np.where(array)
  indices = np.vstack(non_zero).T
  values = array[non_zero]
  shape = array.shape
  return indices, values, shape


class PartialFlattenTest(tf.test.TestCase):

  def testDensePartialFlatten(self):
    """Test `_inner_flatten` on `Tensor`s."""
    shape = [2, 3, 4, 5, 6]
    np.random.seed(5446)
    inputs = np.random.randint(0, 100, size=shape)

    for new_rank in [1, 2, 3, 4, 5]:
      expected_new_shape = (shape[:new_rank - 1] +
                            [np.prod(shape[new_rank - 1:])])
      expected_flattened = np.reshape(inputs, expected_new_shape)

      flattened_t = _layers._inner_flatten(inputs, new_rank)
      static_shape = flattened_t.get_shape().as_list()
      self.assertEqual(static_shape, expected_new_shape)
      with self.test_session() as sess:
        flattened = sess.run(flattened_t)
      np.testing.assert_array_equal(expected_flattened, flattened)

  def testSparsePartialFlatten(self):
    """Test `_inner_flatten` on `SparseTensor`s."""
    shape = [4, 3, 11, 6, 1, 3]
    np.random.seed(10301)
    random_ = np.random.rand(*shape)
    indices, values, _ = _sparsify(random_)

    for new_rank in [1, 2, 3, 4, 5]:
      expected_shape = (shape[:new_rank - 1] + [np.prod(shape[new_rank - 1:])])
      reshaped_random_ = np.reshape(random_, expected_shape)
      expected_indices, expected_values, _ = _sparsify(reshaped_random_)

      inputs_t = tf.SparseTensor(indices, values, shape)

      flattened_t = _layers._inner_flatten(inputs_t, new_rank)

      with self.test_session() as sess:
        flattened = sess.run(flattened_t)

      np.testing.assert_array_equal(expected_indices, flattened.indices)
      np.testing.assert_array_equal(expected_values, flattened.values)
      np.testing.assert_array_equal(expected_shape, flattened.shape)

  def testIncompleteShape(self):
    """Test `_inner_flatten` shape inference for incomplete shapes."""
    shape = [2, None, 4, None, 5, 6]
    inputs = tf.placeholder(tf.int32)
    inputs.set_shape(shape)

    flattened1 = _layers._inner_flatten(inputs, 1)
    self.assertEqual([None], flattened1.get_shape().as_list())

    flattened2 = _layers._inner_flatten(inputs, 2)
    self.assertEqual([2, None], flattened2.get_shape().as_list())

    flattened3 = _layers._inner_flatten(inputs, 3)
    self.assertEqual([2, None, None], flattened3.get_shape().as_list())

    flattened4 = _layers._inner_flatten(inputs, 4)
    self.assertEqual([2, None, 4, None], flattened4.get_shape().as_list())

    flattened5 = _layers._inner_flatten(inputs, 5)
    self.assertEqual([2, None, 4, None, 30], flattened5.get_shape().as_list())


class FCTest(tf.test.TestCase):

  def testCreateFC(self):
    height, width = 3, 3
    for layer_fn in (tf.contrib.layers.fully_connected, tf.contrib.layers.relu):
      with tf.Graph().as_default() as g, self.test_session(g):
        inputs = np.random.uniform(size=(5, height * width * 3))
        output = layer_fn(inputs, 32)
        self.assertEqual(output.op.name, 'fully_connected/Relu')
        self.assertListEqual(output.get_shape().as_list(), [5, 32])
        weights = tf.contrib.framework.get_variables_by_name('weights')[0]
        self.assertListEqual(weights.get_shape().as_list(), [3 * 3 * 3, 32])
        biases = tf.contrib.framework.get_variables_by_name('biases')[0]
        self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateFCWithScope(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      output = tf.contrib.layers.fully_connected(inputs, 32, scope='fc1')
      self.assertEqual(output.op.name, 'fc1/Relu')

  def testCreateFCWithCollection(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with tf.name_scope('fe'):
      fc = tf.contrib.layers.fully_connected(inputs, 7,
                                             outputs_collections='outputs',
                                             scope='fc')
    output_collected = tf.get_collection('outputs')[0]
    self.assertEqual(output_collected.alias, 'fe/fc')
    self.assertEqual(output_collected, fc)

  def testCreateFcCreatesWeightsAndBiasesVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      self.assertFalse(tf.contrib.framework.get_variables('fc1/weights'))
      self.assertFalse(tf.contrib.framework.get_variables('fc1/biases'))
      tf.contrib.layers.fully_connected(inputs, 32, scope='fc1')
      self.assertTrue(tf.contrib.framework.get_variables('fc1/weights'))
      self.assertTrue(tf.contrib.framework.get_variables('fc1/biases'))

  def testReuseVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      tf.contrib.layers.fully_connected(inputs, 32, scope='fc1')
      self.assertEqual(len(tf.contrib.framework.get_variables('fc1')), 2)
      tf.contrib.layers.fully_connected(inputs, 32, scope='fc1', reuse=True)
      self.assertEqual(len(tf.contrib.framework.get_variables('fc1')), 2)

  def testNonReuseVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      tf.contrib.layers.fully_connected(inputs, 32)
      self.assertEqual(
          len(tf.contrib.framework.get_variables('fully_connected')), 2)
      tf.contrib.layers.fully_connected(inputs, 32)
      self.assertEqual(
          len(tf.contrib.framework.get_variables('fully_connected')), 4)

  def testCreateFCWithoutActivation(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      output = tf.contrib.layers.fully_connected(inputs, 32, activation_fn=None)
      self.assertEqual(output.op.name, 'fully_connected/BiasAdd')

  def testCreateFCWithWD(self):
    height, width = 3, 3
    with self.test_session() as sess:
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      weight_decay = tf.contrib.layers.l2_regularizer(0.01)
      tf.contrib.layers.fully_connected(inputs, 32,
                                        weights_regularizer=weight_decay)
      wd = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertEqual(wd.op.name,
                        'fully_connected/weights/Regularizer/l2_regularizer')
      sess.run(tf.global_variables_initializer())
      self.assertLess(sess.run(wd), 0.4)

  def testCreateNoRegularizers(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      tf.contrib.layers.fully_connected(inputs, 32)
      self.assertEqual(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), [])

  def testReuseFCWithWD(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      weight_decay = tf.contrib.layers.l2_regularizer(0.01)
      tf.contrib.layers.fully_connected(inputs, 32,
                                        weights_regularizer=weight_decay,
                                        scope='FC')
      self.assertEqual(len(tf.contrib.framework.get_variables()), 2)
      self.assertEqual(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)
      tf.contrib.layers.fully_connected(inputs, 32,
                                        weights_regularizer=weight_decay,
                                        scope='FC',
                                        reuse=True)
      self.assertEqual(len(tf.contrib.framework.get_variables()), 2)
      self.assertEqual(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)

  def testFCWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height * width * 3), seed=1)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.fully_connected],
          normalizer_fn=tf.contrib.layers.batch_norm,
          normalizer_params={'decay': 0.9}):
        net = tf.contrib.layers.fully_connected(images, 27)
        net = tf.contrib.layers.fully_connected(net, 27)
      self.assertEqual(len(tf.contrib.framework.get_variables()), 8)
      self.assertEqual(len(tf.contrib.framework.get_variables(
          'fully_connected/BatchNorm')), 3)
      self.assertEqual(len(tf.contrib.framework.get_variables(
          'fully_connected_1/BatchNorm')), 3)

  def testReuseFCWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height * width * 3), seed=1)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.fully_connected],
          normalizer_fn=tf.contrib.layers.batch_norm,
          normalizer_params={'decay': 0.9}):
        net = tf.contrib.layers.fully_connected(images, 27, scope='fc1')
        net = tf.contrib.layers.fully_connected(net, 27, scope='fc1',
                                                reuse=True)
      self.assertEqual(len(tf.contrib.framework.get_variables()), 4)
      self.assertEqual(
          len(tf.contrib.framework.get_variables('fc1/BatchNorm')), 3)


class BatchNormTest(tf.test.TestCase):

  def _addBesselsCorrection(self, sample_size, expected_var):
    correction_factor = sample_size / (sample_size - 1)
    expected_var *= correction_factor
    return  expected_var, correction_factor

  def testUnknownShape(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32)
      with self.assertRaisesRegexp(ValueError, 'undefined rank'):
        tf.contrib.layers.batch_norm(inputs)

  def testInvalidDataFormat(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32)
      with self.assertRaisesRegexp(
          ValueError, 'data_format has to be either NCHW or NHWC.'):
        tf.contrib.layers.batch_norm(inputs, data_format='CHWN')

  def testUnknownChannelsDimNHWC(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32)
      inputs.set_shape(tf.TensorShape((5, 3, 3, None)))
      with self.assertRaisesRegexp(ValueError, 'undefined channels dimension'):
        tf.contrib.layers.batch_norm(inputs, data_format='NHWC')

  def testUnknownChannelsDimNCHW(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32)
      inputs.set_shape(tf.TensorShape((5, None, 3, 3)))
      with self.assertRaisesRegexp(ValueError, 'undefined channels dimension'):
        tf.contrib.layers.batch_norm(inputs, data_format='NCHW')

  def testWeightedMomentsFused(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32, shape=(5, 3, 3, 7))
      batch_weights = tf.placeholder(dtype=tf.float32)
      with self.assertRaisesRegexp(ValueError,
                                   'Weighted mean and variance'):
        tf.contrib.layers.batch_norm(
            inputs, batch_weights=batch_weights, fused=True)

  def _testCreateOp(self, fused):
    height, width = 3, 3
    with self.test_session():
      images = np.random.uniform(size=(5, height, width, 3)).astype('f')
      output = tf.contrib.layers.batch_norm(images, fused=fused)
      expected_name = ('BatchNorm/FusedBatchNorm' if fused else
                       'BatchNorm/batchnorm')
      self.assertTrue(output.op.name.startswith(expected_name))
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testCreateOpDefault(self):
    self._testCreateOp(False)

  def testCreateOpFused(self):
    self._testCreateOp(True)

  def testCreateVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.batch_norm(images, scale=True)
      beta = tf.contrib.framework.get_variables_by_name('beta')[0]
      gamma = tf.contrib.framework.get_variables_by_name('gamma')[0]
      self.assertEqual(beta.op.name, 'BatchNorm/beta')
      self.assertEqual(gamma.op.name, 'BatchNorm/gamma')
      moving_mean = tf.contrib.framework.get_variables_by_name('moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables_by_name(
          'moving_variance')[0]
      self.assertEqual(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEqual(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testMovingAverageVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.batch_norm(images, scale=True)
      moving_mean = tf.contrib.framework.get_variables_by_name('moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables_by_name(
          'moving_variance')[0]
      self.assertEqual(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEqual(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testUpdatesCollection(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.batch_norm(images, updates_collections='my_update_ops')
      update_layers = tf.get_collection('my_update_ops')
      update_moving_mean = update_layers[0]
      update_moving_variance = update_layers[1]
      self.assertEqual(update_moving_mean.op.name,
                        'BatchNorm/AssignMovingAvg')
      self.assertEqual(update_moving_variance.op.name,
                        'BatchNorm/AssignMovingAvg_1')

  def testReuseVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.batch_norm(images, scale=True, scope='bn')
      tf.contrib.layers.batch_norm(images, scale=True, scope='bn', reuse=True)
      beta = tf.contrib.framework.get_variables_by_name('beta')
      gamma = tf.contrib.framework.get_variables_by_name('gamma')
      self.assertEqual(len(beta), 1)
      self.assertEqual(len(gamma), 1)
      moving_mean = tf.contrib.framework.get_variables_by_name('moving_mean')
      moving_variance = tf.contrib.framework.get_variables_by_name(
          'moving_variance')
      moving_vars = moving_mean + moving_variance
      self.assertEqual(len(moving_vars), 2)

  def testReuseUpdateOps(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                          updates_collections='update_ops'):
        tf.contrib.layers.batch_norm(images, scope='bn')
        self.assertEqual(len(tf.get_collection('update_ops')), 2)
        tf.contrib.layers.batch_norm(images, scope='bn', reuse=True)
        self.assertEqual(len(tf.get_collection('update_ops')), 4)

  def testCreateMovingVars(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      _ = tf.contrib.layers.batch_norm(images)
      moving_mean = tf.contrib.framework.get_variables('BatchNorm/moving_mean')
      self.assertEqual(len(moving_mean), 1)
      self.assertEqual(moving_mean[0].op.name, 'BatchNorm/moving_mean')
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')
      self.assertEqual(len(moving_variance), 1)
      self.assertEqual(moving_variance[0].op.name, 'BatchNorm/moving_variance')

  def _testNoneUpdatesCollections(self, fused, data_format='NHWC'):
    height, width = 2, 2
    batch_size = 10
    channels = 3
    np.random.seed(1)
    use_gpu = fused
    with self.test_session(use_gpu=use_gpu) as sess:
      if data_format == 'NHWC':
        image_shape = (batch_size, height, width, channels)
        axis = (0, 1, 2)
      else:
        image_shape = (batch_size, channels, height, width)
        axis = (0, 2, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=axis)
      expected_var = np.var(image_values, axis=axis)
      if fused:
        # Add Bessel's correction
        expected_var, _ = self._addBesselsCorrection(
            batch_size * height * width, expected_var)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(
          images,
          decay=0.1,
          updates_collections=None,
          fused=fused,
          data_format=data_format)
      # updates_ops are not added to UPDATE_OPS collection.
      self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * channels)
      self.assertAllClose(variance, [1] * channels)
      for _ in range(10):
        sess.run([output])
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # After 10 updates with decay 0.1 moving_mean == expected_mean and
      # moving_variance == expected_var.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)

  def testNoneUpdatesCollectionsNHWC(self):
    self._testNoneUpdatesCollections(False, data_format='NHWC')

  def testNoneUpdatesCollectionsNCHW(self):
    self._testNoneUpdatesCollections(False, data_format='NCHW')

  def testNoneUpdatesCollectionsFusedNCHW(self):
    if tf.test.is_gpu_available():
      self._testNoneUpdatesCollections(True, data_format='NCHW')

  def testNoneUpdatesCollectionsFusedNHWC(self):
    self._testNoneUpdatesCollections(True, data_format='NHWC')

  def _testDelayedUpdateMovingVars(self, fused, data_format='NHWC'):
    height, width = 2, 2
    batch_size = 10
    channels = 3
    np.random.seed(1)
    use_gpu = fused
    with self.test_session(use_gpu=use_gpu) as sess:
      if data_format == 'NHWC':
        image_shape = (batch_size, height, width, channels)
        axis = (0, 1, 2)
      else:
        image_shape = (batch_size, channels, height, width)
        axis = (0, 2, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=axis)
      expected_var = np.var(image_values, axis=axis)
      if fused:
        # Add Bessel's correction
        expected_var, correction_factor = self._addBesselsCorrection(
            batch_size * height * width, expected_var)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(
          images, decay=0.1, fused=fused, data_format=data_format)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # updates_ops are added to UPDATE_OPS collection.
      self.assertEqual(len(update_ops), 2)
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name='barrier')
      output = control_flow_ops.with_dependencies([barrier], output)
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * channels)
      self.assertAllClose(variance, [1] * channels)
      for _ in range(10):
        sess.run([output])
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # After 10 updates with decay 0.1 moving_mean == expected_mean and
      # moving_variance == expected_var.
      self.assertAllClose(mean, expected_mean)
      if fused:
        # Add Bessel's correction
        moving_variance_corrected = moving_variance / correction_factor
        correct_moving_variance = state_ops.assign(moving_variance,
                                                   moving_variance_corrected)
        sess.run(correct_moving_variance)
      self.assertAllClose(variance, expected_var)

  def testDelayedUpdateMovingVarsNHWC(self):
    self._testDelayedUpdateMovingVars(False, data_format='NHWC')

  def testDelayedUpdateMovingVarsNCHW(self):
    self._testDelayedUpdateMovingVars(False, data_format='NCHW')

  def testDelayedUpdateMovingVarsFusedNCHW(self):
    if tf.test.is_gpu_available():
      self._testDelayedUpdateMovingVars(True, data_format='NCHW')

  def testDelayedUpdateMovingVarsFusedNHWC(self):
    self._testDelayedUpdateMovingVars(True, data_format='NHWC')

  def testDelayedUpdateMovingVars(self):
    self._testDelayedUpdateMovingVars(False)

  def testEvalMovingVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(images,
                                            decay=0.1,
                                            is_training=False)
      self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      # Simulate assigment from saver restore.
      init_assigns = [tf.assign(moving_mean, expected_mean),
                      tf.assign(moving_variance, expected_var)]
      sess.run(init_assigns)
      for _ in range(10):
        sess.run([output], {images: np.random.rand(*image_shape)})
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # Although we feed different images, the moving_mean and moving_variance
      # shouldn't change.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)

  def testEvalMovingVarsWithPartitioner(self):
    # This test makes sure that the moving-mean and moving-variance logic works
    # when `batch_norm` is called within a variable-scope that has a variable
    # partitioner.
    partitioner = tf.fixed_size_partitioner(2, axis=0)
    with tf.variable_scope(tf.get_variable_scope(), partitioner=partitioner):
      self.testEvalMovingVars()

  def _testReuseVars(self, fused):
    height, width = 3, 3
    batch_size = 10
    channels = 3
    with self.test_session() as sess:
      image_shape = (batch_size, height, width, channels)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      if fused:
        # Add Bessel's correction
        expected_var, correction_factor = self._addBesselsCorrection(
            batch_size * height * width, expected_var)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output_train = tf.contrib.layers.batch_norm(
          images, decay=0.1, is_training=True, scope='BN', fused=fused)
      output_eval = tf.contrib.layers.batch_norm(
          images,
          decay=0.1,
          is_training=False,
          scope='BN',
          reuse=True,
          fused=fused)
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BN/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BN/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * channels)
      self.assertAllClose(variance, [1] * channels)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name='barrier')
      train_op = control_flow_ops.with_dependencies([barrier], output_train)
      # Before updates the outputs are different for train and eval.
      self.assertFalse(np.allclose(sess.run([output_train]),
                                   sess.run([output_eval])))
      for _ in range(10):
        sess.run([train_op])
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # After 10 updates with decay 0.1 moving_mean == expected_mean and
      # moving_variance == expected_var.
      self.assertAllClose(mean, expected_mean)
      if fused:
        # Add Bessel's correction
        moving_variance_corrected = moving_variance / correction_factor
        correct_moving_variance = state_ops.assign(moving_variance,
                                                   moving_variance_corrected)
        sess.run(correct_moving_variance)
      self.assertAllClose(variance, expected_var)
      # After convergence output_train and output_eval should be the same.
      self.assertAllClose(sess.run([output_train]), sess.run([output_eval]))

  def testReuseVarsDefault(self):
    self._testReuseVars(False)

  def testReuseVarsFused(self):
    self._testReuseVars(True)

  def _testIsTrainingVariable(self, fused, data_format='NHWC'):
    height, width = 2, 2
    batch_size = 10
    channels = 3
    np.random.seed(1)
    use_gpu = fused
    np.random.seed(1)
    with self.test_session(use_gpu=use_gpu) as sess:
      if data_format == 'NHWC':
        image_shape = (batch_size, height, width, channels)
        axis = (0, 1, 2)
      else:
        image_shape = (batch_size, channels, height, width)
        axis = (0, 2, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=axis)
      expected_var = np.var(image_values, axis=axis)
      if fused:
        # Add Bessel's correction
        expected_var, correction_factor = self._addBesselsCorrection(
            batch_size * height * width, expected_var)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      is_training = tf.Variable(True)
      output = tf.contrib.layers.batch_norm(
          images,
          decay=0.1,
          is_training=is_training,
          fused=fused,
          data_format=data_format)
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * channels)
      self.assertAllClose(variance, [1] * channels)
      # Before updates the outputs are different depending of is_training.
      output_true = sess.run([output], {is_training: True})
      output_false = sess.run([output], {is_training: False})
      self.assertFalse(np.allclose(output_true, output_false))
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name='barrier')
      train_op = control_flow_ops.with_dependencies([barrier], output)
      for _ in range(10):
        sess.run([train_op])
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # After 10 updates with decay 0.1 moving_mean == expected_mean and
      # moving_variance == expected_var.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)
      # After updates to convergence the outputs don't depend on is_training.
      output_true = sess.run([output], {is_training: True})
      if fused:
        # Add Bessel's correction
        moving_variance_corrected = moving_variance / correction_factor
        correct_moving_variance = state_ops.assign(moving_variance,
                                                   moving_variance_corrected)
        sess.run(correct_moving_variance)
      output_false = sess.run([output], {is_training: False})
      self.assertAllClose(output_true, output_false)

  def testIsTrainingVariableNHWC(self):
    self._testIsTrainingVariable(False, data_format='NHWC')

  def testIsTrainingVariableNCHW(self):
    self._testIsTrainingVariable(False, data_format='NCHW')

  def testIsTrainingVariableFusedNCHW(self):
    if tf.test.is_gpu_available():
      self._testIsTrainingVariable(True, data_format='NCHW')

  def testIsTrainingVariableFusedNHWC(self):
    self._testIsTrainingVariable(True, data_format='NHWC')

  def testNoUpdatesWhenIsTrainingFalse(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(images,
                                            decay=0.1,
                                            is_training=False)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # updates_ops are not added to UPDATE_OPS collection.
      self.assertEqual(len(update_ops), 0)
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      # When is_training is False batch_norm doesn't update moving_vars.
      for _ in range(10):
        sess.run([output])
      self.assertAllClose(moving_mean.eval(), [0] * 3)
      self.assertAllClose(moving_variance.eval(), [1] * 3)

  def testNoneUpdatesCollectionNoTraining(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(images,
                                            decay=0.1,
                                            updates_collections=None,
                                            is_training=False)
      # updates_ops are not added to UPDATE_OPS collection.
      self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      # When is_training is False batch_norm doesn't update moving_vars.
      for _ in range(10):
        sess.run([output])
      self.assertAllClose(moving_mean.eval(), [0] * 3)
      self.assertAllClose(moving_variance.eval(), [1] * 3)

  def _testNoneUpdatesCollectionIsTrainingVariable(self,
                                                   fused,
                                                   data_format='NHWC'):
    height, width = 2, 2
    batch_size = 10
    channels = 3
    np.random.seed(1)
    use_gpu = fused
    with self.test_session(use_gpu=use_gpu) as sess:
      if data_format == 'NHWC':
        image_shape = (batch_size, height, width, channels)
        axis = (0, 1, 2)
      else:
        image_shape = (batch_size, channels, height, width)
        axis = (0, 2, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=axis)
      expected_var = np.var(image_values, axis=axis)
      if fused:
        # Add Bessel's correction
        expected_var, correction_factor = self._addBesselsCorrection(
            batch_size * height * width, expected_var)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      is_training = tf.Variable(True)
      output = tf.contrib.layers.batch_norm(
          images,
          decay=0.1,
          updates_collections=None,
          is_training=is_training,
          fused=fused,
          data_format=data_format)
      # updates_ops are not added to UPDATE_OPS collection.
      self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * channels)
      self.assertAllClose(variance, [1] * channels)
      # When is_training is False batch_norm doesn't update moving_vars.
      for _ in range(10):
        sess.run([output], {is_training: False})
      self.assertAllClose(moving_mean.eval(), [0] * channels)
      self.assertAllClose(moving_variance.eval(), [1] * channels)
      # Before updates the outputs are different depending of is_training.
      output_true = sess.run([output], {is_training: True})
      output_false = sess.run([output], {is_training: False})
      self.assertFalse(np.allclose(output_true, output_false))
      # When is_training is True update moving_vars.
      for _ in range(10):
        sess.run([output], {is_training: True})
      # After 10 updates with decay 0.1 moving_mean == expected_mean and
      # moving_variance == expected_var.
      self.assertAllClose(moving_mean.eval(), expected_mean)
      self.assertAllClose(moving_variance.eval(), expected_var)
      # After updates to convergence the outputs don't depend on is_training.
      output_true = sess.run([output], {is_training: True})
      if fused:
        # Add Bessel's correction
        moving_variance_corrected = moving_variance / correction_factor
        correct_moving_variance = state_ops.assign(moving_variance,
                                                   moving_variance_corrected)
        sess.run(correct_moving_variance)
      output_false = sess.run([output], {is_training: False})
      self.assertTrue(np.allclose(output_true, output_false))

  def testNoneUpdatesCollectionIsTrainingVariableNHWC(self):
    self._testNoneUpdatesCollectionIsTrainingVariable(False, data_format='NHWC')

  def testNoneUpdatesCollectionIsTrainingVariableNCHW(self):
    self._testNoneUpdatesCollectionIsTrainingVariable(False, data_format='NCHW')

  def testNoneUpdatesCollectionIsTrainingVariableFusedNCHW(self):
    if tf.test.is_gpu_available():
      self._testNoneUpdatesCollectionIsTrainingVariable(
          True, data_format='NCHW')

  def testNoneUpdatesCollectionIsTrainingVariableFusedNHWC(self):
    self._testNoneUpdatesCollectionIsTrainingVariable(True, data_format='NHWC')

  def _testTrainMovingVars(self, fused, data_format='NHWC'):
    # Test that the gradients are stable while the moving_mean is updated.
    # Since the moving_mean is used as shift to compute the tf.momments, the
    # gradients could diverge, this test checks that gradients remains stable
    # while the moving_mean is updated.
    height, width = 7, 7
    batch_size = 10
    channels = 32
    np.random.seed(1)
    use_gpu = fused
    with self.test_session(use_gpu=use_gpu) as sess:
      if data_format == 'NHWC':
        image_shape = (batch_size, height, width, channels)
        axis = (0, 1, 2)
      else:
        image_shape = (batch_size, channels, height, width)
        axis = (0, 2, 3)
      image_values = np.random.rand(*image_shape) + 2
      expected_mean = np.mean(image_values, axis=axis)
      expected_var = np.var(image_values, axis=axis)
      if fused:
        # Add Bessel's correction
        expected_var, _ = self._addBesselsCorrection(
            batch_size * height * width, expected_var)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(
          images,
          decay=0.2,
          updates_collections=None,
          is_training=True,
          fused=fused,
          data_format=data_format)
      self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])

      objective = tf.reduce_sum(output)

      [images_gradients] = tf.gradients(objective, images)
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * channels)
      self.assertAllClose(variance, [1] * channels)

      # Initial input gradients.
      images_gradients_value = sess.run(images_gradients)
      for _ in range(10):
        np_output, new_images_gradients = sess.run([output, images_gradients])
        # The outputs should be close to 0.0 mean and 1.0 variance
        self.assertAllClose(
            np.mean(
                np_output, axis=axis), [0] * channels, rtol=0.1, atol=0.1)
        self.assertAllClose(
            np.var(np_output, axis=axis), [1] * channels, rtol=0.1, atol=0.1)
        # The gradients should change slowly while updating moving_mean.
        max_diff = np.max(np.abs(images_gradients_value - new_images_gradients))
        self.assertGreaterEqual(max_diff, 0.0)
        self.assertLess(max_diff, 5e-5)
      self.assertAllClose(moving_mean.eval(), expected_mean)
      self.assertAllClose(moving_variance.eval(), expected_var)

  def testTrainMovingVarsNHWC(self):
    self._testTrainMovingVars(False, data_format='NHWC')

  def testTrainMovingVarsNCHW(self):
    self._testTrainMovingVars(False, data_format='NCHW')

  def testTrainMovingVarsFusedNCHW(self):
    if tf.test.is_gpu_available():
      self._testTrainMovingVars(True, data_format='NCHW')

  def testTrainMovingVarsFusedNHWC(self):
    self._testTrainMovingVars(True, data_format='NHWC')

  def testCustomInitializer(self):
    height, width = 3, 3
    channels = 3
    with self.test_session() as sess:
      images = (np.ones((5, height, width, channels)) * 9.0).astype('f')
      beta = tf.constant_initializer((np.ones(channels) * 5.0).astype('f'))
      gamma = tf.constant_initializer((np.ones(channels) * 2.0).astype('f'))
      mean = tf.constant_initializer((np.ones(channels) * 5.0).astype('f'))
      variance = tf.constant_initializer((np.ones(channels) * 4.0).astype('f'))
      output = tf.contrib.layers.batch_norm(images,
                                            is_training=False,
                                            scale=True,
                                            epsilon=0.0,
                                            param_initializers={
                                              'beta': beta,
                                              'gamma': gamma,
                                              'moving_mean': mean,
                                              'moving_variance': variance,
                                            })
      sess.run(tf.global_variables_initializer())
      outs = sess.run(output)
      self.assertAllClose(outs, images)

  def _runBatchNormalizationWithFormat(self, shape, data_format, is_training):
    channels = shape[-1]
    with self.test_session() as sess:
      images = np.arange(np.product(shape), dtype=np.float32).reshape(shape)
      beta = tf.constant_initializer(
          np.arange(2, channels + 2, dtype=np.float32))
      gamma = tf.constant_initializer(
          np.arange(10, channels + 10, dtype=np.float32) * 2.0)
      mean = tf.constant_initializer(
          np.arange(3, channels + 3, dtype=np.float32) * 5.0)
      variance = tf.constant_initializer(
          np.arange(1, channels + 1, dtype=np.float32) * 4.0)
      if data_format == 'NCHW':
        # Reshape inputs from NHWC to NCHW format.
        images = tf.transpose(
            images,
            [0, len(shape) - 1] + list(range(1, len(shape) - 1)))
      output = tf.contrib.layers.batch_norm(
          images,
          is_training=is_training,
          scale=True,
          epsilon=0.5,
          param_initializers={
              'beta': beta,
              'gamma': gamma,
              'moving_mean': mean,
              'moving_variance': variance,
          },
          data_format=data_format)
      if data_format == 'NCHW':
        # Reshape outputs from NCHW back to NHWC format.
        output = tf.transpose(
            output, [0] + list(range(2, len(shape))) + [1])
      sess.run(tf.global_variables_initializer())
      return sess.run(output)

  def testNHWCAndNCHWInferenceProduceSameOutput(self):
    for shape in [[7, 3, 5], [5, 2, 3, 4], [11, 3, 2, 4, 5]]:
      nhwc = self._runBatchNormalizationWithFormat(
          data_format='NHWC', shape=shape, is_training=False)
      nchw = self._runBatchNormalizationWithFormat(
          data_format='NCHW', shape=shape, is_training=False)
      self.assertAllClose(nhwc, nchw, atol=1e-4, rtol=1e-4)

  def testNHWCAndNCHWTrainingProduceSameOutput(self):
    for shape in [[7, 3, 5], [5, 2, 3, 4], [11, 3, 2, 4, 5]]:
      nhwc = self._runBatchNormalizationWithFormat(
          data_format='NHWC', shape=shape, is_training=True)
      nchw = self._runBatchNormalizationWithFormat(
          data_format='NCHW', shape=shape, is_training=True)
      self.assertAllClose(nhwc, nchw, atol=1e-4, rtol=1e-4)


class LayerNormTest(tf.test.TestCase):

  def testUnknownShape(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32)
      with self.assertRaisesRegexp(ValueError, 'undefined rank'):
        tf.contrib.layers.layer_norm(inputs)

  def testUnknownLastDim(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      inputs = tf.placeholder(dtype=tf.float32)
      inputs.set_shape(tf.TensorShape((5, 3, 3, None)))
      with self.assertRaisesRegexp(ValueError, 'undefined last dimension'):
        tf.contrib.layers.layer_norm(inputs)

  def testCreateOp(self):
    height, width = 3, 3
    with self.test_session():
      images = np.random.uniform(size=(5, height, width, 3))
      output = tf.contrib.layers.layer_norm(images)
      self.assertTrue(output.op.name.startswith('LayerNorm/batchnorm'))
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testCreateVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.layer_norm(images)
      beta = tf.contrib.framework.get_variables_by_name('beta')[0]
      gamma = tf.contrib.framework.get_variables_by_name('gamma')[0]
      self.assertEqual(beta.op.name, 'LayerNorm/beta')
      self.assertEqual(gamma.op.name, 'LayerNorm/gamma')

  def testReuseVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.layer_norm(images, scope='ln')
      tf.contrib.layers.layer_norm(images, scope='ln', reuse=True)
      beta = tf.contrib.framework.get_variables_by_name('beta')
      gamma = tf.contrib.framework.get_variables_by_name('gamma')
      self.assertEqual(len(beta), 1)
      self.assertEqual(len(gamma), 1)

  def testReuseVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output_train = tf.contrib.layers.layer_norm(images, scope='LN')
      output_eval = tf.contrib.layers.layer_norm(images,
                                                 scope='LN',
                                                 reuse=True)
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      # output_train and output_eval should be the same.
      self.assertAllClose(sess.run([output_train]), sess.run([output_eval]))

  def doOutputTest(self, input_shape):
    with self.test_session() as sess:
      input_values = np.random.rand(*input_shape)
      inputs = tf.constant(input_values, shape=input_shape, dtype=tf.float32)
      output_op = tf.contrib.layers.layer_norm(inputs, scope='LN')
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      # The mean and variance of the output should be close to 0 and 1
      # respectively.
      moments_axis = tuple([i for i in range(1, len(input_shape))])
      outputs = sess.run(output_op)
      expected_mean = np.zeros(input_shape[0])
      expected_var = np.ones(input_shape[0])
      mean = np.mean(outputs, axis=moments_axis)
      var = np.var(outputs, axis=moments_axis)
      tol = 1e-5
      self.assertAllClose(mean, expected_mean, rtol=tol, atol=tol)
      self.assertAllClose(var, expected_var, rtol=tol, atol=tol)

  def testOutput2DInput(self):
    self.doOutputTest((10, 300))

  def testOutput4DInput(self):
    self.doOutputTest((100, 10, 10, 3))

class MaxPool2DTest(tf.test.TestCase):

  def testInvalidDataFormat(self):
    height, width = 3, 6
    images = np.random.uniform(size=(5, height, width, 3))
    with self.assertRaisesRegexp(
        ValueError, 'data_format has to be either NCHW or NHWC.'):
      tf.contrib.layers.max_pool2d(images, [3, 3], data_format='CHWN')

  def testCreateMaxPool(self):
    height, width = 3, 6
    images = np.random.uniform(size=(5, height, width, 3)).astype(np.float32)
    output = tf.contrib.layers.max_pool2d(images, [3, 3])
    self.assertEqual(output.op.name, 'MaxPool2D/MaxPool')
    self.assertListEqual(output.get_shape().as_list(), [5, 1, 2, 3])

  def testCreateMaxPoolNCHW(self):
    height, width = 3, 6
    images = np.random.uniform(size=(5, 3, height, width)).astype(np.float32)
    output = tf.contrib.layers.max_pool2d(images, [3, 3], data_format='NCHW')
    self.assertEquals(output.op.name, 'MaxPool2D/MaxPool')
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 1, 2])

  def testCollectOutputs(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.max_pool2d(images, [3, 3],
                                          outputs_collections='outputs')
    output_collected = tf.get_collection('outputs')[0]
    self.assertEqual(output_collected.alias, 'MaxPool2D')
    self.assertEqual(output_collected, output)

  def testCreateSquareMaxPool(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.max_pool2d(images, 3)
    self.assertEqual(output.op.name, 'MaxPool2D/MaxPool')
    self.assertListEqual(output.get_shape().as_list(), [5, 1, 2, 3])

  def testCreateMaxPoolWithScope(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.max_pool2d(images, [3, 3], scope='pool1')
    self.assertEqual(output.op.name, 'pool1/MaxPool')

  def testCreateMaxPoolWithSamePadding(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.max_pool2d(images, [3, 3], padding='SAME')
    self.assertListEqual(output.get_shape().as_list(), [5, 2, 3, 3])

  def testCreateMaxPoolWithSamePaddingNCHW(self):
    height, width = 3, 6
    images = tf.random_uniform((5, 3, height, width), seed=1)
    output = tf.contrib.layers.max_pool2d(images, [3, 3], padding='SAME',
                                          data_format='NCHW')
    self.assertListEqual(output.get_shape().as_list(), [5, 3, 2, 3])

  def testCreateMaxPoolStrideWithSamePadding(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.max_pool2d(images, [3, 3], stride=1,
                                          padding='SAME')
    self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testGlobalMaxPool(self):
    height, width = 3, 6
    images = tf.random_uniform((5, height, width, 3), seed=1)
    output = tf.contrib.layers.max_pool2d(images, images.get_shape()[1:3],
                                          stride=1)
    self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])


class OneHotEncodingTest(tf.test.TestCase):

  def testOneHotEncodingCreate(self):
    with self.test_session():
      labels = np.array([0, 1, 2])
      output = tf.contrib.layers.one_hot_encoding(labels, num_classes=3)
      self.assertEqual(output.op.name, 'OneHotEncoding/one_hot')
      self.assertListEqual(output.get_shape().as_list(), [3, 3])

  def testCollectOutputs(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2])
      output = tf.contrib.layers.one_hot_encoding(labels, num_classes=3,
                                                  outputs_collections='outputs')
      c_output = tf.get_collection('outputs')[0]
      self.assertEqual(c_output.alias, 'OneHotEncoding')
      self.assertEqual(c_output, output)

  def testOneHotEncoding(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2])
      one_hot_labels = tf.constant([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
      output = tf.contrib.layers.one_hot_encoding(labels, num_classes=3)
      self.assertAllClose(output.eval(), one_hot_labels.eval())

  def testOneHotEncodingInt32(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2], dtype=tf.int32)
      one_hot_labels = tf.constant([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
      output = tf.contrib.layers.one_hot_encoding(labels, num_classes=3)
      self.assertAllClose(output.eval(), one_hot_labels.eval())


class RepeatTests(tf.test.TestCase):

  def testRepeat(self):
    height, width = 3, 3
    with self.test_session():
      images = np.random.uniform(size=(5, height, width, 3))
      output = tf.contrib.layers.repeat(images, 3,
                                        tf.contrib.layers.conv2d, 32, [3, 3])
      self.assertEqual(output.op.name, 'Repeat/convolution_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 3, 3, 32])

  def testRepeatWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.repeat(images, 3,
                                        tf.contrib.layers.conv2d, 32, [3, 3],
                                        scope='conv1')
      self.assertEqual(output.op.name, 'conv1/conv1_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 3, 3, 32])


class SeparableConv2dTest(tf.test.TestCase):

  def testCreateConvInt32(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform(
          (5, height, width, 3), seed=1, dtype=tf.int32, maxval=12345)
      with self.assertRaisesRegexp(TypeError, 'non-floating point type'):
        tf.contrib.layers.separable_conv2d(images, 32, [3, 3], 2)

  def testCreateConvFloat32(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform(
          (5, height, width, 3), seed=1, dtype=tf.float32)
      output = tf.contrib.layers.separable_conv2d(images, 32, [3, 3], 2)
      self.assertEqual(output.op.name, 'SeparableConv2d/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateConvFloat64(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform(
          (5, height, width, 3), seed=1, dtype=tf.float64)
      output = tf.contrib.layers.separable_conv2d(images, 32, [3, 3], 2)
      self.assertEqual(output.op.name, 'SeparableConv2d/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateDepthwiseConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.separable_conv2d(images, None, [3, 3], 2)
      self.assertEqual(output.op.name, 'SeparableConv2d/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 6])

  def testCreateConvCreatesWeightsAndBiasesVars(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with self.test_session():
      self.assertFalse(
          tf.contrib.framework.get_variables('conv1/depthwise_weights'))
      self.assertFalse(
          tf.contrib.framework.get_variables('conv1/pointwise_weights'))
      self.assertFalse(tf.contrib.framework.get_variables('conv1/biases'))
      tf.contrib.layers.separable_conv2d(images, 32, [3, 3], 4, scope='conv1')
      self.assertTrue(
          tf.contrib.framework.get_variables('conv1/depthwise_weights'))
      self.assertTrue(
          tf.contrib.framework.get_variables('conv1/pointwise_weights'))
      self.assertTrue(tf.contrib.framework.get_variables('conv1/biases'))

  def testCreateDepthwiseConvCreatesWeightsAndBiasesVars(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with self.test_session():
      self.assertFalse(
          tf.contrib.framework.get_variables('conv1/depthwise_weights'))
      self.assertFalse(
          tf.contrib.framework.get_variables('conv1/pointwise_weights'))
      self.assertFalse(tf.contrib.framework.get_variables('conv1/biases'))
      tf.contrib.layers.separable_conv2d(images, None, [3, 3], 4, scope='conv1')
      self.assertTrue(
          tf.contrib.framework.get_variables('conv1/depthwise_weights'))
      self.assertFalse(
          tf.contrib.framework.get_variables('conv1/pointwise_weights'))
      self.assertTrue(tf.contrib.framework.get_variables('conv1/biases'))

  def testCreateConvWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.separable_conv2d(
          images, 32, [3, 3], 6, scope='conv1')
      self.assertEqual(output.op.name, 'conv1/Relu')

  def testCreateConvWithoutActivation(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.separable_conv2d(
          images, 32, [3, 3], 8, activation_fn=None)
      self.assertEqual(output.op.name, 'SeparableConv2d/BiasAdd')

  def testCreateConvValid(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.separable_conv2d(
          images, 32, [3, 3], 2, padding='VALID')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 32])

  def testCreateDepthwiseConvValid(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.separable_conv2d(
          images, None, [3, 3], 2, padding='VALID')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 6])

  def testCreateConvWithWeightDecay(self):
    tf.set_random_seed(0)
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1)
      regularizer = tf.contrib.layers.l2_regularizer(0.01)
      tf.contrib.layers.separable_conv2d(
          images, 32, [3, 3], 2, weights_regularizer=regularizer)
      self.assertEqual(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 2)
      weight_decay = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertEqual(
          weight_decay.op.name,
          'SeparableConv2d/depthwise_weights/Regularizer/l2_regularizer')
      sess.run(tf.global_variables_initializer())
      self.assertLessEqual(sess.run(weight_decay), 0.05)
      weight_decay = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[1]
      self.assertEqual(
          weight_decay.op.name,
          'SeparableConv2d/pointwise_weights/Regularizer/l2_regularizer')
      self.assertLessEqual(sess.run(weight_decay), 0.05)

  def testReuseConvWithWeightDecay(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      regularizer = tf.contrib.layers.l2_regularizer(0.01)
      tf.contrib.layers.separable_conv2d(
          images, 32, [3, 3], 2,
          weights_regularizer=regularizer,
          scope='conv1')
      self.assertEqual(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 2)
      tf.contrib.layers.separable_conv2d(
          images, 32, [3, 3], 2,
          weights_regularizer=regularizer,
          scope='conv1', reuse=True)
      self.assertEqual(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 2)

  def testConvWithBatchNorm(self):
    height, width = 3, 3
    batch_norm_collection = 'moving_vars'
    normalizer_params = {
        'variables_collections': {
            'beta': [batch_norm_collection],
            'gamma': [batch_norm_collection],
            'moving_mean': [batch_norm_collection],
            'moving_variance': [batch_norm_collection],
        }
    }
    images = tf.random_uniform((5, height, width, 3), seed=1)
    net = tf.contrib.layers.separable_conv2d(
        images, 8, [3, 3], 2,
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params=normalizer_params,
        scope='conv1')
    net = tf.contrib.layers.separable_conv2d(
        net, 32, [3, 3], 2,
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params=normalizer_params,
        scope='conv2')
    self.assertEqual(len(tf.get_collection(batch_norm_collection)), 6)
    self.assertEqual(
        len(tf.contrib.framework.get_variables('conv1/BatchNorm')), 3)
    self.assertEqual(
        len(tf.contrib.framework.get_variables('conv2/BatchNorm')), 3)

  def testConvWithInputsViaPlaceHolder(self):
    height, width = 3, 3
    images_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    net = tf.contrib.layers.separable_conv2d(
        images_placeholder, 8, [3, 3], 2,
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={},
        scope='conv1')
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      images = np.random.rand(5, height, width, 3)
      sess.run(init_op)
      sess.run(net, feed_dict={images_placeholder: images})


class SoftmaxTests(tf.test.TestCase):

  def setUp(self):
    self.low = 1 / (1 + math.e)
    self.high = math.e / (1 + math.e)

  def testSoftmax2D(self):
    logits = tf.constant([[0.0, 1], [1, 1], [1, 0]])
    prediction = tf.contrib.layers.softmax(logits)
    exp_prediction = np.array([[self.low, self.high],
                               [0.5, 0.5],
                               [self.high, self.low]])

    with self.test_session() as sess:
      prediction = sess.run(prediction)
      self.assertAllClose(exp_prediction, prediction)

  def testSoftmax3D(self):
    logits = np.ones((2, 3, 2))
    logits[0, 0, 0] = 0
    logits[1, 1, 1] = 0
    logits = tf.constant(logits)
    exp_prediction = 0.5 * np.ones((2, 3, 2))
    exp_prediction[0, 0, 0] = self.low
    exp_prediction[0, 0, 1] = self.high
    exp_prediction[1, 1, 0] = self.high
    exp_prediction[1, 1, 1] = self.low

    prediction = tf.contrib.layers.softmax(logits)
    with self.test_session() as sess:
      prediction = sess.run(prediction)
      self.assertAllClose(exp_prediction, prediction)

  def testSoftmax3DUnknownSize(self):
    logits = np.ones((2, 3, 2))
    logits[0, 0, 0] = 0
    logits[1, 1, 1] = 0
    logit_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))
    feed_dict = {logit_placeholder: logits}
    exp_prediction = 0.5 * np.ones((2, 3, 2))
    exp_prediction[0, 0, 0] = self.low
    exp_prediction[0, 0, 1] = self.high
    exp_prediction[1, 1, 0] = self.high
    exp_prediction[1, 1, 1] = self.low

    prediction = tf.contrib.layers.softmax(logit_placeholder)
    with self.test_session() as sess:
      prediction = sess.run(prediction, feed_dict=feed_dict)
      self.assertAllClose(exp_prediction, prediction)

  def testSoftmaxUndefinedNthDimension(self):
    logits = tf.placeholder(tf.float32)
    with self.assertRaises(ValueError):
      tf.contrib.layers.softmax(logits)


class StackTests(tf.test.TestCase):

  def testStackFullyConnected(self):
    height, width = 3, 3
    with self.test_session():
      images = np.random.uniform(size=(5, height * width * 3))
      output = tf.contrib.layers.stack(images,
                                       tf.contrib.layers.fully_connected,
                                       [10, 20, 30])
      self.assertEqual(output.op.name, 'Stack/fully_connected_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 30])

  def testStackRelu(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height * width * 3), seed=1, name='images')
      output = tf.contrib.layers.stack(images,
                                       tf.contrib.layers.relu,
                                       [10, 20, 30])
      self.assertEqual(output.op.name, 'Stack/fully_connected_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 30])

  def testStackConvolution2d(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.stack(images,
                                       tf.contrib.layers.convolution2d,
                                       [10, 20, 30],
                                       kernel_size=[3, 3],
                                       padding='SAME')
      self.assertEqual(output.op.name, 'Stack/convolution_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 3, 3, 30])

  def testStackWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.stack(images,
                                       tf.contrib.layers.convolution2d,
                                       [10, 20, 30],
                                       kernel_size=[3, 3],
                                       padding='SAME',
                                       scope='conv1')
      self.assertEqual(output.op.name, 'conv1/conv1_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 3, 3, 30])


class UnitNormTests(tf.test.TestCase):

  def testUnitNormWithRandomMatrix(self):
    height, width = 2, 3

    for dim in range(3):
      tf.set_random_seed(0)
      image = tf.random_uniform((height, width, 3))
      output = tf.contrib.layers.unit_norm(image, dim=dim, epsilon=1e-6)
      norms = tf.sqrt(tf.reduce_sum(tf.square(output), reduction_indices=dim))

      shape = [height, width, 3]
      del shape[dim]
      expected = np.ones(shape)

      with self.test_session():
        actual = norms.eval()
        self.assertAllClose(expected, actual, 1e-4, 1e-4)

  def testDimEqualToRankRaisesError(self):
    height, width = 2, 3

    tf.set_random_seed(0)
    image = tf.random_uniform((height, width, 3))

    with self.assertRaises(ValueError):
      tf.contrib.layers.unit_norm(image, dim=3, epsilon=1e-6)

  def testUnknownRankRaisesError(self):
    image = tf.placeholder(tf.float32)
    with self.assertRaises(ValueError):
      tf.contrib.layers.unit_norm(image, dim=2)

  def testKnownRankUnknownDimsSucceeds(self):
    height, width = 2, 3

    for dim in range(3):
      placeholder_value = np.ones((height, width, 3))
      shape = [height, width, 3]
      del shape[dim]
      expected = np.ones(shape)

      image = tf.placeholder(tf.float32, (None, None, 3))
      output = tf.contrib.layers.unit_norm(image, dim=dim, epsilon=1e-6)
      norms = tf.sqrt(tf.reduce_sum(tf.square(output), reduction_indices=dim))

      with self.test_session():
        actual = norms.eval({image: placeholder_value})
        self.assertAllClose(expected, actual, 1e-4, 1e-4)


# TODO(b/28426988): Add separate tests for non-legacy versions.
class LegacyFullyConnectedTest(tf.test.TestCase):

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.set_random_seed(1234)
    self.input = tf.constant([[1., 2., 3.], [-4., 15., -6.]])
    self.input_3_dim_arr = [[[1., 1.1, 1.2],
                             [2., 2.1, 2.2],
                             [3., 3.1, 3.2],
                             [4., 4.1, 4.2]],
                            [[5., 5.1, 5.2],
                             [6., 6.1, 6.2],
                             [7., 7.1, 7.2],
                             [8., 8.1, 8.2]]]
    self.input_3_dim = tf.constant(self.input_3_dim_arr)

    assert not tf.get_collection(tf.GraphKeys.SUMMARIES)

  def _fully_connected_basic_use(self, x, num_output_units, expected_shape):
    output = tf.contrib.layers.legacy_fully_connected(x,
                                                      num_output_units,
                                                      activation_fn=tf.nn.relu)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.global_variables_initializer().run()
      out_value, shape_value = sess.run([output, tf.shape(output)])

    self.assertAllClose(shape_value, expected_shape)
    self.assertEqual(output.get_shape().as_list(), expected_shape)
    self.assertTrue(np.all(out_value >= 0),
                    'Relu should have all values >= 0.')

    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

  def test_fully_connected_basic_use(self):
    self._fully_connected_basic_use(self.input, 8, [2, 8])

  def test_fully_connected_basic_use_multi_dim(self):
    for last_dim in [1, 3]:
      self.setUp()
      self._fully_connected_basic_use(
          self.input_3_dim, last_dim, [2, 4, last_dim])

  def test_relu_layer_basic_use(self):
    output = tf.contrib.layers.legacy_relu(self.input, 8)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.global_variables_initializer().run()
      out_value = sess.run(output)

    self.assertEqual(output.get_shape().as_list(), [2, 8])
    self.assertTrue(np.all(out_value >= 0),
                    'Relu should have all values >= 0.')

    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

  def test_variable_reuse_with_scope(self):
    with tf.variable_scope('test') as vs:
      output1 = tf.contrib.layers.legacy_relu(self.input, 8)
      output2 = tf.contrib.layers.legacy_relu(self.input, 8)

    with tf.variable_scope(vs, reuse=True):
      output3 = tf.contrib.layers.legacy_relu(self.input, 8)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      out_value1, out_value2, out_value3 = sess.run([output1, output2, output3])

    self.assertFalse(np.allclose(out_value1, out_value2))
    self.assertAllClose(out_value1, out_value3)

  def test_variable_reuse_with_template(self):
    tmpl1 = tf.make_template('test',
                             tf.contrib.layers.legacy_fully_connected,
                             num_output_units=8)
    output1 = tmpl1(self.input)
    output2 = tmpl1(self.input)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      out_value1, out_value2 = sess.run([output1, output2])
    self.assertAllClose(out_value1, out_value2)

  def _custom_initializers(self, x, num_output_units, expected_outputs):
    output = tf.contrib.layers.legacy_relu(
        x,
        num_output_units,
        weight_init=tf.constant_initializer(2.0),
        bias_init=tf.constant_initializer(1.0))

    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      out_value = sess.run(output)

    self.assertAllClose(np.array(expected_outputs), out_value)

  def test_custom_initializers(self):
    self._custom_initializers(
        self.input, 2, [[13.0, 13.0], [11.0, 11.0]])

  def test_custom_initializers_multi_dim(self):
    self._custom_initializers(self.input_3_dim,
                              2,
                              [[[7.6, 7.6],
                                [13.6, 13.6],
                                [19.6, 19.6],
                                [25.6, 25.6]],
                               [[31.6, 31.6],
                                [37.6, 37.6],
                                [43.6, 43.6],
                                [49.6, 49.6]]])

  def test_custom_collections(self):
    tf.contrib.layers.legacy_relu(self.input,
                                  2,
                                  weight_collections=['unbiased'],
                                  bias_collections=['biased'],
                                  output_collections=['output'])

    self.assertEqual(1, len(tf.get_collection('unbiased')))
    self.assertEqual(1, len(tf.get_collection('biased')))
    self.assertEqual(1, len(tf.get_collection('output')))
    self.assertEqual(2, len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

  def test_all_custom_collections(self):
    tf.contrib.layers.legacy_relu(self.input,
                                  2,
                                  weight_collections=['unbiased', 'all'],
                                  bias_collections=['biased', 'all'])

    self.assertEqual(1, len(tf.get_collection('unbiased')))
    self.assertEqual(1, len(tf.get_collection('biased')))
    self.assertEqual(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                      tf.get_collection('all'))

  def test_no_bias(self):
    tf.contrib.layers.legacy_relu(self.input, 2, bias_init=None)
    self.assertEqual(1, len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

  def test_no_activation(self):
    y = tf.contrib.layers.legacy_fully_connected(self.input, 2)
    self.assertEqual(2, len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    self.assertEqual('BiasAdd', y.op.type)

  def test_no_activation_no_bias(self):
    y = tf.contrib.layers.legacy_fully_connected(self.input, 2, bias_init=None)
    self.assertEqual(1, len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    self.assertEqual('MatMul', y.op.type)

  def test_regularizer(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    tf.contrib.layers.legacy_fully_connected(self.input,
                                             2,
                                             weight_regularizer=test_fn)

    self.assertEqual([tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(1, cnt[0])

  def test_regularizer_with_multiple_variables(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    tf.contrib.layers.legacy_fully_connected(self.input,
                                             2,
                                             weight_regularizer=test_fn)
    tf.contrib.layers.legacy_fully_connected(self.input,
                                             2,
                                             weight_regularizer=test_fn)

    self.assertEqual([tensor, tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(2, cnt[0])

  def test_regularizer_with_variable_reuse(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    with tf.variable_scope('test') as vs:
      tf.contrib.layers.legacy_fully_connected(self.input,
                                               2,
                                               weight_regularizer=test_fn)

    with tf.variable_scope(vs, reuse=True):
      tf.contrib.layers.legacy_fully_connected(self.input,
                                               2,
                                               weight_regularizer=test_fn)

    self.assertEqual([tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(1, cnt[0])

  def test_empty_x_results_in_empty_output(self):
    # Empty x is common if someone masks their input with tf.boolean_mask in
    # order to drop missing entries, and in a particular batch all entries are
    # missing.
    with self.test_session():
      x = np.array([]).reshape(0, 3)
      self.assertEqual(0, tf.size(x).eval())
      y = tf.contrib.layers.legacy_fully_connected(x,
                                                   2,
                                                   activation_fn=tf.nn.softmax)
      tf.global_variables_initializer().run()
      expected_y = np.array([]).reshape(0, 2)
      np.testing.assert_array_equal(expected_y, y.eval())

  def test_shapes_variable_first_dim(self):
    # first dimension is not known statically.
    x = tf.placeholder(tf.float32, shape=[None, 4, 3])
    y = tf.contrib.layers.legacy_fully_connected(x, 1)
    # in the output we still only know the 2nd and 3rd dimensions statically.
    self.assertEqual(y.get_shape().as_list(), [None, 4, 1])
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      # we can feed in input with first dimension 2
      shape_value = sess.run(tf.shape(y), feed_dict={x: self.input_3_dim_arr})
      self.assertAllClose(shape_value, [2, 4, 1])
      # we can feed in input with first dimension 1
      shape_value = sess.run(tf.shape(y),
                             feed_dict={x: [self.input_3_dim_arr[0]]})
      self.assertAllClose(shape_value, [1, 4, 1])
      # we cannot feed in input with inconsistent dimensions
      with self.assertRaises(ValueError):
        sess.run(tf.shape(y), feed_dict={x: [[[]]]})

  def _unknown_dim_invalid_input(self, last_dim):
    x = tf.placeholder(tf.float32, shape=[3, last_dim])
    tf.contrib.layers.legacy_fully_connected(x, 2, activation_fn=None)

  def test_known_dim_valid_input(self):
    self._unknown_dim_invalid_input(last_dim=3)

  def test_unknown_dim_invalid_input(self):
    with self.assertRaisesRegexp(
        ValueError, 'last dimension of x must be known but is None'):
      self._unknown_dim_invalid_input(last_dim=None)

  def test_1d_invalid_input(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError,
                                   'rank of x must be at least 2 not: 1'):
        x = tf.constant([[]], shape=[0])
        tf.contrib.layers.legacy_fully_connected(x,
                                                 2,
                                                 activation_fn=tf.nn.softmax)


if __name__ == '__main__':
  tf.test.main()
