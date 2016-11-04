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

"""A `Transform` that computes the sum of two `Series`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import series
from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops


@series.Series.register_binary_op("__add__")
class Sum(transform.TensorFlowTransform):
  """Adds two `Series`."""

  def __init__(self):
    super(Sum, self).__init__()

  @property
  def name(self):
    return "sum"

  @property
  def input_valency(self):
    return 2

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors, **kwargs):
    pair_sparsity = (isinstance(input_tensors[0], sparse_tensor.SparseTensor),
                     isinstance(input_tensors[1], sparse_tensor.SparseTensor))

    if pair_sparsity == (False, False):
      result = input_tensors[0] + input_tensors[1]
    # note tf.sparse_add accepts the mixed cases,
    # so long as at least one input is sparse.
    else:
      result = sparse_ops.sparse_add(input_tensors[0], input_tensors[1])

    # pylint: disable=not-callable
    return self.return_type(result)
