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
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest


__all__ = [
    "BasicDecoderOutput",
    "BasicDecoder",
]


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
  pass


class BasicDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, initial_state):
    """Initialize BasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.

    Raises:
      TypeError: if `cell` is not an instance of `RNNCell` or `helper`
        is not an instance of `Helper`.
    """
    if not isinstance(cell, core_rnn_cell.RNNCell):
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state

  @property
  def batch_size(self):
    return self._helper.batch_size

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._cell.output_size,
        sample_id=tensor_shape.TensorShape([]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._cell.output_size),
        dtypes.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)
