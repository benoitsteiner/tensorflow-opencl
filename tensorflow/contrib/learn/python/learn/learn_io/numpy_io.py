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
"""Methods to allow dict of numpy arrays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_functions

# Key name to pack the target into dict of `features`. See
# `_get_unique_target_key` for details.
_TARGET_KEY = '__target_key__'

def _get_unique_target_key(features):
  """Returns a key not existed in the input dict `features`.

  Caller of `input_fn` usually provides `features` (dict of numpy arrays) and
  `target`, but the underlying feeding module expects a single dict of numpy
  arrays as input. So, the `target` needs to be packed into the `features`
  temporarily and unpacked after calling the feeding function. Toward this goal,
  this function returns a key not existed in the `features` to pack the
  `target`.
  """
  target_key = _TARGET_KEY
  while target_key in features:
    target_key += '_n'
  return target_key

def numpy_input_fn(x,
                   y=None,
                   batch_size=128,
                   num_epochs=1,
                   shuffle=True,
                   queue_capacity=1000,
                   num_threads=1):
  """Returns input function that would feed dict of numpy arrays into the model.

  This returns a function outputting `features` and `target` based on the dict
  of numpy arrays. The dict `features` has the same keys as the `x`.

  Example:
  ```python
  age = np.arange(4) * 1.0
  height = np.arange(32, 36)
  x = {'age': age, 'height': height}
  y = np.arange(-32, -28)

  with tf.Session() as session:
    input_fn = numpy_io.numpy_input_fn(
        x, y, batch_size=2, shuffle=False, num_epochs=1)
  ```

  Args:
    x: dict of numpy array object.
    y: numpy array object.
    batch_size: Integer, size of batches to return.
    num_epochs: Integer, number of epochs to iterate over data. If `None` will
      run forever.
    shuffle: Boolean, if True shuffles the queue. Avoid shuffle at prediction
      time.
    queue_capacity: Integer, size of queue to accumulate.
    num_threads: Integer, number of threads used for reading and enqueueing.

  Returns:
    Function, that has signature of ()->(dict of `features`, `target`)

  Raises:
    ValueError: if the shape of `y` mismatches the shape of values in `x` (i.e.,
      values in `x` have same shape).
    TypeError: `x` is not a dict.
  """

  def input_fn():
    """Numpy input function."""
    if not isinstance(x, dict):
      raise TypeError('x must be dict; got {}'.format(type(x).__name__))

    unique_target_key = _get_unique_target_key(x)
    if y is not None:
      x[unique_target_key] = y

    if len(set(v.shape for v in x.values())) != 1:
      shape_dict_of_x = {k: x[k].shape for k in x.keys()}
      shape_of_y = None if y is None else y.shape
      raise ValueError('Shape of x and y are mismatch, this will lead to '
                       'missing values. Please make sure each value in x have '
                       'the same shape as y.\n'
                       'Shape for x: {}\n'
                       'Shape for y: {}\n'.format(shape_dict_of_x, shape_of_y))

    # Ensure the order of iteration is consistent.
    ordered_dict_x = collections.OrderedDict(
        sorted(x.items(), key=lambda t: t[0]))

    queue = feeding_functions.enqueue_data(
        ordered_dict_x,
        queue_capacity,
        shuffle=shuffle,
        num_threads=num_threads,
        enqueue_size=batch_size,
        num_epochs=num_epochs)

    features = (queue.dequeue_many(batch_size) if num_epochs is None
                else queue.dequeue_up_to(batch_size))

    # Remove the first `Tensor` in `features`, which is the row number.
    if len(features) > 0:
      features.pop(0)

    features = dict(zip(ordered_dict_x.keys(), features))
    if y is not None:
      target = features.pop(unique_target_key)
      return features, target
    return features

  return input_fn
