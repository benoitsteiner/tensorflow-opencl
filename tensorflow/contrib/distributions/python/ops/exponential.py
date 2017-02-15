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
"""The Exponential distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import gamma
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops


__all__ = [
    "Exponential",
    "ExponentialWithSoftplusRate",
]


class Exponential(gamma.Gamma):
  """Exponential distribution.

  The Exponential distribution is parameterized by an event `rate` parameter.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; lambda, x > 0) = exp(-lambda x) / Z
  Z = 1 / lambda
  ```

  where `rate = lambda` and `Z` is the normalizaing constant.

  The Exponential distribution is a special case of the Gamma distribution,
  i.e.,

  ```python
  Exponential(rate) = Gamma(concentration=1., rate)
  ```

  The Exponential distribution uses a `rate` parameter, or "inverse scale",
  which can be intuited as,

  ```none
  X ~ Exponential(rate=1)
  Y = X / rate
  ```

  """

  def __init__(self,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="Exponential"):
    """Construct Exponential distribution with parameter `rate`.

    Args:
      rate: Floating point tensor, equivalent to `1 / mean`. Must contain only
        positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = locals()
    # Even though all statistics of are defined for valid inputs, this is not
    # true in the parent class "Gamma."  Therefore, passing
    # allow_nan_stats=True
    # through to the parent class results in unnecessary asserts.
    with ops.name_scope(name, values=[rate]) as ns:
      self._rate = ops.convert_to_tensor(rate, name="rate")
    super(Exponential, self).__init__(
        concentration=array_ops.ones([], dtype=self._rate.dtype),
        rate=self._rate,
        allow_nan_stats=allow_nan_stats,
        validate_args=validate_args,
        name=ns)
    # While the Gamma distribution is not reparameterizable, the exponential
    # distribution is.
    self._reparameterization_type = True
    self._parameters = parameters
    self._graph_parents += [self._rate]

  @staticmethod
  def _param_shapes(sample_shape):
    return {"rate": ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

  @property
  def rate(self):
    return self._rate

  def _sample_n(self, n, seed=None):
    shape = array_ops.concat([[n], array_ops.shape(self._rate)], 0)
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use `np.finfo(self.dtype.as_numpy_dtype).tiny`
    # because it is the smallest, positive, "normal" number. A "normal" number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    sampled = random_ops.random_uniform(
        shape,
        minval=np.finfo(self.dtype.as_numpy_dtype).tiny,
        maxval=1.,
        seed=seed,
        dtype=self.dtype)
    return -math_ops.log(sampled) / self._rate


class ExponentialWithSoftplusRate(Exponential):
  """Exponential with softplus transform on `rate`."""

  def __init__(self,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="ExponentialWithSoftplusRate"):
    parameters = locals()
    with ops.name_scope(name, values=[rate]) as ns:
      super(ExponentialWithSoftplusRate, self).__init__(
          rate=nn.softplus(rate, name="softplus_rate"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters
