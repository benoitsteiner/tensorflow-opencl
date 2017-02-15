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
"""Multivariate Normal distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import linalg
from tensorflow.contrib.distributions.python.ops import bijector as bijectors
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.distributions.python.ops import normal
from tensorflow.contrib.distributions.python.ops import transformed_distribution
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


__all__ = [
    "MultivariateNormalLinearOperator",
]


_mvn_sample_note = """
`value` is a batch vector with compatible shape if `value` is a `Tensor` whose
shape can be broadcast up to either:

```python
self.batch_shape + self.event_shape
```

or

```python
[M1, ..., Mm] + self.batch_shape + self.event_shape
```

"""


# TODO(b/35290280): Import in `../../__init__.py` after adding unit-tests.
class MultivariateNormalLinearOperator(
    transformed_distribution.TransformedDistribution):
  """The multivariate normal distribution on `R^k`.

  The Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
  `scale` matrix; `covariance = scale @ scale.T` where `@` denotes
  matrix-multiplication.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z,
  y = inv(scale) @ (x - loc),
  Z = (2 pi)**(0.5 k) |det(scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||y||**2` denotes the squared Euclidean norm of `y`.

  The MultivariateNormal distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
  Y = scale @ X + loc
  ```

  #### Examples

  ```python
  ds = tf.contrib.distributions
  la = tf.contrib.linalg

  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  scale = tf.cholesky(cov)
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])

  mvn = ds.MultivariateNormalLinearOperator(
      loc=mu,
      scale=la.LinearOperatorTriL(scale))

  # Covariance agrees with cholesky(cov) parameterization.
  mvn.covariance().eval()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  # Compute the pdf of an`R^3` observation; return a scalar.
  mvn.prob([-1., 0, 1]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  scale_diag = [[1., 2, 3],
                [0.5, 1, 1.5]]     # shape: [2, 3]

  mvn = ds.MultivariateNormalLinearOperator(
      loc=mu,
      scale=la.LinearOperatorDiag(scale_diag))

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x).eval()    # shape: [2]
  ```

  """

  def __init__(self,
               loc=None,
               scale=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalLinearOperator"):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by the last dimension of `loc` or the last
    dimension of the matrix implied by `scale`.

    Recall that `covariance = scale @ scale.T`.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale: Instance of `LinearOperator` with same `dtype` as `loc` and shape
        `[B1, ..., Bb, k, k]`.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      ValueError: if `scale` is unspecified.
      TypeError: if not `scale.dtype.is_floating`
    """
    parameters = locals()
    if scale is None:
      raise ValueError("Missing required `scale` parameter.")
    if not scale.dtype.is_floating:
      raise TypeError("`scale` parameter must have floating-point dtype.")

    # Since expand_dims doesn't preserve constant-ness, we obtain the
    # non-dynamic value if possible.
    event_shape = scale.domain_dimension_tensor()
    if tensor_util.constant_value(event_shape) is not None:
      event_shape = tensor_util.constant_value(event_shape)
    event_shape = event_shape[array_ops.newaxis]

    super(MultivariateNormalLinearOperator, self).__init__(
        distribution=normal.Normal(
            loc=array_ops.zeros([], dtype=scale.dtype),
            scale=array_ops.ones([], dtype=scale.dtype)),
        bijector=bijectors.AffineLinearOperator(
            shift=loc, scale=scale, validate_args=validate_args),
        batch_shape=scale.batch_shape_tensor(),
        event_shape=event_shape,
        validate_args=validate_args,
        name=name)
    self._parameters = parameters

  @property
  def loc(self):
    """The `loc` `Tensor` in `Y = scale @ X + loc`."""
    return self.bijector.shift

  @property
  def scale(self):
    """The `scale` `LinearOperator` in `Y = scale @ X + loc`."""
    return self.bijector.scale

  def log_det_covariance(self, name="log_det_covariance"):
    """Log of determinant of covariance matrix."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.scale.graph_parents):
        return 2. * self.scale.log_abs_determinant()

  def det_covariance(self, name="det_covariance"):
    """Determinant of covariance matrix."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.scale.graph_parents):
        return math_ops.exp(2.* self.scale.log_abs_determinant())

  @distribution_util.AppendDocstring(_mvn_sample_note)
  def _log_prob(self, x):
    return super(MultivariateNormalLinearOperator, self)._log_prob(x)

  @distribution_util.AppendDocstring(_mvn_sample_note)
  def _prob(self, x):
    return super(MultivariateNormalLinearOperator, self)._prob(x)

  def _mean(self):
    if self.loc is None:
      shape = array_ops.concat([
          self.batch_shape_tensor(),
          self.event_shape_tensor(),
      ], 0)
      return array_ops.zeros(shape, self.dtype)
    return array_ops.identity(self.loc)

  def _covariance(self):
    if (isinstance(self.scale, linalg.LinearOperatorIdentity) or
        isinstance(self.scale, linalg.LinearOperatorScaledIdentity) or
        isinstance(self.scale, linalg.LinearOperatorDiag)):
      return array_ops.matrix_diag(math_ops.square(self.scale.diag_part()))
    else:
      # TODO(b/35040238): Remove transpose once LinOp supports `transpose`.
      return self.scale.apply(array_ops.matrix_transpose(self.scale.to_dense()))

  def _variance(self):
    if (isinstance(self.scale, linalg.LinearOperatorIdentity) or
        isinstance(self.scale, linalg.LinearOperatorScaledIdentity) or
        isinstance(self.scale, linalg.LinearOperatorDiag)):
      return math_ops.square(self.scale.diag_part())
    elif (isinstance(self.scale, linalg.LinearOperatorUDVHUpdate)
          and self.scale.is_self_adjoint):
      return array_ops.matrix_diag_part(
          self.scale.apply(self.scale.to_dense()))
    else:
      # TODO(b/35040238): Remove transpose once LinOp supports `transpose`.
      return array_ops.matrix_diag_part(
          self.scale.apply(array_ops.matrix_transpose(self.scale.to_dense())))

  def _stddev(self):
    if (isinstance(self.scale, linalg.LinearOperatorIdentity) or
        isinstance(self.scale, linalg.LinearOperatorScaledIdentity) or
        isinstance(self.scale, linalg.LinearOperatorDiag)):
      return math_ops.abs(self.scale.diag_part())
    elif (isinstance(self.scale, linalg.LinearOperatorUDVHUpdate)
          and self.scale.is_self_adjoint):
      return math_ops.sqrt(array_ops.matrix_diag_part(
          self.scale.apply(self.scale.to_dense())))
    else:
      # TODO(b/35040238): Remove transpose once LinOp supports `transpose`.
      return math_ops.sqrt(array_ops.matrix_diag_part(
          self.scale.apply(array_ops.matrix_transpose(self.scale.to_dense()))))

  def _mode(self):
    return self._mean()


@kullback_leibler.RegisterKL(MultivariateNormalLinearOperator,
                             MultivariateNormalLinearOperator)
def _kl_brute_force(a, b, name=None):
  """Batched KL divergence `KL(a || b)` for multivariate Normals.

  With `X`, `Y` both multivariate Normals in `R^k` with means `mu_a`, `mu_b` and
  covariance `C_a`, `C_b` respectively,

  ```
  KL(a || b) = 0.5 * ( L - k + T + Q ),
  L := Log[Det(C_b)] - Log[Det(C_a)]
  T := trace(C_b^{-1} C_a),
  Q := (mu_b - mu_a)^T C_b^{-1} (mu_b - mu_a),
  ```

  This `Op` computes the trace by solving `C_b^{-1} C_a`. Although efficient
  methods for solving systems with `C_b` may be available, a dense version of
  (the square root of) `C_a` is used, so performance is `O(B s k**2)` where `B`
  is the batch size, and `s` is the cost of solving `C_b x = y` for vectors `x`
  and `y`.

  Args:
    a: Instance of `MultivariateNormalLinearOperator`.
    b: Instance of `MultivariateNormalLinearOperator`.
    name: (optional) name to use for created ops. Default "kl_mvn".

  Returns:
    Batchwise `KL(a || b)`.
  """

  def squared_frobenius_norm(x):
    """Helper to make KL calculation slightly more readable."""
    # http://mathworld.wolfram.com/FrobeniusNorm.html
    return math_ops.square(linalg_ops.norm(x, ord="fro", axis=[-2, -1]))

  # TODO(b/35041439): See also b/35040945. Remove this function once LinOp
  # supports something like:
  #   A.inverse().solve(B).norm(order='fro', axis=[-1, -2])
  def is_diagonal(x):
    """Helper to identify if `LinearOperator` has only a diagonal component."""
    return (isinstance(x, linalg.LinearOperatorIdentity) or
            isinstance(x, linalg.LinearOperatorScaledIdentity) or
            isinstance(x, linalg.LinearOperatorDiag))

  with ops.name_scope(name, "kl_mvn", values=[a.loc, b.loc] +
                      a.scale.graph_parents + b.scale.graph_parents):
    # Calculation is based on:
    # http://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    # and,
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    # i.e.,
    #   If Ca = AA', Cb = BB', then
    #   tr[inv(Cb) Ca] = tr[inv(B)' inv(B) A A']
    #                  = tr[inv(B) A A' inv(B)']
    #                  = tr[(inv(B) A) (inv(B) A)']
    #                  = sum_{ij} (inv(B) A)_{ij}**2
    #                  = ||inv(B) A||_F**2
    # where ||.||_F is the Frobenius norm and the second equality follows from
    # the cyclic permutation property.
    if is_diagonal(a.scale) and is_diagonal(b.scale):
      # Using `stddev` because it handles expansion of Identity cases.
      b_inv_a = (a.stddev() / b.stddev())[..., array_ops.newaxis]
    else:
      b_inv_a = b.scale.solve(a.scale.to_dense())
    kl_div = (b.scale.log_abs_determinant()
              - a.scale.log_abs_determinant()
              + 0.5 * (
                  - math_ops.cast(a.scale.domain_dimension_tensor(), a.dtype)
                  + squared_frobenius_norm(b_inv_a)
                  + squared_frobenius_norm(b.scale.solve(
                      (b.mean() - a.mean())[..., array_ops.newaxis]))))
    kl_div.set_shape(array_ops.broadcast_static_shape(
        a.batch_shape, b.batch_shape))
    return kl_div
