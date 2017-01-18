### `tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)` {#random_gamma}

Draws `shape` samples from each of the given Gamma distribution(s).

`alpha` is the shape parameter describing the distribution(s), and `beta` is
the inverse scale parameter(s).

Example:

  samples = tf.random_gamma([10], [0.5, 1.5])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution

  samples = tf.random_gamma([7, 5], [0.5, 1.5])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions

  samples = tf.random_gamma([30], [[1.],[3.],[5.]], beta=[[3., 4.]])
  # samples has shape [30, 3, 2], with 30 samples each of 3x2 distributions.

  Note that for small alpha values, there is a chance you will draw a value of
  exactly 0, which gets worse for lower-precision dtypes, even though zero is
  not in the support of the gamma distribution.

  Relevant cdfs (~chance you will draw a exactly-0 value):
  ```
    stats.gamma(.01).cdf(np.finfo(np.float16).tiny)
        0.91269738769897879
    stats.gamma(.01).cdf(np.finfo(np.float32).tiny)
        0.41992668622045726
    stats.gamma(.01).cdf(np.finfo(np.float64).tiny)
        0.00084322740680686662
    stats.gamma(.35).cdf(np.finfo(np.float16).tiny)
        0.037583276135263931
    stats.gamma(.35).cdf(np.finfo(np.float32).tiny)
        5.9514895726818067e-14
    stats.gamma(.35).cdf(np.finfo(np.float64).tiny)
        2.3529843400647272e-108
  ```

##### Args:


*  <b>`shape`</b>: A 1-D integer Tensor or Python array. The shape of the output samples
    to be drawn per alpha/beta-parameterized distribution.
*  <b>`alpha`</b>: A Tensor or Python value or N-D array of type `dtype`. `alpha`
    provides the shape parameter(s) describing the gamma distribution(s) to
    sample. Must be broadcastable with `beta`.
*  <b>`beta`</b>: A Tensor or Python value or N-D array of type `dtype`. Defaults to 1.
    `beta` provides the inverse scale parameter(s) of the gamma
    distribution(s) to sample. Must be broadcastable with `alpha`.
*  <b>`dtype`</b>: The type of alpha, beta, and the output: `float16`, `float32`, or
    `float64`.
*  <b>`seed`</b>: A Python integer. Used to create a random seed for the distributions.
    See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `tf.concat(shape, tf.shape(alpha + beta))`
    with values of type `dtype`.

