### `tf.contrib.losses.mean_squared_error(*args, **kwargs)` {#mean_squared_error}

Adds a Sum-of-Squares loss to the training procedure. (deprecated arguments) (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.mean_squared_error instead.

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`targets` is being deprecated, use `labels`. `weight` is being deprecated, use `weights`.

`weight` acts as a coefficient for the loss. If a scalar is provided, then the
loss is simply scaled by the given value. If `weight` is a tensor of size
[batch_size], then the total loss for each sample of the batch is rescaled
by the corresponding element in the `weight` vector. If the shape of
`weight` matches the shape of `predictions`, then the loss of each
measurable element of `predictions` is scaled by the corresponding value of
`weight`.

##### Args:


*  <b>`predictions`</b>: The predicted outputs.
*  <b>`labels`</b>: The ground truth output tensor, same dimensions as 'predictions'.
*  <b>`weights`</b>: Coefficients for the loss a scalar, a tensor of shape
    [batch_size] or a tensor whose shape matches `predictions`.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.
*  <b>`targets`</b>: Deprecated alias for `labels`.
*  <b>`weight`</b>: Deprecated alias for `weights`.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` doesn't match that of `labels` or
    if the shape of `weight` is invalid.

