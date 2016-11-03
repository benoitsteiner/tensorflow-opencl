### `tf.contrib.layers.convolution2d_transpose(*args, **kwargs)` {#convolution2d_transpose}

Adds a convolution2d_transpose with an optional batch normalization layer.

The function creates a variable called `weights`, representing the
kernel, that is convolved with the input. If `batch_norm_params` is `None`, a
second variable called 'biases' is added to the result of the operation.

##### Args:


*  <b>`inputs`</b>: A 4-D `Tensor` of type `float` and shape
    `[batch, height, width, in_channels]` for `NHWC` data format or
    `[batch, in_channels, height, width]` for `NCHW` data format.
*  <b>`num_outputs`</b>: integer, the number of output filters.
*  <b>`kernel_size`</b>: a list of length 2 holding the [kernel_height, kernel_width] of
    of the filters. Can be an int if both values are the same.
*  <b>`stride`</b>: a list of length 2: [stride_height, stride_width].
    Can be an int if both strides are the same.  Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: one of 'VALID' or 'SAME'.
*  <b>`data_format`</b>: A string. `NHWC` (default) and `NCHW` are supported.
*  <b>`activation_fn`</b>: activation function, set to None to skip it and maintain
    a linear activation.
*  <b>`normalizer_fn`</b>: normalization function to use instead of `biases`. If
    `normalizer_fn` is provided then `biases_initializer` and
    `biases_regularizer` are ignored and `biases` are not created nor added.
    default set to None for no normalizer function
*  <b>`normalizer_params`</b>: normalization function parameters.
*  <b>`weights_initializer`</b>: An initializer for the weights.
*  <b>`weights_regularizer`</b>: Optional regularizer for the weights.
*  <b>`biases_initializer`</b>: An initializer for the biases. If None skip biases.
*  <b>`biases_regularizer`</b>: Optional regularizer for the biases.
*  <b>`reuse`</b>: whether or not the layer and its variables should be reused. To be
    able to reuse the layer scope must be given.
*  <b>`variables_collections`</b>: optional list of collections for all the variables or
    a dictionary containing a different list of collection per variable.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`trainable`</b>: whether or not the variables should be trainable or not.
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  a tensor representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if 'kernel_size' is not a list of length 2.
*  <b>`ValueError`</b>: if `data_format` is neither `NHWC` nor `NCHW`.
*  <b>`ValueError`</b>: if `C` dimension of `inputs` is None.

