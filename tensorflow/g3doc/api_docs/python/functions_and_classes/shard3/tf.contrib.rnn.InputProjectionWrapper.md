Operator adding an input projection to the given cell.

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your inputs in time,
do the projection on this batch-concatenated sequence, then split it.
- - -

#### `tf.contrib.rnn.InputProjectionWrapper.__call__(inputs, state, scope=None)` {#InputProjectionWrapper.__call__}

Run the input projection and then the cell.


- - -

#### `tf.contrib.rnn.InputProjectionWrapper.__init__(cell, num_proj, input_size=None)` {#InputProjectionWrapper.__init__}

Create a cell with input projection.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection of inputs is added before it.
*  <b>`num_proj`</b>: Python integer.  The dimension to project to.
*  <b>`input_size`</b>: Deprecated and unused.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.


- - -

#### `tf.contrib.rnn.InputProjectionWrapper.output_size` {#InputProjectionWrapper.output_size}




- - -

#### `tf.contrib.rnn.InputProjectionWrapper.state_size` {#InputProjectionWrapper.state_size}




- - -

#### `tf.contrib.rnn.InputProjectionWrapper.zero_state(batch_size, dtype)` {#InputProjectionWrapper.zero_state}

Return zero-filled state tensor(s).

##### Args:


*  <b>`batch_size`</b>: int, float, or unit Tensor representing the batch size.
*  <b>`dtype`</b>: the data type to use for the state.

##### Returns:

  If `state_size` is an int or TensorShape, then the return value is a
  `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

  If `state_size` is a nested list or tuple, then the return value is
  a nested list or tuple (of the same structure) of `2-D` tensors with
the shapes `[batch_size x s]` for each s in `state_size`.


