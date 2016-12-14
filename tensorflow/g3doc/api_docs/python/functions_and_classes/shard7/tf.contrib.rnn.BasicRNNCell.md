The most basic RNN cell.
- - -

#### `tf.contrib.rnn.BasicRNNCell.__call__(inputs, state, scope=None)` {#BasicRNNCell.__call__}

Most basic RNN: output = new_state = act(W * input + U * state + B).


- - -

#### `tf.contrib.rnn.BasicRNNCell.__init__(num_units, input_size=None, activation=tanh)` {#BasicRNNCell.__init__}




- - -

#### `tf.contrib.rnn.BasicRNNCell.output_size` {#BasicRNNCell.output_size}




- - -

#### `tf.contrib.rnn.BasicRNNCell.state_size` {#BasicRNNCell.state_size}




- - -

#### `tf.contrib.rnn.BasicRNNCell.zero_state(batch_size, dtype)` {#BasicRNNCell.zero_state}

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


