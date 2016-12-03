<!-- This file is machine generated: DO NOT EDIT! -->

# Neural Network RNN Cells
[TOC]

Module for constructing RNN Cells.

## Base interface for all RNN Cells

- - -

### `class tf.nn.rnn_cell.RNNCell` {#RNNCell}

Abstract object representing an RNN cell.

The definition of cell in this package differs from the definition used in the
literature. In the literature, cell refers to an object with a single scalar
output. The definition in this package refers to a horizontal array of such
units.

An RNN cell, in the most abstract setting, is anything that has
a state and performs some operation that takes a matrix of inputs.
This operation results in an output matrix with `self.output_size` columns.
If `self.state_size` is an integer, this operation also results in a new
state matrix with `self.state_size` columns.  If `self.state_size` is a
tuple of integers, then it results in a tuple of `len(state_size)` state
matrices, each with a column size corresponding to values in `state_size`.

This module provides a number of basic commonly used RNN cells, such as
LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
of operators that allow add dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`,
or by calling the `rnn` ops several times. Every `RNNCell` must have the
properties below and and implement `__call__` with the following signature.
- - -

#### `tf.nn.rnn_cell.RNNCell.__call__(inputs, state, scope=None)` {#RNNCell.__call__}

Run this RNN cell on inputs, starting from the given state.

##### Args:


*  <b>`inputs`</b>: `2-D` tensor with shape `[batch_size x input_size]`.
*  <b>`state`</b>: if `self.state_size` is an integer, this should be a `2-D Tensor`
    with shape `[batch_size x self.state_size]`.  Otherwise, if
    `self.state_size` is a tuple of integers, this should be a tuple
    with shapes `[batch_size x s] for s in self.state_size`.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to class name.

##### Returns:

  A pair containing:

  - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
  - New state: Either a single `2-D` tensor, or a tuple of tensors matching
    the arity and shapes of `state`.


- - -

#### `tf.nn.rnn_cell.RNNCell.output_size` {#RNNCell.output_size}

Integer or TensorShape: size of outputs produced by this cell.


- - -

#### `tf.nn.rnn_cell.RNNCell.state_size` {#RNNCell.state_size}

size(s) of state(s) used by this cell.

It can be represented by an Integer, a TensorShape or a tuple of Integers
or TensorShapes.


- - -

#### `tf.nn.rnn_cell.RNNCell.zero_state(batch_size, dtype)` {#RNNCell.zero_state}

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




## RNN Cells for use with TensorFlow's core RNN methods

- - -

### `class tf.nn.rnn_cell.BasicRNNCell` {#BasicRNNCell}

The most basic RNN cell.
- - -

#### `tf.nn.rnn_cell.BasicRNNCell.__call__(inputs, state, scope=None)` {#BasicRNNCell.__call__}

Most basic RNN: output = new_state = act(W * input + U * state + B).


- - -

#### `tf.nn.rnn_cell.BasicRNNCell.__init__(num_units, input_size=None, activation=tanh)` {#BasicRNNCell.__init__}




- - -

#### `tf.nn.rnn_cell.BasicRNNCell.output_size` {#BasicRNNCell.output_size}




- - -

#### `tf.nn.rnn_cell.BasicRNNCell.state_size` {#BasicRNNCell.state_size}




- - -

#### `tf.nn.rnn_cell.BasicRNNCell.zero_state(batch_size, dtype)` {#BasicRNNCell.zero_state}

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



- - -

### `class tf.nn.rnn_cell.BasicLSTMCell` {#BasicLSTMCell}

Basic LSTM recurrent network cell.

The implementation is based on: http://arxiv.org/abs/1409.2329.

We add forget_bias (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.

It does not allow cell clipping, a projection layer, and does not
use peep-hole connections: it is the basic baseline.

For advanced models, please use the full LSTMCell that follows.
- - -

#### `tf.nn.rnn_cell.BasicLSTMCell.__call__(inputs, state, scope=None)` {#BasicLSTMCell.__call__}

Long short-term memory cell (LSTM).


- - -

#### `tf.nn.rnn_cell.BasicLSTMCell.__init__(num_units, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tanh)` {#BasicLSTMCell.__init__}

Initialize the basic LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell.
*  <b>`forget_bias`</b>: float, The bias added to forget gates (see above).
*  <b>`input_size`</b>: Deprecated and unused.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are 2-tuples of
    the `c_state` and `m_state`.  If False, they are concatenated
    along the column axis.  The latter behavior will soon be deprecated.
*  <b>`activation`</b>: Activation function of the inner states.


- - -

#### `tf.nn.rnn_cell.BasicLSTMCell.output_size` {#BasicLSTMCell.output_size}




- - -

#### `tf.nn.rnn_cell.BasicLSTMCell.state_size` {#BasicLSTMCell.state_size}




- - -

#### `tf.nn.rnn_cell.BasicLSTMCell.zero_state(batch_size, dtype)` {#BasicLSTMCell.zero_state}

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



- - -

### `class tf.nn.rnn_cell.GRUCell` {#GRUCell}

Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
- - -

#### `tf.nn.rnn_cell.GRUCell.__call__(inputs, state, scope=None)` {#GRUCell.__call__}

Gated recurrent unit (GRU) with nunits cells.


- - -

#### `tf.nn.rnn_cell.GRUCell.__init__(num_units, input_size=None, activation=tanh)` {#GRUCell.__init__}




- - -

#### `tf.nn.rnn_cell.GRUCell.output_size` {#GRUCell.output_size}




- - -

#### `tf.nn.rnn_cell.GRUCell.state_size` {#GRUCell.state_size}




- - -

#### `tf.nn.rnn_cell.GRUCell.zero_state(batch_size, dtype)` {#GRUCell.zero_state}

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



- - -

### `class tf.nn.rnn_cell.LSTMCell` {#LSTMCell}

Long short-term memory unit (LSTM) recurrent network cell.

The default non-peephole implementation is based on:

  http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

S. Hochreiter and J. Schmidhuber.
"Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

The peephole implementation is based on:

  https://research.google.com/pubs/archive/43905.pdf

Hasim Sak, Andrew Senior, and Francoise Beaufays.
"Long short-term memory recurrent neural network architectures for
 large scale acoustic modeling." INTERSPEECH, 2014.

The class uses optional peep-hole connections, optional cell clipping, and
an optional projection layer.
- - -

#### `tf.nn.rnn_cell.LSTMCell.__call__(inputs, state, scope=None)` {#LSTMCell.__call__}

Run one step of LSTM.

##### Args:


*  <b>`inputs`</b>: input Tensor, 2D, batch x num_units.
*  <b>`state`</b>: if `state_is_tuple` is False, this must be a state Tensor,
    `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
    tuple of state Tensors, both `2-D`, with column sizes `c_state` and
    `m_state`.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "lstm_cell".

##### Returns:

  A tuple containing:

  - A `2-D, [batch x output_dim]`, Tensor representing the output of the
    LSTM after reading `inputs` when previous state was `state`.
    Here output_dim is:
       num_proj if num_proj was set,
       num_units otherwise.
  - Tensor(s) representing the new state of LSTM after reading `inputs` when
    the previous state was `state`.  Same type and shape(s) as `state`.

##### Raises:


*  <b>`ValueError`</b>: If input size cannot be inferred from inputs via
    static shape inference.


- - -

#### `tf.nn.rnn_cell.LSTMCell.__init__(num_units, input_size=None, use_peepholes=False, cell_clip=None, initializer=None, num_proj=None, proj_clip=None, num_unit_shards=None, num_proj_shards=None, forget_bias=1.0, state_is_tuple=True, activation=tanh)` {#LSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`input_size`</b>: Deprecated and unused.
*  <b>`use_peepholes`</b>: bool, set True to enable diagonal/peephole connections.
*  <b>`cell_clip`</b>: (optional) A float value, if provided the cell state is clipped
    by this value prior to the cell output activation.
*  <b>`initializer`</b>: (optional) The initializer to use for the weight and
    projection matrices.
*  <b>`num_proj`</b>: (optional) int, The output dimensionality for the projection
    matrices.  If None, no projection is performed.
*  <b>`proj_clip`</b>: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
    provided, then the projected values are clipped elementwise to within
    `[-proj_clip, proj_clip]`.
*  <b>`num_unit_shards`</b>: Deprecated, will be removed by Jan. 2017.
    Use a variable_scope partitioner instead.
*  <b>`num_proj_shards`</b>: Deprecated, will be removed by Jan. 2017.
    Use a variable_scope partitioner instead.
*  <b>`forget_bias`</b>: Biases of the forget gate are initialized by default to 1
    in order to reduce the scale of forgetting at the beginning of
    the training.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are 2-tuples of
    the `c_state` and `m_state`.  If False, they are concatenated
    along the column axis.  This latter behavior will soon be deprecated.
*  <b>`activation`</b>: Activation function of the inner states.


- - -

#### `tf.nn.rnn_cell.LSTMCell.output_size` {#LSTMCell.output_size}




- - -

#### `tf.nn.rnn_cell.LSTMCell.state_size` {#LSTMCell.state_size}




- - -

#### `tf.nn.rnn_cell.LSTMCell.zero_state(batch_size, dtype)` {#LSTMCell.zero_state}

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




## Classes storing split `RNNCell` state

- - -

### `class tf.nn.rnn_cell.LSTMStateTuple` {#LSTMStateTuple}

Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

Stores two elements: `(c, h)`, in that order.

Only used when `state_is_tuple=True`.
- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.__getnewargs__()` {#LSTMStateTuple.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.__getstate__()` {#LSTMStateTuple.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.__new__(_cls, c, h)` {#LSTMStateTuple.__new__}

Create new instance of LSTMStateTuple(c, h)


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.__repr__()` {#LSTMStateTuple.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.c` {#LSTMStateTuple.c}

Alias for field number 0


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.dtype` {#LSTMStateTuple.dtype}




- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.h` {#LSTMStateTuple.h}

Alias for field number 1




## RNN Cell wrappers (RNNCells that wrap other RNNCells)

- - -

### `class tf.nn.rnn_cell.MultiRNNCell` {#MultiRNNCell}

RNN cell composed sequentially of multiple simple cells.
- - -

#### `tf.nn.rnn_cell.MultiRNNCell.__call__(inputs, state, scope=None)` {#MultiRNNCell.__call__}

Run this multi-layer cell on inputs, starting from state.


- - -

#### `tf.nn.rnn_cell.MultiRNNCell.__init__(cells, state_is_tuple=True)` {#MultiRNNCell.__init__}

Create a RNN cell composed sequentially of a number of RNNCells.

##### Args:


*  <b>`cells`</b>: list of RNNCells that will be composed in this order.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are n-tuples, where
    `n = len(cells)`.  If False, the states are all
    concatenated along the column axis.  This latter behavior will soon be
    deprecated.

##### Raises:


*  <b>`ValueError`</b>: if cells is empty (not allowed), or at least one of the cells
    returns a state tuple but the flag `state_is_tuple` is `False`.


- - -

#### `tf.nn.rnn_cell.MultiRNNCell.output_size` {#MultiRNNCell.output_size}




- - -

#### `tf.nn.rnn_cell.MultiRNNCell.state_size` {#MultiRNNCell.state_size}




- - -

#### `tf.nn.rnn_cell.MultiRNNCell.zero_state(batch_size, dtype)` {#MultiRNNCell.zero_state}

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



- - -

### `class tf.nn.rnn_cell.DropoutWrapper` {#DropoutWrapper}

Operator adding dropout to inputs and outputs of the given cell.
- - -

#### `tf.nn.rnn_cell.DropoutWrapper.__call__(inputs, state, scope=None)` {#DropoutWrapper.__call__}

Run the cell with the declared dropouts.


- - -

#### `tf.nn.rnn_cell.DropoutWrapper.__init__(cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)` {#DropoutWrapper.__init__}

Create a cell with added input and/or output dropout.

Dropout is never used on the state.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection to output_size is added to it.
*  <b>`input_keep_prob`</b>: unit Tensor or float between 0 and 1, input keep
    probability; if it is float and 1, no input dropout will be added.
*  <b>`output_keep_prob`</b>: unit Tensor or float between 0 and 1, output keep
    probability; if it is float and 1, no output dropout will be added.
*  <b>`seed`</b>: (optional) integer, the randomness seed.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if keep_prob is not between 0 and 1.


- - -

#### `tf.nn.rnn_cell.DropoutWrapper.output_size` {#DropoutWrapper.output_size}




- - -

#### `tf.nn.rnn_cell.DropoutWrapper.state_size` {#DropoutWrapper.state_size}




- - -

#### `tf.nn.rnn_cell.DropoutWrapper.zero_state(batch_size, dtype)` {#DropoutWrapper.zero_state}

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



- - -

### `class tf.nn.rnn_cell.EmbeddingWrapper` {#EmbeddingWrapper}

Operator adding input embedding to the given cell.

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your inputs in time,
do the embedding on this batch-concatenated sequence, then split it and
feed into your RNN.
- - -

#### `tf.nn.rnn_cell.EmbeddingWrapper.__call__(inputs, state, scope=None)` {#EmbeddingWrapper.__call__}

Run the cell on embedded inputs.


- - -

#### `tf.nn.rnn_cell.EmbeddingWrapper.__init__(cell, embedding_classes, embedding_size, initializer=None)` {#EmbeddingWrapper.__init__}

Create a cell with an added input embedding.

##### Args:


*  <b>`cell`</b>: an RNNCell, an embedding will be put before its inputs.
*  <b>`embedding_classes`</b>: integer, how many symbols will be embedded.
*  <b>`embedding_size`</b>: integer, the size of the vectors we embed into.
*  <b>`initializer`</b>: an initializer to use when creating the embedding;
    if None, the initializer from variable scope or a default one is used.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if embedding_classes is not positive.


- - -

#### `tf.nn.rnn_cell.EmbeddingWrapper.output_size` {#EmbeddingWrapper.output_size}




- - -

#### `tf.nn.rnn_cell.EmbeddingWrapper.state_size` {#EmbeddingWrapper.state_size}




- - -

#### `tf.nn.rnn_cell.EmbeddingWrapper.zero_state(batch_size, dtype)` {#EmbeddingWrapper.zero_state}

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



- - -

### `class tf.nn.rnn_cell.InputProjectionWrapper` {#InputProjectionWrapper}

Operator adding an input projection to the given cell.

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your inputs in time,
do the projection on this batch-concatenated sequence, then split it.
- - -

#### `tf.nn.rnn_cell.InputProjectionWrapper.__call__(inputs, state, scope=None)` {#InputProjectionWrapper.__call__}

Run the input projection and then the cell.


- - -

#### `tf.nn.rnn_cell.InputProjectionWrapper.__init__(cell, num_proj, input_size=None)` {#InputProjectionWrapper.__init__}

Create a cell with input projection.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection of inputs is added before it.
*  <b>`num_proj`</b>: Python integer.  The dimension to project to.
*  <b>`input_size`</b>: Deprecated and unused.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.


- - -

#### `tf.nn.rnn_cell.InputProjectionWrapper.output_size` {#InputProjectionWrapper.output_size}




- - -

#### `tf.nn.rnn_cell.InputProjectionWrapper.state_size` {#InputProjectionWrapper.state_size}




- - -

#### `tf.nn.rnn_cell.InputProjectionWrapper.zero_state(batch_size, dtype)` {#InputProjectionWrapper.zero_state}

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



- - -

### `class tf.nn.rnn_cell.OutputProjectionWrapper` {#OutputProjectionWrapper}

Operator adding an output projection to the given cell.

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your outputs in time,
do the projection on this batch-concatenated sequence, then split it
if needed or directly feed into a softmax.
- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.__call__(inputs, state, scope=None)` {#OutputProjectionWrapper.__call__}

Run the cell and output projection on inputs, starting from state.


- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.__init__(cell, output_size)` {#OutputProjectionWrapper.__init__}

Create a cell with output projection.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection to output_size is added to it.
*  <b>`output_size`</b>: integer, the size of the output after projection.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if output_size is not positive.


- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.output_size` {#OutputProjectionWrapper.output_size}




- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.state_size` {#OutputProjectionWrapper.state_size}




- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.zero_state(batch_size, dtype)` {#OutputProjectionWrapper.zero_state}

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



