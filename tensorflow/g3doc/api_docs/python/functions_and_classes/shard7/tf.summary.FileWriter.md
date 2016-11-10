Writes `Summary` protocol buffers to event files.

The `FileWriter` class provides a mechanism to create an event file in a
given directory and add summaries and events to it. The class updates the
file contents asynchronously. This allows a training program to call methods
to add data to the file directly from the training loop, without slowing down
training.

- - -

#### `tf.summary.FileWriter.__init__(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)` {#FileWriter.__init__}

Creates a `FileWriter` and an event file.

On construction the summary writer creates a new event file in `logdir`.
This event file will contain `Event` protocol buffers constructed when you
call one of the following functions: `add_summary()`, `add_session_log()`,
`add_event()`, or `add_graph()`.

If you pass a `Graph` to the constructor it is added to
the event file. (This is equivalent to calling `add_graph()` later).

TensorBoard will pick the graph from the file and display it graphically so
you can interactively explore the graph you built. You will usually pass
the graph from the session in which you launched it:

```python
...create a graph...
# Launch the graph in a session.
sess = tf.Session()
# Create a summary writer, add the 'graph' to the event file.
writer = tf.train.SummaryWriter(<some-directory>, sess.graph)
```

The other arguments to the constructor control the asynchronous writes to
the event file:

*  `flush_secs`: How often, in seconds, to flush the added summaries
   and events to disk.
*  `max_queue`: Maximum number of summaries or events pending to be
   written to disk before one of the 'add' calls block.

##### Args:


*  <b>`logdir`</b>: A string. Directory where event file will be written.
*  <b>`graph`</b>: A `Graph` object, such as `sess.graph`.
*  <b>`max_queue`</b>: Integer. Size of the queue for pending events and summaries.
*  <b>`flush_secs`</b>: Number. How often, in seconds, to flush the
    pending events and summaries to disk.
*  <b>`graph_def`</b>: DEPRECATED: Use the `graph` argument instead.



- - -

#### `tf.summary.FileWriter.add_summary(summary, global_step=None)` {#FileWriter.add_summary}

Adds a `Summary` protocol buffer to the event file.

This method wraps the provided summary in an `Event` protocol buffer
and adds it to the event file.

You can pass the result of evaluating any summary op, using
[`Session.run()`](client.md#Session.run) or
[`Tensor.eval()`](framework.md#Tensor.eval), to this
function. Alternatively, you can pass a `tf.Summary` protocol
buffer that you populate with your own data. The latter is
commonly done to report evaluation results in event files.

##### Args:


*  <b>`summary`</b>: A `Summary` protocol buffer, optionally serialized as a string.
*  <b>`global_step`</b>: Number. Optional global step value to record with the
    summary.


- - -

#### `tf.summary.FileWriter.add_session_log(session_log, global_step=None)` {#FileWriter.add_session_log}

Adds a `SessionLog` protocol buffer to the event file.

This method wraps the provided session in an `Event` protocol buffer
and adds it to the event file.

##### Args:


*  <b>`session_log`</b>: A `SessionLog` protocol buffer.
*  <b>`global_step`</b>: Number. Optional global step value to record with the
    summary.


- - -

#### `tf.summary.FileWriter.add_event(event)` {#FileWriter.add_event}

Adds an event to the event file.

##### Args:


*  <b>`event`</b>: An `Event` protocol buffer.


- - -

#### `tf.summary.FileWriter.add_graph(graph, global_step=None, graph_def=None)` {#FileWriter.add_graph}

Adds a `Graph` to the event file.

The graph described by the protocol buffer will be displayed by
TensorBoard. Most users pass a graph in the constructor instead.

##### Args:


*  <b>`graph`</b>: A `Graph` object, such as `sess.graph`.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    graph.
*  <b>`graph_def`</b>: DEPRECATED. Use the `graph` parameter instead.

##### Raises:


*  <b>`ValueError`</b>: If both graph and graph_def are passed to the method.


- - -

#### `tf.summary.FileWriter.add_run_metadata(run_metadata, tag, global_step=None)` {#FileWriter.add_run_metadata}

Adds a metadata information for a single session.run() call.

##### Args:


*  <b>`run_metadata`</b>: A `RunMetadata` protobuf object.
*  <b>`tag`</b>: The tag name for this metadata.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    StepStats.

##### Raises:


*  <b>`ValueError`</b>: If the provided tag was already used for this type of event.


- - -

#### `tf.summary.FileWriter.get_logdir()` {#FileWriter.get_logdir}

Returns the directory where event file will be written.



- - -

#### `tf.summary.FileWriter.flush()` {#FileWriter.flush}

Flushes the event file to disk.

Call this method to make sure that all pending events have been written to
disk.


- - -

#### `tf.summary.FileWriter.close()` {#FileWriter.close}

Flushes the event file to disk and close the file.

Call this method when you do not need the summary writer anymore.



#### Other Methods
- - -

#### `tf.summary.FileWriter.reopen()` {#FileWriter.reopen}

Reopens the EventFileWriter.

Can be called after `close()` to add more events in the same directory.
The events will go into a new events file.

Does nothing if the EventFileWriter was not closed.


