# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Reads Summaries from and writes Summaries to event files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import threading
import time

import six

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


class SummaryWriter(object):
  """Writes `Summary` protocol buffers to event files.

  The `SummaryWriter` class provides a mechanism to create an event file in a
  given directory and add summaries and events to it. The class updates the
  file contents asynchronously. This allows a training program to call methods
  to add data to the file directly from the training loop, without slowing down
  training.

  @@__init__

  @@add_summary
  @@add_session_log
  @@add_event
  @@add_graph
  @@add_run_metadata
  @@get_logdir

  @@flush
  @@close
  """

  def __init__(self, logdir, graph=None, max_queue=10, flush_secs=120,
               graph_def=None):
    """Creates a `SummaryWriter` and an event file.

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
    writer = tf.summary.FileWriter(<some-directory>, sess.graph)
    ```

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      graph: A `Graph` object, such as `sess.graph`.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      graph_def: DEPRECATED: Use the `graph` argument instead.
    """
    self._logdir = logdir
    if not gfile.IsDirectory(self._logdir):
      gfile.MakeDirs(self._logdir)
    self._event_queue = six.moves.queue.Queue(max_queue)
    self._ev_writer = pywrap_tensorflow.EventsWriter(
        compat.as_bytes(os.path.join(self._logdir, "events")))
    self._closed = False
    self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                      flush_secs)
    # For storing used tags for session.run() outputs.
    self._session_run_tags = {}
    self._worker.start()
    if graph is not None or graph_def is not None:
      # Calling it with both graph and graph_def for backward compatibility.
      self.add_graph(graph=graph, graph_def=graph_def)

  def get_logdir(self):
    """Returns the directory where event file will be written."""
    return self._logdir

  def reopen(self):
    """Reopens the summary writer.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the summary writer was not closed.
    """
    if self._closed:
      self._closed = False

  def add_summary(self, summary, global_step=None):
    """Adds a `Summary` protocol buffer to the event file.

    This method wraps the provided summary in an `Event` protocol buffer
    and adds it to the event file.

    You can pass the result of evaluating any summary op, using
    [`Session.run()`](client.md#Session.run) or
    [`Tensor.eval()`](framework.md#Tensor.eval), to this
    function. Alternatively, you can pass a `tf.Summary` protocol
    buffer that you populate with your own data. The latter is
    commonly done to report evaluation results in event files.

    Args:
      summary: A `Summary` protocol buffer, optionally serialized as a string.
      global_step: Number. Optional global step value to record with the
        summary.
    """
    if isinstance(summary, bytes):
      summ = summary_pb2.Summary()
      summ.ParseFromString(summary)
      summary = summ
    event = event_pb2.Event(wall_time=time.time(), summary=summary)
    if global_step is not None:
      event.step = int(global_step)
    self.add_event(event)

  def add_session_log(self, session_log, global_step=None):
    """Adds a `SessionLog` protocol buffer to the event file.

    This method wraps the provided session in an `Event` protocol buffer
    and adds it to the event file.

    Args:
      session_log: A `SessionLog` protocol buffer.
      global_step: Number. Optional global step value to record with the
        summary.
    """
    event = event_pb2.Event(wall_time=time.time(), session_log=session_log)
    if global_step is not None:
      event.step = int(global_step)
    self.add_event(event)

  def add_event(self, event):
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    if not self._closed:
      self._event_queue.put(event)

  def _add_graph_def(self, graph_def, global_step=None):
    graph_bytes = graph_def.SerializeToString()
    event = event_pb2.Event(wall_time=time.time(), graph_def=graph_bytes)
    if global_step is not None:
      event.step = int(global_step)
    self._event_queue.put(event)

  def add_graph(self, graph, global_step=None, graph_def=None):
    """Adds a `Graph` to the event file.

    The graph described by the protocol buffer will be displayed by
    TensorBoard. Most users pass a graph in the constructor instead.

    Args:
      graph: A `Graph` object, such as `sess.graph`.
      global_step: Number. Optional global step counter to record with the
        graph.
      graph_def: DEPRECATED. Use the `graph` parameter instead.

    Raises:
      ValueError: If both graph and graph_def are passed to the method.
    """

    if graph is not None and graph_def is not None:
      raise ValueError("Please pass only graph, or graph_def (deprecated), "
                       "but not both.")

    if isinstance(graph, ops.Graph) or isinstance(graph_def, ops.Graph):
      # The user passed a `Graph`.

      # Check if the user passed it via the graph or the graph_def argument and
      # correct for that.
      if not isinstance(graph, ops.Graph):
        logging.warning("When passing a `Graph` object, please use the `graph`"
                        " named argument instead of `graph_def`.")
        graph = graph_def

      # Serialize the graph with additional info.
      true_graph_def = graph.as_graph_def(add_shapes=True)
    elif (isinstance(graph, graph_pb2.GraphDef)
          or isinstance(graph_def, graph_pb2.GraphDef)):
      # The user passed a `GraphDef`.
      logging.warning("Passing a `GraphDef` to the SummaryWriter is deprecated."
                      " Pass a `Graph` object instead, such as `sess.graph`.")

      # Check if the user passed it via the graph or the graph_def argument and
      # correct for that.
      if isinstance(graph, graph_pb2.GraphDef):
        true_graph_def = graph
      else:
        true_graph_def = graph_def

    else:
      # The user passed neither `Graph`, nor `GraphDef`.
      raise TypeError("The passed graph must be an instance of `Graph` "
                      "or the deprecated `GraphDef`")
    # Finally, add the graph_def to the summary writer.
    self._add_graph_def(true_graph_def, global_step)

  def add_run_metadata(self, run_metadata, tag, global_step=None):
    """Adds a metadata information for a single session.run() call.

    Args:
      run_metadata: A `RunMetadata` protobuf object.
      tag: The tag name for this metadata.
      global_step: Number. Optional global step counter to record with the
        StepStats.

    Raises:
      ValueError: If the provided tag was already used for this type of event.
    """
    if tag in self._session_run_tags:
      raise ValueError("The provided tag was already used for this event type")
    self._session_run_tags[tag] = True

    tagged_metadata = event_pb2.TaggedRunMetadata()
    tagged_metadata.tag = tag
    # Store the `RunMetadata` object as bytes in order to have postponed
    # (lazy) deserialization when used later.
    tagged_metadata.run_metadata = run_metadata.SerializeToString()
    event = event_pb2.Event(wall_time=time.time(),
                            tagged_run_metadata=tagged_metadata)
    if global_step is not None:
      event.step = int(global_step)
    self._event_queue.put(event)

  def flush(self):
    """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
    self._event_queue.join()
    self._ev_writer.Flush()

  def close(self):
    """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
    self.flush()
    self._ev_writer.Close()
    self._closed = True


class _EventLoggerThread(threading.Thread):
  """Thread that logs events."""

  def __init__(self, queue, ev_writer, flush_secs):
    """Creates an _EventLoggerThread.

    Args:
      queue: A Queue from which to dequeue events.
      ev_writer: An event writer. Used to log brain events for
       the visualizer.
      flush_secs: How often, in seconds, to flush the
        pending file to disk.
    """
    threading.Thread.__init__(self)
    self.daemon = True
    self._queue = queue
    self._ev_writer = ev_writer
    self._flush_secs = flush_secs
    # The first event will be flushed immediately.
    self._next_event_flush_time = 0

  def run(self):
    while True:
      event = self._queue.get()
      try:
        self._ev_writer.WriteEvent(event)
        # Flush the event writer every so often.
        now = time.time()
        if now > self._next_event_flush_time:
          self._ev_writer.Flush()
          # Do it again in two minutes.
          self._next_event_flush_time = now + self._flush_secs
      finally:
        self._queue.task_done()


def summary_iterator(path):
  # pylint: disable=line-too-long
  """An iterator for reading `Event` protocol buffers from an event file.

  You can use this function to read events written to an event file. It returns
  a Python iterator that yields `Event` protocol buffers.

  Example: Print the contents of an events file.

  ```python
  for e in tf.train.summary_iterator(path to events file):
      print(e)
  ```

  Example: Print selected summary values.

  ```python
  # This example supposes that the events file contains summaries with a
  # summary value tag 'loss'.  These could have been added by calling
  # `add_summary()`, passing the output of a scalar summary op created with
  # with: `tf.summary.scalar('loss', loss_tensor)`.
  for e in tf.train.summary_iterator(path to events file):
      for v in e.summary.value:
          if v.tag == 'loss':
              print(v.simple_value)
  ```

  See the protocol buffer definitions of
  [Event](https://www.tensorflow.org/code/tensorflow/core/util/event.proto)
  and
  [Summary](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  for more information about their attributes.

  Args:
    path: The path to an event file created by a `SummaryWriter`.

  Yields:
    `Event` protocol buffers.
  """
  # pylint: enable=line-too-long
  for r in tf_record.tf_record_iterator(path):
    yield event_pb2.Event.FromString(r)


class SummaryWriterCache(object):
  """Cache for summary writers.

  This class caches summary writers, one per directory.
  """
  # Cache, keyed by directory.
  _cache = {}

  # Lock protecting _SUMMARY_WRITERS.
  _lock = threading.RLock()

  @staticmethod
  def clear():
    """Clear cached summary writers. Currently only used for unit tests."""
    with SummaryWriterCache._lock:
      SummaryWriterCache._cache = {}

  @staticmethod
  def get(logdir):
    """Returns the SummaryWriter for the specified directory.

    Args:
      logdir: str, name of the directory.

    Returns:
      A `SummaryWriter`.
    """
    with SummaryWriterCache._lock:
      if logdir not in SummaryWriterCache._cache:
        SummaryWriterCache._cache[logdir] = SummaryWriter(
            logdir, graph=ops.get_default_graph())
      return SummaryWriterCache._cache[logdir]
