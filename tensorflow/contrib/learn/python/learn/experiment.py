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

"""Experiment class collecting information needed for a single training run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import time

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import monitors
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


__all__ = ["Experiment"]


class Experiment(object):
  """Experiment is a class containing all information needed to train a model.

  After an experiment is created (by passing an Estimator and inputs for
  training and evaluation), an Experiment instance knows how to invoke training
  and eval loops in a sensible fashion for distributed training.
  """

  @deprecated_arg_values(
      "2016-10-23",
      "local_eval_frequency is deprecated as local_run will be renamed to "
      "train_and_evaluate. Use min_eval_frequency and call train_and_evaluate "
      "instead. Note, however, that the default for min_eval_frequency is 1, "
      "meaning models will be evaluated every time a new checkpoint is "
      "available. In contrast, the default for local_eval_frequency is None, "
      "resulting in evaluation occurring only after training has completed. "
      "min_eval_frequency is ignored when calling the deprecated local_run.",
      local_eval_frequency=None)
  def __init__(self,
               estimator,
               train_input_fn,
               eval_input_fn,
               eval_metrics=None,
               train_steps=None,
               eval_steps=100,
               train_monitors=None,
               local_eval_frequency=None,
               eval_delay_secs=120,
               continuous_eval_throttle_secs=60,
               min_eval_frequency=1):
    """Constructor for `Experiment`.

    Creates an Experiment instance. None of the functions passed to this
    constructor are executed at construction time. They are stored and used
    when a method is executed which requires it.

    Args:
      estimator: Object implementing `Trainable` and `Evaluable`.
      train_input_fn: function, returns features and labels for training.
      eval_input_fn: function, returns features and labels for evaluation. If
        `eval_steps` is `None`, this should be configured only to produce for a
        finite number of batches (generally, 1 epoch over the evaluation data).
      eval_metrics: `dict` of string, metric function. If `None`, default set
        is used.
      train_steps: Perform this many steps of training. `None`, the default,
        means train forever.
      eval_steps: `evaluate` runs until input is exhausted (or another exception
        is raised), or for `eval_steps` steps, if specified.
      train_monitors: A list of monitors to pass to the `Estimator`'s `fit`
        function.
      local_eval_frequency: Frequency of running eval in steps,
        when running locally. If `None`, runs evaluation only at the end of
        training.
      eval_delay_secs: Start evaluating after waiting for this many seconds.
      continuous_eval_throttle_secs: Do not re-evaluate unless the last
        evaluation was started at least this many seconds ago for
        continuous_eval().
      min_eval_frequency: (applies only to train_and_evaluate). the minimum
        number of steps between evaluations. Of course, evaluation does not
        occur if no new snapshot is available, hence, this is the minimum.

    Raises:
      ValueError: if `estimator` does not implement `Evaluable` and `Trainable`.
    """
    if not isinstance(estimator, evaluable.Evaluable):
      raise ValueError("`estimator` must implement `Evaluable`.")
    if not isinstance(estimator, trainable.Trainable):
      raise ValueError("`estimator` must implement `Trainable`.")
    super(Experiment, self).__init__()
    self._estimator = estimator
    self._train_input_fn = train_input_fn
    self._eval_input_fn = eval_input_fn
    self._eval_metrics = eval_metrics
    self._train_steps = train_steps
    self._eval_steps = eval_steps
    self._train_monitors = train_monitors
    self._local_eval_frequency = local_eval_frequency
    self._eval_delay_secs = eval_delay_secs
    self._continuous_eval_throttle_secs = continuous_eval_throttle_secs
    self._min_eval_frequency = min_eval_frequency

  @property
  def estimator(self):
    return self._estimator

  def train(self, delay_secs=None):
    """Fit the estimator using the training data.

    Train the estimator for `self._train_steps` steps, after waiting for
    `delay_secs` seconds. If `self._train_steps` is `None`, train forever.

    Args:
      delay_secs: Start training after this many seconds.

    Returns:
      The trained estimator.
    """
    start = time.time()

    # Start the server, if needed. It's important to start the server before
    # we (optionally) sleep for the case where no device_filters are set.
    # Otherwise, the servers will wait to connect to each other before starting
    # to train. We might as well start as soon as we can.
    if self._estimator.config.cluster_spec and self._estimator.config.master:
      self._start_server()

    if delay_secs is None:
      task_id = self._estimator.config.task or 0
      delay_secs = min(60, task_id * 5)

    if delay_secs:
      elapsed_secs = time.time() - start
      remaining = delay_secs - elapsed_secs
      logging.info("Waiting %d secs before starting training.", remaining)
      time.sleep(delay_secs)

    return self._estimator.fit(input_fn=self._train_input_fn,
                               max_steps=self._train_steps,
                               monitors=self._train_monitors)

  def evaluate(self, delay_secs=None):
    """Evaluate on the evaluation data.

    Runs evaluation on the evaluation data and returns the result. Runs for
    `self._eval_steps` steps, or if it's `None`, then run until input is
    exhausted or another exception is raised. Start the evaluation after
    `delay_secs` seconds, or if it's `None`, defaults to using
    `self._eval_delay_secs` seconds.

    Args:
      delay_secs: Start evaluating after this many seconds. If `None`, defaults
        to using `self._eval_delays_secs`.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    if delay_secs is None:
      delay_secs = self._eval_delay_secs

    if delay_secs:
      logging.info("Waiting %d secs before starting eval.", delay_secs)
      time.sleep(delay_secs)

    return self._estimator.evaluate(input_fn=self._eval_input_fn,
                                    steps=self._eval_steps,
                                    metrics=self._eval_metrics,
                                    name="one_pass")

  @deprecated(
      "2016-10-23",
      "local_run will be renamed to train_and_evaluate and the new default "
      "behavior will be to run evaluation every time there is a new "
      "checkpoint.")
  def local_run(self):
    with _new_attr_context(self, "_min_eval_frequency"):
      self._min_eval_frequency = self._local_eval_frequency
      return self.train_and_evaluate()

  def _continuous_eval(self,
                       input_fn,
                       name,
                       delay_secs,
                       throttle_delay_secs):
    """Run continuous eval.

    Runs infinite eval on the evaluation data set. This function starts
    evaluating after `delay_secs` seconds and then runs no more than one
    evaluation (with `self._eval_steps` steps each time) per
    `throttle_delay_secs`. It never returns.

    Args:
      input_fn: The input to use for this eval.
      name: A string appended to the folder name of evaluation results.
      delay_secs: Start evaluating after this many seconds. If None, defaults to
        self._eval_delay_secs.
      throttle_delay_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago. If None, defaults to
        self._continuous_eval_throttle_secs.
    """
    if delay_secs is None:
      delay_secs = self._eval_delay_secs
    if throttle_delay_secs is None:
      throttle_delay_secs = self._continuous_eval_throttle_secs

    if delay_secs:
      logging.info("Waiting %f secs before starting eval.", delay_secs)
      time.sleep(delay_secs)

    last_fitted_error_time = 0
    while True:
      start = time.time()
      try:
        self._estimator.evaluate(input_fn=input_fn,
                                 steps=self._eval_steps,
                                 metrics=self._eval_metrics,
                                 name=name)
      except NotFittedError:
        # Print warning message every 10 mins.
        if time.time() - last_fitted_error_time > 600:
          logging.warning(
              "Estimator is not fitted yet. "
              "Will start an evaluation when a checkpoint will be ready.")
          last_fitted_error_time = time.time()

      duration = time.time() - start
      if duration < throttle_delay_secs:
        difference = throttle_delay_secs - duration
        logging.info("Waiting %f secs before starting next eval run.",
                     difference)
        time.sleep(difference)

  def continuous_eval(self, delay_secs=None, throttle_delay_secs=None):
    self._continuous_eval(self._eval_input_fn,
                          name="continuous",
                          delay_secs=delay_secs,
                          throttle_delay_secs=throttle_delay_secs)

  def continuous_eval_on_train_data(self,
                                    delay_secs=None,
                                    throttle_delay_secs=None):
    self._continuous_eval(self._train_input_fn,
                          name="continuous_on_train_data",
                          delay_secs=delay_secs,
                          throttle_delay_secs=throttle_delay_secs)

  def train_and_evaluate(self):
    """Interleaves training and evaluation.

    The frequency of evaluation is controlled by the contructor arg
    `min_eval_frequency`. When this parameter is None or 0, evaluation happens
    only after training has completed. Note that evaluation cannot happen
    more frequently than checkpoints are taken. If no new snapshots are
    available when evaluation is supposed to occur, then evaluation doesn't
    happen for another `min_eval_frequency` steps (assuming a checkpoint is
    available at that point). Thus, settings `min_eval_frequency` to 1 means
    that the model will be evaluated everytime there is a new checkpoint.

    This is particular useful for a "Master" task in the cloud, whose
    responsibility it is to take checkpoints, evaluate those checkpoints,
    and write out summaries. Participating in training as the supervisor
    allows such a task to accomplish the first and last items, while
    performing evaluation allows for the second.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    # The directory to which evaluation summaries are written are determined
    # by adding a suffix to 'eval'; that suffix is the 'name' parameter to
    # the various evaluate(...) methods. By setting it to None, we force
    # the directory name to simply be 'eval'.
    eval_dir_suffix = None

    # We set every_n_steps to 1, but evaluation only occurs when a new
    # snapshot is available. If, by the time we finish evaluation
    # there is a new snapshot, then we just evaluate again. Otherwise,
    # we keep training until one becomes available.
    with _new_attr_context(self, "_train_monitors"):
      self._train_monitors = self._train_monitors or []
      if self._min_eval_frequency:
        self._train_monitors += [monitors.ValidationMonitor(
            input_fn=self._eval_input_fn, eval_steps=self._eval_steps,
            metrics=self._eval_metrics, every_n_steps=self._min_eval_frequency,
            name=eval_dir_suffix,
        )]
      self.train(delay_secs=0)

    return self._estimator.evaluate(input_fn=self._eval_input_fn,
                                    steps=self._eval_steps,
                                    metrics=self._eval_metrics,
                                    name=eval_dir_suffix)


  def run_std_server(self):
    """Starts a TensorFlow server and joins the serving thread.

    Typically used for parameter servers.

    Raises:
      ValueError: if not enough information is available in the estimator's
        config to create a server.
    """
    self._start_server().join()

  def test(self):
    """Tests training and evaluating the estimator both for a single step.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    self._estimator.fit(input_fn=self._train_input_fn,
                        steps=1,
                        monitors=self._train_monitors)

    return self._estimator.evaluate(input_fn=self._eval_input_fn,
                                    steps=1,
                                    metrics=self._eval_metrics,
                                    name="one_pass")

  def _start_server(self):
    """Creates, starts, and returns a server_lib.Server."""
    config = self._estimator.config
    if (not config.cluster_spec or not config.job_name or not config.master or
        config.task is None):
      raise ValueError("Could not start server; be sure to specify "
                       "cluster_spec, job_name, master, and task in "
                       "RunConfig or set the TF_CONFIG environment variable.")
    server = server_lib.Server(
        config.cluster_spec,
        job_name=config.job_name,
        task_index=config.task,
        config=config.tf_config,
        start=False)
    server.start()
    return server


@contextlib.contextmanager
def _new_attr_context(obj, attr):
  """Creates a new context in which an object's attribute can be changed.

  This creates a context in which an object's attribute can be changed.
  Once the context is exited, the attribute reverts to its original value.

  Example usage:
    my_obj.x = 1
    with _new_attr_context(my_obj, "x"):
      my_obj.x = 2
      print(my_obj.x)
    print(my_obj.x)
  """
  saved = getattr(obj, attr)
  try:
    yield
  finally:
    setattr(obj, attr, saved)
