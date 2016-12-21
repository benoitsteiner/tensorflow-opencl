# pylint: disable=g-bad-file-header
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
"""Tests for basic_session_run_hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile
import threading
import time

from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.testing.python.framework import fake_summary_writer
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook


class MockCheckpointSaverListener(
    basic_session_run_hooks.CheckpointSaverListener):

  def __init__(self):
    self.begin_count = 0
    self.before_save_count = 0
    self.after_save_count = 0
    self.end_count = 0

  def begin(self):
    self.begin_count += 1

  def before_save(self, session, global_step):
    self.before_save_count += 1

  def after_save(self, session, global_step):
    self.after_save_count += 1

  def end(self, session, global_step):
    self.end_count += 1

  def get_counts(self):
    return {
        'begin': self.begin_count,
        'before_save': self.before_save_count,
        'after_save': self.after_save_count,
        'end': self.end_count
    }


class SecondOrStepTimerTest(test.TestCase):

  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks._SecondOrStepTimer(every_secs=2.0, every_steps=10)

  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks._SecondOrStepTimer()

  def test_every_secs(self):
    timer = basic_session_run_hooks._SecondOrStepTimer(every_secs=1.0)
    self.assertTrue(timer.should_trigger_for_step(1))

    timer.update_last_triggered_step(1)
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertFalse(timer.should_trigger_for_step(2))

    time.sleep(1.0)
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertTrue(timer.should_trigger_for_step(2))

  def test_every_steps(self):
    timer = basic_session_run_hooks._SecondOrStepTimer(every_steps=3)
    self.assertTrue(timer.should_trigger_for_step(1))

    timer.update_last_triggered_step(1)
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertFalse(timer.should_trigger_for_step(2))
    self.assertFalse(timer.should_trigger_for_step(3))
    self.assertTrue(timer.should_trigger_for_step(4))

  def test_update_last_triggered_step(self):
    timer = basic_session_run_hooks._SecondOrStepTimer(every_steps=1)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(1)
    self.assertEqual(None, elapsed_secs)
    self.assertEqual(None, elapsed_steps)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(5)
    self.assertLess(0, elapsed_secs)
    self.assertEqual(4, elapsed_steps)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(7)
    self.assertLess(0, elapsed_secs)
    self.assertEqual(2, elapsed_steps)


class StopAtStepTest(test.TestCase):

  def test_raise_in_both_last_step_and_num_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.StopAtStepHook(num_steps=10, last_step=20)

  def test_stop_based_on_last_step(self):
    h = basic_session_run_hooks.StopAtStepHook(last_step=10)
    with ops.Graph().as_default():
      global_step = variables.get_or_create_global_step()
      no_op = control_flow_ops.no_op()
      h.begin()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(state_ops.assign(global_step, 5))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 9))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 10))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 11))
        mon_sess._should_stop = False
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())

  def test_stop_based_on_num_step(self):
    h = basic_session_run_hooks.StopAtStepHook(num_steps=10)

    with ops.Graph().as_default():
      global_step = variables.get_or_create_global_step()
      no_op = control_flow_ops.no_op()
      h.begin()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(state_ops.assign(global_step, 5))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 13))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 14))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 15))
        mon_sess._should_stop = False
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())


class LoggingTensorHookTest(test.TestCase):

  def setUp(self):
    # Mock out logging calls so we can verify whether correct tensors are being
    # monitored.
    self._actual_log = tf_logging.info
    self.logged_message = None

    def mock_log(*args, **kwargs):
      self.logged_message = args
      self._actual_log(*args, **kwargs)

    tf_logging.info = mock_log

  def tearDown(self):
    tf_logging.info = self._actual_log

  def test_illegal_args(self):
    with self.assertRaisesRegexp(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=0)
    with self.assertRaisesRegexp(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=-10)
    with self.assertRaisesRegexp(ValueError, 'xactly one of'):
      basic_session_run_hooks.LoggingTensorHook(
          tensors=['t'], every_n_iter=5, every_n_secs=5)
    with self.assertRaisesRegexp(ValueError, 'xactly one of'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'])

  def test_print_every_n_steps(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      hook = basic_session_run_hooks.LoggingTensorHook(
          tensors=[t.name], every_n_iter=10)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      sess.run(variables_lib.global_variables_initializer())
      mon_sess.run(train_op)
      self.assertRegexpMatches(str(self.logged_message), t.name)
      for j in range(3):
        _ = j
        self.logged_message = ''
        for i in range(9):
          _ = i
          mon_sess.run(train_op)
          # assertNotRegexpMatches is not supported by python 3.1 and later
          self.assertEqual(str(self.logged_message).find(t.name), -1)
        mon_sess.run(train_op)
        self.assertRegexpMatches(str(self.logged_message), t.name)

  def test_print_every_n_secs(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)

      hook = basic_session_run_hooks.LoggingTensorHook(
          tensors=[t.name], every_n_secs=1.0)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      sess.run(variables_lib.global_variables_initializer())

      mon_sess.run(train_op)
      self.assertRegexpMatches(str(self.logged_message), t.name)

      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.logged_message = ''
      mon_sess.run(train_op)
      self.assertEqual(str(self.logged_message).find(t.name), -1)
      time.sleep(1.0)

      self.logged_message = ''
      mon_sess.run(train_op)
      self.assertRegexpMatches(str(self.logged_message), t.name)


class CheckpointSaverHookTest(test.TestCase):

  def setUp(self):
    self.model_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    with self.graph.as_default():
      self.scaffold = monitored_session.Scaffold()
      self.global_step = variables.get_or_create_global_step()
      self.train_op = state_ops.assign_add(self.global_step, 1)

  def tearDown(self):
    shutil.rmtree(self.model_dir, ignore_errors=True)

  def test_raise_when_saver_and_scaffold_both_missing(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(self.model_dir)

  def test_raise_when_saver_and_scaffold_both_present(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, saver=self.scaffold.saver, scaffold=self.scaffold)

  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=10, save_steps=20)

  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(self.model_dir)

  def test_save_secs_saves_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_secs_calls_listeners_at_begin_and_end(self):
    with self.graph.as_default():
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_secs=2,
          scaffold=self.scaffold,
          listeners=[listener])
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)  # hook runs here
        mon_sess.run(self.train_op)  # hook won't run here, so it does at end
        hook.end(sess)  # hook runs here
      self.assertEqual({
          'begin': 1,
          'before_save': 2,
          'after_save': 2,
          'end': 1
      }, listener.get_counts())

  def test_save_secs_saves_periodically(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        time.sleep(2.5)
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        time.sleep(2.5)
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(6,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_secs_calls_listeners_periodically(self):
    with self.graph.as_default():
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_secs=2,
          scaffold=self.scaffold,
          listeners=[listener])
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)  # hook runs here
        mon_sess.run(self.train_op)
        time.sleep(2.5)
        mon_sess.run(self.train_op)  # hook runs here
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        time.sleep(2.5)
        mon_sess.run(self.train_op)  # hook runs here
        mon_sess.run(self.train_op)  # hook won't run here, so it does at end
        hook.end(sess)  # hook runs here
      self.assertEqual({
          'begin': 1,
          'before_save': 4,
          'after_save': 4,
          'end': 1
      }, listener.get_counts())

  def test_save_steps_saves_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_steps_saves_periodically(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_saves_at_end(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        hook.end(sess)
        self.assertEqual(2,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_summary_writer_defs(self):
    fake_summary_writer.FakeSummaryWriter.install()
    writer_cache.FileWriterCache.clear()
    summary_writer = writer_cache.FileWriterCache.get(self.model_dir)

    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.model_dir,
          expected_added_meta_graphs=[
              meta_graph.create_meta_graph_def(
                  graph_def=self.graph.as_graph_def(add_shapes=True),
                  saver_def=self.scaffold.saver.saver_def)
          ])

    fake_summary_writer.FakeSummaryWriter.uninstall()


class StepCounterHookTest(test.TestCase):

  def setUp(self):
    self.log_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.log_dir, ignore_errors=True)

  def test_step_counter_every_n_steps(self):
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      global_step = variables.get_or_create_global_step()
      train_op = state_ops.assign_add(global_step, 1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=10)
      hook.begin()
      sess.run(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(30):
        time.sleep(0.01)
        mon_sess.run(train_op)
      hook.end(sess)
      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([11, 21], summary_writer.summaries.keys())
      for step in [11, 21]:
        summary_value = summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  def test_step_counter_every_n_secs(self):
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      global_step = variables.get_or_create_global_step()
      train_op = state_ops.assign_add(global_step, 1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=None, every_n_secs=0.1)

      hook.begin()
      sess.run(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)
      time.sleep(0.2)
      mon_sess.run(train_op)
      time.sleep(0.2)
      mon_sess.run(train_op)
      hook.end(sess)

      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertTrue(summary_writer.summaries, 'No summaries were created.')
      self.assertItemsEqual([2, 3], summary_writer.summaries.keys())
      for summary in summary_writer.summaries.values():
        summary_value = summary[0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  def test_global_step_name(self):
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      with variable_scope.variable_scope('bar'):
        foo_step = variable_scope.get_variable(
            'foo',
            initializer=0,
            trainable=False,
            collections=[
                ops.GraphKeys.GLOBAL_STEP, ops.GraphKeys.GLOBAL_VARIABLES
            ])
      train_op = state_ops.assign_add(foo_step, 1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=1, every_n_secs=None)

      hook.begin()
      sess.run(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)
      mon_sess.run(train_op)
      hook.end(sess)

      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertTrue(summary_writer.summaries, 'No summaries were created.')
      self.assertItemsEqual([2], summary_writer.summaries.keys())
      summary_value = summary_writer.summaries[2][0].value[0]
      self.assertEqual('bar/foo/sec', summary_value.tag)


class SummarySaverHookTest(test.TestCase):

  def setUp(self):
    test.TestCase.setUp(self)

    self.log_dir = 'log/dir'
    self.summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir)

    var = variables_lib.Variable(0.0)
    tensor = state_ops.assign_add(var, 1.0)
    tensor2 = tensor * 2
    self.summary_op = summary_lib.scalar('my_summary', tensor)
    self.summary_op2 = summary_lib.scalar('my_summary2', tensor2)

    global_step = variables.get_or_create_global_step()
    self.train_op = state_ops.assign_add(global_step, 1)

  def test_raise_when_scaffold_and_summary_op_both_missing(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook()

  def test_raise_when_scaffold_and_summary_op_both_present(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          scaffold=monitored_session.Scaffold(), summary_op=self.summary_op)

  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          save_secs=10, save_steps=20, summary_writer=self.summary_writer)

  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          save_secs=None, save_steps=None, summary_writer=self.summary_writer)

  def test_save_steps(self):
    hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=8,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.test_session() as sess:
      hook.begin()
      sess.run(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(30):
        mon_sess.run(self.train_op)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            9: {
                'my_summary': 2.0
            },
            17: {
                'my_summary': 3.0
            },
            25: {
                'my_summary': 4.0
            },
        })

  def test_multiple_summaries(self):
    hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=8,
        summary_writer=self.summary_writer,
        summary_op=[self.summary_op, self.summary_op2])

    with self.test_session() as sess:
      hook.begin()
      sess.run(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(10):
        mon_sess.run(self.train_op)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0,
                'my_summary2': 2.0
            },
            9: {
                'my_summary': 2.0,
                'my_summary2': 4.0
            },
        })

  def test_save_secs_saving_once_every_step(self):
    hook = basic_session_run_hooks.SummarySaverHook(
        save_secs=0.5,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.test_session() as sess:
      hook.begin()
      sess.run(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(4):
        mon_sess.run(self.train_op)
        time.sleep(0.5)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            2: {
                'my_summary': 2.0
            },
            3: {
                'my_summary': 3.0
            },
            4: {
                'my_summary': 4.0
            },
        })

  def test_save_secs_saving_once_every_three_steps(self):
    hook = basic_session_run_hooks.SummarySaverHook(
        save_secs=0.9,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.test_session() as sess:
      hook.begin()
      sess.run(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(8):
        mon_sess.run(self.train_op)
        time.sleep(0.3)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            4: {
                'my_summary': 2.0
            },
            7: {
                'my_summary': 3.0
            },
        })


class GlobalStepWaiterHookTest(test.TestCase):

  def test_not_wait_for_step_zero(self):
    with ops.Graph().as_default():
      variables.get_or_create_global_step()
      hook = basic_session_run_hooks.GlobalStepWaiterHook(wait_until_step=0)
      hook.begin()
      with session_lib.Session() as sess:
        # Before run should return without waiting gstep increment.
        hook.before_run(
            session_run_hook.SessionRunContext(
                original_args=None, session=sess))

  def test_wait_for_step(self):
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      hook = basic_session_run_hooks.GlobalStepWaiterHook(wait_until_step=1000)
      hook.begin()
      with session_lib.Session() as sess:
        sess.run(variables_lib.global_variables_initializer())
        waiter = threading.Thread(
            target=hook.before_run,
            args=(session_run_hook.SessionRunContext(
                original_args=None, session=sess),))
        waiter.daemon = True
        waiter.start()
        time.sleep(1.0)
        self.assertTrue(waiter.is_alive())
        sess.run(state_ops.assign(gstep, 500))
        time.sleep(1.0)
        self.assertTrue(waiter.is_alive())
        sess.run(state_ops.assign(gstep, 1100))
        time.sleep(1.2)
        self.assertFalse(waiter.is_alive())


if __name__ == '__main__':
  test.main()
