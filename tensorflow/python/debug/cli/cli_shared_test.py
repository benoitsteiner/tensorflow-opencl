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
"""Unit tests for the shared functions and classes for tfdbg CLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class GetRunStartIntroAndDescriptionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.const_a = constant_op.constant(11.0, name="a")
    self.const_b = constant_op.constant(22.0, name="b")
    self.const_c = constant_op.constant(33.0, name="c")

  def tearDown(self):
    ops.reset_default_graph()

  def testSingleFetchNoFeeds(self):
    run_start_intro = cli_shared.get_run_start_intro(12, self.const_a, None, {})

    # Verify line about run() call number.
    self.assertEqual("About to enter Session run() call #12:",
                     run_start_intro.lines[1])

    # Verify line about fetch.
    const_a_name_line = run_start_intro.lines[4]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())

    # Verify line about feeds.
    feeds_line = run_start_intro.lines[7]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify lines about possible commands and their font attributes.
    self.assertEqual("run:", run_start_intro.lines[11][2:])
    self.assertEqual([(2, 5, "bold")], run_start_intro.font_attr_segs[11])
    self.assertEqual("run -n:", run_start_intro.lines[13][2:])
    self.assertEqual([(2, 8, "bold")], run_start_intro.font_attr_segs[13])
    self.assertEqual("run -f <filter_name>:", run_start_intro.lines[15][2:])
    self.assertEqual([(2, 22, "bold")], run_start_intro.font_attr_segs[15])

    # Verify short description.
    description = cli_shared.get_run_short_description(12, self.const_a, None)
    self.assertEqual("run #12: 1 fetch (a:0); 0 feeds", description)

  def testTwoFetchesListNoFeeds(self):
    fetches = [self.const_a, self.const_b]
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_a_name_line = run_start_intro.lines[4]
    const_b_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())
    self.assertEqual(self.const_b.name, const_b_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testNestedListAsFetches(self):
    fetches = [self.const_c, [self.const_a, self.const_b]]
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    # Verify lines about the fetches.
    self.assertEqual(self.const_c.name, run_start_intro.lines[4].strip())
    self.assertEqual(self.const_a.name, run_start_intro.lines[5].strip())
    self.assertEqual(self.const_b.name, run_start_intro.lines[6].strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 3 fetches; 0 feeds", description)

  def testNestedDictAsFetches(self):
    fetches = {"c": self.const_c, "ab": {"a": self.const_a, "b": self.const_b}}
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    # Verify lines about the fetches. The ordering of the dict keys is
    # indeterminate.
    fetch_names = set()
    fetch_names.add(run_start_intro.lines[4].strip())
    fetch_names.add(run_start_intro.lines[5].strip())
    fetch_names.add(run_start_intro.lines[6].strip())

    self.assertEqual({"a:0", "b:0", "c:0"}, fetch_names)

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 3 fetches; 0 feeds", description)

  def testTwoFetchesAsTupleNoFeeds(self):
    fetches = (self.const_a, self.const_b)
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_a_name_line = run_start_intro.lines[4]
    const_b_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())
    self.assertEqual(self.const_b.name, const_b_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testTwoFetchesAsNamedTupleNoFeeds(self):
    fetches_namedtuple = namedtuple("fetches", "x y")
    fetches = fetches_namedtuple(self.const_b, self.const_c)
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_b_name_line = run_start_intro.lines[4]
    const_c_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_b.name, const_b_name_line.strip())
    self.assertEqual(self.const_c.name, const_c_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testWithFeedDict(self):
    feed_dict = {
        self.const_a: 10.0,
        self.const_b: 20.0,
    }

    run_start_intro = cli_shared.get_run_start_intro(1, self.const_c, feed_dict,
                                                     {})

    const_c_name_line = run_start_intro.lines[4]
    self.assertEqual(self.const_c.name, const_c_name_line.strip())

    # Verify lines about the feed dict.
    feed_a_line = run_start_intro.lines[7]
    feed_b_line = run_start_intro.lines[8]
    self.assertEqual(self.const_a.name, feed_a_line.strip())
    self.assertEqual(self.const_b.name, feed_b_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, self.const_c,
                                                       feed_dict)
    self.assertEqual("run #1: 1 fetch (c:0); 2 feeds", description)

  def testTensorFilters(self):
    feed_dict = {self.const_a: 10.0}
    tensor_filters = {
        "filter_a": lambda x: True,
        "filter_b": lambda x: False,
    }

    run_start_intro = cli_shared.get_run_start_intro(1, self.const_c, feed_dict,
                                                     tensor_filters)

    # Verify the listed names of the tensor filters.
    filter_names = set()
    filter_names.add(run_start_intro.lines[18].split(" ")[-1])
    filter_names.add(run_start_intro.lines[19].split(" ")[-1])

    self.assertEqual({"filter_a", "filter_b"}, filter_names)

    # Verify short description.
    description = cli_shared.get_run_short_description(1, self.const_c,
                                                       feed_dict)
    self.assertEqual("run #1: 1 fetch (c:0); 1 feed (a:0)", description)


class GetErrorIntroTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.var_a = variables.Variable(42.0, name="a")

  def tearDown(self):
    ops.reset_default_graph()

  def testShapeError(self):
    tf_error = errors.OpError(None, self.var_a.initializer, "foo description",
                              None)

    error_intro = cli_shared.get_error_intro(tf_error)

    self.assertEqual("!!! An error occurred during the run !!!",
                     error_intro.lines[1])
    self.assertEqual([(0, len(error_intro.lines[1]), "blink")],
                     error_intro.font_attr_segs[1])

    self.assertEqual(2, error_intro.lines[4].index("ni a/Assign"))
    self.assertEqual([(2, 13, "bold")], error_intro.font_attr_segs[4])

    self.assertEqual(2, error_intro.lines[6].index("li -r a/Assign"))
    self.assertEqual([(2, 16, "bold")], error_intro.font_attr_segs[6])

    self.assertEqual(2, error_intro.lines[8].index("lt"))
    self.assertEqual([(2, 4, "bold")], error_intro.font_attr_segs[8])

    self.assertTrue(error_intro.lines[11].startswith("Op name:"))
    self.assertTrue(error_intro.lines[11].endswith("a/Assign"))

    self.assertTrue(error_intro.lines[12].startswith("Error type:"))
    self.assertTrue(error_intro.lines[12].endswith(str(type(tf_error))))

    self.assertEqual("Details:", error_intro.lines[14])
    self.assertTrue(error_intro.lines[15].startswith("foo description"))


if __name__ == "__main__":
  googletest.main()
