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
"""TensorBoard Plugin abstract base class.

Every plugin in TensorBoard must extend and implement the abstract methods of
this base class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod


class TBPlugin(object):
  """TensorBoard plugin interface. Every plugin must extend from this class."""
  __metaclass__ = ABCMeta

  @abstractmethod
  def get_plugin_handlers(self, run_paths, logdir):
    """Returns a set of http handlers that the plugin implements.

    Each handler gets registered with the tensorboard handler and is served
    under a prefix path that includes the name of the plugin.

    Args:
      run_paths: A dict mapping a run name to an event file path.
      logdir: The logging directory TensorBoard was started with.

    Returns:
      A dict mapping route paths to http handler methods.
    """
    raise NotImplementedError()
