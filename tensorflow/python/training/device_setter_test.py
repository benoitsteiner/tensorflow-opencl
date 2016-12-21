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
"""Tests for device function for replicated training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import server_lib


class DeviceSetterTest(test.TestCase):

  _cluster_spec = server_lib.ClusterSpec({
      "ps": ["ps0:2222", "ps1:2222"],
      "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
  })

  def testCPUOverride(self):
    with ops.device(
        device_setter.replica_device_setter(cluster=self._cluster_spec)):
      with ops.device("/cpu:0"):
        v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      with ops.device("/cpu:0"):
        a = v + w
      self.assertDeviceEqual("/job:ps/task:0/cpu:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0/cpu:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker/cpu:0", a.device)

  def testPS2TasksWithClusterSpecClass(self):
    with ops.device(
        device_setter.replica_device_setter(cluster=self._cluster_spec)):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  def testPS2TasksWithClusterSpecDict(self):
    with ops.device(
        device_setter.replica_device_setter(cluster=self._cluster_spec.as_dict(
        ))):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  def testPS2TasksWithClusterDef(self):
    with ops.device(
        device_setter.replica_device_setter(
            cluster=self._cluster_spec.as_cluster_def())):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  def testPS2TasksWithDevice(self):
    cluster_spec = server_lib.ClusterSpec({
        "sun": ["sun0:2222", "sun1:2222", "sun2:2222"],
        "moon": ["moon0:2222", "moon1:2222"]
    })

    with ops.device(
        device_setter.replica_device_setter(
            ps_device="/job:moon",
            worker_device="/job:sun",
            cluster=cluster_spec.as_cluster_def())):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:moon/task:0", v.device)
      self.assertDeviceEqual("/job:moon/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:moon/task:1", w.device)
      self.assertDeviceEqual("/job:moon/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:sun", a.device)

  def testPS2TasksWithCPUConstraint(self):
    cluster_spec = server_lib.ClusterSpec({
        "sun": ["sun0:2222", "sun1:2222", "sun2:2222"],
        "moon": ["moon0:2222", "moon1:2222"]
    })

    with ops.device(
        device_setter.replica_device_setter(
            ps_device="/job:moon/cpu:0",
            worker_device="/job:sun",
            cluster=cluster_spec.as_cluster_def())):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:moon/task:0/cpu:0", v.device)
      self.assertDeviceEqual("/job:moon/task:0/cpu:0", v.initializer.device)
      self.assertDeviceEqual("/job:moon/task:1/cpu:0", w.device)
      self.assertDeviceEqual("/job:moon/task:1/cpu:0", w.initializer.device)
      self.assertDeviceEqual("/job:sun", a.device)


if __name__ == "__main__":
  test.main()
