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
"""Ops to use variables as resources."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_resource_variable_ops import *
# pylint: enable=wildcard-import

ops.RegisterShape("VarHandleOp")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("CreateVariableOp")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ReadVariableOp")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("AssignVariableOp")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("AssignAddVariableOp")(common_shapes.call_cpp_shape_fn)
