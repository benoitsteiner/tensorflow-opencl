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
"""Deprecated endpoints that we aren't yet ready to remove entirely.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# When the endpoints are removed from core TensorFlow, the old implementations
# will move to this file.
# pylint: disable=unused-import,line-too-long
from tensorflow.python.ops.logging_ops import audio_summary
from tensorflow.python.ops.logging_ops import histogram_summary
from tensorflow.python.ops.logging_ops import image_summary
from tensorflow.python.ops.logging_ops import merge_all_summaries
from tensorflow.python.ops.logging_ops import merge_summary
from tensorflow.python.ops.logging_ops import scalar_summary
# pylint: enable=unused-import,line-too-long
