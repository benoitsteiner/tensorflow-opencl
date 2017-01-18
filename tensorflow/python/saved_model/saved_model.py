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
"""Convenience functions to save a model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=unused-import
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
# pylint: enable=unused-import

from tensorflow.python.util.all_util import remove_undocumented


_allowed_symbols = [
    "builder",
    "loader",
    "signature_constants",
    "signature_def_utils",
    "tag_constants",
]
remove_undocumented(__name__, _allowed_symbols)
