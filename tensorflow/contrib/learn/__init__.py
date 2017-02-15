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

# TODO(ptucker,ipolosukhin): Improve descriptions.
"""High level API for learning. See the @{$python/contrib.learn} guide.

@@BaseEstimator
@@Estimator
@@Trainable
@@Evaluable
@@KMeansClustering
@@ModeKeys
@@ModelFnOps
@@MetricSpec
@@PredictionKey
@@DNNClassifier
@@DNNRegressor
@@DNNLinearCombinedRegressor
@@DNNLinearCombinedClassifier
@@LinearClassifier
@@LinearRegressor
@@LogisticRegressor

@@Experiment
@@ExportStrategy
@@TaskType

@@NanLossDuringTrainingError
@@RunConfig
@@evaluate
@@infer
@@run_feeds
@@run_n
@@train

@@extract_dask_data
@@extract_dask_labels
@@extract_pandas_data
@@extract_pandas_labels
@@extract_pandas_matrix
@@infer_real_valued_columns_from_input
@@infer_real_valued_columns_from_input_fn
@@read_batch_examples
@@read_batch_features
@@read_batch_record_features

@@InputFnOps
@@ProblemType
@@build_parsing_serving_input_fn
@@make_export_strategy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow.contrib.learn.python.learn import *
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['datasets', 'head', 'io', 'models',
                    'monitors', 'NotFittedError', 'ops', 'preprocessing',
                    'utils', 'graph_actions']

remove_undocumented(__name__, _allowed_symbols)
