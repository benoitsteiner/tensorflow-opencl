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

"""Linear Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re

import six

from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training as train


# The default learning rate of 0.2 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.2


def _get_optimizer(spec):
  if isinstance(spec, six.string_types):
    return layers.OPTIMIZER_CLS_NAMES[spec](
        learning_rate=_LEARNING_RATE)
  elif callable(spec):
    return spec()
  return spec


# TODO(ispir): Remove this function by fixing '_infer_model' with single outputs
# and as_iteable case.
def _as_iterable(preds, output):
  for pred in preds:
    yield pred[output]


def _add_bias_column(feature_columns, columns_to_tensors, bias_variable,
                     labels, columns_to_variables):
  # TODO(b/31008490): Move definition to a common constants place.
  bias_column_name = "tf_virtual_bias_column"
  if any(col.name is bias_column_name for col in feature_columns):
    raise ValueError("%s is a reserved column name." % bias_column_name)
  bias_column = layers.real_valued_column(bias_column_name)
  columns_to_tensors[bias_column] = array_ops.ones_like(labels,
                                                        dtype=dtypes.float32)
  columns_to_variables[bias_column] = [bias_variable]


def _linear_model_fn(features, labels, mode, params, config=None):
  """A model_fn for linear models that use a gradient-based optimizer.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * optimizer: string, `Optimizer` object, or callable that defines the
          optimizer to use for training. If `None`, will use a FTRL optimizer.
      * gradient_clip_norm: A float > 0. If provided, gradients are
          clipped to their global norm with this clipping ratio.
      * num_ps_replicas: The number of parameter server replicas.
      * joint_weights: If True, the weights for all columns will be stored in a
        single (possibly partitioned) variable. It's more efficient, but it's
        incompatible with SDCAOptimizer, and requires all feature columns are
        sparse and use the 'sum' combiner.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    A `ModelFnOps` instance.

  Raises:
    ValueError: If mode is not any of the `ModeKeys`.
  """
  head = params["head"]
  feature_columns = params["feature_columns"]
  optimizer = params.get("optimizer") or _get_default_optimizer(feature_columns)
  gradient_clip_norm = params.get("gradient_clip_norm", None)
  num_ps_replicas = config.num_ps_replicas if config else 0
  joint_weights = params.get("joint_weights", False)

  if not isinstance(features, dict):
    features = {"": features}

  parent_scope = "linear"
  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas,
      min_slice_size=64 << 20)

  with variable_scope.variable_scope(
      parent_scope, values=features.values(), partitioner=partitioner) as scope:
    if joint_weights:
      logits, _, _ = (
          layers.joint_weighted_sum_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=feature_columns,
              num_outputs=head.logits_dimension,
              weight_collections=[parent_scope],
              scope=scope))
    else:
      logits, _, _ = (
          layers.weighted_sum_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=feature_columns,
              num_outputs=head.logits_dimension,
              weight_collections=[parent_scope],
              scope=scope))

  def _train_op_fn(loss):
    global_step = contrib_variables.get_global_step()
    my_vars = ops.get_collection("linear")
    grads = gradients.gradients(loss, my_vars)
    if gradient_clip_norm:
      grads, _ = clip_ops.clip_by_global_norm(grads, gradient_clip_norm)
    return (_get_optimizer(optimizer).apply_gradients(
        zip(grads, my_vars), global_step=global_step))

  return head.head_ops(features, labels, mode, _train_op_fn, logits)


def sdca_model_fn(features, labels, mode, params):
  """A model_fn for linear models that use the SDCA optimizer.

  Args:
    features: A dict of `Tensor` keyed by column name.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance. Type must be one of `_BinarySvmHead`,
          `_RegressionHead` or `_MultiClassHead`.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * optimizer: An `SDCAOptimizer` instance.
      * weight_column_name: A string defining the weight feature column, or
          None if there are no weights.
      * update_weights_hook: A `SessionRunHook` object or None. Used to update
          model weights.

  Returns:
    A `ModelFnOps` instance.

  Raises:
    ValueError: If `optimizer` is not an `SDCAOptimizer` instance.
    ValueError: If the type of head is neither `_BinarySvmHead`, nor
      `_RegressionHead` nor `_MultiClassHead`.
    ValueError: If mode is not any of the `ModeKeys`.
  """
  head = params["head"]
  feature_columns = params["feature_columns"]
  optimizer = params["optimizer"]
  weight_column_name = params["weight_column_name"]
  update_weights_hook = params.get("update_weights_hook", None)

  if not isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
    raise ValueError("Optimizer must be of type SDCAOptimizer")

  # pylint: disable=protected-access
  if isinstance(head, head_lib._BinarySvmHead):
    loss_type = "hinge_loss"
  elif isinstance(
      head, (head_lib._MultiClassHead, head_lib._BinaryLogisticHead)):
    loss_type = "logistic_loss"
  elif isinstance(head, head_lib._RegressionHead):
    loss_type = "squared_loss"
  else:
    raise ValueError("Unsupported head type: {}".format(head))
  # pylint: enable=protected-access

  parent_scope = "linear"

  with variable_scope.variable_op_scope(
      features.values(), parent_scope) as scope:
    logits, columns_to_variables, bias = (
        layers.weighted_sum_from_feature_columns(
            columns_to_tensors=features,
            feature_columns=feature_columns,
            num_outputs=1,
            scope=scope))

    _add_bias_column(feature_columns, features, bias, labels,
                     columns_to_variables)

  def _train_op_fn(unused_loss):
    global_step = contrib_variables.get_global_step()
    sdca_model, train_op = optimizer.get_train_step(columns_to_variables,
                                                    weight_column_name,
                                                    loss_type, features,
                                                    labels, global_step)
    if update_weights_hook is not None:
      update_weights_hook.set_parameters(sdca_model, train_op)
    return train_op

  model_fn_ops = head.head_ops(features, labels, mode, _train_op_fn, logits)
  if update_weights_hook is not None:
    return model_fn_ops._replace(
        training_chief_hooks=(model_fn_ops.training_chief_hooks +
                              [update_weights_hook]))
  return model_fn_ops


# Ensures consistency with LinearComposableModel.
def _get_default_optimizer(feature_columns):
  learning_rate = min(_LEARNING_RATE, 1.0 / math.sqrt(len(feature_columns)))
  return train.FtrlOptimizer(learning_rate=learning_rate)


class _SdcaUpdateWeightsHook(session_run_hook.SessionRunHook):
  """SessionRunHook to update and shrink SDCA model weights."""

  def __init__(self):
    pass

  def set_parameters(self, sdca_model, train_op):
    self._sdca_model = sdca_model
    self._train_op = train_op

  def begin(self):
    """Construct the update_weights op.

    The op is implicitly added to the default graph.
    """
    self._update_op = self._sdca_model.update_weights(self._train_op)

  def before_run(self, run_context):
    """Return the update_weights op so that it is executed during this run."""
    return session_run_hook.SessionRunArgs(self._update_op)


class LinearClassifier(estimator.Estimator):
  """Linear classifier model.

  Train a linear model to classify instances into one of multiple possible
  classes. When number of possible classes is 2, this is binary classification.

  Example:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  # Estimator using the default optimizer.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
      optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Or estimator using the SDCAOptimizer.
  estimator = LinearClassifier(
     feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
     optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
       example_id_column='example_id',
       num_loss_partitions=...,
       symmetric_l2_regularization=2.0
     ))

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    ...
  def input_fn_eval: # returns x, y (where y represents label's class index).
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x) # returns predicted labels (i.e. label's class index).
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.
  """

  def __init__(self,  # _joint_weight pylint: disable=invalid-name
               feature_columns,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               gradient_clip_norm=None,
               enable_centered_bias=False,
               _joint_weight=False,
               config=None,
               feature_engineering_fn=None):
    """Construct a `LinearClassifier` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        Note that class labels are integers representing the class index (i.e.
        values from 0 to n_classes-1). For arbitrary label values (e.g. string
        labels), convert to class indices first.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: The optimizer used to train the model. If specified, it should
        be either an instance of `tf.Optimizer` or the SDCAOptimizer. If `None`,
        the Ftrl optimizer will be used.
      gradient_clip_norm: A `float` > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      _joint_weight: If True, the weights for all columns will be stored in a
        single (possibly partitioned) variable. It's more efficient, but it's
        incompatible with SDCAOptimizer, and requires all feature columns are
        sparse and use the 'sum' combiner.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.

    Returns:
      A `LinearClassifier` estimator.

    Raises:
      ValueError: if n_classes < 2.
    """
    # TODO(zoy): Give an unsupported error if enable_centered_bias is
    #    requested for SDCA once its default changes to False.
    self._feature_columns = tuple(feature_columns or [])
    assert self._feature_columns
    self._optimizer = optimizer

    chief_hook = None
    if (isinstance(optimizer, sdca_optimizer.SDCAOptimizer) and
        enable_centered_bias):
      enable_centered_bias = False
      logging.warning("centered_bias is not supported with SDCA, "
                      "please disable it explicitly.")
    head = head_lib._multi_class_head(  # pylint: disable=protected-access
        n_classes,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias)
    params = {
        "head": head,
        "feature_columns": feature_columns,
        "optimizer": optimizer,
    }

    if isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
      assert not _joint_weight, ("_joint_weight is incompatible with the"
                                 " SDCAOptimizer")
      assert n_classes == 2, "SDCA only applies to binary classification."

      model_fn = sdca_model_fn
      # The model_fn passes the model parameters to the chief_hook. We then use
      # the hook to update weights and shrink step only on the chief.
      chief_hook = _SdcaUpdateWeightsHook()
      params.update({
          "weight_column_name": weight_column_name,
          "update_weights_hook": chief_hook,
      })
    else:
      model_fn = _linear_model_fn
      params.update({
          "gradient_clip_norm": gradient_clip_norm,
          "joint_weights": _joint_weight,
      })

    super(LinearClassifier, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        feature_engineering_fn=feature_engineering_fn)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=True):
    """Runs inference to determine the predicted class (i.e. class index)."""
    return self.predict_classes(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        as_iterable=as_iterable)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_classes(self, x=None, input_fn=None, batch_size=None,
                      as_iterable=True):
    """Runs inference to determine the predicted class (i.e. class index)."""
    key = prediction_key.PredictionKey.CLASSES
    preds = super(LinearClassifier, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key]

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_proba(self, x=None, input_fn=None, batch_size=None, outputs=None,
                    as_iterable=True):
    """Runs inference to determine the class probability predictions."""
    key = prediction_key.PredictionKey.PROBABILITIES
    preds = super(LinearClassifier, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key]

  def export(self,
             export_dir,
             input_fn=None,
             input_feature_key=None,
             use_deprecated_input_fn=True,
             signature_fn=None,
             default_batch_size=1,
             exports_to_keep=None):
    """See BaseEstimator.export."""
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)

    return super(LinearClassifier, self).export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=(signature_fn or
                      export.classification_signature_fn_with_prob),
        prediction_key=prediction_key.PredictionKey.PROBABILITIES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def weights_(self):
    values = {}
    if self._optimizer and not callable(self._optimizer):
      optimizer_name = _get_optimizer(self._optimizer).get_name()
    elif self._optimizer and callable(self._optimizer):
      raise ValueError("Callable optimizer is not supported in this method.")
    else:
      optimizer_name = _get_default_optimizer(self._feature_columns).get_name()
    optimizer_regex = r".*/" + optimizer_name + r"(_\d)?$"
    for name in self.get_variable_names():
      if (name.startswith("linear/") and
          name != "linear/bias_weight" and
          not re.match(optimizer_regex, name)):
        values[name] = self.get_variable_value(name)
    if len(values) == 1:
      return values[list(values.keys())[0]]
    return values

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def bias_(self):
    return self.get_variable_value("linear/bias_weight")


class LinearRegressor(estimator.Estimator):
  """Linear regressor model.

  Train a linear regression model to predict label value given observation of
  feature values.

  Example:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  estimator = LinearRegressor(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a KeyError:

  * if `weight_column_name` is not `None`:
    key=weight_column_name, value=a `Tensor`
  * for column in `feature_columns`:
    - if isinstance(column, `SparseColumn`):
        key=column.name, value=a `SparseTensor`
    - if isinstance(column, `WeightedSparseColumn`):
        {key=id column name, value=a `SparseTensor`,
         key=weight column name, value=a `SparseTensor`}
    - if isinstance(column, `RealValuedColumn`):
        key=column.name, value=a `Tensor`
  """

  def __init__(self,  # _joint_weights: pylint: disable=invalid-name
               feature_columns,
               model_dir=None,
               weight_column_name=None,
               optimizer=None,
               gradient_clip_norm=None,
               enable_centered_bias=False,
               label_dimension=1,
               _joint_weights=False,
               config=None,
               feature_engineering_fn=None):
    """Construct a `LinearRegressor` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph, etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Ftrl optimizer.
      gradient_clip_norm: A `float` > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      label_dimension: Dimension of the label for multilabels. Defaults to 1.
      _joint_weights: If True use a single (possibly partitioned) variable to
        store the weights. It's faster, but requires all feature columns are
        sparse and have the 'sum' combiner. Incompatible with SDCAOptimizer.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.

    Returns:
      A `LinearRegressor` estimator.
    """
    self._feature_columns = tuple(feature_columns or [])
    assert self._feature_columns
    self._optimizer = optimizer

    chief_hook = None
    if (isinstance(optimizer, sdca_optimizer.SDCAOptimizer) and
        enable_centered_bias):
      enable_centered_bias = False
      logging.warning("centered_bias is not supported with SDCA, "
                      "please disable it explicitly.")
    head = head_lib._regression_head(  # pylint: disable=protected-access
        weight_column_name=weight_column_name,
        label_dimension=label_dimension,
        enable_centered_bias=enable_centered_bias)
    params = {
        "head": head,
        "feature_columns": feature_columns,
        "optimizer": optimizer,
    }

    if isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
      assert label_dimension == 1, "SDCA only applies for label_dimension=1."
      assert not _joint_weights, ("_joint_weights is incompatible with"
                                  " SDCAOptimizer.")

      model_fn = sdca_model_fn
      # The model_fn passes the model parameters to the chief_hook. We then use
      # the hook to update weights and shrink step only on the chief.
      chief_hook = _SdcaUpdateWeightsHook()
      params.update({
          "weight_column_name": weight_column_name,
          "update_weights_hook": chief_hook,
      })
    else:
      model_fn = _linear_model_fn
      params.update({
          "gradient_clip_norm": gradient_clip_norm,
          "joint_weights": _joint_weights,
      })

    super(LinearRegressor, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        feature_engineering_fn=feature_engineering_fn)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=True):
    """Runs inference to determine the predicted scores."""
    return self.predict_scores(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        as_iterable=as_iterable)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_scores(self, x=None, input_fn=None, batch_size=None,
                     as_iterable=True):
    """Runs inference to determine the predicted scores."""
    key = prediction_key.PredictionKey.SCORES
    preds = super(LinearRegressor, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key]

  def export(self,
             export_dir,
             input_fn=None,
             input_feature_key=None,
             use_deprecated_input_fn=True,
             signature_fn=None,
             default_batch_size=1,
             exports_to_keep=None):
    """See BaseEstimator.export."""
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)

    return super(LinearRegressor, self).export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=(signature_fn or export.regression_signature_fn),
        prediction_key=prediction_key.PredictionKey.SCORES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def weights_(self):
    values = {}
    if self._optimizer and not callable(self._optimizer):
      optimizer_name = _get_optimizer(self._optimizer).get_name()
    elif self._optimizer and callable(self._optimizer):
      raise ValueError("Callable optimizer is not supported in this method.")
    else:
      optimizer_name = _get_default_optimizer(self._feature_columns).get_name()
    optimizer_regex = r".*/" + optimizer_name + r"(_\d)?$"
    for name in self.get_variable_names():
      if (name.startswith("linear/") and
          name != "linear/bias_weight" and
          not re.match(optimizer_regex, name)):
        values[name] = self.get_variable_value(name)
    if len(values) == 1:
      return values[list(values.keys())[0]]
    return values

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def bias_(self):
    return self.get_variable_value("linear/bias_weight")
