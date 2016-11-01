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

"""TensorFlow estimators for Linear and DNN joined training models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.learn.python.learn.estimators import composable_model
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops


class _DNNLinearCombinedBaseEstimator(estimator.BaseEstimator):
  """An estimator for TensorFlow Linear and DNN joined training models.

    Input of `fit`, `train`, and `evaluate` should have following features,
      otherwise there will be a `KeyError`:
        if `weight_column_name` is not `None`, a feature with
          `key=weight_column_name` whose value is a `Tensor`.
        for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
        - if `column` is a `SparseColumn`, a feature with `key=column.name`
          whose `value` is a `SparseTensor`.
        - if `column` is a `WeightedSparseColumn`, two features: the first with
          `key` the id column name, the second with `key` the weight column
          name. Both features' `value` must be a `SparseTensor`.
        - if `column` is a `RealValuedColumn, a feature with `key=column.name`
          whose `value` is a `Tensor`.
  """

  def __init__(self,  # _joint_linear_weights pylint: disable=invalid-name
               head,
               model_dir=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               _joint_linear_weights=False,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               gradient_clip_norm=None,
               config=None,
               feature_engineering_fn=None,
               default_prediction_key=None,
               enable_centered_bias=False):
    """Initializes a _DNNLinearCombinedBaseEstimator instance.

    Args:
      head: A _Head object.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set should be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
      _joint_linear_weights: If True will use a single (possibly partitioned)
        variable to store all weights for the linear model. More efficient if
        there are many columns, however requires all columns are sparse and
        have the 'sum' combiner.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set should be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. If `None`, will use an Adagrad optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If `None`,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out
        a given coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      config: RunConfig object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.
      default_prediction_key: Default prediction key to use with metrics.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
    super(_DNNLinearCombinedBaseEstimator, self).__init__(
        model_dir=model_dir, config=config)

    num_ps_replicas = config.num_ps_replicas if config else 0

    self._linear_model = composable_model.LinearComposableModel(
        num_label_columns=head.logits_dimension,
        optimizer=linear_optimizer,
        _joint_weights=_joint_linear_weights,
        gradient_clip_norm=gradient_clip_norm,
        num_ps_replicas=num_ps_replicas)

    self._dnn_model = composable_model.DNNComposableModel(
        num_label_columns=head.logits_dimension,
        hidden_units=dnn_hidden_units,
        optimizer=dnn_optimizer,
        activation_fn=dnn_activation_fn,
        dropout=dnn_dropout,
        gradient_clip_norm=gradient_clip_norm,
        num_ps_replicas=num_ps_replicas) if dnn_hidden_units else None

    self._linear_feature_columns = linear_feature_columns
    self._linear_optimizer = linear_optimizer
    self._dnn_feature_columns = dnn_feature_columns
    self._dnn_hidden_units = dnn_hidden_units
    self._head = head
    self._default_prediction_key = default_prediction_key
    self._feature_engineering_fn = (
        feature_engineering_fn or
        (lambda features, labels: (features, labels)))
    self._enable_centered_bias = enable_centered_bias

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def linear_weights_(self):
    """Returns weights per feature of the linear part."""
    return self._linear_model.get_weights(model_dir=self._model_dir)

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def linear_bias_(self):
    """Returns bias of the linear part."""
    if not self._enable_centered_bias:
      return self._linear_model.get_bias(model_dir=self._model_dir)
    return (self._linear_model.get_bias(model_dir=self._model_dir) +
            self.get_variable_value("centered_bias_weight"))

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def dnn_weights_(self):
    """Returns weights of deep neural network part."""
    return self._dnn_model.get_weights(model_dir=self._model_dir)

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def dnn_bias_(self):
    """Returns bias of deep neural network part."""
    if not self._enable_centered_bias:
      return self._dnn_model.get_bias(model_dir=self._model_dir)
    return (self._dnn_model.get_bias(model_dir=self._model_dir) +
            [self._get_centered_bias_value()])

  # TODO(zakaria): Remove this function once export. export_estimator is
  #   obsolete.
  def _create_signature_fn(self):
    """Returns a function to create export signature of this Estimator."""
    # pylint: disable=protected-access
    return self._head._create_signature_fn()

  def _get_feature_dict(self, features):
    if isinstance(features, dict):
      return features
    return {"": features}

  def _get_train_ops(self, features, labels):
    """See base class."""

    features = self._get_feature_dict(features)
    features, labels = self._feature_engineering_fn(features, labels)
    logits = self._logits(features, is_training=True)

    def _make_training_op(training_loss):
      global_step = contrib_variables.get_global_step()
      assert global_step

      linear_train_step = self._linear_model.get_train_step(training_loss)
      dnn_train_step = (self._dnn_model.get_train_step(training_loss) if
                        self._dnn_model else [])
      with ops.control_dependencies(linear_train_step + dnn_train_step):
        with ops.get_default_graph().colocate_with(global_step):
          return state_ops.assign_add(global_step, 1).op

    model_fn_ops = self._head.head_ops(features, labels,
                                       estimator.ModeKeys.TRAIN,
                                       _make_training_op,
                                       logits=logits)
    return model_fn_ops.training_op, model_fn_ops.loss

  def _get_eval_ops(self, features, labels, metrics=None):
    """See base class."""
    features = self._get_feature_dict(features)
    features, labels = self._feature_engineering_fn(features, labels)
    logits = self._logits(features)

    model_fn_ops = self._head.head_ops(features, labels,
                                       estimator.ModeKeys.EVAL, None,
                                       logits=logits)
    all_metrics = model_fn_ops.default_metrics
    if metrics:
      for name, metric in six.iteritems(metrics):
        if not isinstance(name, tuple):
          # TODO(zakaria): remove once deprecation is finished (b/31229024)
          all_metrics[(name, self._default_prediction_key)] = metric
        else:
          all_metrics[name] = metric
    # TODO(zakaria): Remove this once we refactor this class to delegate
    #   to estimator.
    # pylint: disable=protected-access
    result = estimator._make_metrics_ops(all_metrics, features, labels,
                                         model_fn_ops.predictions)
    return result

  def _get_predict_ops(self, features):
    """See base class."""
    features = self._get_feature_dict(features)
    features, _ = self._feature_engineering_fn(features, None)
    logits = self._logits(features)
    model_fn_ops = self._head.head_ops(features, None, estimator.ModeKeys.INFER,
                                       None, logits=logits)
    return model_fn_ops.predictions

  @deprecated(
      "2016-09-23",
      "The signature of the input_fn accepted by export is changing to be "
      "consistent with what's used by tf.Learn Estimator's train/evaluate, "
      "which makes this function useless. This will be removed after the "
      "deprecation date.")
  def _get_feature_ops_from_example(self, examples_batch):
    column_types = layers.create_feature_spec_for_parsing((
        self._get_linear_feature_columns() or []) + (
            self._get_dnn_feature_columns() or []))
    features = parsing_ops.parse_example(examples_batch, column_types)
    return features

  def _get_linear_feature_columns(self):
    if not self._linear_feature_columns:
      return None
    feature_column_ops.check_feature_columns(self._linear_feature_columns)
    return sorted(set(self._linear_feature_columns), key=lambda x: x.key)

  def _get_dnn_feature_columns(self):
    if not self._dnn_feature_columns:
      return None
    feature_column_ops.check_feature_columns(self._dnn_feature_columns)
    return sorted(set(self._dnn_feature_columns), key=lambda x: x.key)

  def _dnn_logits(self, features, is_training):
    return self._dnn_model.build_model(
        features, self._dnn_feature_columns, is_training)

  def _linear_logits(self, features, is_training):
    return self._linear_model.build_model(
        features, self._linear_feature_columns, is_training)

  def _logits(self, features, is_training=False):
    linear_feature_columns = self._get_linear_feature_columns()
    dnn_feature_columns = self._get_dnn_feature_columns()
    if not (linear_feature_columns or dnn_feature_columns):
      raise ValueError("Either linear_feature_columns or dnn_feature_columns "
                       "should be defined.")

    if linear_feature_columns and dnn_feature_columns:
      logits = (self._linear_logits(features, is_training) +
                self._dnn_logits(features, is_training))
    elif dnn_feature_columns:
      logits = self._dnn_logits(features, is_training)
    else:
      logits = self._linear_logits(features, is_training)

    return logits


class DNNLinearCombinedClassifier(_DNNLinearCombinedBaseEstimator):
  """A classifier for TensorFlow Linear and DNN joined training models.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)
  education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                   combiner="sum")
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")

  estimator = DNNLinearCombinedClassifier(
      # common settings
      n_classes=n_classes,
      weight_column_name=weight_column_name,
      # wide settings
      linear_feature_columns=[education_x_occupation],
      linear_optimizer=tf.train.FtrlOptimizer(...),
      # deep settings
      dnn_feature_columns=[education_emb, occupation_emb],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.train.AdagradOptimizer(...))

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
    otherwise there will be a `KeyError`:
      if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
      for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `WeightedSparseColumn`, two features: the first with
        `key` the id column name, the second with `key` the weight column name.
        Both features' `value` must be a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  """

  def __init__(self,  # _joint_linear_weights pylint: disable=invalid-name
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               _joint_linear_weights=False,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=False,
               config=None,
               feature_engineering_fn=None):
    """Constructs a DNNLinearCombinedClassifier instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training.
        It will be multiplied by the loss of the example.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
      _joint_linear_weights: If True a single (possibly partitioned) variable
        will be used to store the linear model weights. It's faster, but
        requires all columns are sparse and have the 'sum' combiner.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set must be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. If `None`, will use an Adagrad optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If `None`,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out
        a given coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      config: RunConfig object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.

    Raises:
      ValueError: If `n_classes` < 2.
      ValueError: If both `linear_feature_columns` and `dnn_features_columns`
        are empty at the same time.
    """

    if n_classes < 2:
      raise ValueError("n_classes should be greater than 1. Given: {}".format(
          n_classes))
    head = head_lib._multi_class_head(  # pylint: disable=protected-access
        n_classes=n_classes,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias)
    super(DNNLinearCombinedClassifier, self).__init__(
        model_dir=model_dir,
        linear_feature_columns=linear_feature_columns,
        linear_optimizer=linear_optimizer,
        _joint_linear_weights=_joint_linear_weights,
        dnn_feature_columns=dnn_feature_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation_fn=dnn_activation_fn,
        dnn_dropout=dnn_dropout,
        gradient_clip_norm=gradient_clip_norm,
        head=head,
        config=config,
        feature_engineering_fn=feature_engineering_fn,
        default_prediction_key=head_lib.PredictionKey.CLASSES,
        enable_centered_bias=enable_centered_bias)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=True):
    """Returns predicted classes for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted classes (or an iterable of predicted classes if
      as_iterable is True).
    """
    predictions = self.predict_proba(
        x=x, input_fn=input_fn, batch_size=batch_size, as_iterable=as_iterable)
    if as_iterable:
      return (np.argmax(p, axis=0) for p in predictions)
    else:
      return np.argmax(predictions, axis=1)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_proba(
      self, x=None, input_fn=None, batch_size=None, as_iterable=True):
    """Returns prediction probabilities for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted probabilities (or an iterable of predicted
      probabilities if as_iterable is True).
    """
    return super(DNNLinearCombinedClassifier, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size, as_iterable=as_iterable)

  def _get_predict_ops(self, features):
    """See base class."""
    return super(DNNLinearCombinedClassifier, self)._get_predict_ops(features)[
        head_lib.PredictionKey.PROBABILITIES]


class DNNLinearCombinedRegressor(_DNNLinearCombinedBaseEstimator):
  """A regressor for TensorFlow Linear and DNN joined training models.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)
  education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                   combiner="sum")
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")

  estimator = DNNLinearCombinedRegressor(
      # common settings
      weight_column_name=weight_column_name,
      # wide settings
      linear_feature_columns=[education_x_occupation],
      linear_optimizer=tf.train.FtrlOptimizer(...),
      # deep settings
      dnn_feature_columns=[education_emb, occupation_emb],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.train.ProximalAdagradOptimizer(...))

  # To apply L1 and L2 regularization, you can set optimizers as follows:
  tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001,
      l2_regularization_strength=0.001)
  # It is same for FtrlOptimizer.

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  estimator.train(input_fn_train)
  estimator.evaluate(input_fn_eval)
  estimator.predict(x)
  ```

  Input of `fit`, `train`, and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
      if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
      for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `WeightedSparseColumn`, two features: the first with
        `key` the id column name, the second with `key` the weight column name.
        Both features' `value` must be a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  """

  def __init__(self,  # _joint_linear_weights pylint: disable=invalid-name
               model_dir=None,
               weight_column_name=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               _joint_linear_weights=False,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=False,
               label_dimension=1,
               config=None,
               feature_engineering_fn=None):
    """Initializes a DNNLinearCombinedRegressor instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
      _joint_linear_weights: If True a single (possibly partitioned) variable
        will be used to store the linear model weights. It's faster, but
        requires that all columns are sparse and have the 'sum' combiner.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set must be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. If `None`, will use an Adagrad optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If None,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out
        a given coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      label_dimension: TODO(zakaria): dimension of the label for multilabels.
      config: RunConfig object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
    head = head_lib._regression_head(  # pylint: disable=protected-access
        weight_column_name=weight_column_name,
        label_dimension=label_dimension,
        enable_centered_bias=enable_centered_bias)
    super(DNNLinearCombinedRegressor, self).__init__(
        model_dir=model_dir,
        linear_feature_columns=linear_feature_columns,
        linear_optimizer=linear_optimizer,
        _joint_linear_weights=_joint_linear_weights,
        dnn_feature_columns=dnn_feature_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation_fn=dnn_activation_fn,
        dnn_dropout=dnn_dropout,
        gradient_clip_norm=gradient_clip_norm,
        head=head,
        config=config,
        feature_engineering_fn=feature_engineering_fn,
        default_prediction_key=head_lib.PredictionKey.SCORES,
        enable_centered_bias=enable_centered_bias)

  def _get_predict_ops(self, features):
    """See base class."""
    return super(DNNLinearCombinedRegressor, self)._get_predict_ops(features)[
        head_lib.PredictionKey.SCORES]


