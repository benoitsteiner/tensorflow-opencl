A regressor for TensorFlow DNN models.

Example:

```python
education = sparse_column_with_hash_bucket(column_name="education",
                                           hash_bucket_size=1000)
occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)

education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                 combiner="sum")
occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                 combiner="sum")

estimator = DNNRegressor(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256])

# Or estimator using the ProximalAdagradOptimizer optimizer with
# regularization.
estimator = DNNRegressor(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Input builders
def input_fn_train: # returns x, Y
  pass
estimator.fit(input_fn=input_fn_train)

def input_fn_eval: # returns x, Y
  pass
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x)
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
- - -

#### `tf.contrib.learn.DNNRegressor.__init__(hidden_units, feature_columns, model_dir=None, weight_column_name=None, optimizer=None, activation_fn=relu, dropout=None, gradient_clip_norm=None, enable_centered_bias=False, config=None, feature_engineering_fn=None)` {#DNNRegressor.__init__}

Initializes a `DNNRegressor` instance.

##### Args:


*  <b>`hidden_units`</b>: List of hidden units per layer. All layers are fully
    connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
    has 32.
*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator to
    continue training a previously saved model.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
*  <b>`optimizer`</b>: An instance of `tf.Optimizer` used to train the model. If
    `None`, will use an Adagrad optimizer.
*  <b>`activation_fn`</b>: Activation function applied to each layer. If `None`, will
    use `tf.nn.relu`.
*  <b>`dropout`</b>: When not `None`, the probability we will drop out a given
    coordinate.
*  <b>`gradient_clip_norm`</b>: A `float` > 0. If provided, gradients are clipped
    to their global norm with this clipping ratio. See
    `tf.clip_by_global_norm` for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.

##### Returns:

  A `DNNRegressor` estimator.


- - -

#### `tf.contrib.learn.DNNRegressor.__repr__()` {#DNNRegressor.__repr__}




- - -

#### `tf.contrib.learn.DNNRegressor.bias_` {#DNNRegressor.bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNRegressor.config` {#DNNRegressor.config}




- - -

#### `tf.contrib.learn.DNNRegressor.dnn_bias_` {#DNNRegressor.dnn_bias_}

Returns bias of deep neural network part. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNRegressor.dnn_weights_` {#DNNRegressor.dnn_weights_}

Returns weights of deep neural network part. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNRegressor.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#DNNRegressor.evaluate}

See `Evaluable`.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`.


- - -

#### `tf.contrib.learn.DNNRegressor.export(*args, **kwargs)` {#DNNRegressor.export}

Exports inference graph into given dir. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn (and in most cases, input_feature_key) will become required args, and use_deprecated_input_fn will default to False and be removed altogether.

    Args:
      export_dir: A string containing a directory to write the exported graph
        and checkpoints.
      input_fn: If `use_deprecated_input_fn` is true, then a function that given
        `Tensor` of `Example` strings, parses it into features that are then
        passed to the model. Otherwise, a function that takes no argument and
        returns a tuple of (features, labels), where features is a dict of
        string key to `Tensor` and labels is a `Tensor` that's currently not
        used (and so can be `None`).
      input_feature_key: Only used if `use_deprecated_input_fn` is false. String
        key into the features dict returned by `input_fn` that corresponds to a
        the raw `Example` strings `Tensor` that the exported model will take as
        input. Can only be `None` if you're using a custom `signature_fn` that
        does not use the first arg (examples).
      use_deprecated_input_fn: Determines the signature format of `input_fn`.
      signature_fn: Function that returns a default signature and a named
        signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
        for features and `Tensor` or `dict` of `Tensor`s for predictions.
      prediction_key: The key for a tensor in the `predictions` dict (output
        from the `model_fn`) to use as the `predictions` input to the
        `signature_fn`. Optional. If `None`, predictions will pass to
        `signature_fn` without filtering.
      default_batch_size: Default batch size of the `Example` placeholder.
      exports_to_keep: Number of exports to keep.

    Returns:
      The string path to the exported directory. NB: this functionality was
      added ca. 2016/09/25; clients that depend on the return value may need
      to handle the case where this function returns None because subclasses
      are not returning a value.


- - -

#### `tf.contrib.learn.DNNRegressor.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#DNNRegressor.fit}

See `Trainable`.

##### Raises:


*  <b>`ValueError`</b>: If `x` or `y` are not `None` while `input_fn` is not `None`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.


- - -

#### `tf.contrib.learn.DNNRegressor.get_params(deep=True)` {#DNNRegressor.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional

    If `True`, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.DNNRegressor.get_variable_names()` {#DNNRegressor.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.DNNRegressor.get_variable_value(name)` {#DNNRegressor.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.DNNRegressor.linear_bias_` {#DNNRegressor.linear_bias_}

Returns bias of the linear part. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNRegressor.linear_weights_` {#DNNRegressor.linear_weights_}

Returns weights per feature of the linear part. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.DNNRegressor.model_dir` {#DNNRegressor.model_dir}




- - -

#### `tf.contrib.learn.DNNRegressor.partial_fit(x=None, y=None, input_fn=None, steps=1, batch_size=None, monitors=None)` {#DNNRegressor.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`x`</b>: Matrix of shape [n_samples, n_features...]. Can be iterator that
     returns arrays of features. The training input samples for fitting the
     model. If set, `input_fn` must be `None`.
*  <b>`y`</b>: Vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
     iterator that returns array of labels. The training label values
     (class labels in classification, real numbers in regression). If set,
     `input_fn` must be `None`.
*  <b>`input_fn`</b>: Input function. If set, `x`, `y`, and `batch_size` must be
    `None`.
*  <b>`steps`</b>: Number of steps for which to train model. If `None`, train forever.
*  <b>`batch_size`</b>: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.

##### Returns:

  `self`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: If at least one of `x` and `y` is provided, and `input_fn` is
      provided.


- - -

#### `tf.contrib.learn.DNNRegressor.predict(*args, **kwargs)` {#DNNRegressor.predict}

Returns predictions for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

    Args:
      x: Matrix of shape [n_samples, n_features...]. Can be iterator that
         returns arrays of features. The training input samples for fitting the
         model. If set, `input_fn` must be `None`.
      input_fn: Input function. If set, `x` and 'batch_size' must be `None`.
      batch_size: Override default batch size. If set, 'input_fn' must be
        'None'.
      outputs: list of `str`, name of the output to predict.
        If `None`, returns all.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      A numpy array of predicted classes or regression values if the
      constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict`
      of numpy arrays if `model_fn` returns a `dict`. Returns an iterable of
      predictions if as_iterable is True.

    Raises:
      ValueError: If x and input_fn are both provided or both `None`.


- - -

#### `tf.contrib.learn.DNNRegressor.set_params(**params)` {#DNNRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Args:


*  <b>`**params`</b>: Parameters.

##### Returns:

  self

##### Raises:


*  <b>`ValueError`</b>: If params contain invalid names.


- - -

#### `tf.contrib.learn.DNNRegressor.weights_` {#DNNRegressor.weights_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


