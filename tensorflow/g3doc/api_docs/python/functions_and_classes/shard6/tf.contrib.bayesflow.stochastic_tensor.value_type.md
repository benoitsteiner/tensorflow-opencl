### `tf.contrib.bayesflow.stochastic_tensor.value_type(dist_value_type)` {#value_type}

Creates a value type context for any StochasticTensor created within.

Typical usage:

```
with sg.value_type(sg.MeanValue(stop_gradients=True)):
  st = sg.StochasticTensor(tf.contrib.distributions.Normal, mu=mu,
                           sigma=sigma)
```

In the example above, `st.value()` (or equivalently, `tf.identity(st)`) will
be the mean value of the Normal distribution, i.e., `mu` (possibly
broadcasted to the shape of `sigma`).  Furthermore, because the `MeanValue`
was marked with `stop_gradients=True`, this value will have been wrapped
in a `stop_gradients` call to disable any possible backpropagation.

##### Args:


*  <b>`dist_value_type`</b>: An instance of `MeanValue`, `SampleValue`, or
    any other stochastic value type.

##### Yields:

  A context for `StochasticTensor` objects that controls the
  value created when they are initialized.

##### Raises:


*  <b>`TypeError`</b>: if `dist_value_type` is not an instance of a stochastic value
    type.

