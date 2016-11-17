### `tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)` {#reduce_mean}

Computes the mean of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

For example:

```python
# 'x' is [[1., 1.]
#         [2., 2.]]
tf.reduce_mean(x) ==> 1.5
tf.reduce_mean(x, 0) ==> [1.5, 1.5]
tf.reduce_mean(x, 1) ==> [1.,  2.]
```

##### Args:


*  <b>`input_tensor`</b>: The tensor to reduce. Should have numeric type.
*  <b>`axis`</b>: The dimensions to reduce. If `None` (the default),
    reduces all dimensions.
*  <b>`keep_dims`</b>: If true, retains reduced dimensions with length 1.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`reduction_indices`</b>: The old (deprecated) name for axis.

##### Returns:

  The reduced tensor.

@compatibility(numpy)
Equivalent to np.mean
@end_compatibility

