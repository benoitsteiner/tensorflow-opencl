### `tf.real(input, name=None)` {#real}

Returns the real part of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float32` or `float64` that is the real part of each element in `input`.
All elements in `input` must be complex numbers of the form \\(a + bj\\),
where *a* is the real part returned by this operation and *b* is the
imaginary part.

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(input) ==> [-2.25, 3.25]
```

If `input` is already real, it is returned unchanged.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must have numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32` or `float64`.

