### `tf.unsorted_segment_max(data, segment_ids, num_segments, name=None)` {#unsorted_segment_max}

Computes the Max along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

This operator is similar to the [unsorted segment sum operator](../../api_docs/python/math_ops.md#UnsortedSegmentSum).
Instead of computing the sum over segments, it computes the maximum
such that:

\\(output_i = \max_j data_j\\) where max is over `j` such
that `segment_ids[j] == i`.

If the maximum is empty for a given segment ID `i`, it outputs the smallest possible value for specific numeric type,
 `output[i] = numeric_limits<T>::min()`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/UnsortedSegmentSum.png" alt>
</div>

##### Args:


*  <b>`data`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`segment_ids`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A 1-D tensor whose rank is equal to the rank of `data`'s
    first dimension.
*  <b>`num_segments`</b>: A `Tensor` of type `int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `data`.
  Has same shape as data, except for dimension 0 which
  has size `num_segments`.

