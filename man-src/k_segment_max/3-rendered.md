Computes the max of segments in a tensor.

@description

# Examples

```r
data <- k_convert_to_tensor(c(1, 2, 10, 20, 100, 200))
segment_ids <- k_array(c(1, 1, 2, 2, 3, 3), "int32")
num_segments <- 3
k_segment_max(data, segment_ids, num_segments)
```

```
## tf.Tensor([  2.  20. 200.], shape=(3), dtype=float32)
```

```r
# array([2, 20, 200], dtype=int32)
```

@returns
A tensor containing the max of segments, where each element
represents the max of the corresponding segment in `data`.

@param data
Input tensor.

@param segment_ids
A 1-D tensor containing segment indices for each
element in `data`.

@param num_segments
An integer representing the total number of
segments. If not specified, it is inferred from the maximum
value in `segment_ids`.

@param sorted
A boolean indicating whether `segment_ids` is sorted.
Defaults to`FALSE`.

@export
@family math ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#segmentmax-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/segment_max>
