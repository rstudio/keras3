keras.ops.segment_sum
__signature__
(
  data,
  segment_ids,
  num_segments=None,
  sorted=False
)
__doc__
Computes the sum of segments in a tensor.

Args:
    data: Input tensor.
    segment_ids: A 1-D tensor containing segment indices for each
        element in `data`.
    num_segments: An integer representing the total number of
        segments. If not specified, it is inferred from the maximum
        value in `segment_ids`.
    sorted: A boolean indicating whether `segment_ids` is sorted.
        Defaults to`False`.

Returns:
    A tensor containing the sum of segments, where each element
    represents the sum of the corresponding segment in `data`.

Example:

>>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
>>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
>>> num_segments = 3
>>> keras.ops.segment_sum(data, segment_ids,num_segments)
array([3, 30, 300], dtype=int32)
