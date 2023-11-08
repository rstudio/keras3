keras.ops.trace
__signature__
(
  x,
  offset=0,
  axis1=0,
  axis2=1
)
__doc__
Return the sum along diagonals of the tensor.

If `x` is 2-D, the sum along its diagonal with the given offset is
returned, i.e., the sum of elements `x[i, i+offset]` for all `i`.

If a has more than two dimensions, then the axes specified by `axis1`
and `axis2` are used to determine the 2-D sub-arrays whose traces are
returned.

The shape of the resulting tensor is the same as that of `x` with `axis1`
and `axis2` removed.

Args:
    x: Input tensor.
    offset: Offset of the diagonal from the main diagonal. Can be
        both positive and negative. Defaults to `0`.
    axis1: Axis to be used as the first axis of the 2-D sub-arrays.
        Defaults to `0`.(first axis).
    axis2: Axis to be used as the second axis of the 2-D sub-arrays.
        Defaults to `1` (second axis).

Returns:
    If `x` is 2-D, the sum of the diagonal is returned. If `x` has
    larger dimensions, then a tensor of sums along diagonals is
    returned.
