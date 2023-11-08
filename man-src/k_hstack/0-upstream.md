keras.ops.hstack
__signature__
(xs)
__doc__
Stack tensors in sequence horizontally (column wise).

This is equivalent to concatenation along the first axis for 1-D tensors,
and along the second axis for all other tensors.

Args:
    xs: Sequence of tensors.

Returns:
    The tensor formed by stacking the given tensors.
