keras.ops.max
__signature__
(
  x,
  axis=None,
  keepdims=False,
  initial=None
)
__doc__
Return the maximum of a tensor or maximum along an axis.

Args:
    x: Input tensor.
    axis: Axis or axes along which to operate. By default, flattened input
        is used.
    keepdims: If this is set to `True`, the axes which are reduced are left
        in the result as dimensions with size one. Defaults to`False`.
    initial: The minimum value of an output element. Defaults to`None`.

Returns:
    Maximum of `x`.
