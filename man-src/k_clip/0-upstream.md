keras.ops.clip
__signature__
(x, x_min, x_max)
__doc__
Clip (limit) the values in a tensor.

Given an interval, values outside the interval are clipped to the
interval edges. For example, if an interval of `[0, 1]` is specified,
values smaller than 0 become 0, and values larger than 1 become 1.

Args:
    x: Input tensor.
    x_min: Minimum value.
    x_max: Maximum value.
Returns:
    The clipped tensor.
