Return whether a tensor is finite, element-wise.

Real values are finite when they are not NaN, not positive infinity, and
not negative infinity. Complex values are finite when both their real
and imaginary parts are finite.

Args:
    x: Input tensor.

Returns:
    Output boolean tensor.
