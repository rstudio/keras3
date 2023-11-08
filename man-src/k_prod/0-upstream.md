keras.ops.prod
__signature__
(
  x,
  axis=None,
  keepdims=False,
  dtype=None
)
__doc__
Return the product of tensor elements over a given axis.

Args:
    x: Input tensor.
    axis: Axis or axes along which a product is performed. The default,
        `axis=None`, will compute the product of all elements
        in the input tensor.
    keepdims: If this is set to `True`, the axes which are reduce
        are left in the result as dimensions with size one.
    dtype: Data type of the returned tensor.

Returns:
    Product of elements of `x` over the given axis or axes.
