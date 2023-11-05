Return the product of tensor elements over a given axis.

# Returns
    Product of elements of `x` over the given axis or axes.

@param x Input tensor.
@param axis Axis or axes along which a product is performed. The default,
    `axis=None`, will compute the product of all elements
    in the input tensor.
@param keepdims If this is set to `True`, the axes which are reduce
    are left in the result as dimensions with size one.
@param dtype Data type of the returned tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/prod>
