Sum of a tensor over the given axes.

@returns
    Output tensor containing the sum.

@param x Input tensor.
@param axis Axis or axes along which the sum is computed. The default is to
    compute the sum of the flattened tensor.
@param keepdims If this is set to `True`, the axes which are reduced are left
    in the result as dimensions with size one.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sum>
