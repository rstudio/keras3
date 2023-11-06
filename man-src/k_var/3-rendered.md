Compute the variance along the specified axes.

@returns
    Output tensor containing the variance.

@param x Input tensor.
@param axis Axis or axes along which the variance is computed. The default
    is to compute the variance of the flattened tensor.
@param keepdims If this is set to `True`, the axes which are reduced are left
    in the result as dimensions with size one.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/var>
