Return the maximum of a tensor or maximum along an axis.

# Returns
    Maximum of `x`.

@param x Input tensor.
@param axis Axis or axes along which to operate. By default, flattened input
    is used.
@param keepdims If this is set to `True`, the axes which are reduced are left
    in the result as dimensions with size one. Defaults to`False`.
@param initial The minimum value of an output element. Defaults to`None`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/max>
