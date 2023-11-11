Compute the arithmetic mean along the specified axes.

@returns
    Output tensor containing the mean values.

@param x Input tensor.
@param axis Axis or axes along which the means are computed. The default
    is to compute the mean of the flattened tensor.
@param keepdims If this is set to `TRUE`, the axes which are reduced are left
    in the result as dimensions with size one.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#mean-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/mean>
