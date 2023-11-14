Clip (limit) the values in a tensor.

@description
Given an interval, values outside the interval are clipped to the
interval edges. For example, if an interval of `[0, 1]` is specified,
values smaller than 0 become 0, and values larger than 1 become 1.

@returns
    The clipped tensor.

@param x
Input tensor.

@param x_min
Minimum value.

@param x_max
Maximum value.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#clip-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/clip>
