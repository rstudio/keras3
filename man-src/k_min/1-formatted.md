Return the minimum of a tensor or minimum along an axis.

@returns
    Minimum of `x`.

@param x
Input tensor.

@param axis
Axis or axes along which to operate. By default, flattened input
is used.

@param keepdims
If this is set to `True`, the axes which are reduced are left
in the result as dimensions with size one. Defaults to`False`.

@param initial
The maximum value of an output element. Defaults to`None`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#min-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/min>
