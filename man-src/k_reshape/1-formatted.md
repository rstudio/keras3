Gives a new shape to a tensor without changing its data.

@returns
    The reshaped tensor.

@param x
Input tensor.

@param new_shape
The new shape should be compatible with the original shape.
One shape dimension can be -1 in which case the value is
inferred from the length of the array and remaining dimensions.

@export
@family numpy ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#reshape-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/reshape>
