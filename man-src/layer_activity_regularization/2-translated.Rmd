Layer that applies an update to the cost function based input activity.

@description

# Input Shape
Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

# Output Shape
    Same shape as input.

@param l1 L1 regularization factor (positive float).
@param l2 L2 regularization factor (positive float).
@param object Object to compose the layer with. A tensor, array, or sequential model.
@param ... Passed on to the Python callable

@export
@family regularization layers
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ActivityRegularization>
