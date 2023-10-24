Applies an activation function to an output.

Args:
    activation: Activation function. It could be a callable, or the name of
        an activation from the `keras.activations` namespace.
    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

Example:

>>> layer = keras.layers.Activation('relu')
>>> layer([-3.0, -1.0, 0.0, 2.0])
[0.0, 0.0, 0.0, 2.0]
>>> layer = keras.layers.Activation(keras.activations.relu)
>>> layer([-3.0, -1.0, 0.0, 2.0])
[0.0, 0.0, 0.0, 2.0]
