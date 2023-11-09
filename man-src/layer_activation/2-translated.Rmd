Applies an activation function to an output.

@description

# Examples
```python
layer = keras.layers.Activation('relu')
layer([-3.0, -1.0, 0.0, 2.0])
# [0.0, 0.0, 0.0, 2.0]
layer = keras.layers.Activation(keras.activations.relu)
layer([-3.0, -1.0, 0.0, 2.0])
# [0.0, 0.0, 0.0, 2.0]
```

@param activation Activation function. It could be a callable, or the name of
    an activation from the `keras.activations` namespace.
@param ... Base layer keyword arguments, such as `name` and `dtype`.
@param object Object to compose the layer with. A tensor, array, or sequential model.

@export
@family activations layers
@seealso
+ <https:/keras.io/api/layers/core_layers/activation#activation-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation>
