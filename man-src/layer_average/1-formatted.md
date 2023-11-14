Averages a list of inputs element-wise..

@description
It takes as input a list of tensors, all of the same shape,
and returns a single tensor (also of the same shape).

# Examples
```python
input_shape = (2, 3, 4)
x1 = np.random.rand(*input_shape)
x2 = np.random.rand(*input_shape)
y = keras.layers.Average()([x1, x2])
```

Usage in a Keras model:

```python
input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# equivalent to `y = keras.layers.average([x1, x2])`
y = keras.layers.Average()([x1, x2])
out = keras.layers.Dense(4)(y)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

@param ...
Passed on to the Python callable

@param inputs
layers to combine

@export
@family merging layers
@family layers
@seealso
+ <https:/keras.io/api/layers/merging_layers/average#average-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Average>
