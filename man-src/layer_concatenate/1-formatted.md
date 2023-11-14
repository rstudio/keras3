Concatenates a list of inputs.

@description
It takes as input a list of tensors, all of the same shape except
for the concatenation axis, and returns a single tensor that is the
concatenation of all inputs.

# Examples
```python
x = np.arange(20).reshape(2, 2, 5)
y = np.arange(20, 30).reshape(2, 1, 5)
keras.layers.Concatenate(axis=1)([x, y])
```

Usage in a Keras model:

```python
x1 = keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
x2 = keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
y = keras.layers.Concatenate()([x1, x2])
```

@returns
    A tensor, the concatenation of the inputs alongside axis `axis`.

@param axis
Axis along which to concatenate.

@param ...
Standard layer keyword arguments.

@param inputs
layers to combine

@export
@family merging layers
@seealso
+ <https:/keras.io/api/layers/merging_layers/concatenate#concatenate-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate>
