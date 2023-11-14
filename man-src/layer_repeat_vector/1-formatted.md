Repeats the input n times.

@description

# Examples
```python
x = keras.Input(shape=(32,))
y = keras.layers.RepeatVector(3)(x)
y.shape
# (None, 3, 32)
```

# Input Shape
2D tensor with shape `(batch_size, features)`.

# Output Shape
    3D tensor with shape `(batch_size, n, features)`.

@param n
Integer, repetition factor.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family vector repeat reshaping layers
@family repeat reshaping layers
@family reshaping layers
@family layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/repeat_vector#repeatvector-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RepeatVector>
