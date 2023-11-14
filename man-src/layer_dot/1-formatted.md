Computes element-wise dot product of two tensors.

@description
It takes a list of inputs of size 2, and the axes
corresponding to each input along with the dot product
is to be performed.

Let's say `x` and `y` are the two input tensors with shapes
`(2, 3, 5)` and `(2, 10, 3)`. The batch dimension should be
of same size for both the inputs, and `axes` should correspond
to the dimensions that have the same size in the corresponding
inputs. e.g. with `axes=(1, 2)`, the dot product of `x`, and `y`
will result in a tensor with shape `(2, 5, 10)`

# Examples
```python
x = np.arange(10).reshape(1, 5, 2)
y = np.arange(10, 20).reshape(1, 2, 5)
keras.layers.Dot(axes=(1, 2))([x, y])
```

Usage in a Keras model:

```python
x1 = keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
x2 = keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
y = keras.layers.Dot(axes=1)([x1, x2])
```

@returns
    A tensor, the dot product of the samples from the inputs.

@param axes
Integer or tuple of integers, axis or axes along which to
take the dot product. If a tuple, should be two integers
corresponding to the desired axis from the first input and the
desired axis from the second input, respectively. Note that the
size of the two selected axes must match.

@param normalize
Whether to L2-normalize samples along the dot product axis
before taking the dot product. If set to `True`, then
the output of the dot product is the cosine proximity
between the two samples.

@param ...
Standard layer keyword arguments.

@param inputs
layers to combine

@export
@family merging layers
@family layers
@seealso
+ <https:/keras.io/api/layers/merging_layers/dot#dot-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot>
