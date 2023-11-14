Unit normalization layer.

@description
Normalize a batch of inputs so that each input in the batch has a L2 norm
equal to 1 (across the axes specified in `axis`).

# Examples
```python
data = np.arange(6).reshape(2, 3)
normalized_data = keras.layers.UnitNormalization()(data)
print(np.sum(normalized_data[0, :] ** 2)
# 1.0
```

@param axis
Integer or list/tuple. The axis or axes to normalize across.
Typically, this is the features axis or axes. The left-out axes are
typically the batch axis or axes. `-1` is the last dimension
in the input. Defaults to `-1`.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family unit normalization layers
@family normalization layers
@family layers
@seealso
+ <https:/keras.io/api/layers/normalization_layers/unit_normalization#unitnormalization-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UnitNormalization>
