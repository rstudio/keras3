Softmax activation layer.

@description
Formula:
``` python
exp_x = exp(x - max(x))
f(x) = exp_x / sum(exp_x)
```

# Examples
```python
>>>softmax_layer = keras.layers.activations.Softmax()
# >>>input = np.array([1.0, 2.0, 1.0])
# >>>result = softmax_layer(input)
# [0.21194157, 0.5761169, 0.21194157]
```

# Call Arguments
- `inputs`: The inputs (logits) to the softmax layer.
- `mask`: A boolean mask of the same shape as `inputs`. The mask
    specifies 1 to keep and 0 to mask. Defaults to `None`.

@returns
    Softmaxed output with the same shape as `inputs`.

@param axis Integer, or list of Integers, axis along which the softmax
    normalization is applied.
@param ... Base layer keyword arguments, such as `name` and `dtype`.
@param object Object to compose the layer with. A tensor, array, or sequential model.

@export
@family activations layers
@seealso
+ <https:/keras.io/api/layers/activation_layers/softmax#softmax-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Softmax>
