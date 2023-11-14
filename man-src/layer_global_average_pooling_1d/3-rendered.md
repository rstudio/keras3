Global average pooling operation for temporal data.

@description

# Call Arguments
- `inputs`: A 3D tensor.
- `mask`: Binary tensor of shape `(batch_size, steps)` indicating whether
    a given step should be masked (excluded from the average).

# Input Shape
- If `data_format='channels_last'`:
    3D tensor with shape:
    `(batch_size, steps, features)`
- If `data_format='channels_first'`:
    3D tensor with shape:
    `(batch_size, features, steps)`

# Output Shape
- If `keepdims=FALSE`:
    2D tensor with shape `(batch_size, features)`.
- If `keepdims=TRUE`:
    - If `data_format="channels_last"`:
        3D tensor with shape `(batch_size, 1, features)`
    - If `data_format="channels_first"`:
        3D tensor with shape `(batch_size, features, 1)`

# Examples

```r
x <- random_uniform(c(2, 3, 4))
y <- x |> layer_global_average_pooling_1d()
shape(y)
```

```
## (2, 4)
```

@param data_format
string, either `"channels_last"` or `"channels_first"`.
The ordering of the dimensions in the inputs. `"channels_last"`
corresponds to inputs with shape `(batch, steps, features)`
while `"channels_first"` corresponds to inputs with shape
`(batch, features, steps)`. It defaults to the `image_data_format`
value found in your Keras config file at `~/.keras/keras.json`.
If you never set it, then it will be `"channels_last"`.

@param keepdims
A boolean, whether to keep the temporal dimension or not.
If `keepdims` is `FALSE` (default), the rank of the tensor is
reduced for spatial dimensions. If `keepdims` is `TRUE`, the
temporal dimension are retained with length 1.
The behavior is the same as for `tf$reduce_mean()` or `k_mean()`.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family global pooling layers
@family pooling layers
@family layers
@seealso
+ <https:/keras.io/api/layers/pooling_layers/global_average_pooling1d#globalaveragepooling1d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D>
