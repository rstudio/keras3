Global max pooling operation for 2D data.

@description

# Input Shape
- If `data_format='channels_last'`:
    4D tensor with shape:
    `(batch_size, height, width, channels)`
- If `data_format='channels_first'`:
    4D tensor with shape:
    `(batch_size, channels, height, width)`

# Output Shape
- If `keepdims=FALSE`:
    2D tensor with shape `(batch_size, channels)`.
- If `keepdims=TRUE`:
    - If `data_format="channels_last"`:
        4D tensor with shape `(batch_size, 1, 1, channels)`
    - If `data_format="channels_first"`:
        4D tensor with shape `(batch_size, channels, 1, 1)`

# Examples

```r
x <- random_uniform(c(2, 4, 5, 3))
y <- x |> layer_global_max_pooling_2d()
y$shape
```

```
## TensorShape([2, 3])
```

@param data_format string, either `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs. `"channels_last"`
    corresponds to inputs with shape `(batch, height, width, channels)`
    while `"channels_first"` corresponds to inputs with shape
    `(batch, features, height, weight)`. It defaults to the
    `image_data_format` value found in your Keras config file at
    `~/.keras/keras.json`. If you never set it, then it will be
    `"channels_last"`.
@param keepdims A boolean, whether to keep the temporal dimension or not.
    If `keepdims` is `FALSE` (default), the rank of the tensor is
    reduced for spatial dimensions. If `keepdims` is `TRUE`, the
    spatial dimension are retained with length 1.
    The behavior is the same as for `tf$reduce_mean()` or `k_mean()`.
@param object Object to compose the layer with. A tensor, array, or sequential model.
@param ... Passed on to the Python callable

@export
@family pooling layers
@seealso
+ <https:/keras.io/api/layers/pooling_layers/global_max_pooling2d#globalmaxpooling2d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPooling2D>
