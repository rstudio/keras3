Global average pooling operation for 3D data.

@description

# Input Shape
- If `data_format='channels_last'`:
    5D tensor with shape:
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`:
    5D tensor with shape:
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

# Output Shape
- If `keepdims=FALSE`:
    2D tensor with shape `(batch_size, channels)`.
- If `keepdims=TRUE`:
    - If `data_format="channels_last"`:
        5D tensor with shape `(batch_size, 1, 1, 1, channels)`
    - If `data_format="channels_first"`:
        5D tensor with shape `(batch_size, channels, 1, 1, 1)`

# Examples

```r
x <- random_uniform(c(2, 4, 5, 4, 3))
y <- x |> layer_global_average_pooling_3d()
y$shape
```

```
## TensorShape([2, 3])
```

@param data_format
string, either `"channels_last"` or `"channels_first"`.
The ordering of the dimensions in the inputs. `"channels_last"`
corresponds to inputs with shape
`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
while `"channels_first"` corresponds to inputs with shape
`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
It defaults to the `image_data_format` value found in your Keras
config file at `~/.keras/keras.json`. If you never set it, then it
will be `"channels_last"`.

@param keepdims
A boolean, whether to keep the temporal dimension or not.
If `keepdims` is `FALSE` (default), the rank of the tensor is
reduced for spatial dimensions. If `keepdims` is `TRUE`, the
spatial dimension are retained with length 1.
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
+ <https:/keras.io/api/layers/pooling_layers/global_average_pooling3d#globalaveragepooling3d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling3D>
