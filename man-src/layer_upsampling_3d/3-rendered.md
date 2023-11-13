Upsampling layer for 3D inputs.

@description
Repeats the 1st, 2nd and 3rd dimensions
of the data by `size[0]`, `size[1]` and `size[2]` respectively.

# Examples

```r
input_shape <- c(2, 1, 2, 1, 3)
x <- array(1, dim = input_shape)
y <- layer_upsampling_3d(x, size = c(2, 2, 2))
y$shape
```

```
## TensorShape([2, 2, 4, 2, 3])
```

# Input Shape
5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch_size, dim1, dim2, dim3, channels)`
- If `data_format` is `"channels_first"`:
    `(batch_size, channels, dim1, dim2, dim3)`

# Output Shape
5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3,
    channels)`
- If `data_format` is `"channels_first"`:
    `(batch_size, channels, upsampled_dim1, upsampled_dim2,
    upsampled_dim3)`

@param size Int, or list of 3 integers.
    The upsampling factors for dim1, dim2 and dim3.
@param data_format A string,
    one of `"channels_last"` (default) or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `"channels_first"` corresponds to inputs with shape
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    When unspecified, uses
    `image_data_format` value found in your Keras config file at
     `~/.keras/keras.json` (if exists) else `"channels_last"`.
    Defaults to `"channels_last"`.
@param object Object to compose the layer with. A tensor, array, or sequential model.
@param ... Passed on to the Python callable

@export
@family reshaping layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/up_sampling3d#upsampling3d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling3D>

