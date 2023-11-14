Cropping layer for 3D data (e.g. spatial or spatio-temporal).

@description

# Examples

```r
input_shape <- c(2, 28, 28, 10, 3)
x <- input_shape %>% { k_reshape(seq(prod(.)), .) }
y <- x |> layer_cropping_3d(cropping = c(2, 4, 2))
y$shape
```

```
## TensorShape([2, 24, 20, 6, 3])
```

# Input Shape
5D tensor with shape:
- If `data_format` is `"channels_last"`:
  `(batch_size, first_axis_to_crop, second_axis_to_crop,
  third_axis_to_crop, channels)`
- If `data_format` is `"channels_first"`:
  `(batch_size, channels, first_axis_to_crop, second_axis_to_crop,
  third_axis_to_crop)`

# Output Shape
5D tensor with shape:
- If `data_format` is `"channels_last"`:
  `(batch_size, first_cropped_axis, second_cropped_axis,
  third_cropped_axis, channels)`
- If `data_format` is `"channels_first"`:
  `(batch_size, channels, first_cropped_axis, second_cropped_axis,
  third_cropped_axis)`

@param cropping
Int, or list of 3 ints, or list of 3 lists of 2 ints.
- If int: the same symmetric cropping is applied to depth, height,
  and width.
- If list of 3 ints: interpreted as three different symmetric
  cropping values for depth, height, and width:
  `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
- If list of 3 lists of 2 ints: interpreted as
  `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
  right_dim2_crop), (left_dim3_crop, right_dim3_crop))`.

@param data_format
A string, one of `"channels_last"` (default) or
`"channels_first"`. The ordering of the dimensions in the inputs.
`"channels_last"` corresponds to inputs with shape
`(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
while `"channels_first"` corresponds to inputs with shape
`(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
When unspecified, uses `image_data_format` value found in your Keras
config file at `~/.keras/keras.json` (if exists). Defaults to
`"channels_last"`.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family reshaping layers
@family layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/cropping3d#cropping3d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping3D>
