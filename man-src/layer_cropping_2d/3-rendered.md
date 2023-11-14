Cropping layer for 2D input (e.g. picture).

@description
It crops along spatial dimensions, i.e. height and width.

# Examples

```r
input_shape <- c(2, 28, 28, 3)
x <- k_arange(prod(input_shape), dtype ='int32') |> k_reshape(input_shape)
y <- x |> layer_cropping_2d(cropping=list(c(2, 2), c(4, 4)))
y$shape
```

```
## TensorShape([2, 24, 20, 3])
```

# Input Shape
4D tensor with shape:
- If `data_format` is `"channels_last"`:
  `(batch_size, height, width, channels)`
- If `data_format` is `"channels_first"`:
  `(batch_size, channels, height, width)`

# Output Shape
4D tensor with shape:
- If `data_format` is `"channels_last"`:
  `(batch_size, cropped_height, cropped_width, channels)`
- If `data_format` is `"channels_first"`:
  `(batch_size, channels, cropped_height, cropped_width)`

@param cropping
Int, or list of 2 ints, or list of 2 lists of 2 ints.
- If int: the same symmetric cropping is applied to height and
  width.
- If list of 2 ints: interpreted as two different symmetric
  cropping values for height and width:
  `(symmetric_height_crop, symmetric_width_crop)`.
- If list of 2 lists of 2 ints: interpreted as
  `((top_crop, bottom_crop), (left_crop, right_crop))`.

@param data_format
A string, one of `"channels_last"` (default) or
`"channels_first"`. The ordering of the dimensions in the inputs.
`"channels_last"` corresponds to inputs with shape
`(batch_size, height, width, channels)` while `"channels_first"`
corresponds to inputs with shape
`(batch_size, channels, height, width)`.
When unspecified, uses `image_data_format` value found in your Keras
config file at `~/.keras/keras.json` (if exists). Defaults to
`"channels_last"`.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family cropping2d reshaping layers
@family reshaping layers
@family layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/cropping2d#cropping2d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D>
