Upsampling layer for 2D inputs.

@description
The implementation uses interpolative resizing, given the resize method
(specified by the `interpolation` argument). Use `interpolation=nearest`
to repeat the rows and columns of the data.

# Examples

```r
input_shape <- c(2, 2, 1, 3)
x <- k_reshape(seq_len(prod(input_shape)), input_shape)
print(x)
```

```
## tf.Tensor(
## [[[[ 1  2  3]]
##
##   [[ 4  5  6]]]
##
##
##  [[[ 7  8  9]]
##
##   [[10 11 12]]]], shape=(2, 2, 1, 3), dtype=int64)
```

```r
y <- layer_upsampling_2d(x, size = c(1, 2))
print(y)
```

```
## tf.Tensor(
## [[[[ 1  2  3]
##    [ 1  2  3]]
##
##   [[ 4  5  6]
##    [ 4  5  6]]]
##
##
##  [[[ 7  8  9]
##    [ 7  8  9]]
##
##   [[10 11 12]
##    [10 11 12]]]], shape=(2, 2, 2, 3), dtype=int64)
```

# Input Shape
4D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch_size, rows, cols, channels)`
- If `data_format` is `"channels_first"`:
    `(batch_size, channels, rows, cols)`

# Output Shape
4D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch_size, upsampled_rows, upsampled_cols, channels)`
- If `data_format` is `"channels_first"`:
    `(batch_size, channels, upsampled_rows, upsampled_cols)`

@param size
Int, or list of 2 integers.
The upsampling factors for rows and columns.

@param data_format
A string,
one of `"channels_last"` (default) or `"channels_first"`.
The ordering of the dimensions in the inputs.
`"channels_last"` corresponds to inputs with shape
`(batch_size, height, width, channels)` while `"channels_first"`
corresponds to inputs with shape
`(batch_size, channels, height, width)`.
When unspecified, uses
`image_data_format` value found in your Keras config file at
`~/.keras/keras.json` (if exists) else `"channels_last"`.
Defaults to `"channels_last"`.

@param interpolation
A string, one of `"bicubic"`, `"bilinear"`, `"lanczos3"`,
`"lanczos5"`, `"nearest"`.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family reshaping layers
@family layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/up_sampling2d#upsampling2d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling2D>

