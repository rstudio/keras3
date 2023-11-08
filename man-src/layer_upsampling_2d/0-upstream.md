keras.layers.UpSampling2D
__signature__
(
  size=(2,
  2),
  data_format=None,
  interpolation='nearest',
  **kwargs
)
__doc__
Upsampling layer for 2D inputs.

The implementation uses interpolative resizing, given the resize method
(specified by the `interpolation` argument). Use `interpolation=nearest`
to repeat the rows and columns of the data.

Examples:

>>> input_shape = (2, 2, 1, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> print(x)
[[[[ 0  1  2]]
  [[ 3  4  5]]]
 [[[ 6  7  8]]
  [[ 9 10 11]]]]
>>> y = keras.layers.UpSampling2D(size=(1, 2))(x)
>>> print(y)
[[[[ 0  1  2]
   [ 0  1  2]]
  [[ 3  4  5]
   [ 3  4  5]]]
 [[[ 6  7  8]
   [ 6  7  8]]
  [[ 9 10 11]
   [ 9 10 11]]]]

Args:
    size: Int, or tuple of 2 integers.
        The upsampling factors for rows and columns.
    data_format: A string,
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
    interpolation: A string, one of `"bicubic"`, `"bilinear"`, `"lanczos3"`,
        `"lanczos5"`, `"nearest"`.

Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`

Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_rows, upsampled_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_rows, upsampled_cols)`
