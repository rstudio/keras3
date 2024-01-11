

#' Cropping layer for 1D input (e.g. temporal sequence).
#'
#' @description
#' It crops along the time dimension (axis 2).
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 3, 2)
#' x <- op_arange(prod(input_shape)) |> op_reshape(input_shape)
#' x
#'
#' y <- x |> layer_cropping_1d(cropping = 1)
#' y
#' ```
#'
#' # Input Shape
#' 3D tensor with shape `(batch_size, axis_to_crop, features)`
#'
#' # Output Shape
#' 3D tensor with shape `(batch_size, cropped_axis, features)`
#'
#' @param cropping
#' Int, or list of int (length 2).
#' - If int: how many units should be trimmed off at the beginning and
#'   end of the cropping dimension (axis 1).
#' - If list of 2 ints: how many units should be trimmed off at the
#'   beginning and end of the cropping dimension
#'   (`(left_crop, right_crop)`).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/cropping1d#cropping1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping1D>
#' @tether keras.layers.Cropping1D
layer_cropping_1d <-
function (object, cropping = list(1L, 1L), ...)
{
    args <- capture_args(list(cropping = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Cropping1D, object, args)
}


#' Cropping layer for 2D input (e.g. picture).
#'
#' @description
#' It crops along spatial dimensions, i.e. height and width.
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 28, 28, 3)
#' x <- op_arange(prod(input_shape), dtype ='int32') |> op_reshape(input_shape)
#' y <- x |> layer_cropping_2d(cropping=list(c(2, 2), c(4, 4)))
#' shape(y)
#' ```
#'
#' # Input Shape
#' 4D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'   `(batch_size, height, width, channels)`
#' - If `data_format` is `"channels_first"`:
#'   `(batch_size, channels, height, width)`
#'
#' # Output Shape
#' 4D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'   `(batch_size, cropped_height, cropped_width, channels)`
#' - If `data_format` is `"channels_first"`:
#'   `(batch_size, channels, cropped_height, cropped_width)`
#'
#' @param cropping
#' Int, or list of 2 ints, or list of 2 lists of 2 ints.
#' - If int: the same symmetric cropping is applied to height and
#'   width.
#' - If list of 2 ints: interpreted as two different symmetric
#'   cropping values for height and width:
#'   `(symmetric_height_crop, symmetric_width_crop)`.
#' - If list of 2 lists of 2 ints: interpreted as
#'   `((top_crop, bottom_crop), (left_crop, right_crop))`.
#'
#' @param data_format
#' A string, one of `"channels_last"` (default) or
#' `"channels_first"`. The ordering of the dimensions in the inputs.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch_size, height, width, channels)` while `"channels_first"`
#' corresponds to inputs with shape
#' `(batch_size, channels, height, width)`.
#' When unspecified, uses `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json` (if exists). Defaults to
#' `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/cropping2d#cropping2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D>
#' @tether keras.layers.Cropping2D
layer_cropping_2d <-
function (object, cropping = list(list(0L, 0L), list(0L, 0L)),
    data_format = NULL, ...)
{
    args <- capture_args(list(cropping = function (x)
    normalize_cropping(x, 2L), input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Cropping2D, object, args)
}


#' Cropping layer for 3D data (e.g. spatial or spatio-temporal).
#'
#' @description
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 28, 28, 10, 3)
#' x <- input_shape %>% { op_reshape(seq(prod(.)), .) }
#' y <- x |> layer_cropping_3d(cropping = c(2, 4, 2))
#' shape(y)
#' ```
#'
#' # Input Shape
#' 5D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'   `(batch_size, first_axis_to_crop, second_axis_to_crop,
#'   third_axis_to_crop, channels)`
#' - If `data_format` is `"channels_first"`:
#'   `(batch_size, channels, first_axis_to_crop, second_axis_to_crop,
#'   third_axis_to_crop)`
#'
#' # Output Shape
#' 5D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'   `(batch_size, first_cropped_axis, second_cropped_axis,
#'   third_cropped_axis, channels)`
#' - If `data_format` is `"channels_first"`:
#'   `(batch_size, channels, first_cropped_axis, second_cropped_axis,
#'   third_cropped_axis)`
#'
#' @param cropping
#' Int, or list of 3 ints, or list of 3 lists of 2 ints.
#' - If int: the same symmetric cropping is applied to depth, height,
#'   and width.
#' - If list of 3 ints: interpreted as three different symmetric
#'   cropping values for depth, height, and width:
#'   `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
#' - If list of 3 lists of 2 ints: interpreted as
#'   `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
#'   right_dim2_crop), (left_dim3_crop, right_dim3_crop))`.
#'
#' @param data_format
#' A string, one of `"channels_last"` (default) or
#' `"channels_first"`. The ordering of the dimensions in the inputs.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
#' When unspecified, uses `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json` (if exists). Defaults to
#' `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/cropping3d#cropping3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping3D>
#' @tether keras.layers.Cropping3D
layer_cropping_3d <-
function (object, cropping = list(list(1L, 1L), list(1L, 1L),
    list(1L, 1L)), data_format = NULL, ...)
{
    args <- capture_args(list(cropping = function (x)
    normalize_cropping(x, 3L), input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Cropping3D, object, args)
}


#' Flattens the input. Does not affect the batch size.
#'
#' @description
#'
#' # Note
#' If inputs are shaped `(batch)` without a feature axis, then
#' flattening adds an extra channel dimension and output shape is `(batch, 1)`.
#'
#' # Examples
#' ```{r}
#' x <- layer_input(shape=c(10, 64))
#' y <- x |> layer_flatten()
#' shape(y)
#' ```
#'
#' @param data_format
#' A string, one of `"channels_last"` (default) or
#' `"channels_first"`. The ordering of the dimensions in the inputs.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, ..., channels)` while `"channels_first"` corresponds to
#' inputs with shape `(batch, channels, ...)`.
#' When unspecified, uses `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json` (if exists). Defaults to
#' `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/flatten#flatten-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten>
#' @tether keras.layers.Flatten
layer_flatten <-
function (object, data_format = NULL, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Flatten, object, args)
}


#' Permutes the dimensions of the input according to a given pattern.
#'
#' @description
#' Useful e.g. connecting RNNs and convnets.
#'
#' # Input Shape
#' Arbitrary.
#'
#' # Output Shape
#' Same as the input shape, but with the dimensions re-ordered according
#' to the specified pattern.
#'
#' # Examples
#' ```{r}
#' x <- layer_input(shape=c(10, 64))
#' y <- layer_permute(x, c(2, 1))
#' shape(y)
#' ```
#'
#' @param dims
#' List of integers. Permutation pattern does not include the
#' batch dimension. Indexing starts at 1.
#' For instance, `c(2, 1)` permutes the first and second dimensions
#' of the input.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/permute#permute-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Permute>
#'
#' @tether keras.layers.Permute
layer_permute <-
function (object, dims, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape,
        dims = function (x)
        tuple(lapply(x, as_integer))), ignore = "object")
    create_layer(keras$layers$Permute, object, args)
}


#' Repeats the input n times.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- layer_input(shape = 32)
#' y <- layer_repeat_vector(x, n = 3)
#' shape(y)
#' ```
#'
#' # Input Shape
#' 2D tensor with shape `(batch_size, features)`.
#'
#' # Output Shape
#' 3D tensor with shape `(batch_size, n, features)`.
#'
#' @param n
#' Integer, repetition factor.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/repeat_vector#repeatvector-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RepeatVector>
#'
#' @tether keras.layers.RepeatVector
layer_repeat_vector <-
function (object, n, ...)
{
    args <- capture_args(list(n = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$RepeatVector, object, args)
}


#' Layer that reshapes inputs into the given shape.
#'
#' @description
#'
#' # Input Shape
#' Arbitrary, although all dimensions in the input shape must be
#' known/fixed. Use the keyword argument `input_shape` (list of integers,
#' does not include the samples/batch size axis) when using this layer as
#' the first layer in a model.
#'
#' # Output Shape
#' `(batch_size, *target_shape)`
#'
#' # Examples
#' ```{r}
#' x <- layer_input(shape = 12)
#' y <- layer_reshape(x, c(3, 4))
#' shape(y)
#' ```
#'
#' ```{r}
#' # also supports shape inference using `-1` as dimension
#' y <- layer_reshape(x, c(-1, 2, 2))
#' shape(y)
#' ```
#'
#' @param target_shape
#' Target shape. List of integers, does not include the
#' samples dimension (batch size).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/reshape#reshape-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape>
#'
#' @tether keras.layers.Reshape
layer_reshape <-
function (object, target_shape, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape,
        target_shape = as_integer), ignore = "object")
    create_layer(keras$layers$Reshape, object, args)
}


#' Upsampling layer for 1D inputs.
#'
#' @description
#' Repeats each temporal step `size` times along the time axis.
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 2, 3)
#' x <- seq_len(prod(input_shape)) %>% op_reshape(input_shape)
#' x
#' y <- layer_upsampling_1d(x, size = 2)
#' y
#' ```
#'
#'  `[[ 6.  7.  8.]`
#'   `[ 6.  7.  8.]`
#'   `[ 9. 10. 11.]`
#'   `[ 9. 10. 11.]]]`
#'
#' # Input Shape
#' 3D tensor with shape: `(batch_size, steps, features)`.
#'
#' # Output Shape
#' 3D tensor with shape: `(batch_size, upsampled_steps, features)`.
#'
#' @param size
#' Integer. Upsampling factor.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/up_sampling1d#upsampling1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling1D>
#'
#' @tether keras.layers.UpSampling1D
layer_upsampling_1d <-
function (object, size = 2L, ...)
{
    args <- capture_args(list(size = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$UpSampling1D, object, args)
}


#' Upsampling layer for 2D inputs.
#'
#' @description
#' The implementation uses interpolative resizing, given the resize method
#' (specified by the `interpolation` argument). Use `interpolation=nearest`
#' to repeat the rows and columns of the data.
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 2, 1, 3)
#' x <- op_reshape(seq_len(prod(input_shape)), input_shape)
#' print(x)
#' y <- layer_upsampling_2d(x, size = c(1, 2))
#' print(y)
#' ```
#'
#' # Input Shape
#' 4D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'     `(batch_size, rows, cols, channels)`
#' - If `data_format` is `"channels_first"`:
#'     `(batch_size, channels, rows, cols)`
#'
#' # Output Shape
#' 4D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'     `(batch_size, upsampled_rows, upsampled_cols, channels)`
#' - If `data_format` is `"channels_first"`:
#'     `(batch_size, channels, upsampled_rows, upsampled_cols)`
#'
#' @param size
#' Int, or list of 2 integers.
#' The upsampling factors for rows and columns.
#'
#' @param data_format
#' A string,
#' one of `"channels_last"` (default) or `"channels_first"`.
#' The ordering of the dimensions in the inputs.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch_size, height, width, channels)` while `"channels_first"`
#' corresponds to inputs with shape
#' `(batch_size, channels, height, width)`.
#' When unspecified, uses
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json` (if exists) else `"channels_last"`.
#' Defaults to `"channels_last"`.
#'
#' @param interpolation
#' A string, one of `"bicubic"`, `"bilinear"`, `"lanczos3"`,
#' `"lanczos5"`, `"nearest"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/up_sampling2d#upsampling2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling2D>
#'
#' @tether keras.layers.UpSampling2D
layer_upsampling_2d <-
function (object, size = list(2L, 2L), data_format = NULL, interpolation = "nearest",
    ...)
{
    args <- capture_args(list(size = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$UpSampling2D, object, args)
}


#' Upsampling layer for 3D inputs.
#'
#' @description
#' Repeats the 1st, 2nd and 3rd dimensions
#' of the data by `size[0]`, `size[1]` and `size[2]` respectively.
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 1, 2, 1, 3)
#' x <- array(1, dim = input_shape)
#' y <- layer_upsampling_3d(x, size = c(2, 2, 2))
#' shape(y)
#' ```
#'
#' # Input Shape
#' 5D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'     `(batch_size, dim1, dim2, dim3, channels)`
#' - If `data_format` is `"channels_first"`:
#'     `(batch_size, channels, dim1, dim2, dim3)`
#'
#' # Output Shape
#' 5D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'     `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3,
#'     channels)`
#' - If `data_format` is `"channels_first"`:
#'     `(batch_size, channels, upsampled_dim1, upsampled_dim2,
#'     upsampled_dim3)`
#'
#' @param size
#' Int, or list of 3 integers.
#' The upsampling factors for dim1, dim2 and dim3.
#'
#' @param data_format
#' A string,
#' one of `"channels_last"` (default) or `"channels_first"`.
#' The ordering of the dimensions in the inputs.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
#' When unspecified, uses
#' `image_data_format` value found in your Keras config file at
#'  `~/.keras/keras.json` (if exists) else `"channels_last"`.
#' Defaults to `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/up_sampling3d#upsampling3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling3D>
#'
#' @tether keras.layers.UpSampling3D
layer_upsampling_3d <-
function (object, size = list(2L, 2L, 2L), data_format = NULL,
    ...)
{
    args <- capture_args(list(size = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$UpSampling3D, object, args)
}


#' Zero-padding layer for 1D input (e.g. temporal sequence).
#'
#' @description
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 2, 3)
#' x <- op_reshape(seq_len(prod(input_shape)), input_shape)
#' x
#' y <- layer_zero_padding_1d(x, padding = 2)
#' y
#' ```
#'
#' # Input Shape
#' 3D tensor with shape `(batch_size, axis_to_pad, features)`
#'
#' # Output Shape
#' 3D tensor with shape `(batch_size, padded_axis, features)`
#'
#' @param padding
#' Int, or list of int (length 2), or named listionary.
#' - If int: how many zeros to add at the beginning and end of
#'   the padding dimension (axis 1).
#' - If list of 2 ints: how many zeros to add at the beginning and the
#'   end of the padding dimension (`(left_pad, right_pad)`).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/zero_padding1d#zeropadding1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D>
#'
#' @tether keras.layers.ZeroPadding1D
layer_zero_padding_1d <-
function (object, padding = 1L, ...)
{
    args <- capture_args(list(padding = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$ZeroPadding1D, object, args)
}


#' Zero-padding layer for 2D input (e.g. picture).
#'
#' @description
#' This layer can add rows and columns of zeros at the top, bottom, left and
#' right side of an image tensor.
#'
#' # Examples
#' ```{r}
#' input_shape <- c(1, 1, 2, 2)
#' x <- op_reshape(seq_len(prod(input_shape)), input_shape)
#' x
#' y <- layer_zero_padding_2d(x, padding = 1)
#' y
#' ```
#'
#' # Input Shape
#' 4D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'   `(batch_size, height, width, channels)`
#' - If `data_format` is `"channels_first"`:
#'   `(batch_size, channels, height, width)`
#'
#' # Output Shape
#' 4D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'   `(batch_size, padded_height, padded_width, channels)`
#' - If `data_format` is `"channels_first"`:
#'   `(batch_size, channels, padded_height, padded_width)`
#'
#' @param padding
#' Int, or list of 2 ints, or list of 2 lists of 2 ints.
#' - If int: the same symmetric padding is applied to height and width.
#' - If list of 2 ints: interpreted as two different symmetric padding
#'   values for height and width:
#'   `(symmetric_height_pad, symmetric_width_pad)`.
#' - If list of 2 lists of 2 ints: interpreted as
#'  `((top_pad, bottom_pad), (left_pad, right_pad))`.
#'
#' @param data_format
#' A string, one of `"channels_last"` (default) or
#' `"channels_first"`. The ordering of the dimensions in the inputs.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch_size, height, width, channels)` while `"channels_first"`
#' corresponds to inputs with shape
#' `(batch_size, channels, height, width)`.
#' When unspecified, uses `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json` (if exists). Defaults to
#' `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/zero_padding2d#zeropadding2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D>
#'
#' @tether keras.layers.ZeroPadding2D
layer_zero_padding_2d <-
function (object, padding = list(1L, 1L), data_format = NULL,
    ...)
{
    args <- capture_args(list(padding = function (x)
    normalize_padding(x, 2L), input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$ZeroPadding2D, object, args)
}


#' Zero-padding layer for 3D data (spatial or spatio-temporal).
#'
#' @description
#'
#' # Examples
#' ```{r}
#' input_shape <- c(1, 1, 2, 2, 3)
#' x <- op_reshape(seq_len(prod(input_shape)), input_shape)
#' x
#' y <- layer_zero_padding_3d(x, padding = 2)
#' shape(y)
#' ```
#'
#' # Input Shape
#' 5D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'   `(batch_size, first_axis_to_pad, second_axis_to_pad,
#'   third_axis_to_pad, depth)`
#' - If `data_format` is `"channels_first"`:
#'   `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
#'   third_axis_to_pad)`
#'
#' # Output Shape
#' 5D tensor with shape:
#' - If `data_format` is `"channels_last"`:
#'   `(batch_size, first_padded_axis, second_padded_axis,
#'   third_axis_to_pad, depth)`
#' - If `data_format` is `"channels_first"`:
#'   `(batch_size, depth, first_padded_axis, second_padded_axis,
#'   third_axis_to_pad)`
#'
#' @param padding
#' Int, or list of 3 ints, or list of 3 lists of 2 ints.
#' - If int: the same symmetric padding is applied to depth, height,
#'   and width.
#' - If list of 3 ints: interpreted as three different symmetric
#'   padding values for depth, height, and width:
#'   `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
#' - If list of 3 lists of 2 ints: interpreted as
#'   `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
#'   right_dim2_pad), (left_dim3_pad, right_dim3_pad))`.
#'
#' @param data_format
#' A string, one of `"channels_last"` (default) or
#' `"channels_first"`. The ordering of the dimensions in the inputs.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
#' When unspecified, uses `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json` (if exists). Defaults to
#' `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family reshaping layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/reshaping_layers/zero_padding3d#zeropadding3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D>
#'
#' @tether keras.layers.ZeroPadding3D
layer_zero_padding_3d <-
function (object, padding = list(list(1L, 1L), list(1L, 1L),
    list(1L, 1L)), data_format = NULL, ...)
{
    args <- capture_args(list(padding = function (x)
    normalize_padding(x, 3L), input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$ZeroPadding3D, object, args)
}
