


#' Average pooling for temporal data.
#'
#' @description
#' Downsamples the input representation by taking the average value over the
#' window defined by `pool_size`. The window is shifted by `strides`.  The
#' resulting output when using "valid" padding option has a shape of:
#' `output_shape = (input_shape - pool_size + 1) / strides)`
#'
#' The resulting output shape when using the "same" padding option is:
#' `output_shape = input_shape / strides`
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     3D tensor with shape `(batch_size, steps, features)`.
#' - If `data_format="channels_first"`:
#'     3D tensor with shape `(batch_size, features, steps)`.
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     3D tensor with shape `(batch_size, downsampled_steps, features)`.
#' - If `data_format="channels_first"`:
#'     3D tensor with shape `(batch_size, features, downsampled_steps)`.
#'
#' # Examples
#' `strides=1` and `padding="valid"`:
#'
#' ```{r}
#' x <- op_array(c(1., 2., 3., 4., 5.)) |> op_reshape(c(1, 5, 1))
#' output <- x |>
#'   layer_average_pooling_1d(pool_size = 2,
#'                            strides = 1,
#'                            padding = "valid")
#' output
#' ```
#'
#' `strides=2` and `padding="valid"`:
#'
#' ```{r}
#' x <- op_array(c(1., 2., 3., 4., 5.)) |> op_reshape(c(1, 5, 1))
#' output <- x |>
#'   layer_average_pooling_1d(pool_size = 2,
#'                            strides = 2,
#'                            padding = "valid")
#' output
#' ```
#'
#' `strides=1` and `padding="same"`:
#'
#' ```{r}
#' x <- op_array(c(1., 2., 3., 4., 5.)) |> op_reshape(c(1, 5, 1))
#' output <- x |>
#'   layer_average_pooling_1d(pool_size = 2,
#'                            strides = 1,
#'                            padding = "same")
#' output
#' ```
#'
#' @param pool_size
#' int, size of the max pooling window.
#'
#' @param strides
#' int or `NULL`. Specifies how much the pooling window moves
#' for each pooling step. If `NULL`, it will default to `pool_size`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the same
#' height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, steps, features)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, steps)`. It defaults to the `image_data_format`
#' value found in your Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/average_pooling1d#averagepooling1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling1D>
#' @tether keras.layers.AveragePooling1D
layer_average_pooling_1d <-
function (object, pool_size, strides = NULL, padding = "valid",
    data_format = NULL, name = NULL, ...)
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$AveragePooling1D, object, args)
}


#' Average pooling operation for 2D spatial data.
#'
#' @description
#' Downsamples the input along its spatial dimensions (height and width)
#' by taking the average value over an input window
#' (of size defined by `pool_size`) for each channel of the input.
#' The window is shifted by `strides` along each dimension.
#'
#' The resulting output when using the `"valid"` padding option has a spatial
#' shape (number of rows or columns) of:
#' `output_shape = math.floor((input_shape - pool_size) / strides) + 1`
#' (when `input_shape >= pool_size`)
#'
#' The resulting output shape when using the `"same"` padding option is:
#' `output_shape = math.floor((input_shape - 1) / strides) + 1`
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     4D tensor with shape `(batch_size, height, width, channels)`.
#' - If `data_format="channels_first"`:
#'     4D tensor with shape `(batch_size, channels, height, width)`.
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     4D tensor with shape
#'     `(batch_size, pooled_height, pooled_width, channels)`.
#' - If `data_format="channels_first"`:
#'     4D tensor with shape
#'     `(batch_size, channels, pooled_height, pooled_width)`.
#'
#' # Examples
#' `strides=(1, 1)` and `padding="valid"`:
#'
#' ```{r}
#' x <- op_array(1:9, "float32") |> op_reshape(c(1, 3, 3, 1))
#' output <- x |>
#'   layer_average_pooling_2d(pool_size = c(2, 2),
#'                            strides = c(1, 1),
#'                            padding = "valid")
#' output
#' ```
#'
#' `strides=(2, 2)` and `padding="valid"`:
#'
#' ```{r}
#' x <- op_array(1:12, "float32") |> op_reshape(c(1, 3, 4, 1))
#' output <- x |>
#'   layer_average_pooling_2d(pool_size = c(2, 2),
#'                            strides = c(2, 2),
#'                            padding = "valid")
#' output
#' ```
#'
#' `stride=(1, 1)` and `padding="same"`:
#'
#' ```{r}
#' x <- op_array(1:9, "float32") |> op_reshape(c(1, 3, 3, 1))
#' output <- x |>
#'   layer_average_pooling_2d(pool_size = c(2, 2),
#'                            strides = c(1, 1),
#'                            padding = "same")
#' output
#' ```
#'
#' @param pool_size
#' int or list of 2 integers, factors by which to downscale
#' (dim1, dim2). If only one integer is specified, the same
#' window length will be used for all dimensions.
#'
#' @param strides
#' int or list of 2 integers, or `NULL`. Strides values. If `NULL`,
#' it will default to `pool_size`. If only one int is specified, the
#' same stride size will be used for all dimensions.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the same
#' height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, height, width, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, height, width)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/average_pooling2d#averagepooling2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D>
#' @tether keras.layers.AveragePooling2D
layer_average_pooling_2d <-
function (object, pool_size, strides = NULL, padding = "valid",
    data_format = "channels_last", name = NULL, ...)
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$AveragePooling2D, object, args)
}


#' Average pooling operation for 3D data (spatial or spatio-temporal).
#'
#' @description
#' Downsamples the input along its spatial dimensions (depth, height, and
#' width) by taking the average value over an input window (of size defined by
#' `pool_size`) for each channel of the input. The window is shifted by
#' `strides` along each dimension.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     5D tensor with shape:
#'     `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' - If `data_format="channels_first"`:
#'     5D tensor with shape:
#'     `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     5D tensor with shape:
#'     `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
#' - If `data_format="channels_first"`:
#'     5D tensor with shape:
#'     `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
#'
#' # Examples
#' ```{r}
#' depth <- height <- width <- 30
#' channels <- 3
#'
#' inputs <- layer_input(shape = c(depth, height, width, channels))
#' outputs <- inputs |> layer_average_pooling_3d(pool_size = 3)
#' outputs # Shape: (batch_size, 10, 10, 10, 3)
#' ```
#'
#' @param pool_size
#' int or list of 3 integers, factors by which to downscale
#' (dim1, dim2, dim3). If only one integer is specified, the same
#' window length will be used for all dimensions.
#'
#' @param strides
#' int or list of 3 integers, or `NULL`. Strides values. If `NULL`,
#' it will default to `pool_size`. If only one int is specified, the
#' same stride size will be used for all dimensions.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the same
#' height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape
#' `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while
#' `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
#' It defaults to the `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json`. If you never set it, then it
#' will be `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/average_pooling3d#averagepooling3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling3D>
#' @tether keras.layers.AveragePooling3D
layer_average_pooling_3d <-
function (object, pool_size, strides = NULL, padding = "valid",
    data_format = "channels_last", name = NULL, ...)
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$AveragePooling3D, object, args)
}


#' Global average pooling operation for temporal data.
#'
#' @description
#'
#' # Call Arguments
#' - `inputs`: A 3D tensor.
#' - `mask`: Binary tensor of shape `(batch_size, steps)` indicating whether
#'     a given step should be masked (excluded from the average).
#'
#' # Input Shape
#' - If `data_format='channels_last'`:
#'     3D tensor with shape:
#'     `(batch_size, steps, features)`
#' - If `data_format='channels_first'`:
#'     3D tensor with shape:
#'     `(batch_size, features, steps)`
#'
#' # Output Shape
#' - If `keepdims=FALSE`:
#'     2D tensor with shape `(batch_size, features)`.
#' - If `keepdims=TRUE`:
#'     - If `data_format="channels_last"`:
#'         3D tensor with shape `(batch_size, 1, features)`
#'     - If `data_format="channels_first"`:
#'         3D tensor with shape `(batch_size, features, 1)`
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 3, 4))
#' y <- x |> layer_global_average_pooling_1d()
#' shape(y)
#' ```
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, steps, features)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, steps)`. It defaults to the `image_data_format`
#' value found in your Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param keepdims
#' A boolean, whether to keep the temporal dimension or not.
#' If `keepdims` is `FALSE` (default), the rank of the tensor is
#' reduced for spatial dimensions. If `keepdims` is `TRUE`, the
#' temporal dimension are retained with length 1.
#' The behavior is the same as for `tf$reduce_mean()` or `op_mean()`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/global_average_pooling1d#globalaveragepooling1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D>
#' @tether keras.layers.GlobalAveragePooling1D
layer_global_average_pooling_1d <-
function (object, data_format = NULL, keepdims = FALSE, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GlobalAveragePooling1D, object,
        args)
}


#' Global average pooling operation for 2D data.
#'
#' @description
#'
#' # Input Shape
#' - If `data_format='channels_last'`:
#'     4D tensor with shape:
#'     `(batch_size, height, width, channels)`
#' - If `data_format='channels_first'`:
#'     4D tensor with shape:
#'     `(batch_size, channels, height, width)`
#'
#' # Output Shape
#' - If `keepdims=FALSE`:
#'     2D tensor with shape `(batch_size, channels)`.
#' - If `keepdims=TRUE`:
#'     - If `data_format="channels_last"`:
#'         4D tensor with shape `(batch_size, 1, 1, channels)`
#'     - If `data_format="channels_first"`:
#'         4D tensor with shape `(batch_size, channels, 1, 1)`
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 4, 5, 3))
#' y <- x |> layer_global_average_pooling_2d()
#' shape(y)
#' ```
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, height, width, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, height, weight)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param keepdims
#' A boolean, whether to keep the temporal dimension or not.
#' If `keepdims` is `FALSE` (default), the rank of the tensor is
#' reduced for spatial dimensions. If `keepdims` is `TRUE`, the
#' spatial dimension are retained with length 1.
#' The behavior is the same as for `tf$reduce_mean()` or `op_mean()`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/global_average_pooling2d#globalaveragepooling2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D>
#' @tether keras.layers.GlobalAveragePooling2D
layer_global_average_pooling_2d <-
function (object, data_format = NULL, keepdims = FALSE, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GlobalAveragePooling2D, object,
        args)
}


#' Global average pooling operation for 3D data.
#'
#' @description
#'
#' # Input Shape
#' - If `data_format='channels_last'`:
#'     5D tensor with shape:
#'     `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' - If `data_format='channels_first'`:
#'     5D tensor with shape:
#'     `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
#'
#' # Output Shape
#' - If `keepdims=FALSE`:
#'     2D tensor with shape `(batch_size, channels)`.
#' - If `keepdims=TRUE`:
#'     - If `data_format="channels_last"`:
#'         5D tensor with shape `(batch_size, 1, 1, 1, channels)`
#'     - If `data_format="channels_first"`:
#'         5D tensor with shape `(batch_size, channels, 1, 1, 1)`
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 4, 5, 4, 3))
#' y <- x |> layer_global_average_pooling_3d()
#' shape(y)
#' ```
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape
#' `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
#' It defaults to the `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json`. If you never set it, then it
#' will be `"channels_last"`.
#'
#' @param keepdims
#' A boolean, whether to keep the temporal dimension or not.
#' If `keepdims` is `FALSE` (default), the rank of the tensor is
#' reduced for spatial dimensions. If `keepdims` is `TRUE`, the
#' spatial dimension are retained with length 1.
#' The behavior is the same as for `tf$reduce_mean()` or `op_mean()`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/global_average_pooling3d#globalaveragepooling3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling3D>
#' @tether keras.layers.GlobalAveragePooling3D
layer_global_average_pooling_3d <-
function (object, data_format = NULL, keepdims = FALSE, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GlobalAveragePooling3D, object,
        args)
}


#' Global max pooling operation for temporal data.
#'
#' @description
#'
#' # Input Shape
#' - If `data_format='channels_last'`:
#'     3D tensor with shape:
#'     `(batch_size, steps, features)`
#' - If `data_format='channels_first'`:
#'     3D tensor with shape:
#'     `(batch_size, features, steps)`
#'
#' # Output Shape
#' - If `keepdims=FALSE`:
#'     2D tensor with shape `(batch_size, features)`.
#' - If `keepdims=TRUE`:
#'     - If `data_format="channels_last"`:
#'         3D tensor with shape `(batch_size, 1, features)`
#'     - If `data_format="channels_first"`:
#'         3D tensor with shape `(batch_size, features, 1)`
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 3, 4))
#' y <- x |> layer_global_max_pooling_1d()
#' shape(y)
#' ```
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, steps, features)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, steps)`. It defaults to the `image_data_format`
#' value found in your Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param keepdims
#' A boolean, whether to keep the temporal dimension or not.
#' If `keepdims` is `FALSE` (default), the rank of the tensor is
#' reduced for spatial dimensions. If `keepdims` is `TRUE`, the
#' temporal dimension are retained with length 1.
#' The behavior is the same as for `tf$reduce_mean()` or `op_mean()`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/global_max_pooling1d#globalmaxpooling1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPooling1D>
#' @tether keras.layers.GlobalMaxPooling1D
layer_global_max_pooling_1d <-
function (object, data_format = NULL, keepdims = FALSE, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GlobalMaxPooling1D, object, args)
}


#' Global max pooling operation for 2D data.
#'
#' @description
#'
#' # Input Shape
#' - If `data_format='channels_last'`:
#'     4D tensor with shape:
#'     `(batch_size, height, width, channels)`
#' - If `data_format='channels_first'`:
#'     4D tensor with shape:
#'     `(batch_size, channels, height, width)`
#'
#' # Output Shape
#' - If `keepdims=FALSE`:
#'     2D tensor with shape `(batch_size, channels)`.
#' - If `keepdims=TRUE`:
#'     - If `data_format="channels_last"`:
#'         4D tensor with shape `(batch_size, 1, 1, channels)`
#'     - If `data_format="channels_first"`:
#'         4D tensor with shape `(batch_size, channels, 1, 1)`
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 4, 5, 3))
#' y <- x |> layer_global_max_pooling_2d()
#' shape(y)
#' ```
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, height, width, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, height, weight)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param keepdims
#' A boolean, whether to keep the temporal dimension or not.
#' If `keepdims` is `FALSE` (default), the rank of the tensor is
#' reduced for spatial dimensions. If `keepdims` is `TRUE`, the
#' spatial dimension are retained with length 1.
#' The behavior is the same as for `tf$reduce_mean()` or `op_mean()`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/global_max_pooling2d#globalmaxpooling2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPooling2D>
#' @tether keras.layers.GlobalMaxPooling2D
layer_global_max_pooling_2d <-
function (object, data_format = NULL, keepdims = FALSE, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GlobalMaxPooling2D, object, args)
}


#' Global max pooling operation for 3D data.
#'
#' @description
#'
#' # Input Shape
#' - If `data_format='channels_last'`:
#'     5D tensor with shape:
#'     `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' - If `data_format='channels_first'`:
#'     5D tensor with shape:
#'     `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
#'
#' # Output Shape
#' - If `keepdims=FALSE`:
#'     2D tensor with shape `(batch_size, channels)`.
#' - If `keepdims=TRUE`:
#'     - If `data_format="channels_last"`:
#'         5D tensor with shape `(batch_size, 1, 1, 1, channels)`
#'     - If `data_format="channels_first"`:
#'         5D tensor with shape `(batch_size, channels, 1, 1, 1)`
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 4, 5, 4, 3))
#' y <- x |> layer_global_max_pooling_3d()
#' shape(y)
#' ```
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape
#' `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
#' It defaults to the `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json`. If you never set it, then it
#' will be `"channels_last"`.
#'
#' @param keepdims
#' A boolean, whether to keep the temporal dimension or not.
#' If `keepdims` is `FALSE` (default), the rank of the tensor is
#' reduced for spatial dimensions. If `keepdims` is `TRUE`, the
#' spatial dimension are retained with length 1.
#' The behavior is the same as for `tf$reduce_mean()` or `op_mean()`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/global_max_pooling3d#globalmaxpooling3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPooling3D>
#' @tether keras.layers.GlobalMaxPooling3D
layer_global_max_pooling_3d <-
function (object, data_format = NULL, keepdims = FALSE, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GlobalMaxPooling3D, object, args)
}


#' Max pooling operation for 1D temporal data.
#'
#' @description
#' Downsamples the input representation by taking the maximum value over a
#' spatial window of size `pool_size`. The window is shifted by `strides`.
#'
#' The resulting output when using the `"valid"` padding option has a shape of:
#' `output_shape = (input_shape - pool_size + 1) / strides)`.
#'
#' The resulting output shape when using the `"same"` padding option is:
#' `output_shape = input_shape / strides`
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     3D tensor with shape `(batch_size, steps, features)`.
#' - If `data_format="channels_first"`:
#'     3D tensor with shape `(batch_size, features, steps)`.
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     3D tensor with shape `(batch_size, downsampled_steps, features)`.
#' - If `data_format="channels_first"`:
#'     3D tensor with shape `(batch_size, features, downsampled_steps)`.
#'
#' # Examples
#' `strides=1` and `padding="valid"`:
#'
#' ```{r}
#' x <- op_reshape(c(1, 2, 3, 4, 5),
#'                c(1, 5, 1))
#' max_pool_1d <- layer_max_pooling_1d(pool_size = 2,
#'                                     strides = 1,
#'                                     padding = "valid")
#' max_pool_1d(x)
#' ```
#'
#' `strides=2` and `padding="valid"`:
#'
#' ```{r}
#' x <- op_reshape(c(1, 2, 3, 4, 5),
#'                c(1, 5, 1))
#' max_pool_1d <- layer_max_pooling_1d(pool_size = 2,
#'                                     strides = 2,
#'                                     padding = "valid")
#' max_pool_1d(x)
#' ```
#'
#' `strides=1` and `padding="same"`:
#'
#' ```{r}
#' x <- op_reshape(c(1, 2, 3, 4, 5),
#'                c(1, 5, 1))
#' max_pool_1d <- layer_max_pooling_1d(pool_size = 2,
#'                                     strides = 1,
#'                                     padding = "same")
#' max_pool_1d(x)
#' ```
#'
#' @param pool_size
#' int, size of the max pooling window.
#'
#' @param strides
#' int or `NULL`. Specifies how much the pooling window moves
#' for each pooling step. If `NULL`, it will default to `pool_size`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the same
#' height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, steps, features)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, features, steps)`. It defaults to the `image_data_format`
#' value found in your Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/max_pooling1d#maxpooling1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling1D>
#' @tether keras.layers.MaxPooling1D
layer_max_pooling_1d <-
function (object, pool_size = 2L, strides = NULL, padding = "valid",
    data_format = NULL, name = NULL, ...)
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$MaxPooling1D, object, args)
}


#' Max pooling operation for 2D spatial data.
#'
#' @description
#' Downsamples the input along its spatial dimensions (height and width)
#' by taking the maximum value over an input window
#' (of size defined by `pool_size`) for each channel of the input.
#' The window is shifted by `strides` along each dimension.
#'
#' The resulting output when using the `"valid"` padding option has a spatial
#' shape (number of rows or columns) of:
#' `output_shape = floor((input_shape - pool_size) / strides) + 1`
#' (when `input_shape >= pool_size`)
#'
#' The resulting output shape when using the `"same"` padding option is:
#' `output_shape = floor((input_shape - 1) / strides) + 1`
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     4D tensor with shape `(batch_size, height, width, channels)`.
#' - If `data_format="channels_first"`:
#'     4D tensor with shape `(batch_size, channels, height, width)`.
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     4D tensor with shape
#'     `(batch_size, pooled_height, pooled_width, channels)`.
#' - If `data_format="channels_first"`:
#'     4D tensor with shape
#'     `(batch_size, channels, pooled_height, pooled_width)`.
#'
#' # Examples
#' `strides = (1, 1)` and `padding = "valid"`:
#'
#' ```{r}
#' x <- rbind(c(1., 2., 3.),
#'            c(4., 5., 6.),
#'            c(7., 8., 9.)) |> op_reshape(c(1, 3, 3, 1))
#' max_pool_2d <- layer_max_pooling_2d(pool_size = c(2, 2),
#'                                     strides = c(1, 1),
#'                                     padding = "valid")
#' max_pool_2d(x)
#' ```
#'
#' `strides = c(2, 2)` and `padding = "valid"`:
#'
#' ```{r}
#' x <- rbind(c(1., 2., 3., 4.),
#'            c(5., 6., 7., 8.),
#'            c(9., 10., 11., 12.)) |> op_reshape(c(1, 3, 4, 1))
#' max_pool_2d <- layer_max_pooling_2d(pool_size = c(2, 2),
#'                                     strides = c(2, 2),
#'                                     padding = "valid")
#' max_pool_2d(x)
#' ```
#'
#' `stride = (1, 1)` and `padding = "same"`:
#'
#' ```{r}
#' x <- rbind(c(1., 2., 3.),
#'            c(4., 5., 6.),
#'            c(7., 8., 9.)) |> op_reshape(c(1, 3, 3, 1))
#' max_pool_2d <- layer_max_pooling_2d(pool_size = c(2, 2),
#'                                     strides = c(1, 1),
#'                                     padding = "same")
#' max_pool_2d(x)
#' ```
#'
#' @param pool_size
#' int or list of 2 integers, factors by which to downscale
#' (dim1, dim2). If only one integer is specified, the same
#' window length will be used for all dimensions.
#'
#' @param strides
#' int or list of 2 integers, or `NULL`. Strides values. If `NULL`,
#' it will default to `pool_size`. If only one int is specified, the
#' same stride size will be used for all dimensions.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the same
#' height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, height, width, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, height, width)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/max_pooling2d#maxpooling2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling2D>
#' @tether keras.layers.MaxPooling2D
layer_max_pooling_2d <-
function (object, pool_size = list(2L, 2L), strides = NULL, padding = "valid",
    data_format = NULL, name = NULL, ...)
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$MaxPooling2D, object, args)
}


#' Max pooling operation for 3D data (spatial or spatio-temporal).
#'
#' @description
#' Downsamples the input along its spatial dimensions (depth, height, and
#' width) by taking the maximum value over an input window (of size defined by
#' `pool_size`) for each channel of the input. The window is shifted by
#' `strides` along each dimension.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     5D tensor with shape:
#'     `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' - If `data_format="channels_first"`:
#'     5D tensor with shape:
#'     `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     5D tensor with shape:
#'     `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
#' - If `data_format="channels_first"`:
#'     5D tensor with shape:
#'     `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
#'
#' # Examples
#' ```{r}
#' depth <- 30
#' height <- 30
#' width <- 30
#' channels <- 3
#'
#' inputs <- layer_input(shape=c(depth, height, width, channels))
#' layer <- layer_max_pooling_3d(pool_size=3)
#' outputs <- inputs |> layer()
#' outputs
#' ```
#'
#' @param pool_size
#' int or list of 3 integers, factors by which to downscale
#' (dim1, dim2, dim3). If only one integer is specified, the same
#' window length will be used for all dimensions.
#'
#' @param strides
#' int or list of 3 integers, or `NULL`. Strides values. If `NULL`,
#' it will default to `pool_size`. If only one int is specified, the
#' same stride size will be used for all dimensions.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input such that output has the same
#' height/width dimension as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape
#' `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while
#' `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
#' It defaults to the `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json`. If you never set it, then it
#' will be `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/max_pooling3d#maxpooling3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling3D>
#' @tether keras.layers.MaxPooling3D
layer_max_pooling_3d <-
function (object, pool_size = list(2L, 2L, 2L), strides = NULL,
    padding = "valid", data_format = NULL, name = NULL, ...)
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$MaxPooling3D, object, args)
}
