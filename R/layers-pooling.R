

#' Max pooling operation for temporal data.
#'
#' @inheritParams layer_dense
#'
#' @param pool_size Integer, size of the max pooling windows.
#' @param strides Integer, or NULL. Factor by which to downscale. E.g. 2 will
#'   halve the input. If NULL, it will default to `pool_size`.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of "channels_last" (default) or
#'   "channels_first". The ordering of the dimensions in the inputs.
#'   channels_last corresponds to inputs with shape `(batch, steps, features)`
#'   while channels_first corresponds to inputs with shape `(batch, features, steps)`.
#'
#' @section Input Shape:
#' If data_format='channels_last': 3D tensor with shape `(batch_size, steps, features)`.
#' If data_format='channels_first': 3D tensor with shape `(batch_size, features, steps)`.
#'
#' @section Output shape:
#' If data_format='channels_last': 3D tensor with shape `(batch_size, downsampled_steps, features)`.
#' If data_format='channels_first': 3D tensor with shape `(batch_size, features, downsampled_steps)`.
#'
#' @family pooling layers
#'
#' @export
layer_max_pooling_1d <- function(object, pool_size = 2L, strides = NULL, padding = "valid",
                                 data_format='channels_last',
                                 batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {

  create_layer(keras$layers$MaxPooling1D, object, list(
    pool_size = as.integer(pool_size),
    strides = as_nullable_integer(strides),
    padding = padding,
    data_format = match.arg(data_format, c("channels_last", "channels_first")),
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
}



#' Max pooling operation for spatial data.
#'
#' @inheritParams layer_conv_2d
#' @inheritParams layer_max_pooling_1d
#'
#' @param pool_size integer or list of 2 integers, factors by which to downscale
#'   (vertical, horizontal). (2, 2) will halve the input in both spatial
#'   dimension. If only one integer is specified, the same window length will be
#'   used for both dimensions.
#' @param strides Integer, list of 2 integers, or NULL. Strides values. If NULL,
#'   it will default to `pool_size`.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#'
#' @section Input shape:
#' - If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
#' - If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`
#'
#' @section Output shape:
#' - If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, pooled_rows, pooled_cols, channels)`
#' - If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, pooled_rows, pooled_cols)`
#'
#' @family pooling layers
#'
#' @export
layer_max_pooling_2d <- function(object, pool_size = c(2L, 2L), strides = NULL, padding = "valid", data_format = NULL,
                                 batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {

  create_layer(keras$layers$MaxPooling2D, object, list(
    pool_size = as.integer(pool_size),
    strides = as_nullable_integer(strides),
    padding = padding,
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))

}


#' Max pooling operation for 3D data (spatial or spatio-temporal).
#'
#' @inheritParams layer_max_pooling_1d
#'
#' @param pool_size list of 3 integers, factors by which to downscale (dim1,
#'   dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each
#'   dimension.
#' @param strides list of 3 integers, or NULL. Strides values.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, spatial_dim1,
#'   spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds
#'   to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2,
#'   spatial_dim3)`. It defaults to the `image_data_format` value found in your
#'   Keras config file at `~/.keras/keras.json`. If you never set it, then it
#'   will be "channels_last".
#'
#' @section Input shape:
#' - If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' - If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
#'
#' @section Output shape:
#' - If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
#' - If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
#'
#' @family pooling layers
#'
#' @export
layer_max_pooling_3d <- function(object, pool_size = c(2L, 2L, 2L), strides = NULL, padding = "valid", data_format = NULL,
                                 batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {

  create_layer(keras$layers$MaxPooling3D, object, list(
    pool_size = as.integer(pool_size),
    strides = as_nullable_integer(strides),
    padding = padding,
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))

}


#' Average pooling for temporal data.
#'
#' @inheritParams layer_max_pooling_1d
#'
#' @param pool_size Integer, size of the average pooling windows.
#' @param strides Integer, or NULL. Factor by which to downscale. E.g. 2 will
#'   halve the input. If NULL, it will default to `pool_size`.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format One of `channels_last` (default) or `channels_first`.
#'   The ordering of the dimensions in the inputs.
#'
#' @section Input shape: 3D tensor with shape: `(batch_size, steps, features)`.
#'
#' @section Output shape: 3D tensor with shape: `(batch_size, downsampled_steps,
#'   features)`.
#'
#' @family pooling layers
#'
#' @export
layer_average_pooling_1d <- function(object, pool_size = 2L, strides = NULL, padding = "valid",
                                     data_format = "channels_last",
                                     batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {

  args <- list(
    pool_size = as.integer(pool_size),
    strides = as_nullable_integer(strides),
    padding = padding,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  )

  if (keras_version() >= "2.2.3")
    args$data_format <- data_format

  create_layer(keras$layers$AveragePooling1D, object, args)

}

#' Average pooling operation for spatial data.
#'
#' @inheritParams layer_conv_2d
#' @inheritParams layer_average_pooling_1d
#'
#' @param pool_size integer or list of 2 integers, factors by which to downscale
#'   (vertical, horizontal). (2, 2) will halve the input in both spatial
#'   dimension. If only one integer is specified, the same window length will be
#'   used for both dimensions.
#' @param strides Integer, list of 2 integers, or NULL. Strides values. If NULL,
#'   it will default to `pool_size`.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#'
#' @section Input shape:
#' - If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
#' - If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`
#'
#' @section Output shape:
#' - If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, pooled_rows, pooled_cols, channels)`
#' - If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, pooled_rows, pooled_cols)`
#'
#' @family pooling layers
#'
#' @export
layer_average_pooling_2d <- function(object, pool_size = c(2L, 2L), strides = NULL, padding = "valid", data_format = NULL,
                                     batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {

  create_layer(keras$layers$AveragePooling2D, object, list(
    pool_size = as.integer(pool_size),
    strides = as_nullable_integer(strides),
    padding = padding,
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))

}

#' Average pooling operation for 3D data (spatial or spatio-temporal).
#'
#' @inheritParams layer_average_pooling_1d
#'
#' @param pool_size list of 3 integers, factors by which to downscale (dim1,
#'   dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each
#'   dimension.
#' @param strides list of 3 integers, or NULL. Strides values.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, spatial_dim1,
#'   spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds
#'   to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2,
#'   spatial_dim3)`. It defaults to the `image_data_format` value found in your
#'   Keras config file at `~/.keras/keras.json`. If you never set it, then it
#'   will be "channels_last".
#'
#' @section Input shape:
#' - If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' - If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
#'
#' @section Output shape:
#' - If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
#' - If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
#'
#' @family pooling layers
#'
#' @export
layer_average_pooling_3d <- function(object, pool_size = c(2L, 2L, 2L), strides = NULL, padding = "valid", data_format = NULL,
                                     batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$AveragePooling3D, object, list(
    pool_size = as.integer(pool_size),
    strides = as_nullable_integer(strides),
    padding = padding,
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))

}


#' Global max pooling operation for temporal data.
#'
#' @inheritParams layer_dense
#' @inheritParams layer_average_pooling_1d
#'
#' @param keepdims A boolean, whether to keep the spatial dimensions or not. If
#'   `keepdims` is `FALSE` (default), the rank of the tensor is reduced for
#'   spatial dimensions. If `keepdims` is `TRUE`, the spatial dimensions are
#'   retained with length 1. The behavior is the same as for `tf.reduce_mean` or
#'   `np.mean`.
#'
#' @param ... standard layer arguments.
#'
#' @section Input shape:
#' 3D tensor with shape: `(batch_size, steps, features)`.
#'
#' @section Output shape:
#' 2D tensor with shape: `(batch_size, channels)`
#'
#' @family pooling layers
#'
#' @export
layer_global_max_pooling_1d <-
function(object, data_format = "channels_last", keepdims = FALSE, ...)
{
  args <- capture_args(match.call(),
                       list(batch_size = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$GlobalMaxPooling1D, object, args)
}


#' Global average pooling operation for temporal data.
#'
#' @inheritParams layer_global_max_pooling_1d
#'
#' @section Input shape:
#' 3D tensor with shape: `(batch_size, steps, features)`.
#'
#' @section Output shape:
#' 2D tensor with shape: `(batch_size, channels)`
#'
#' @family pooling layers
#'
#' @export
layer_global_average_pooling_1d <-
function(object, data_format = "channels_last", keepdims = FALSE, ...)
{
  args <- capture_args(match.call(),
                       list(batch_size = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$GlobalAveragePooling1D, object, args)
}


#' Global max pooling operation for spatial data.
#'
#' @inheritParams layer_conv_2d
#' @inheritParams layer_global_max_pooling_1d
#'
#' @section Input shape:
#' - If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
#' - If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`
#'
#' @section Output shape: 2D tensor with shape: `(batch_size, channels)`
#'
#' @family pooling layers
#'
#' @export
layer_global_max_pooling_2d <-
function(object, data_format = NULL, keepdims = FALSE, ...)
{
  args <- capture_args(match.call(),
                       list(batch_size = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$GlobalMaxPooling2D, object, args)
}


#' Global average pooling operation for spatial data.
#'
#' @inheritParams layer_conv_2d
#' @inheritParams layer_global_average_pooling_1d
#'
#' @section Input shape:
#' - If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
#' - If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`
#'
#' @section Output shape: 2D tensor with shape: `(batch_size, channels)`
#'
#' @family pooling layers
#'
#' @export
layer_global_average_pooling_2d <-
function(object, data_format = NULL, keepdims = FALSE, ...)
{
  args <- capture_args(match.call(),
                       list(batch_size = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$GlobalAveragePooling2D, object, args)
}

# TODO: standard layer args used to contain:
# batch_size = NULL, name = NULL, trainable = NULL, weights = NULL



#' Global Max pooling operation for 3D data.
#'
#' @inheritParams layer_global_max_pooling_2d
#'
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, spatial_dim1,
#'   spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds
#'   to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2,
#'   spatial_dim3)`. It defaults to the `image_data_format` value found in your
#'   Keras config file at `~/.keras/keras.json`. If you never set it, then it
#'   will be "channels_last".
#'
#' @section Input shape:
#' - If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' - If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
#'
#' @section Output shape: 2D tensor with shape: `(batch_size, channels)`
#'
#' @family pooling layers
#'
#' @export
layer_global_max_pooling_3d <-
function(object, data_format = NULL, keepdims = FALSE, ...)
{
  args <- capture_args(match.call(),
                       list(batch_size = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$GlobalMaxPooling3D, object, args)
}


#' Global Average pooling operation for 3D data.
#'
#' @inheritParams layer_global_average_pooling_2d
#'
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, spatial_dim1,
#'   spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds
#'   to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2,
#'   spatial_dim3)`. It defaults to the `image_data_format` value found in your
#'   Keras config file at `~/.keras/keras.json`. If you never set it, then it
#'   will be "channels_last".
#'
#' @section Input shape:
#' - If `data_format='channels_last'`: 5D tensor with shape: `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' - If `data_format='channels_first'`: 5D tensor with shape: `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
#'
#' @section Output shape: 2D tensor with shape: `(batch_size, channels)`
#'
#' @family pooling layers
#'
#' @export
layer_global_average_pooling_3d <-
function(object, data_format = NULL, keepdims = FALSE, ...)
{
  args <- capture_args(match.call(),
                       list(batch_size = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$GlobalAveragePooling3D, object, args)
}
