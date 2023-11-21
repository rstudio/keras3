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
#' ```python
#' depth = 30
#' height = 30
#' width = 30
#' channels = 3
#'
#' inputs = keras.layers.Input(shape=(depth, height, width, channels))
#' layer = keras.layers.MaxPooling3D(pool_size=3)
#' outputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)
#' ```
#'
#' @param pool_size
#' int or tuple of 3 integers, factors by which to downscale
#' (dim1, dim2, dim3). If only one integer is specified, the same
#' window length will be used for all dimensions.
#'
#' @param strides
#' int or tuple of 3 integers, or None. Strides values. If None,
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
#' Passed on to the Python callable
#'
#' @export
#' @family pooling layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/pooling_layers/max_pooling3d#maxpooling3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling3D>
layer_max_pooling_3d <-
function (object, pool_size = list(2L, 2L, 2L), strides = NULL,
    padding = "valid", data_format = NULL, name = NULL, ...)
{
}
