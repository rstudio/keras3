#' 3D convolution layer.
#'
#' @description
#' This layer creates a convolution kernel that is convolved with the layer
#' input over a single spatial (or temporal) dimension to produce a tensor of
#' outputs. If `use_bias` is True, a bias vector is created and added to the
#' outputs. Finally, if `activation` is not `None`, it is applied to the
#' outputs as well.
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
#'     `(batch_size, new_spatial_dim1, new_spatial_dim2, new_spatial_dim3,
#'     filters)`
#' - If `data_format="channels_first"`:
#'     5D tensor with shape:
#'     `(batch_size, filters, new_spatial_dim1, new_spatial_dim2,
#'     new_spatial_dim3)`
#'
#' # Returns
#' A 5D tensor representing `activation(conv3d(inputs, kernel) + bias)`.
#'
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # Examples
#' ```python
#' x = np.random.rand(4, 10, 10, 10, 128)
#' y = keras.layers.Conv3D(32, 3, activation='relu')(x)
#' print(y.shape)
#' # (4, 8, 8, 8, 32)
#' ```
#'
#' @param filters int, the dimension of the output space (the number of filters
#'     in the convolution).
#' @param kernel_size int or tuple/list of 3 integer, specifying the size of the
#'     convolution window.
#' @param strides int or tuple/list of 3 integer, specifying the stride length
#'     of the convolution. `strides > 1` is incompatible with
#'     `dilation_rate > 1`.
#' @param padding string, either `"valid"` or `"same"` (case-insensitive).
#'     `"valid"` means no padding. `"same"` results in padding evenly to
#'     the left/right or up/down of the input such that output has the same
#'     height/width dimension as the input.
#' @param data_format string, either `"channels_last"` or `"channels_first"`.
#'     The ordering of the dimensions in the inputs. `"channels_last"`
#'     corresponds to inputs with shape
#'     `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#'     while `"channels_first"` corresponds to inputs with shape
#'     `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
#'     It defaults to the `image_data_format` value found in your Keras
#'     config file at `~/.keras/keras.json`. If you never set it, then it
#'     will be `"channels_last"`.
#' @param dilation_rate int or tuple/list of 3 integers, specifying the dilation
#'     rate to use for dilated convolution.
#' @param groups A positive int specifying the number of groups in which the
#'     input is split along the channel axis. Each group is convolved
#'     separately with `filters // groups` filters. The output is the
#'     concatenation of all the `groups` results along the channel axis.
#'     Input channels and `filters` must both be divisible by `groups`.
#' @param activation Activation function. If `None`, no activation is applied.
#' @param use_bias bool, if `True`, bias will be added to the output.
#' @param kernel_initializer Initializer for the convolution kernel. If `None`,
#'     the default initializer (`"glorot_uniform"`) will be used.
#' @param bias_initializer Initializer for the bias vector. If `None`, the
#'     default initializer (`"zeros"`) will be used.
#' @param kernel_regularizer Optional regularizer for the convolution kernel.
#' @param bias_regularizer Optional regularizer for the bias vector.
#' @param activity_regularizer Optional regularizer function for the output.
#' @param kernel_constraint Optional projection function to be applied to the
#'     kernel after being updated by an `Optimizer` (e.g. used to implement
#'     norm constraints or value constraints for layer weights). The
#'     function must take as input the unprojected variable and must return
#'     the projected variable (which must have the same shape). Constraints
#'     are not safe to use when doing asynchronous distributed training.
#' @param bias_constraint Optional projection function to be applied to the
#'     bias after being updated by an `Optimizer`.
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family convolutional layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D>
layer_conv_3d <-
function (object, filters, kernel_size, strides = list(1L, 1L,
    1L), padding = "valid", data_format = NULL, dilation_rate = list(
    1L, 1L, 1L), groups = 1L, activation = NULL, use_bias = TRUE,
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args2(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        groups = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Conv3D, object, args)
}
