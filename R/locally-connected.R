
#' Locally-connected layer for 1D inputs.
#' 
#' `layer_locally_connected_1d()` works similarly to [layer_conv_1d()] , except
#' that weights are unshared, that is, a different set of filters is applied at
#' each different patch of the input. 
#' 
#' @inheritParams layer_conv_1d
#' 
#' @param filters Integer, the dimensionality of the output space (i.e. the
#'   number output of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying the
#'   length of the 1D convolution window.
#' @param strides An integer or list of a single integer, specifying the stride
#'   length of the convolution. Specifying any stride value != 1 is incompatible
#'   with specifying any `dilation_rate` value != 1.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format data_format
#' @param activation Activation function to use. If you don't specify anything,
#'   no activation is applied (ie. "linear" activation: `a(x) = x`).
#' @param use_bias Boolean, whether the layer uses a bias vector.
#' @param kernel_initializer Initializer for the `kernel` weights matrix.
#' @param bias_initializer Initializer for the bias vector.
#' @param kernel_regularizer Regularizer function applied to the `kernel`
#'   weights matrix.
#' @param bias_regularizer Regularizer function applied to the bias vector.
#' @param activity_regularizer Regularizer function applied to the output of the
#'   layer (its "activation")..
#' @param kernel_constraint Constraint function applied to the kernel matrix.
#' @param bias_constraint Constraint function applied to the bias vector.
#'   
#' @section Input shape: 3D tensor with shape: `(batch_size, steps, input_dim)`
#'   
#' @section Output shape: 3D tensor with shape: `(batch_size, new_steps,
#'   filters)` `steps` value might have changed due to padding or strides.
#'   
#' @export
layer_locally_connected_1d <- function(x, filters, kernel_size, strides = 1L, padding = "valid", data_format = NULL, 
                                       activation = NULL, use_bias = TRUE, kernel_initializer = "glorot_uniform", 
                                       bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL, 
                                       activity_regularizer = NULL, kernel_constraint = NULL, bias_constraint = NULL) {
  call_layer(tf$contrib$keras$layers$LocallyConnected1D, x, list(
    filters = as.integer(filters),
    kernel_size = as_integer_tuple(kernel_size),
    strides = as_integer_tuple(strides),
    padding = padding,
    data_format = data_format,
    activation = activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    bias_initializer = bias_initializer,
    kernel_regularizer = kernel_regularizer,
    bias_regularizer = bias_regularizer,
    activity_regularizer = activity_regularizer,
    kernel_constraint = kernel_constraint,
    bias_constraint = bias_constraint
  ))
}



#' Locally-connected layer for 2D inputs.
#' 
#' `layer_locally_connected_2d` works similarly to [layer_conv_2d()], except
#' that weights are unshared, that is, a different set of filters is applied at
#' each different patch of the input.
#' 
#' @inheritParams layer_conv_2d
#' 
#' @param filters Integer, the dimensionality of the output space (i.e. the
#'   number output of filters in the convolution).
#' @param kernel_size An integer or list of 2 integers, specifying the width and
#'   height of the 2D convolution window. Can be a single integer to specify the
#'   same value for all spatial dimensions.
#' @param strides An integer or list of 2 integers, specifying the strides of
#'   the convolution along the width and height. Can be a single integer to
#'   specify the same value for all spatial dimensions. Specifying any stride
#'   value != 1 is incompatible with specifying any `dilation_rate` value != 1.
#' @param padding one of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, width, height,
#'   channels)` while `channels_first` corresponds to inputs with shape `(batch,
#'   channels, width, height)`. It defaults to the `image_data_format` value
#'   found in your Keras config file at `~/.keras/keras.json`. If you never set
#'   it, then it will be "channels_last".
#' @param activation Activation function to use. If you don't specify anything,
#'   no activation is applied (ie. "linear" activation: `a(x) = x`).
#' @param use_bias Boolean, whether the layer uses a bias vector.
#' @param kernel_initializer Initializer for the `kernel` weights matrix.
#' @param bias_initializer Initializer for the bias vector.
#' @param kernel_regularizer Regularizer function applied to the `kernel`
#'   weights matrix.
#' @param bias_regularizer Regularizer function applied to the bias vector.
#' @param activity_regularizer Regularizer function applied to the output of the
#'   layer (its "activation")..
#' @param kernel_constraint Constraint function applied to the kernel matrix.
#' @param bias_constraint Constraint function applied to the bias vector.
#'   
#' @section Input shape: 4D tensor with shape: `(samples, channels, rows, cols)`
#'   if data_format='channels_first' or 4D tensor with shape: `(samples, rows,
#'   cols, channels)` if data_format='channels_last'.
#'   
#' @section Output shape: 4D tensor with shape: `(samples, filters, new_rows,
#'   new_cols)` if data_format='channels_first' or 4D tensor with shape:
#'   `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
#'   `rows` and `cols` values might have changed due to padding.
#'   
#' @export
layer_locally_connected_2d <- function(x, filters, kernel_size, strides = c(1L, 1L), padding = "valid", data_format = NULL, 
                                       activation = NULL, use_bias = TRUE, kernel_initializer = "glorot_uniform", 
                                       bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL, 
                                       activity_regularizer = NULL, kernel_constraint = NULL, bias_constraint = NULL) {
  call_layer(tf$contrib$keras$layers$LocallyConnected2D, x, list(
    filters = as.integer(filters),
    kernel_size = as_integer_tuple(kernel_size),
    strides = as_integer_tuple(strides),
    padding = padding,
    data_format = data_format,
    activation = activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    bias_initializer = bias_initializer,
    kernel_regularizer = kernel_regularizer,
    bias_regularizer = bias_regularizer,
    activity_regularizer = activity_regularizer,
    kernel_constraint = kernel_constraint,
    bias_constraint = bias_constraint
  ))
}


