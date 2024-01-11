


#' 1D convolution layer (e.g. temporal convolution).
#'
#' @description
#' This layer creates a convolution kernel that is convolved with the layer
#' input over a single spatial (or temporal) dimension to produce a tensor of
#' outputs. If `use_bias` is `TRUE`, a bias vector is created and added to the
#' outputs. Finally, if `activation` is not `NULL`, it is applied to the
#' outputs as well.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     A 3D tensor with shape: `(batch_shape, steps, channels)`
#' - If `data_format="channels_first"`:
#'     A 3D tensor with shape: `(batch_shape, channels, steps)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     A 3D tensor with shape: `(batch_shape, new_steps, filters)`
#' - If `data_format="channels_first"`:
#'     A 3D tensor with shape: `(batch_shape, filters, new_steps)`
#'
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # Examples
#' ```{r}
#' # The inputs are 128-length vectors with 10 timesteps, and the
#' # batch size is 4.
#' x <- random_uniform(c(4, 10, 128))
#' y <- x |> layer_conv_1d(32, 3, activation='relu')
#' shape(y)
#' ```
#'
#' @returns
#' A 3D tensor representing `activation(conv1d(inputs, kernel) + bias)`.
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the convolution).
#'
#' @param kernel_size
#' int or list of 1 integer, specifying the size of the
#' convolution window.
#'
#' @param strides
#' int or list of 1 integer, specifying the stride length
#' of the convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, `"valid"`, `"same"` or `"causal"`(case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
#' `"causal"` results in causal (dilated) convolutions, e.g. `output[t]`
#' does not depend on`tail(input, t+1)`. Useful when modeling temporal data
#' where the model should not violate the temporal order.
#' See [WaveNet: A Generative Model for Raw Audio, section2.1](
#' https://arxiv.org/abs/1609.03499).
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
#' @param dilation_rate
#' int or list of 1 integers, specifying the dilation
#' rate to use for dilated convolution.
#'
#' @param groups
#' A positive int specifying the number of groups in which the
#' input is split along the channel axis. Each group is convolved
#' separately with `filters // groups` filters. The output is the
#' concatenation of all the `groups` results along the channel axis.
#' Input channels and `filters` must both be divisible by `groups`.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param kernel_initializer
#' Initializer for the convolution kernel. If `NULL`,
#' the default initializer (`"glorot_uniform"`) will be used.
#'
#' @param bias_initializer
#' Initializer for the bias vector. If `NULL`, the
#' default initializer (`"zeros"`) will be used.
#'
#' @param kernel_regularizer
#' Optional regularizer for the convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param kernel_constraint
#' Optional projection function to be applied to the
#' kernel after being updated by an `Optimizer` (e.g. used to implement
#' norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape). Constraints
#' are not safe to use when doing asynchronous distributed training.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/convolution1d#conv1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D>
#' @tether keras.layers.Conv1D
layer_conv_1d <-
function (object, filters, kernel_size, strides = 1L, padding = "valid",
    data_format = NULL, dilation_rate = 1L, groups = 1L, activation = NULL,
    use_bias = TRUE, kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        groups = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Conv1D, object, args)
}


#' 1D transposed convolution layer.
#'
#' @description
#' The need for transposed convolutions generally arise from the desire to use
#' a transformation going in the opposite direction of a normal convolution,
#' i.e., from something that has the shape of the output of some convolution
#' to something that has the shape of its input while maintaining a
#' connectivity pattern that is compatible with said convolution.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     A 3D tensor with shape: `(batch_shape, steps, channels)`
#' - If `data_format="channels_first"`:
#'     A 3D tensor with shape: `(batch_shape, channels, steps)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     A 3D tensor with shape: `(batch_shape, new_steps, filters)`
#' - If `data_format="channels_first"`:
#'     A 3D tensor with shape: `(batch_shape, filters, new_steps)`
#'
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # References
#' - [A guide to convolution arithmetic for deep learning](
#'     https://arxiv.org/abs/1603.07285v1)
#' - [Deconvolutional Networks](
#'     https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 128))
#' y <- x |> layer_conv_1d_transpose(32, 3, 2, activation='relu')
#' shape(y)
#' ```
#'
#' @returns
#' A 3D tensor representing
#' `activation(conv1d_transpose(inputs, kernel) + bias)`.
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the transpose convolution).
#'
#' @param kernel_size
#' int or list of 1 integer, specifying the size of the
#' transposed convolution window.
#'
#' @param strides
#' int or list of 1 integer, specifying the stride length
#' of the transposed convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
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
#' @param dilation_rate
#' int or list of 1 integers, specifying the dilation
#' rate to use for dilated transposed convolution.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param kernel_initializer
#' Initializer for the convolution kernel. If `NULL`,
#' the default initializer (`"glorot_uniform"`) will be used.
#'
#' @param bias_initializer
#' Initializer for the bias vector. If `NULL`, the
#' default initializer (`"zeros"`) will be used.
#'
#' @param kernel_regularizer
#' Optional regularizer for the convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param kernel_constraint
#' Optional projection function to be applied to the
#' kernel after being updated by an `Optimizer` (e.g. used to implement
#' norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape). Constraints
#' are not safe to use when doing asynchronous distributed training.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/convolution1d_transpose#conv1dtranspose-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1DTranspose>
#' @tether keras.layers.Conv1DTranspose
layer_conv_1d_transpose <-
function (object, filters, kernel_size, strides = 1L, padding = "valid",
    data_format = NULL, dilation_rate = 1L, activation = NULL,
    use_bias = TRUE, kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Conv1DTranspose, object, args)
}


#' 2D convolution layer.
#'
#' @description
#' This layer creates a convolution kernel that is convolved with the layer
#' input over a single spatial (or temporal) dimension to produce a tensor of
#' outputs. If `use_bias` is `TRUE`, a bias vector is created and added to the
#' outputs. Finally, if `activation` is not `NULL`, it is applied to the
#' outputs as well.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     A 4D tensor with shape: `(batch_size, height, width, channels)`
#' - If `data_format="channels_first"`:
#'     A 4D tensor with shape: `(batch_size, channels, height, width)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
#' - If `data_format="channels_first"`:
#'     A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`
#'
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 10, 128))
#' y <- x |> layer_conv_2d(32, 3, activation='relu')
#' shape(y)
#' ```
#'
#' @returns
#' A 4D tensor representing `activation(conv2d(inputs, kernel) + bias)`.
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the convolution).
#'
#' @param kernel_size
#' int or list of 2 integer, specifying the size of the
#' convolution window.
#'
#' @param strides
#' int or list of 2 integer, specifying the stride length
#' of the convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape
#' `(batch_size, channels, height, width)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch_size, channels, height, width)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param dilation_rate
#' int or list of 2 integers, specifying the dilation
#' rate to use for dilated convolution.
#'
#' @param groups
#' A positive int specifying the number of groups in which the
#' input is split along the channel axis. Each group is convolved
#' separately with `filters // groups` filters. The output is the
#' concatenation of all the `groups` results along the channel axis.
#' Input channels and `filters` must both be divisible by `groups`.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param kernel_initializer
#' Initializer for the convolution kernel. If `NULL`,
#' the default initializer (`"glorot_uniform"`) will be used.
#'
#' @param bias_initializer
#' Initializer for the bias vector. If `NULL`, the
#' default initializer (`"zeros"`) will be used.
#'
#' @param kernel_regularizer
#' Optional regularizer for the convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param kernel_constraint
#' Optional projection function to be applied to the
#' kernel after being updated by an `Optimizer` (e.g. used to implement
#' norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape). Constraints
#' are not safe to use when doing asynchronous distributed training.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/convolution2d#conv2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D>
#' @tether keras.layers.Conv2D
layer_conv_2d <-
function (object, filters, kernel_size, strides = list(1L, 1L),
    padding = "valid", data_format = NULL, dilation_rate = list(
        1L, 1L), groups = 1L, activation = NULL, use_bias = TRUE,
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        groups = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Conv2D, object, args)
}


#' 2D transposed convolution layer.
#'
#' @description
#' The need for transposed convolutions generally arise from the desire to use
#' a transformation going in the opposite direction of a normal convolution,
#' i.e., from something that has the shape of the output of some convolution
#' to something that has the shape of its input while maintaining a
#' connectivity pattern that is compatible with said convolution.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     A 4D tensor with shape: `(batch_size, height, width, channels)`
#' - If `data_format="channels_first"`:
#'     A 4D tensor with shape: `(batch_size, channels, height, width)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
#' - If `data_format="channels_first"`:
#'     A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`
#'
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # References
#' - [A guide to convolution arithmetic for deep learning](
#'     https://arxiv.org/abs/160307285v1)
#' - [Deconvolutional Networks](
#'     https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 8, 128))
#' y <- x |> layer_conv_2d_transpose(32, 2, 2, activation='relu')
#' shape(y)
#' # (4, 20, 16, 32)
#' ```
#'
#' @returns
#' A 4D tensor representing
#' `activation(conv2d_transpose(inputs, kernel) + bias)`.
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the transposed convolution).
#'
#' @param kernel_size
#' int or list of 1 integer, specifying the size of the
#' transposed convolution window.
#'
#' @param strides
#' int or list of 1 integer, specifying the stride length
#' of the transposed convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape
#' `(batch_size, channels, height, width)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch_size, channels, height, width)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param dilation_rate
#' int or list of 1 integers, specifying the dilation
#' rate to use for dilated transposed convolution.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param kernel_initializer
#' Initializer for the convolution kernel. If `NULL`,
#' the default initializer (`"glorot_uniform"`) will be used.
#'
#' @param bias_initializer
#' Initializer for the bias vector. If `NULL`, the
#' default initializer (`"zeros"`) will be used.
#'
#' @param kernel_regularizer
#' Optional regularizer for the convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param kernel_constraint
#' Optional projection function to be applied to the
#' kernel after being updated by an `Optimizer` (e.g. used to implement
#' norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape). Constraints
#' are not safe to use when doing asynchronous distributed training.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/convolution2d_transpose#conv2dtranspose-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose>
#' @tether keras.layers.Conv2DTranspose
layer_conv_2d_transpose <-
function (object, filters, kernel_size, strides = list(1L, 1L),
    padding = "valid", data_format = NULL, dilation_rate = list(
        1L, 1L), activation = NULL, use_bias = TRUE, kernel_initializer = "glorot_uniform",
    bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL,
    activity_regularizer = NULL, kernel_constraint = NULL, bias_constraint = NULL,
    ...)
{
    args <- capture_args(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Conv2DTranspose, object, args)
}


#' 3D convolution layer.
#'
#' @description
#' This layer creates a convolution kernel that is convolved with the layer
#' input over a single spatial (or temporal) dimension to produce a tensor of
#' outputs. If `use_bias` is `TRUE`, a bias vector is created and added to the
#' outputs. Finally, if `activation` is not `NULL`, it is applied to the
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
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 10, 10, 128))
#' y <- x |> layer_conv_3d(32, 3, activation = 'relu')
#' shape(y)
#' ```
#'
#' @returns
#' A 5D tensor representing `activation(conv3d(inputs, kernel) + bias)`.
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the convolution).
#'
#' @param kernel_size
#' int or list of 3 integer, specifying the size of the
#' convolution window.
#'
#' @param strides
#' int or list of 3 integer, specifying the stride length
#' of the convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape
#' `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
#' It defaults to the `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json`. If you never set it, then it
#' will be `"channels_last"`.
#'
#' @param dilation_rate
#' int or list of 3 integers, specifying the dilation
#' rate to use for dilated convolution.
#'
#' @param groups
#' A positive int specifying the number of groups in which the
#' input is split along the channel axis. Each group is convolved
#' separately with `filters %/% groups` filters. The output is the
#' concatenation of all the `groups` results along the channel axis.
#' Input channels and `filters` must both be divisible by `groups`.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param kernel_initializer
#' Initializer for the convolution kernel. If `NULL`,
#' the default initializer (`"glorot_uniform"`) will be used.
#'
#' @param bias_initializer
#' Initializer for the bias vector. If `NULL`, the
#' default initializer (`"zeros"`) will be used.
#'
#' @param kernel_regularizer
#' Optional regularizer for the convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param kernel_constraint
#' Optional projection function to be applied to the
#' kernel after being updated by an `Optimizer` (e.g. used to implement
#' norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape). Constraints
#' are not safe to use when doing asynchronous distributed training.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/convolution3d#conv3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D>
#' @tether keras.layers.Conv3D
layer_conv_3d <-
function (object, filters, kernel_size, strides = list(1L, 1L,
    1L), padding = "valid", data_format = NULL, dilation_rate = list(
    1L, 1L, 1L), groups = 1L, activation = NULL, use_bias = TRUE,
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        groups = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Conv3D, object, args)
}


#' 3D transposed convolution layer.
#'
#' @description
#' The need for transposed convolutions generally arise from the desire to use
#' a transformation going in the opposite direction of a normal convolution,
#' i.e., from something that has the shape of the output of some convolution
#' to something that has the shape of its input while maintaining a
#' connectivity pattern that is compatible with said convolution.
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
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # References
#' - [A guide to convolution arithmetic for deep learning](
#'     https://arxiv.org/abs/1603.07285v1)
#' - [Deconvolutional Networks](
#'     https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 8, 12, 128))
#' y <- x |> layer_conv_3d_transpose(32, 2, 2, activation = 'relu')
#' shape(y)
#' ```
#'
#' @returns
#' A 5D tensor representing `activation(conv3d(inputs, kernel) + bias)`.
#'
#' @param filters
#' int, the dimension of the output space (the number of filters
#' in the transposed convolution).
#'
#' @param kernel_size
#' int or list of 1 integer, specifying the size of the
#' transposed convolution window.
#'
#' @param strides
#' int or list of 1 integer, specifying the stride length
#' of the transposed convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape
#' `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
#' It defaults to the `image_data_format` value found in your Keras
#' config file at `~/.keras/keras.json`. If you never set it, then it
#' will be `"channels_last"`.
#'
#' @param dilation_rate
#' int or list of 1 integers, specifying the dilation
#' rate to use for dilated transposed convolution.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param kernel_initializer
#' Initializer for the convolution kernel. If `NULL`,
#' the default initializer (`"glorot_uniform"`) will be used.
#'
#' @param bias_initializer
#' Initializer for the bias vector. If `NULL`, the
#' default initializer (`"zeros"`) will be used.
#'
#' @param kernel_regularizer
#' Optional regularizer for the convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param kernel_constraint
#' Optional projection function to be applied to the
#' kernel after being updated by an `Optimizer` (e.g. used to implement
#' norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape). Constraints
#' are not safe to use when doing asynchronous distributed training.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/convolution3d_transpose#conv3dtranspose-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3DTranspose>
#' @tether keras.layers.Conv3DTranspose
layer_conv_3d_transpose <-
function (object, filters, kernel_size, strides = list(1L, 1L,
    1L), padding = "valid", data_format = NULL, dilation_rate = list(
    1L, 1L, 1L), activation = NULL, use_bias = TRUE, kernel_initializer = "glorot_uniform",
    bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL,
    activity_regularizer = NULL, kernel_constraint = NULL, bias_constraint = NULL,
    ...)
{
    args <- capture_args(list(filters = as_integer, kernel_size = as_integer_tuple,
        strides = as_integer_tuple, dilation_rate = as_integer_tuple,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Conv3DTranspose, object, args)
}


#' 1D depthwise convolution layer.
#'
#' @description
#' Depthwise convolution is a type of convolution in which each input channel
#' is convolved with a different kernel (called a depthwise kernel). You can
#' understand depthwise convolution as the first step in a depthwise separable
#' convolution.
#'
#' It is implemented via the following steps:
#'
#' - Split the input into individual channels.
#' - Convolve each channel with an individual depthwise kernel with
#'   `depth_multiplier` output channels.
#' - Concatenate the convolved outputs along the channels axis.
#'
#' Unlike a regular 1D convolution, depthwise convolution does not mix
#' information across different input channels.
#'
#' The `depth_multiplier` argument determines how many filters are applied to
#' one input channel. As such, it controls the amount of output channels that
#' are generated per input channel in the depthwise step.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     A 3D tensor with shape: `(batch_shape, steps, channels)`
#' - If `data_format="channels_first"`:
#'     A 3D tensor with shape: `(batch_shape, channels, steps)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     A 3D tensor with shape:
#'     `(batch_shape, new_steps, channels * depth_multiplier)`
#' - If `data_format="channels_first"`:
#'     A 3D tensor with shape:
#'     `(batch_shape, channels * depth_multiplier, new_steps)`
#'
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 12))
#' y <- x |> layer_depthwise_conv_1d(
#'   kernel_size = 3,
#'   depth_multiplier = 3,
#'   activation = 'relu'
#' )
#' shape(y)
#' ```
#'
#' @returns
#' A 3D tensor representing
#' `activation(depthwise_conv1d(inputs, kernel) + bias)`.
#'
#' @param kernel_size
#' int or list of 1 integer, specifying the size of the
#' depthwise convolution window.
#'
#' @param strides
#' int or list of 1 integer, specifying the stride length
#' of the convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
#'
#' @param depth_multiplier
#' The number of depthwise convolution output channels
#' for each input channel. The total number of depthwise convolution
#' output channels will be equal to `input_channel * depth_multiplier`.
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
#' @param dilation_rate
#' int or list of 1 integers, specifying the dilation
#' rate to use for dilated convolution.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param depthwise_initializer
#' Initializer for the convolution kernel.
#' If `NULL`, the default initializer (`"glorot_uniform"`)
#' will be used.
#'
#' @param bias_initializer
#' Initializer for the bias vector. If `NULL`, the
#' default initializer (`"zeros"`) will be used.
#'
#' @param depthwise_regularizer
#' Optional regularizer for the convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param depthwise_constraint
#' Optional projection function to be applied to the
#' kernel after being updated by an `Optimizer` (e.g. used to implement
#' norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape). Constraints
#' are not safe to use when doing asynchronous distributed training.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv1D>
#' @tether keras.layers.DepthwiseConv1D
layer_depthwise_conv_1d <-
function (object, kernel_size, strides = 1L, padding = "valid",
    depth_multiplier = 1L, data_format = NULL, dilation_rate = 1L,
    activation = NULL, use_bias = TRUE, depthwise_initializer = "glorot_uniform",
    bias_initializer = "zeros", depthwise_regularizer = NULL,
    bias_regularizer = NULL, activity_regularizer = NULL, depthwise_constraint = NULL,
    bias_constraint = NULL, ...)
{
    args <- capture_args(list(kernel_size = as_integer, strides = as_integer,
        depth_multiplier = as_integer, dilation_rate = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$DepthwiseConv1D, object, args)
}


#' 2D depthwise convolution layer.
#'
#' @description
#' Depthwise convolution is a type of convolution in which each input channel
#' is convolved with a different kernel (called a depthwise kernel). You can
#' understand depthwise convolution as the first step in a depthwise separable
#' convolution.
#'
#' It is implemented via the following steps:
#'
#' - Split the input into individual channels.
#' - Convolve each channel with an individual depthwise kernel with
#'   `depth_multiplier` output channels.
#' - Concatenate the convolved outputs along the channels axis.
#'
#' Unlike a regular 2D convolution, depthwise convolution does not mix
#' information across different input channels.
#'
#' The `depth_multiplier` argument determines how many filters are applied to
#' one input channel. As such, it controls the amount of output channels that
#' are generated per input channel in the depthwise step.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     A 4D tensor with shape: `(batch_size, height, width, channels)`
#' - If `data_format="channels_first"`:
#'     A 4D tensor with shape: `(batch_size, channels, height, width)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     A 4D tensor with shape:
#'     `(batch_size, new_height, new_width, channels * depth_multiplier)`
#' - If `data_format="channels_first"`:
#'     A 4D tensor with shape:
#'     `(batch_size, channels * depth_multiplier, new_height, new_width)`
#'
#' # Raises
#' ValueError: when both `strides > 1` and `dilation_rate > 1`.
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 10, 12))
#' y <- x |> layer_depthwise_conv_2d(3, 3, activation = 'relu')
#' shape(y)
#' ```
#'
#' @returns
#' A 4D tensor representing
#' `activation(depthwise_conv2d(inputs, kernel) + bias)`.
#'
#' @param kernel_size
#' int or list of 2 integer, specifying the size of the
#' depthwise convolution window.
#'
#' @param strides
#' int or list of 2 integer, specifying the stride length
#' of the depthwise convolution. `strides > 1` is incompatible with
#' `dilation_rate > 1`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
#'
#' @param depth_multiplier
#' The number of depthwise convolution output channels
#' for each input channel. The total number of depthwise convolution
#' output channels will be equal to `input_channel * depth_multiplier`.
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
#' @param dilation_rate
#' int or list of 2 integers, specifying the dilation
#' rate to use for dilated convolution.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param depthwise_initializer
#' Initializer for the convolution kernel.
#' If `NULL`, the default initializer (`"glorot_uniform"`)
#' will be used.
#'
#' @param bias_initializer
#' Initializer for the bias vector. If `NULL`, the
#' default initializer (`"zeros"`) will be used.
#'
#' @param depthwise_regularizer
#' Optional regularizer for the convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param depthwise_constraint
#' Optional projection function to be applied to the
#' kernel after being updated by an `Optimizer` (e.g. used to implement
#' norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape). Constraints
#' are not safe to use when doing asynchronous distributed training.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/depthwise_convolution2d#depthwiseconv2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D>
#' @tether keras.layers.DepthwiseConv2D
layer_depthwise_conv_2d <-
function (object, kernel_size, strides = list(1L, 1L), padding = "valid",
    depth_multiplier = 1L, data_format = NULL, dilation_rate = list(
        1L, 1L), activation = NULL, use_bias = TRUE, depthwise_initializer = "glorot_uniform",
    bias_initializer = "zeros", depthwise_regularizer = NULL,
    bias_regularizer = NULL, activity_regularizer = NULL, depthwise_constraint = NULL,
    bias_constraint = NULL, ...)
{
    args <- capture_args(list(kernel_size = as_integer, strides = as_integer,
        depth_multiplier = as_integer, dilation_rate = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$DepthwiseConv2D, object, args)
}


#' 1D separable convolution layer.
#'
#' @description
#' This layer performs a depthwise convolution that acts separately on
#' channels, followed by a pointwise convolution that mixes channels.
#' If `use_bias` is TRUE and a bias initializer is provided,
#' it adds a bias vector to the output. It then optionally applies an
#' activation function to produce the final output.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     A 3D tensor with shape: `(batch_shape, steps, channels)`
#' - If `data_format="channels_first"`:
#'     A 3D tensor with shape: `(batch_shape, channels, steps)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     A 3D tensor with shape: `(batch_shape, new_steps, filters)`
#' - If `data_format="channels_first"`:
#'     A 3D tensor with shape: `(batch_shape, filters, new_steps)`
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 12))
#' y <- layer_separable_conv_1d(x, 3, 2, 2, activation='relu')
#' shape(y)
#' ```
#'
#' @returns
#' A 3D tensor representing
#' `activation(separable_conv1d(inputs, kernel) + bias)`.
#'
#' @param filters
#' int, the dimensionality of the output space (i.e. the number
#' of filters in the pointwise convolution).
#'
#' @param kernel_size
#' int or list of 1 integers, specifying the size of the
#' depthwise convolution window.
#'
#' @param strides
#' int or list of 1 integers, specifying the stride length
#' of the depthwise convolution. If only one int is specified, the same
#' stride size will be used for all dimensions. `strides > 1` is
#' incompatible with `dilation_rate > 1`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
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
#' @param dilation_rate
#' int or list of 1 integers, specifying the dilation
#' rate to use for dilated convolution. If only one int is specified,
#' the same dilation rate will be used for all dimensions.
#'
#' @param depth_multiplier
#' The number of depthwise convolution output channels
#' for each input channel. The total number of depthwise convolution
#' output channels will be equal to `input_channel * depth_multiplier`.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param depthwise_initializer
#' An initializer for the depthwise convolution
#' kernel. If NULL, then the default initializer (`"glorot_uniform"`)
#' will be used.
#'
#' @param pointwise_initializer
#' An initializer for the pointwise convolution
#' kernel. If NULL, then the default initializer (`"glorot_uniform"`)
#' will be used.
#'
#' @param bias_initializer
#' An initializer for the bias vector. If NULL, the
#' default initializer ('"zeros"') will be used.
#'
#' @param depthwise_regularizer
#' Optional regularizer for the depthwise
#' convolution kernel.
#'
#' @param pointwise_regularizer
#' Optional regularizer for the pointwise
#' convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param depthwise_constraint
#' Optional projection function to be applied to the
#' depthwise kernel after being updated by an `Optimizer` (e.g. used
#' for norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape).
#'
#' @param pointwise_constraint
#' Optional projection function to be applied to the
#' pointwise kernel after being updated by an `Optimizer`.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/separable_convolution1d#separableconv1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv1D>
#'
#' @tether keras.layers.SeparableConv1D
layer_separable_conv_1d <-
function (object, filters, kernel_size, strides = 1L, padding = "valid",
    data_format = NULL, dilation_rate = 1L, depth_multiplier = 1L,
    activation = NULL, use_bias = TRUE, depthwise_initializer = "glorot_uniform",
    pointwise_initializer = "glorot_uniform", bias_initializer = "zeros",
    depthwise_regularizer = NULL, pointwise_regularizer = NULL,
    bias_regularizer = NULL, activity_regularizer = NULL, depthwise_constraint = NULL,
    pointwise_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args(list(filters = as_integer, kernel_size = as_integer,
        strides = as_integer, dilation_rate = as_integer, depth_multiplier = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$SeparableConv1D, object, args)
}


#' 2D separable convolution layer.
#'
#' @description
#' This layer performs a depthwise convolution that acts separately on
#' channels, followed by a pointwise convolution that mixes channels.
#' If `use_bias` is TRUE and a bias initializer is provided,
#' it adds a bias vector to the output. It then optionally applies an
#' activation function to produce the final output.
#'
#' # Input Shape
#' - If `data_format="channels_last"`:
#'     A 4D tensor with shape: `(batch_size, height, width, channels)`
#' - If `data_format="channels_first"`:
#'     A 4D tensor with shape: `(batch_size, channels, height, width)`
#'
#' # Output Shape
#' - If `data_format="channels_last"`:
#'     A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
#' - If `data_format="channels_first"`:
#'     A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(4, 10, 10, 12))
#' y <- layer_separable_conv_2d(x, 3, c(4, 3), 2, activation='relu')
#' shape(y)
#' ```
#'
#' @returns
#' A 4D tensor representing
#' `activation(separable_conv2d(inputs, kernel) + bias)`.
#'
#' @param filters
#' int, the dimensionality of the output space (i.e. the number
#' of filters in the pointwise convolution).
#'
#' @param kernel_size
#' int or list of 2 integers, specifying the size of the
#' depthwise convolution window.
#'
#' @param strides
#' int or list of 2 integers, specifying the stride length
#' of the depthwise convolution. If only one int is specified, the same
#' stride size will be used for all dimensions. `strides > 1` is
#' incompatible with `dilation_rate > 1`.
#'
#' @param padding
#' string, either `"valid"` or `"same"` (case-insensitive).
#' `"valid"` means no padding. `"same"` results in padding evenly to
#' the left/right or up/down of the input. When `padding="same"` and
#' `strides=1`, the output has the same size as the input.
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
#' @param dilation_rate
#' int or list of 2 integers, specifying the dilation
#' rate to use for dilated convolution. If only one int is specified,
#' the same dilation rate will be used for all dimensions.
#'
#' @param depth_multiplier
#' The number of depthwise convolution output channels
#' for each input channel. The total number of depthwise convolution
#' output channels will be equal to `input_channel * depth_multiplier`.
#'
#' @param activation
#' Activation function. If `NULL`, no activation is applied.
#'
#' @param use_bias
#' bool, if `TRUE`, bias will be added to the output.
#'
#' @param depthwise_initializer
#' An initializer for the depthwise convolution
#' kernel. If NULL, then the default initializer (`"glorot_uniform"`)
#' will be used.
#'
#' @param pointwise_initializer
#' An initializer for the pointwise convolution
#' kernel. If NULL, then the default initializer (`"glorot_uniform"`)
#' will be used.
#'
#' @param bias_initializer
#' An initializer for the bias vector. If NULL, the
#' default initializer ('"zeros"') will be used.
#'
#' @param depthwise_regularizer
#' Optional regularizer for the depthwise
#' convolution kernel.
#'
#' @param pointwise_regularizer
#' Optional regularizer for the pointwise
#' convolution kernel.
#'
#' @param bias_regularizer
#' Optional regularizer for the bias vector.
#'
#' @param activity_regularizer
#' Optional regularizer function for the output.
#'
#' @param depthwise_constraint
#' Optional projection function to be applied to the
#' depthwise kernel after being updated by an `Optimizer` (e.g. used
#' for norm constraints or value constraints for layer weights). The
#' function must take as input the unprojected variable and must return
#' the projected variable (which must have the same shape).
#'
#' @param pointwise_constraint
#' Optional projection function to be applied to the
#' pointwise kernel after being updated by an `Optimizer`.
#'
#' @param bias_constraint
#' Optional projection function to be applied to the
#' bias after being updated by an `Optimizer`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family convolutional layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/convolution_layers/separable_convolution2d#separableconv2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv2D>
#' @tether keras.layers.SeparableConv2D
layer_separable_conv_2d <-
function (object, filters, kernel_size, strides = list(1L, 1L),
    padding = "valid", data_format = NULL, dilation_rate = list(
        1L, 1L), depth_multiplier = 1L, activation = NULL, use_bias = TRUE,
    depthwise_initializer = "glorot_uniform", pointwise_initializer = "glorot_uniform",
    bias_initializer = "zeros", depthwise_regularizer = NULL,
    pointwise_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    depthwise_constraint = NULL, pointwise_constraint = NULL,
    bias_constraint = NULL, ...)
{
    args <- capture_args(list(filters = as_integer, kernel_size = as_integer,
        strides = as_integer, dilation_rate = as_integer, depth_multiplier = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$SeparableConv2D, object, args)
}
