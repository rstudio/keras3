#' 1D convolution layer (e.g. temporal convolution).
#' 
#' This layer creates a convolution kernel that is convolved with the layer 
#' input over a single spatial (or temporal) dimension to produce a tensor of 
#' outputs. If `use_bias` is TRUE, a bias vector is created and added to the 
#' outputs. Finally, if `activation` is not `NULL`, it is applied to the outputs
#' as well. When using this layer as the first layer in a model, provide an 
#' `input_shape` argument (list of integers or `NULL `, e.g. `(10, 128)` for 
#' sequences of 10 vectors of 128-dimensional vectors, or `(NULL, 128)` for 
#' variable-length sequences of 128-dimensional vectors.
#' 
#' @inheritParams layer_dense
#' 
#' @param filters Integer, the dimensionality of the output space (i.e. the 
#'   number output of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying 
#'   the length of the 1D convolution window.
#' @param strides An integer or list of a single integer, specifying the 
#'   stride length of the convolution. Specifying any stride value != 1 is 
#'   incompatible with specifying any `dilation_rate` value != 1.
#' @param padding One of `"valid"`, `"causal"` or `"same"` (case-insensitive). 
#'   `"causal"` results in causal (dilated) convolutions, e.g. `output[t]` depends
#'   solely on `input[:t-1]`. Useful when modeling temporal data where the model 
#'   should not violate the temporal order. See [WaveNet: A Generative Model for
#'   Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).
#' @param dilation_rate an integer or list of a single integer, specifying 
#'   the dilation rate to use for dilated convolution. Currently, specifying any
#'   `dilation_rate` value != 1 is incompatible with specifying any `strides` 
#'   value != 1.
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
#' @param input_shape Dimensionality of the input (integer) not including the 
#'   samples axis. This argument is required when using this layer as the first 
#'   layer in a model.
#'   
#' @section Input shape: 3D tensor with shape: `(batch_size, steps, input_dim)`
#'   
#' @section Output shape: 3D tensor with shape: `(batch_size, new_steps,
#'   filters)` `steps` value might have changed due to padding or strides.
#'   
#' @export
layer_conv1d <- function(x, filters, kernel_size, strides = 1L, padding = "valid", 
                         dilation_rate = 1L, activation = NULL, use_bias = TRUE, 
                         kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                         kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                         kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL) {
  
  call_layer(tf$contrib$keras$layers$Conv1D, x, list(
    filters = as.integer(filters),
    kernel_size = as_integer_tuple(kernel_size),
    strides = as_integer_tuple(strides),
    padding = padding,
    dilation_rate = as_integer_tuple(dilation_rate),
    activation = activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    bias_initializer = bias_initializer,
    kernel_regularizer = kernel_regularizer,
    bias_regularizer = bias_regularizer,
    activity_regularizer = activity_regularizer,
    kernel_constraint = kernel_constraint,
    bias_constraint = bias_constraint,
    input_shape = normalize_shape(input_shape)
  ))
  
}


#' 2D convolution layer (e.g. spatial convolution over images).
#' 
#' This layer creates a convolution kernel that is convolved with the layer
#' input to produce a tensor of outputs. If `use_bias` is TRUE, a bias vector is
#' created and added to the outputs. Finally, if `activation` is not `NULL`, it
#' is applied to the outputs as well. When using this layer as the first layer
#' in a model, provide the keyword argument `input_shape` (list of integers,
#' does not include the sample axis), e.g. `input_shape=c(128, 128, 3)` for
#' 128x128 RGB pictures in `data_format="channels_last"`.
#' 
#' @inheritParams layer_conv1d  
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
#' @param dilation_rate an integer or list of 2 integers, specifying the
#'   dilation rate to use for dilated convolution. Can be a single integer to
#'   specify the same value for all spatial dimensions. Currently, specifying
#'   any `dilation_rate` value != 1 is incompatible with specifying any stride
#'   value != 1.
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
layer_conv2d <- function(x, filters, kernel_size, strides = c(1L, 1L), padding = "valid", data_format = NULL,
                         dilation_rate = c(1L, 1L), activation = NULL, use_bias = TRUE, 
                         kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                         kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                         kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL) {
  
  call_layer(tf$contrib$keras$layers$Conv2D, x, list(
    filters = as.integer(filters),
    kernel_size = as_integer_tuple(kernel_size),
    strides = as_integer_tuple(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = dilation_rate,
    activation = activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    bias_initializer = bias_initializer,
    kernel_regularizer = kernel_regularizer,
    bias_regularizer = bias_regularizer,
    activity_regularizer = activity_regularizer,
    kernel_constraint = kernel_constraint,
    bias_constraint = bias_constraint,
    input_shape = normalize_shape(input_shape)
  ))
  
}

#' 3D convolution layer (e.g. spatial convolution over volumes).
#' 
#' This layer creates a convolution kernel that is convolved with the layer
#' input to produce a tensor of outputs. If `use_bias` is TRUE, a bias vector is
#' created and added to the outputs. Finally, if `activation` is not `NULL`, it
#' is applied to the outputs as well. When using this layer as the first layer
#' in a model, provide the keyword argument `input_shape` (list of integers,
#' does not include the sample axis), e.g. `input_shape=c(128L, 128L, 128L, 3L)`
#' for 128x128x128 volumes with a single channel, in
#' `data_format="channels_last"`.
#' 
#' @inheritParams layer_conv1d  
#' 
#' @param filters Integer, the dimensionality of the output space (i.e. the
#'   number output of filters in the convolution).
#' @param kernel_size An integer or list of 3 integers, specifying the width and
#'   height of the 3D convolution window. Can be a single integer to specify the
#'   same value for all spatial dimensions.
#' @param strides An integer or list of 3 integers, specifying the strides of
#'   the convolution along each spatial dimension. Can be a single integer to
#'   specify the same value for all spatial dimensions. Specifying any stride
#'   value != 1 is incompatible with specifying any `dilation_rate` value != 1.
#' @param padding one of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, spatial_dim1,
#'   spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds
#'   to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2,
#'   spatial_dim3)`. It defaults to the `image_data_format` value found in your
#'   Keras config file at `~/.keras/keras.json`. If you never set it, then it
#'   will be "channels_last".
#' @param dilation_rate an integer or list of 3 integers, specifying the
#'   dilation rate to use for dilated convolution. Can be a single integer to
#'   specify the same value for all spatial dimensions. Currently, specifying
#'   any `dilation_rate` value != 1 is incompatible with specifying any stride
#'   value != 1.
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
#' @section Input shape: 5D tensor with shape: `(samples, channels, conv_dim1,
#'   conv_dim2, conv_dim3)` if data_format='channels_first' or 5D tensor with
#'   shape: `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
#'   data_format='channels_last'.
#'   
#' @section Output shape: 5D tensor with shape: `(samples, filters,
#'   new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
#'   data_format='channels_first' or 5D tensor with shape: `(samples,
#'   new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
#'   data_format='channels_last'. `new_conv_dim1`, `new_conv_dim2` and
#'   `new_conv_dim3` values might have changed due to padding.
#'   
#' @export
layer_conv3d <- function(x, filters, kernel_size, strides = c(1L, 1L, 1L), padding = "valid",
                         data_format = NULL, dilation_rate = c(1L, 1L, 1L), activation = NULL, use_bias = TRUE, 
                         kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                         kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                         kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL) {
  
  call_layer(tf$contrib$keras$layers$Conv3D, x, list(
    filters = as.integer(filters),
    kernel_size = as_integer_tuple(kernel_size),
    strides = as_integer_tuple(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = dilation_rate,
    activation = activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    bias_initializer = bias_initializer,
    kernel_regularizer = kernel_regularizer,
    bias_regularizer = bias_regularizer,
    activity_regularizer = activity_regularizer,
    kernel_constraint = kernel_constraint,
    bias_constraint = bias_constraint,
    input_shape = normalize_shape(input_shape)
  ))
  
}

#' Transposed convolution layer (sometimes called Deconvolution).
#' 
#' The need for transposed convolutions generally arises from the desire to use
#' a transformation going in the opposite direction of a normal convolution,
#' i.e., from something that has the shape of the output of some convolution to
#' something that has the shape of its input while maintaining a connectivity
#' pattern that is compatible with said convolution. When using this layer as
#' the first layer in a model, provide the keyword argument `input_shape` (list
#' of integers, does not include the sample axis), e.g. `input_shape=c(128L,
#' 128L, 3L)` for 128x128 RGB pictures in `data_format="channels_last"`.
#' 
#' @inheritParams layer_conv1d
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
#' @section Input shape: 4D tensor with shape: `(batch, channels, rows, cols)`
#'   if data_format='channels_first' or 4D tensor with shape: `(batch, rows,
#'   cols, channels)` if data_format='channels_last'.
#'   
#' @section Output shape: 4D tensor with shape: `(batch, filters, new_rows,
#'   new_cols)` if data_format='channels_first' or 4D tensor with shape:
#'   `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
#'   `rows` and `cols` values might have changed due to padding.
#'   
#' @section References: 
#'   - [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1) 
#'   - [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
#'   
#' @export
layer_conv2d_transpose <- function(x, filters, kernel_size, strides = c(1L, 1L), padding = "valid", 
                                   data_format = "channels_last", activation = NULL, use_bias = TRUE, 
                                   kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                                   kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                                   kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL) {
  
  call_layer(tf$contrib$keras$layers$Conv2DTranspose, x, list(
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
    bias_constraint = bias_constraint,
    input_shape = normalize_shape(input_shape)
  ))
  
}











