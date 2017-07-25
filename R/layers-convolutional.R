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
#' @param kernel_size An integer or list of a single integer, specifying the
#'   length of the 1D convolution window.
#' @param strides An integer or list of a single integer, specifying the stride
#'   length of the convolution. Specifying any stride value != 1 is incompatible
#'   with specifying any `dilation_rate` value != 1.
#' @param padding One of `"valid"`, `"causal"` or `"same"` (case-insensitive). 
#'   `"causal"` results in causal (dilated) convolutions, e.g. `output[t]` does
#'   not depend on `input[t+1:]`. Useful when modeling temporal data where the
#'   model should not violate the temporal order. See [WaveNet: A Generative
#'   Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).
#' @param dilation_rate an integer or list of a single integer, specifying the
#'   dilation rate to use for dilated convolution. Currently, specifying any 
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
#'   
#' @section Input shape: 3D tensor with shape: `(batch_size, steps, input_dim)`
#'   
#' @section Output shape: 3D tensor with shape: `(batch_size, new_steps, 
#'   filters)` `steps` value might have changed due to padding or strides.
#'   
#' @family convolutional layers
#'   
#' @export
layer_conv_1d <- function(object, filters, kernel_size, strides = 1L, padding = "valid", 
                          dilation_rate = 1L, activation = NULL, use_bias = TRUE, 
                          kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                          kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                          kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL,
                          batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                          name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$Conv1D, object, list(
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
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
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
#' @inheritParams layer_conv_1d  
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
#'   `channels_last` corresponds to inputs with shape `(batch, height, width,
#'   channels)` while `channels_first` corresponds to inputs with shape `(batch,
#'   channels, height, width)`. It defaults to the `image_data_format` value
#'   found in your Keras config file at `~/.keras/keras.json`. If you never set
#'   it, then it will be "channels_last".
#' @param dilation_rate an integer or list of 2 integers, specifying the
#'   dilation rate to use for dilated convolution. Can be a single integer to
#'   specify the same value for all spatial dimensions. Currently, specifying
#'   any `dilation_rate` value != 1 is incompatible with specifying any stride
#'   value != 1.
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
#' @family convolutional layers 
#'    
#' @export
layer_conv_2d <- function(object, filters, kernel_size, strides = c(1L, 1L), padding = "valid", data_format = NULL,
                          dilation_rate = c(1L, 1L), activation = NULL, use_bias = TRUE, 
                          kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                          kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                          kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL,
                          batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                          name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$Conv2D, object, list(
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
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
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
#' @inheritParams layer_conv_2d  
#' 
#' @param filters Integer, the dimensionality of the output space (i.e. the
#'   number output of filters in the convolution).
#' @param kernel_size An integer or list of 3 integers, specifying the depth,
#'   height, and width of the 3D convolution window. Can be a single integer 
#'   to specify the same value for all spatial dimensions.
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
#' @family convolutional layers 
#'   
#' @export
layer_conv_3d <- function(object, filters, kernel_size, strides = c(1L, 1L, 1L), padding = "valid",
                          data_format = NULL, dilation_rate = c(1L, 1L, 1L), activation = NULL, use_bias = TRUE, 
                          kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                          kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                          kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL,
                          batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                          name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$Conv3D, object, list(
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
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}

#' Transposed 2D convolution layer (sometimes called Deconvolution).
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
#' @inheritParams layer_conv_2d
#' 
#' @param filters Integer, the dimensionality of the output space (i.e. the
#'   number of output filters in the convolution).
#' @param kernel_size An integer or list of 2 integers, specifying the width and
#'   height of the 2D convolution window. Can be a single integer to specify the
#'   same value for all spatial dimensions.
#' @param strides An integer or list of 2 integers, specifying the strides of
#'   the convolution along the width and height. Can be a single integer to
#'   specify the same value for all spatial dimensions. Specifying any stride
#'   value != 1 is incompatible with specifying any `dilation_rate` value != 1.
#' @param padding one of `"valid"` or `"same"` (case-insensitive).
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
#'   - [Deconvolutional Networks](https://www.uoguelph.ca/~gwtaylor/publications/mattcvpr2010/deconvolutionalnets.pdf)
#'   
#' @family convolutional layers    
#'   
#' @export
layer_conv_2d_transpose <- function(object, filters, kernel_size, strides = c(1L, 1L), padding = "valid", 
                                    data_format = NULL, activation = NULL, use_bias = TRUE, 
                                    kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                                    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                                    kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL,
                                    batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                                    name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$Conv2DTranspose, object, list(
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
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}


#' Transposed 3D convolution layer (sometimes called Deconvolution).
#'
#' The need for transposed convolutions generally arises from the desire to use
#' a transformation going in the opposite direction of a normal convolution,
#' i.e., from something that has the shape of the output of some convolution to
#' something that has the shape of its input while maintaining a connectivity
#' pattern that is compatible with said convolution. 
#' 
#' When using this layer as the first layer in a model, provide the keyword argument 
#' `input_shape` (list of integers, does not include the sample axis), e.g. 
#' `input_shape = list(128, 128, 128, 3)` for a 128x128x128 volume with 3 channels if
#' `data_format="channels_last"`. 
#'
#' @inheritParams layer_conv_2d
#'
#' @param filters Integer, the dimensionality of the output space (i.e. the
#'   number of output filters in the convolution).
#' @param kernel_size An integer or list of 3 integers, specifying the width and
#'   height of the 3D convolution window. Can be a single integer to specify the
#'   same value for all spatial dimensions.
#' @param strides An integer or list of 3 integers, specifying the strides of
#'   the convolution along the width and height. Can be a single integer to
#'   specify the same value for all spatial dimensions. Specifying any stride
#'   value != 1 is incompatible with specifying any `dilation_rate` value != 1.
#' @param padding one of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, depth, height,
#'   width, channels)` while `channels_first` corresponds to inputs with shape
#'   `(batch, channels, depth, height, width)`. It defaults to the
#'   `image_data_format` value found in your Keras config file at
#'   `~/.keras/keras.json`. If you never set it, then it will be
#'   "channels_last".
#' @param activation Activation function to use. If you don't specify anything, no
#'   activation is applied (ie. "linear" activation: `a(x) = x`).
#' @param use_bias Boolean, whether the layer uses a bias vector.
#' @param kernel_initializer Initializer for the `kernel` weights matrix.
#' @param bias_initializer Initializer for the bias vector.
#' @param kernel_regularizer Regularizer function applied to the `kernel`
#'   weights matrix,
#' @param bias_regularizer Regularizer function applied to the bias vector.
#' @param activity_regularizer Regularizer function applied to the output of the
#'   layer (its "activation").
#' @param kernel_constraint Constraint function applied to the kernel matrix.
#' @param bias_constraint Constraint function applied to the bias vector.
#'
#' @section References:
#'   - [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
#'   - [Deconvolutional Networks](https://www.uoguelph.ca/~gwtaylor/publications/mattcvpr2010/deconvolutionalnets.pdf)
#'
#' @family convolutional layers 
#'
#' @export
layer_conv_3d_transpose <- function(object, filters, kernel_size, strides = c(1, 1, 1), padding = "valid", 
                                    data_format = NULL, activation = NULL, use_bias = TRUE, 
                                    kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
                                    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                                    kernel_constraint = NULL, bias_constraint = NULL, input_shape = NULL,
                                    batch_input_shape = NULL, batch_size = NULL, dtype = NULL, 
                                    name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$Conv3DTranspose, object, list(
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
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
}



#' Depthwise separable 2D convolution.
#' 
#' Separable convolutions consist in first performing a depthwise spatial
#' convolution (which acts on each input channel separately) followed by a
#' pointwise convolution which mixes together the resulting output channels. The
#' `depth_multiplier` argument controls how many output channels are generated
#' per input channel in the depthwise step. Intuitively, separable convolutions
#' can be understood as a way to factorize a convolution kernel into two smaller
#' kernels, or as an extreme version of an Inception block.
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
#' @param depth_multiplier The number of depthwise convolution output channels
#'   for each input channel. The total number of depthwise convolution output
#'   channels will be equal to `filterss_in * depth_multiplier`.
#' @param depthwise_initializer Initializer for the depthwise kernel matrix.
#' @param pointwise_initializer Initializer for the pointwise kernel matrix.
#' @param depthwise_regularizer Regularizer function applied to the depthwise
#'   kernel matrix.
#' @param pointwise_regularizer Regularizer function applied to the depthwise
#'   kernel matrix.
#' @param depthwise_constraint Constraint function applied to the depthwise
#'   kernel matrix.
#' @param pointwise_constraint Constraint function applied to the pointwise
#'   kernel matrix.
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
#' @family convolutional layers 
#'   
#' @export
layer_separable_conv_2d <- function(object, filters, kernel_size, strides = c(1L, 1L), padding = "valid", data_format = NULL, 
                                    depth_multiplier = 1L, activation = NULL, use_bias = TRUE, 
                                    depthwise_initializer = "glorot_uniform", pointwise_initializer = "glorot_uniform", bias_initializer = "zeros", 
                                    depthwise_regularizer = NULL, pointwise_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
                                    depthwise_constraint = NULL, pointwise_constraint = NULL, bias_constraint = NULL,
                                    batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$SeparableConv2D, object, list(
    filters = as.integer(filters),
    kernel_size = as_integer_tuple(kernel_size),
    strides = as_integer_tuple(strides),
    padding = padding,
    data_format = data_format,
    depth_multiplier = as.integer(depth_multiplier),
    activation = activation,
    use_bias = use_bias,
    depthwise_initializer = depthwise_initializer,
    pointwise_initializer = pointwise_initializer,
    bias_initializer = bias_initializer,
    depthwise_regularizer = depthwise_regularizer,
    pointwise_regularizer = pointwise_regularizer,
    bias_regularizer = bias_regularizer,
    activity_regularizer = activity_regularizer,
    depthwise_constraint = depthwise_constraint,
    pointwise_constraint = pointwise_constraint,
    bias_constraint = bias_constraint,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}


#' Upsampling layer for 1D inputs.
#' 
#' Repeats each temporal step `size` times along the time axis.
#' 
#' @inheritParams layer_dense
#' 
#' @param size integer. Upsampling factor.
#'   
#' @section Input shape: 3D tensor with shape: `(batch, steps, features)`.
#'   
#' @section Output shape: 3D tensor with shape: `(batch, upsampled_steps,
#'   features)`.
#' 
#' @family convolutional layers 
#'         
#' @export
layer_upsampling_1d <- function(object, size = 2L,
                                batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$UpSampling1D, object, list(
    size = as.integer(size),
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}


#' Upsampling layer for 2D inputs.
#' 
#' Repeats the rows and columns of the data by `size[[0]]` and `size[[1]]` respectively.
#' 
#' @inheritParams layer_conv_2d
#' 
#' @param size int, or list of 2 integers. The upsampling factors for rows and
#'   columns.
#'   
#' @section Input shape: 
#' 4D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, rows, cols, channels)` 
#' - If `data_format` is `"channels_first"`: `(batch, channels, rows, cols)`
#'   
#' @section Output shape: 
#' 4D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, upsampled_rows, upsampled_cols, channels)` 
#' - If `data_format` is `"channels_first"`: `(batch, channels, upsampled_rows, upsampled_cols)`
#'   
#' @family convolutional layers 
#'   
#' @export
layer_upsampling_2d <- function(object, size = c(2L, 2L), data_format = NULL,
                                batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$UpSampling2D, object, list(
    size = as.integer(size),
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}


#' Upsampling layer for 3D inputs.
#' 
#' Repeats the 1st, 2nd and 3rd dimensions of the data by `size[[0]]`, `size[[1]]` and
#' `size[[2]]` respectively.
#' 
#' @inheritParams layer_upsampling_1d
#'   
#' @param size int, or list of 3 integers. The upsampling factors for dim1, dim2
#'   and dim3.
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
#' 5D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, dim1, dim2, dim3, channels)` 
#' - If `data_format` is `"channels_first"`: `(batch, channels, dim1, dim2, dim3)`
#'   
#' @section Output shape: 
#' 5D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)` 
#' - If `data_format` is `"channels_first"`: `(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
#'
#' @family convolutional layers 
#'         
#' @export
layer_upsampling_3d <- function(object, size= c(2L, 2L, 2L), data_format = NULL,
                                batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$UpSampling3D, object, list(
    size = as.integer(size),
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}

#' Zero-padding layer for 1D input (e.g. temporal sequence).
#'
#' @inheritParams layer_conv_2d
#'  
#' @param padding int, or list of int (length 2)
#' - If int: How many zeros to add at the beginning and end of the padding dimension (axis 1). 
#' - If list of int (length 2): How many zeros to add at the beginning and at the end of the padding dimension (`(left_pad, right_pad)`).
#'
#' @section Input shape:
#' 3D tensor with shape `(batch, axis_to_pad, features)`
#' 
#' @section Output shape:
#' 3D tensor with shape `(batch, padded_axis, features)`
#'
#' @family convolutional layers 
#'
#' @export
layer_zero_padding_1d <- function(object, padding = 1L,
                                  batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$ZeroPadding1D, object, list(
    padding = as.integer(padding),
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
}


#' Zero-padding layer for 2D input (e.g. picture).
#' 
#' This layer can add rows and columns of zeros at the top, bottom, left and
#' right side of an image tensor.
#' 
#' @inheritParams layer_conv_2d
#' @inheritParams layer_zero_padding_1d
#' 
#' @param padding int, or list of 2 ints, or list of 2 lists of 2 ints. 
#' - If int: the same symmetric padding is applied to width and height. 
#' - If list of 2 ints: interpreted as two different symmetric padding values for height
#'   and width: `(symmetric_height_pad, symmetric_width_pad)`. 
#' - If list of 2 lists of 2 ints: interpreted as `((top_pad, bottom_pad), (left_pad,
#'   right_pad))`
#'   
#' @section Input shape: 4D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, rows, cols, channels)` 
#' - If `data_format` is `"channels_first"`: `(batch, channels, rows, cols)`
#'   
#' @section Output shape: 4D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, padded_rows, padded_cols, channels)` 
#' - If `data_format` is `"channels_first"`: `(batch, channels, padded_rows, padded_cols)`
#' 
#' @family convolutional layers 
#'       
#' @export
layer_zero_padding_2d <- function(object, padding = c(1L, 1L), data_format = NULL,
                                  batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {

  create_layer(keras$layers$ZeroPadding2D, object, list(
    padding = normalize_padding(padding, 2L),
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}

#' Zero-padding layer for 3D data (spatial or spatio-temporal).
#' 
#' @inheritParams layer_zero_padding_1d
#' 
#' @param padding int, or list of 3 ints, or list of 3 lists of 2 ints. 
#' - If int: the same symmetric padding is applied to width and height. 
#' - If list of 3 ints: interpreted as three different symmetric padding values: 
#'   `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
#' - If list of 3 lists of 2 ints: interpreted as `((left_dim1_pad,
#'   right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad,
#'   right_dim3_pad))`
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, spatial_dim1,
#'   spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds
#'   to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2,
#'   spatial_dim3)`. It defaults to the `image_data_format` value found in your
#'   Keras config file at `~/.keras/keras.json`. If you never set it, then it
#'   will be "channels_last".
#'   
#' @section Input shape: 5D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, first_axis_to_pad, second_axis_to_pad,
#'   third_axis_to_pad, depth)` 
#' - If `data_format` is `"channels_first"`: `(batch, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`
#'   
#' @section Output shape: 5D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, first_padded_axis, second_padded_axis,
#'   third_axis_to_pad, depth)` 
#' - If `data_format` is `"channels_first"`: `(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`
#'
#' @family convolutional layers 
#'         
#' @export
layer_zero_padding_3d <- function(object,  padding = c(1L, 1L, 1L), data_format = NULL,
                                  batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$ZeroPadding3D, object, list(
    padding = normalize_padding(padding, 3L),
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}


#' Cropping layer for 1D input (e.g. temporal sequence).
#' 
#' It crops along the time dimension (axis 1).
#' 
#' @inheritParams layer_dense
#' 
#' @param cropping int or list of int (length 2) How many units should be
#'   trimmed off at the beginning and end of the cropping dimension (axis 1). If
#'   a single int is provided, the same value will be used for both.
#'   
#' @section Input shape: 3D tensor with shape `(batch, axis_to_crop, features)`
#'   
#' @section Output shape: 3D tensor with shape `(batch, cropped_axis, features)`
#' 
#' @family convolutional layers    
#'  
#' @export
layer_cropping_1d <- function(object, cropping = c(1L, 1L),
                              batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$Cropping1D, object, list(
    cropping = as.integer(cropping),
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
}


#' Cropping layer for 2D input (e.g. picture).
#' 
#' It crops along spatial dimensions, i.e. width and height.
#' 
#' @inheritParams layer_conv_2d
#' @inheritParams layer_cropping_1d
#' 
#' @param cropping int, or list of 2 ints, or list of 2 lists of 2 ints. 
#'   - If int: the same symmetric cropping is applied to width and height. 
#'   - If list of 2 ints: interpreted as two different symmetric cropping values for
#'   height and width: `(symmetric_height_crop, symmetric_width_crop)`. 
#'   - If list of 2 lists of 2 ints: interpreted as `((top_crop, bottom_crop), (left_crop,
#'   right_crop))`
#'   
#' @section Input shape: 4D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, rows, cols, channels)` 
#' - If `data_format` is `"channels_first"`: `(batch, channels, rows, cols)`
#'   
#' @section Output shape: 4D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, cropped_rows, cropped_cols, channels)` 
#' - If `data_format` is `"channels_first"`: `(batch, channels, cropped_rows, cropped_cols)`
#'
#' @family convolutional layers    
#'   
#' @export
layer_cropping_2d <- function(object, cropping = list(c(0L, 0L), c(0L, 0L)), data_format = NULL,
                              batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$Cropping2D, object, list(
    cropping = normalize_cropping(cropping, 2L),
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}


#' Cropping layer for 3D data (e.g. spatial or spatio-temporal).
#' 
#' @inheritParams layer_cropping_1d
#'  
#' @param cropping int, or list of 3 ints, or list of 3 lists of 2 ints. 
#' - If int: the same symmetric cropping is applied to width and height. 
#' - If list of 3 ints: interpreted as two different symmetric cropping values for
#'   height and width: `(symmetric_dim1_crop, symmetric_dim2_crop,
#'   symmetric_dim3_crop)`. 
#' - If list of 3 lists of 2 ints: interpreted as
#'   `((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop),
#'   (left_dim3_crop, right_dim3_crop))`
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, spatial_dim1,
#'   spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds
#'   to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2,
#'   spatial_dim3)`. It defaults to the `image_data_format` value found in your
#'   Keras config file at `~/.keras/keras.json`. If you never set it, then it
#'   will be "channels_last".
#'   
#' @section Input shape: 5D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, first_axis_to_crop, second_axis_to_crop,
#'   third_axis_to_crop, depth)` 
#' - If `data_format` is `"channels_first"`:
#'   `(batch, depth, first_axis_to_crop, second_axis_to_crop,
#'   third_axis_to_crop)`
#'   
#' @section Output shape: 5D tensor with shape: 
#' - If `data_format` is `"channels_last"`: `(batch, first_cropped_axis, second_cropped_axis,
#'   third_cropped_axis, depth)` 
#' - If `data_format` is `"channels_first"`: `(batch, depth, first_cropped_axis, second_cropped_axis,
#'   third_cropped_axis)`
#' 
#' @family convolutional layers 
#'       
#' @export
layer_cropping_3d <- function(object, cropping = list(c(1L, 1L), c(1L, 1L), c(1L, 1L)), data_format = NULL,
                              batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$Cropping3D, object, list(
    cropping = normalize_cropping(cropping, 3L),
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
}


#' Convolutional LSTM.
#' 
#' It is similar to an LSTM layer, but the input transformations and recurrent
#' transformations are both convolutional.
#' 
#' @inheritParams layer_conv_2d
#' 
#' @param filters Integer, the dimensionality of the output space (i.e. the
#'   number output of filters in the convolution).
#' @param kernel_size An integer or list of n integers, specifying the
#'   dimensions of the convolution window.
#' @param strides An integer or list of n integers, specifying the strides of
#'   the convolution. Specifying any stride value != 1 is incompatible with
#'   specifying any `dilation_rate` value != 1.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#'   `channels_first`. The ordering of the dimensions in the inputs.
#'   `channels_last` corresponds to inputs with shape `(batch, time, ...,
#'   channels)` while `channels_first` corresponds to inputs with shape `(batch,
#'   time, channels, ...)`. It defaults to the `image_data_format` value found
#'   in your Keras config file at `~/.keras/keras.json`. If you never set it,
#'   then it will be "channels_last".
#' @param dilation_rate An integer or list of n integers, specifying the
#'   dilation rate to use for dilated convolution. Currently, specifying any
#'   `dilation_rate` value != 1 is incompatible with specifying any `strides`
#'   value != 1.
#' @param activation Activation function to use. If you don't specify anything,
#'   no activation is applied (ie. "linear" activation: `a(x) = x`).
#' @param recurrent_activation Activation function to use for the recurrent
#'   step.
#' @param use_bias Boolean, whether the layer uses a bias vector.
#' @param kernel_initializer Initializer for the `kernel` weights matrix, used
#'   for the linear transformation of the inputs..
#' @param recurrent_initializer Initializer for the `recurrent_kernel` weights
#'   matrix, used for the linear transformation of the recurrent state..
#' @param bias_initializer Initializer for the bias vector.
#' @param unit_forget_bias Boolean. If TRUE, add 1 to the bias of the forget
#'   gate at initialization. Use in combination with `bias_initializer="zeros"`.
#'   This is recommended in [Jozefowicz et
#'   al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
#' @param kernel_regularizer Regularizer function applied to the `kernel`
#'   weights matrix.
#' @param recurrent_regularizer Regularizer function applied to the
#'   `recurrent_kernel` weights matrix.
#' @param bias_regularizer Regularizer function applied to the bias vector.
#' @param activity_regularizer Regularizer function applied to the output of the
#'   layer (its "activation")..
#' @param kernel_constraint Constraint function applied to the `kernel` weights
#'   matrix.
#' @param recurrent_constraint Constraint function applied to the
#'   `recurrent_kernel` weights matrix.
#' @param bias_constraint Constraint function applied to the bias vector.
#' @param return_sequences Boolean. Whether to return the last output in the
#'   output sequence, or the full sequence.
#' @param go_backwards Boolean (default FALSE). If TRUE, rocess the input
#'   sequence backwards.
#' @param stateful Boolean (default FALSE). If TRUE, the last state for each
#'   sample at index i in a batch will be used as initial state for the sample
#'   of index i in the following batch.
#' @param dropout Float between 0 and 1. Fraction of the units to drop for the
#'   linear transformation of the inputs.
#' @param recurrent_dropout Float between 0 and 1. Fraction of the units to drop
#'   for the linear transformation of the recurrent state.
#'   
#' @section Input shape: 
#' - if data_format='channels_first' 5D tensor with shape:
#'   `(samples,time, channels, rows, cols)` 
#'   - if data_format='channels_last' 5D
#'   tensor with shape: `(samples,time, rows, cols, channels)`
#'   
#' @section References: 
#' - [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
#'   The current implementation does not include the feedback loop on the cells
#'   output
#' 
#' @family convolutional layers 
#'       
#' @export
layer_conv_lstm_2d <- function(object, filters, kernel_size, strides = c(1L, 1L), padding = "valid", data_format = NULL, 
                               dilation_rate = c(1L, 1L), activation = "tanh", recurrent_activation = "hard_sigmoid", use_bias = TRUE, 
                               kernel_initializer = "glorot_uniform", recurrent_initializer = "orthogonal", bias_initializer = "zeros", 
                               unit_forget_bias = TRUE, kernel_regularizer = NULL, recurrent_regularizer = NULL, bias_regularizer = NULL, 
                               activity_regularizer = NULL, kernel_constraint = NULL, recurrent_constraint = NULL, bias_constraint = NULL, 
                               return_sequences = FALSE, go_backwards = FALSE, stateful = FALSE, dropout = 0.0, recurrent_dropout = 0.0,
                               batch_size = NULL, name = NULL, trainable = NULL, weights = NULL, input_shape = NULL) {
  
  create_layer(keras$layers$ConvLSTM2D, object, list(
    filters = as.integer(filters),
    kernel_size = as_integer_tuple(kernel_size),
    strides = as_integer_tuple(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate),
    activation = activation,
    recurrent_activation = recurrent_activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    recurrent_initializer = recurrent_initializer,
    bias_initializer = bias_initializer,
    unit_forget_bias = unit_forget_bias,
    kernel_regularizer = kernel_regularizer,
    recurrent_regularizer = recurrent_regularizer,
    bias_regularizer = bias_regularizer,
    activity_regularizer = activity_regularizer,
    kernel_constraint = kernel_constraint,
    recurrent_constraint = recurrent_constraint,
    bias_constraint = bias_constraint,
    return_sequences = return_sequences,
    go_backwards = go_backwards,
    stateful = stateful,
    dropout = dropout,
    recurrent_dropout = recurrent_dropout,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights,
    input_shape = normalize_shape(input_shape)
  ))
  
}






normalize_padding <- function(padding, dims) {
  normalize_scale("padding", padding, dims)
}

normalize_cropping <- function(cropping, dims) {
  normalize_scale("cropping", cropping, dims)
}

normalize_scale <- function(name, scale, dims) {
  
  # validate and marshall scale argument
  throw_invalid_scale <- function() {
    stop(name, " must be a list of ", dims, " integers or list of ", dims,  " lists of 2 integers", 
         call. = FALSE)
  }
  
  # if all of the individual items are numeric then cast to integer vector
  if (all(sapply(scale, function(x) length(x) == 1 && is.numeric(x)))) {
    as.integer(scale)
  } else if (is.list(scale)) {
    lapply(scale, function(x) {
      if (length(x) != 2)
        throw_invalid_scale()
      as.integer(x)
    })
  } else {
    throw_invalid_scale()
  }
}



