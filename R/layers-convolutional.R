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
#' @param filters Integer, the dimensionality of the output space (i.e. the 
#'   number output of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying 
#'   the length of the 1D convolution window.
#' @param strides An integer or list of a single integer, specifying the 
#'   stride length of the convolution. Specifying any stride value != 1 is 
#'   incompatible with specifying any `dilation_rate` value != 1.
#' @param padding One of `"valid"`, `"causal"` or `"same"` (case-insensitive). 
#'   `"causal"` results in causal (dilated) convolutions, e.g. output[t] depends
#'   solely on input[:t-1]. Useful when modeling temporal data where the model 
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
  
  
  # build args
  args <- list(
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
    bias_constraint = bias_constraint
  )
  args$input_shape <- normalize_shape(input_shape)
  
  # call function
  layer <- do.call(tf$contrib$keras$layers$Conv1D, args)
  
  # compose
  compose_layer(x, layer)
}

