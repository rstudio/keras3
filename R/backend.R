

#' Keras backend tensor engine
#' 
#' Obtain a reference to the `keras.backend` Python module used to implement
#' tensor operations.
#'
#' @inheritParams reticulate::import
#'
#' @note See the documentation here <https://keras.io/backend/> for 
#'   additional details on the available functions.
#'
#' @return Reference to Keras backend python module.
#'  
#' @export   
backend <- function(convert = TRUE) {
  if (convert)
    keras$backend
  else
    r_to_py(keras$backend)
}


#' Element-wise absolute value.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_abs <- function(x) {
  keras$backend$abs(
    x = x
  )
}


#' Bitwise reduction (logical AND).
#'
#' @param x Tensor or variable.
#' @param axis Axis along which to perform the reduction (axis indexes are
#'   1-based).
#' @param keepdims whether the drop or broadcast the reduction axes.
#'
#' @return A uint8 tensor (0s and 1s).
#'
#' @template roxlate-keras-backend
#'
#' @export
k_all <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$all(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Bitwise reduction (logical OR).
#' 
#' @param x Tensor or variable.
#' @param axis Axis along which to perform the reduction (axis indexes
#'   are 1-based).
#' @param keepdims whether the drop or broadcast the reduction axes.
#' 
#' @return A uint8 tensor (0s and 1s).
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_any <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$any(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Creates a 1D tensor containing a sequence of integers.
#'
#' The function arguments use the same convention as Theano's arange: if only
#' one argument is provided, it is in fact the "stop" argument. The default
#' type of the returned tensor is `'int32'` to match TensorFlow's default.
#'
#' @param start Start value.
#' @param stop Stop value.
#' @param step Difference between two successive values.
#' @param dtype Integer dtype to use.
#'
#' @return An integer tensor.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_arange <- function(start, stop = NULL, step = 1, dtype = "int32") {
  keras$backend$arange(
    start = as.integer(start),
    stop = as_nullable_integer(stop),
    step = as.integer(step),
    dtype = dtype
  )
}


#' Returns the index of the maximum value along an axis.
#'
#' @param x Tensor or variable.
#' @param axis Axis along which to perform the reduction (axis indexes are
#'   1-based). Pass -1 (the default) to select the last axis.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_argmax <- function(x, axis = -1) {
  keras$backend$argmax(
    x = x,
    axis = as_axis(axis)
  )
}


#' Returns the index of the minimum value along an axis.
#'
#' @param x Tensor or variable.
#' @param axis Axis along which to perform the reduction (axis indexes are
#'   1-based). Pass -1 (the default) to select the last axis.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_argmin <- function(x, axis = -1) {
  keras$backend$argmin(
    x = x,
    axis = as_axis(axis)
  )
}


#' Active Keras backend
#' 
#' @return The name of the backend Keras is currently using.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_backend <- function() {
  keras$backend$backend(
  )
}


#' Batchwise dot product.
#'
#' `batch_dot` is used to compute dot product of `x` and `y` when `x` and `y`
#' are data in batch, i.e. in a shape of `(batch_size)`. `batch_dot` results in
#' a tensor or variable with less dimensions than the input. If the number of
#' dimensions is reduced to 1, we use `expand_dims` to make sure that ndim is
#' at least 2.
#'
#' @param x Keras tensor or variable with 2 more more axes.
#' @param y Keras tensor or variable with 2 or more axes
#' @param axes List of (or single) integer with target dimensions (axis indexes
#'   are 1-based). The lengths of `axes[[1]]` and `axes[[2]]` should be the
#'   same.
#'
#' @return A tensor with shape equal to the concatenation of `x`'s shape (less
#'   the dimension that was summed over) and `y`'s shape (less the batch
#'   dimension and the dimension that was summed over). If the final rank is 1,
#'   we reshape it to `(batch_size, 1)`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_batch_dot <- function(x, y, axes) {
  keras$backend$batch_dot(
    x = x,
    y = y,
    axes = as_axis(axes)
  )
}


#' Turn a nD tensor into a 2D tensor with same 1st dimension.
#' 
#' In other words, it flattens each data samples of a batch.
#' 
#' @param x A tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_batch_flatten <- function(x) {
  keras$backend$batch_flatten(
    x = x
  )
}


#' Returns the value of more than one tensor variable.
#' 
#' @param ops List of ops to evaluate.
#' 
#' @return A list of arrays.
#' 
#' @seealso [k_batch_set_value()]
#' 
#' @template roxlate-keras-backend  
#'
#' @export
k_batch_get_value <- function(ops) {
  keras$backend$batch_get_value(ops)
}


#' Applies batch normalization on x given mean, var, beta and gamma.
#' 
#' i.e. returns
#' `output <- (x - mean) / (sqrt(var) + epsilon) * gamma + beta`
#' 
#' @param x Input tensor or variable.
#' @param mean Mean of batch.
#' @param var Variance of batch.
#' @param beta Tensor with which to center the input.
#' @param gamma Tensor by which to scale the input.
#' @param epsilon Fuzz factor.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_batch_normalization <- function(x, mean, var, beta, gamma, epsilon = 0.001) {
  keras$backend$batch_normalization(
    x = x,
    mean = mean,
    var = var,
    beta = beta,
    gamma = gamma,
    epsilon = epsilon
  )
}


#' Sets the values of many tensor variables at once.
#' 
#' @param lists a list of lists `(tensor, value)`. `value` should be an R array.
#' 
#' @seealso [k_batch_get_value()]
#' 
#' @template roxlate-keras-backend   
#'
#' @export
k_batch_set_value <- function(lists) {
  keras$backend$batch_set_value(
    tuples = lists
  )
}


#' Adds a bias vector to a tensor.
#' 
#' @param x Tensor or variable.
#' @param bias Bias tensor to add.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' 
#' @return Output tensor.
#'  
#' @template roxlate-keras-backend  
#'  
#' @export
k_bias_add <- function(x, bias, data_format = NULL) {
  keras$backend$bias_add(
    x = x,
    bias = bias,
    data_format = data_format
  )
}


#' Binary crossentropy between an output tensor and a target tensor.
#'
#' @param target A tensor with the same shape as `output`.
#' @param output A tensor.
#' @param from_logits Whether `output` is expected to be a logits tensor. By
#'   default, we consider that `output` encodes a probability distribution.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_binary_crossentropy <- function(target, output, from_logits = FALSE) {
  keras$backend$binary_crossentropy(
    target = target,
    output = output,
    from_logits = from_logits
  )
}


#' Casts a tensor to a different dtype and returns it.
#' 
#' You can cast a Keras variable but it still returns a Keras tensor.
#' 
#' @param x Keras tensor (or variable).
#' @param dtype String, either (`'float16'`, `'float32'`, or `'float64'`).
#' 
#' @return Keras tensor with dtype `dtype`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_cast <- function(x, dtype) {
  keras$backend$cast(
    x = x,
    dtype = dtype
  )
}


#' Cast an array to the default Keras float type.
#' 
#' @param x Array.
#' 
#' @return The same array, cast to its new type.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_cast_to_floatx <- function(x) {
  r_to_py(keras$backend)$cast_to_floatx(
    x = x
  )
}


#' Categorical crossentropy between an output tensor and a target tensor.
#'
#' @param target A tensor of the same shape as `output`.
#' @param output A tensor resulting from a softmax (unless `from_logits` is
#'   TRUE, in which case `output` is expected to be the logits).
#' @param from_logits Logical, whether `output` is the result of a softmax, or
#'   is a tensor of logits.
#'
#' @return Output tensor.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_categorical_crossentropy <- function(target, output, from_logits = FALSE) {
  keras$backend$categorical_crossentropy(
    target = target,
    output = output,
    from_logits = from_logits
  )
}


#' Destroys the current TF graph and creates a new one.
#' 
#' Useful to avoid clutter from old models / layers.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_clear_session <- function() {
  keras$backend$clear_session()
}


#' Element-wise value clipping.
#' 
#' @param x Tensor or variable.
#' @param min_value Float or integer.
#' @param max_value Float or integer.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_clip <- function(x, min_value, max_value) {
  keras$backend$clip(
    x = x,
    min_value = min_value,
    max_value = max_value
  )
}


#' Concatenates a list of tensors alongside the specified axis.
#'
#' @param tensors list of tensors to concatenate.
#' @param axis concatenation axis (axis indexes are 1-based). Pass -1 (the
#'   default) to select the last axis.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_concatenate <- function(tensors, axis = -1) {
  keras$backend$concatenate(
    tensors = tensors,
    axis = as_axis(axis)
  )
}


#' Creates a constant tensor.
#' 
#' @param value A constant value
#' @param dtype The type of the elements of the resulting tensor.
#' @param shape Optional dimensions of resulting tensor.
#' @param name Optional name for the tensor.
#' 
#' @return A Constant Tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_constant <- function(value, dtype = NULL, shape = NULL, name = NULL) {
  keras$backend$constant(
    value = value,
    dtype = dtype,
    shape = backend_normalize_shape(shape),
    name = name
  )
}



#' 1D convolution.
#' 
#' @param x Tensor or variable.
#' @param kernel kernel tensor.
#' @param strides stride integer.
#' @param padding string, `"same"`, `"causal"` or `"valid"`.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' @param dilation_rate integer dilate rate.
#' 
#' @return A tensor, result of 1D convolution.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_conv1d <- function(x, kernel, strides = 1, padding = "valid", data_format = NULL, dilation_rate = 1) {
  keras$backend$conv1d(
    x = x,
    kernel = kernel,
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate)
  )
}


#' 2D convolution.
#'
#' @param x Tensor or variable.
#' @param kernel kernel tensor.
#' @param strides strides
#' @param padding string, `"same"` or `"valid"`.
#' @param data_format string, `"channels_last"` or `"channels_first"`. Whether
#'   to use Theano or TensorFlow/CNTK data format for inputs/kernels/outputs.
#' @param dilation_rate vector of 2 integers.
#'
#' @return A tensor, result of 2D convolution.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_conv2d <- function(x, kernel, strides = c(1, 1), padding = "valid", 
                     data_format = NULL, 
                     dilation_rate = c(1, 1)) {
  keras$backend$conv2d(
    x = x,
    kernel = kernel,
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate)
  )
}


#' 2D deconvolution (i.e. transposed convolution).
#'
#' @param x Tensor or variable.
#' @param kernel kernel tensor.
#' @param output_shape 1D int tensor for the output shape.
#' @param strides strides list.
#' @param padding string, `"same"` or `"valid"`.
#' @param data_format string, `"channels_last"` or `"channels_first"`. Whether
#'   to use Theano or TensorFlow/CNTK data format for inputs/kernels/outputs.
#'
#' @return A tensor, result of transposed 2D convolution.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_conv2d_transpose <- function(x, kernel, output_shape, strides = c(1, 1), 
                               padding = "valid", data_format = NULL) {
  keras$backend$conv2d_transpose(
    x = x,
    kernel = kernel,
    output_shape = output_shape,
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format
  )
}


#' 3D convolution.
#'
#' @param x Tensor or variable.
#' @param kernel kernel tensor.
#' @param strides strides
#' @param padding string, `"same"` or `"valid"`.
#' @param data_format string, `"channels_last"` or `"channels_first"`. Whether
#'   to use Theano or TensorFlow/CNTK data format for inputs/kernels/outputs.
#' @param dilation_rate list of 3 integers.
#'
#' @return A tensor, result of 3D convolution.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_conv3d <- function(x, kernel, strides = c(1, 1, 1), padding = "valid", 
                     data_format = NULL, dilation_rate = c(1, 1, 1)) {
  keras$backend$conv3d(
    x = x,
    kernel = kernel,
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate)
  )
}


#' 3D deconvolution (i.e. transposed convolution).
#'
#' @param x input tensor.
#' @param kernel kernel tensor.
#' @param output_shape 1D int tensor for the output shape.
#' @param strides strides
#' @param padding string, "same" or "valid".
#' @param data_format string, `"channels_last"` or `"channels_first"`. Whether
#'   to use Theano or TensorFlow/CNTK data format for inputs/kernels/outputs.
#'
#' @return A tensor, result of transposed 3D convolution.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_conv3d_transpose <- function(x, kernel, output_shape, strides = c(1, 1, 1), 
                               padding = "valid", data_format = NULL) {
  keras$backend$conv3d_transpose(
    x = x,
    kernel = kernel,
    output_shape = output_shape,
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format
  )
}


#' Computes cos of x element-wise.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_cos <- function(x) {
  keras$backend$cos(
    x = x
  )
}


#' Returns the static number of elements in a Keras variable or tensor.
#' 
#' @param x Keras variable or tensor.
#' 
#' @return Integer, the number of elements in `x`, i.e., the product of the array's static dimensions.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_count_params <- function(x) {
  keras$backend$count_params(
    x = x
  )
}


#' Runs CTC loss algorithm on each batch element.
#'
#' @param y_true tensor `(samples, max_string_length)` containing the truth
#'   labels.
#' @param y_pred tensor `(samples, time_steps, num_categories)` containing the
#'   prediction, or output of the softmax.
#' @param input_length tensor `(samples, 1)` containing the sequence length for
#'   each batch item in `y_pred`.
#' @param label_length tensor `(samples, 1)` containing the sequence length for
#'   each batch item in `y_true`.
#'
#' @return Tensor with shape (samples,1) containing the CTC loss of each
#'   element.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_ctc_batch_cost <- function(y_true, y_pred, input_length, label_length) {
  keras$backend$ctc_batch_cost(
    y_true = y_true,
    y_pred = y_pred,
    input_length = input_length,
    label_length = label_length
  )
}


#' Decodes the output of a softmax.
#'
#' Can use either greedy search (also known as best path) or a constrained
#' dictionary search.
#'
#' @param y_pred tensor `(samples, time_steps, num_categories)` containing the
#'   prediction, or output of the softmax.
#' @param input_length tensor `(samples, )` containing the sequence length for
#'   each batch item in `y_pred`.
#' @param greedy perform much faster best-path search if `TRUE`. This does not
#'   use a dictionary.
#' @param beam_width if `greedy` is `FALSE`: a beam search decoder will be used
#'   with a beam of this width.
#' @param top_paths if `greedy` is `FALSE`, how many of the most probable paths
#'   will be returned.
#'
#' @return If `greedy` is `TRUE`, returns a list of one element
#'   that contains the decoded sequence. If `FALSE`, returns the `top_paths`
#'   most probable decoded sequences. Important: blank labels are returned as
#'   `-1`. Tensor `(top_paths)` that contains the log probability of each
#'   decoded sequence.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_ctc_decode <- function(y_pred, input_length, greedy = TRUE, beam_width = 100L, top_paths = 1) {
  keras$backend$ctc_decode(
    y_pred = y_pred,
    input_length = input_length,
    greedy = greedy,
    beam_width = as.integer(beam_width),
    top_paths = as.integer(top_paths)
  )
}


#' Converts CTC labels from dense to sparse.
#' 
#' @param labels dense CTC labels.
#' @param label_lengths length of the labels.
#' 
#' @return A sparse tensor representation of the labels.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_ctc_label_dense_to_sparse <- function(labels, label_lengths) {
  keras$backend$ctc_label_dense_to_sparse(
    labels = labels,
    label_lengths = label_lengths
  )
}


#' Cumulative product of the values in a tensor, alongside the specified axis.
#'
#' @param x A tensor or variable.
#' @param axis An integer, the axis to compute the product (axis indexes are
#'   1-based).
#'
#' @return A tensor of the cumulative product of values of `x` along `axis`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_cumprod <- function(x, axis = 1) {
  keras$backend$cumprod(
    x = x,
    axis = as_axis(axis)
  )
}


#' Cumulative sum of the values in a tensor, alongside the specified axis.
#'
#' @param x A tensor or variable.
#' @param axis An integer, the axis to compute the sum (axis indexes are
#'   1-based).
#'
#' @return A tensor of the cumulative sum of values of `x` along `axis`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_cumsum <- function(x, axis = 1) {
  keras$backend$cumsum(
    x = x,
    axis = as_axis(axis)
  )
}


#' Depthwise 2D convolution with separable filters.
#'
#' @param x input tensor
#' @param depthwise_kernel convolution kernel for the depthwise convolution.
#' @param strides strides (length 2).
#' @param padding string, `"same"` or `"valid"`.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' @param dilation_rate vector of integers, dilation rates for the separable
#'   convolution.
#'
#' @return Output tensor.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_depthwise_conv2d <- function(x, depthwise_kernel, strides = c(1, 1), padding = "valid", 
                               data_format = NULL, dilation_rate = c(1, 1)) {
  keras$backend$depthwise_conv2d(
    x = x,
    depthwise_kernel = depthwise_kernel,
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate)
  )
}


#' Multiplies 2 tensors (and/or variables) and returns a *tensor*.
#' 
#' When attempting to multiply a nD tensor
#' with a nD tensor, it reproduces the Theano behavior.
#' (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A tensor, dot product of `x` and `y`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_dot <- function(x, y) {
  keras$backend$dot(
    x = x,
    y = y
  )
}


#' Sets entries in `x` to zero at random, while scaling the entire tensor.
#'
#' @param x tensor
#' @param level fraction of the entries in the tensor that will be set to 0.
#' @param noise_shape shape for randomly generated keep/drop flags, must be
#'   broadcastable to the shape of `x`
#' @param seed random seed to ensure determinism.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_dropout <- function(x, level, noise_shape = NULL, seed = NULL) {
  keras$backend$dropout(
    x = x,
    level = level,
    noise_shape = noise_shape,
    seed = seed
  )
}


#' Returns the dtype of a Keras tensor or variable, as a string.
#' 
#' @param x Tensor or variable.
#' 
#' @return String, dtype of `x`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_dtype <- function(x) {
  keras$backend$dtype(
    x = x
  )
}


#' Exponential linear unit.
#' 
#' @param x A tensor or variable to compute the activation function for.
#' @param alpha A scalar, slope of negative section.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_elu <- function(x, alpha = 1.0) {
  keras$backend$elu(
    x = x,
    alpha = alpha
  )
}


#' Fuzz factor used in numeric expressions.
#' 
#' @param e float. New value of epsilon.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_epsilon <- function() {
  keras$backend$epsilon(
  )
}

#' @rdname k_epsilon
#' @export
k_set_epsilon <- function(e) {
  keras$backend$set_epsilon(
    e = e
  )
}



#' Element-wise equality between two tensors.
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A bool tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_equal <- function(x, y) {
  keras$backend$equal(
    x = x,
    y = y
  )
}


#' Evaluates the value of a variable.
#' 
#' @param x A variable.
#' 
#' @return An R array.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_eval <- function(x) {
  keras$backend$eval(
    x = x
  )
}


#' Element-wise exponential.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_exp <- function(x) {
  keras$backend$exp(
    x = x
  )
}


#' Adds a 1-sized dimension at index `axis`.
#'
#' @param x A tensor or variable.
#' @param axis Position where to add a new axis (axis indexes are 1-based).
#'   Pass -1 (the default) to select the last axis.
#'
#' @return A tensor with expanded dimensions.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_expand_dims <- function(x, axis = -1) {
  keras$backend$expand_dims(
    x = x,
    axis = as_axis(axis)
  )
}


#' Instantiate an identity matrix and returns it.
#'
#' @param size Integer, number of rows/columns.
#' @param dtype String, data type of returned Keras variable.
#' @param name String, name of returned Keras variable.
#'
#' @return A Keras variable, an identity matrix.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_eye <- function(size, dtype = NULL, name = NULL) {
  keras$backend$eye(
    size = as.integer(size),
    dtype = dtype,
    name = name
  )
}


#' Flatten a tensor.
#' 
#' @param x A tensor or variable.
#' 
#' @return A tensor, reshaped into 1-D
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_flatten <- function(x) {
  keras$backend$flatten(
    x = x
  )
}


#' Default float type
#' 
#' @param floatx String, 'float16', 'float32', or 'float64'.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_floatx <- function() {
  keras$backend$floatx(
  )
}

#' @rdname k_floatx
#' @export
k_set_floatx <- function(floatx) {
  keras$backend$set_floatx(
    floatx = floatx
  )
}


#' Reduce elems using fn to combine them from left to right.
#'
#' @param fn Function that will be called upon each element in elems and an
#'   accumulator
#' @param elems tensor
#' @param initializer The first value used (first element of `elems` in case of
#'   `NULL``)
#' @param name A string name for the foldl node in the graph
#'
#' @return Tensor with same type and shape as `initializer`.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_foldl <- function(fn, elems, initializer = NULL, name = NULL) {
  keras$backend$foldl(
    fn = fn,
    elems = elems,
    initializer = initializer,
    name = name
  )
}


#' Reduce elems using fn to combine them from right to left.
#'
#' @param fn Function that will be called upon each element in elems and an
#'   accumulator
#' @param elems tensor
#' @param initializer The first value used (last element of `elems` in case of
#'   NULL)
#' @param name A string name for the foldr node in the graph
#'
#' @return Tensor with same type and shape as `initializer`.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_foldr <- function(fn, elems, initializer = NULL, name = NULL) {
  keras$backend$foldr(
    fn = fn,
    elems = elems,
    initializer = initializer,
    name = name
  )
}

#' Instantiates a Keras function
#'
#' @param inputs List of placeholder tensors.
#' @param outputs List of output tensors.
#' @param updates List of update ops.
#' @param ... Named arguments passed to `tf$Session$run`.
#'
#' @return Output values as R arrays.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_function <- function(inputs, outputs, updates = NULL, ...) {
  keras$backend$`function`(
    inputs = inputs,
    outputs = outputs,
    updates = updates,
    ...
  )
}


#' Retrieves the elements of indices `indices` in the tensor `reference`.
#' 
#' @param reference A tensor.
#' @param indices An integer tensor of indices.
#' 
#' @return A tensor of same type as `reference`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_gather <- function(reference, indices) {
  keras$backend$gather(
    reference = reference,
    indices = indices
  )
}


#' TF session to be used by the backend.
#'
#' If a default TensorFlow session is available, we will return it. Else, we
#' will return the global Keras session. If no global Keras session exists at
#' this point: we will create a new global session. Note that you can manually
#' set the global session via `k_set_session()`. 
#' 
#' @param session A TensorFlow Session.
#' 
#' @return A TensorFlow session
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_get_session <- function() {
  keras$backend$get_session()
}

#' @rdname k_get_session
#' @export
k_set_session <- function(session) {
  keras$backend$set_session(
    session = session
  )
}


#' Get the uid for the default graph.
#' 
#' @param prefix An optional prefix of the graph.
#' 
#' @return A unique identifier for the graph.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_get_uid <- function(prefix = "") {
  keras$backend$get_uid(
    prefix = prefix
  )
}


#' Returns the value of a variable.
#' 
#' @param x input variable.
#' 
#' @return An R array.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_get_value <- function(x) {
  keras$backend$get_value(
    x = x
  )
}


#' Returns the shape of a variable.
#' 
#' @param x A variable.
#' 
#' @return A vector of integers.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_get_variable_shape <- function(x) {
  keras$backend$get_variable_shape(
    x = x
  )
}


#' Returns the gradients of `variables` w.r.t. `loss`.
#' 
#' @param loss Scalar tensor to minimize.
#' @param variables List of variables.
#' 
#' @return A gradients tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_gradients <- function(loss, variables) {
  keras$backend$gradients(
    loss = loss,
    variables = variables
  )
}


#' Element-wise truth value of (x > y).
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A bool tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_greater <- function(x, y) {
  keras$backend$greater(
    x = x,
    y = y
  )
}


#' Element-wise truth value of (x >= y).
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A bool tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_greater_equal <- function(x, y) {
  keras$backend$greater_equal(
    x = x,
    y = y
  )
}


#' Segment-wise linear approximation of sigmoid.
#' 
#' Faster than sigmoid.
#' Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
#' In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
#' 
#' @param x A tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_hard_sigmoid <- function(x) {
  keras$backend$hard_sigmoid(
    x = x
  )
}


#' Returns a tensor with the same content as the input tensor.
#' 
#' @param x The input tensor.
#' @param name String, name for the variable to create.
#' 
#' @return A tensor of the same shape, type and content.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_identity <- function(x, name = NULL) {
  keras$backend$identity(
    x = x,
    name = name
  )
}


#' Default image data format convention ('channels_first' or 'channels_last').
#' 
#' @param data_format string. `'channels_first'` or `'channels_last'`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_image_data_format <- function() {
  keras$backend$image_data_format(
  )
}

#' @rdname k_image_data_format
#' @export
k_set_image_data_format <- function(data_format) {
  keras$backend$set_image_data_format(
    data_format = data_format
  )
}


#' Selects `x` in test phase, and `alt` otherwise.
#'
#' Note that `alt` should have the *same shape* as `x`.
#'
#' @param x What to return in test phase (tensor or function that returns a
#'   tensor).
#' @param alt What to return otherwise (tensor or function that returns a
#'   tensor).
#' @param training Optional scalar tensor (or R logical or integer) specifying
#'   the learning phase.
#'
#' @return Either `x` or `alt` based on `k_learning_phase()`.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_in_test_phase <- function(x, alt, training = NULL) {
  keras$backend$in_test_phase(
    x = x,
    alt = alt,
    training = training
  )
}


#' Returns whether the `targets` are in the top `k` `predictions`.
#'
#' @param predictions A tensor of shape `(batch_size, classes)` and type
#'   `float32`.
#' @param targets A 1D tensor of length `batch_size` and type `int32` or
#'   `int64`.
#' @param k An `int`, number of top elements to consider.
#'
#' @return A 1D tensor of length `batch_size` and type `bool`. `output[[i]]` is
#'   `TRUE` if `predictions[i, targets[[i]]` is within top-`k` values of
#'   `predictions[[i]]`.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_in_top_k <- function(predictions, targets, k) {
  keras$backend$in_top_k(
    predictions = predictions,
    targets = targets,
    k = as.integer(k)
  )
}


#' Selects `x` in train phase, and `alt` otherwise.
#'
#' Note that `alt` should have the *same shape* as `x`.
#'
#' @param x What to return in train phase (tensor or function that returns a
#'   tensor).
#' @param alt What to return otherwise (tensor or function that returns a
#'   tensor).
#' @param training Optional scalar tensor (or R logical or integer) specifying
#'   the learning phase.
#'
#' @return Either `x` or `alt` based on the `training` flag. the `training`
#'   flag defaults to `k_learning_phase()`.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_in_train_phase <- function(x, alt, training = NULL) {
  keras$backend$in_train_phase(
    x = x,
    alt = alt,
    training = training
  )
}


#' Returns the shape of tensor or variable as a list of int or NULL entries.
#' 
#' @param x Tensor or variable.
#' 
#' @return A list of integers (or NULL entries).
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_int_shape <- function(x) {
  keras$backend$int_shape(
    x = x
  )
}


#' Returns whether `x` is a Keras tensor.
#' 
#' A "Keras tensor" is a tensor that was returned by a Keras layer
#' 
#' @param x A candidate tensor.
#' 
#' @return A logical: Whether the argument is a Keras tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_is_keras_tensor <- function(x) {
  keras$backend$is_keras_tensor(
    x = x
  )
}


#' Returns whether `x` is a placeholder.
#' 
#' @param x A candidate placeholder.
#' 
#' @return A logical
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_is_placeholder <- function(x) {
  keras$backend$is_placeholder(
    x = x
  )
}


#' Returns whether a tensor is a sparse tensor.
#' 
#' @param tensor A tensor instance.
#' 
#' @return A logical
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_is_sparse <- function(tensor) {
  keras$backend$is_sparse(
    tensor = tensor
  )
}


#' Normalizes a tensor wrt the L2 norm alongside the specified axis.
#' 
#' @param x Tensor or variable.
#' @param axis Axis along which to perform normalization (axis indexes
#'   are 1-based)
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_l2_normalize <- function(x, axis = NULL) {
  keras$backend$l2_normalize(
    x = x,
    axis = as_axis(axis)
  )
}


#' Returns the learning phase flag.
#'
#' The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed
#' as input to any Keras function that uses a different behavior at train time
#' and test time. 
#' 
#' @return Learning phase (scalar integer tensor or R integer).
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_learning_phase <- function() {
  keras$backend$learning_phase()
}


#' Element-wise truth value of (x < y).
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A bool tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_less <- function(x, y) {
  keras$backend$less(
    x = x,
    y = y
  )
}


#' Element-wise truth value of (x <= y).
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A bool tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_less_equal <- function(x, y) {
  keras$backend$less_equal(
    x = x,
    y = y
  )
}


#' Apply 1D conv with un-shared weights.
#'
#' @param inputs 3D tensor with shape: (batch_size, steps, input_dim)
#' @param kernel the unshared weight for convolution, with shape
#'   (output_length, feature_dim, filters)
#' @param kernel_size a list of a single integer, specifying the length of the
#'   1D convolution window
#' @param strides a list of a single integer, specifying the stride length of
#'   the convolution
#' @param data_format the data format, channels_first or channels_last
#'
#' @return the tensor after 1d conv with un-shared weights, with shape
#'   (batch_size, output_length, filters)
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_local_conv1d <- function(inputs, kernel, kernel_size, strides, data_format = NULL) {
  keras$backend$local_conv1d(
    inputs = inputs,
    kernel = kernel,
    kernel_size = list(as.integer(kernel_size)),
    strides = list(as.integer(strides)),
    data_format = data_format
  )
}


#' Apply 2D conv with un-shared weights.
#'
#' @param inputs 4D tensor with shape: (batch_size, filters, new_rows,
#'   new_cols) if data_format='channels_first' or 4D tensor with shape:
#'   (batch_size, new_rows, new_cols, filters) if data_format='channels_last'.
#' @param kernel the unshared weight for convolution, with shape (output_items,
#'   feature_dim, filters)
#' @param kernel_size a list of 2 integers, specifying the width and height of
#'   the 2D convolution window.
#' @param strides a list of 2 integers, specifying the strides of the
#'   convolution along the width and height.
#' @param output_shape a list with (output_row, output_col)
#' @param data_format the data format, channels_first or channels_last
#'
#' @return A 4d tensor with shape: (batch_size, filters, new_rows, new_cols) if
#'   data_format='channels_first' or 4D tensor with shape: (batch_size,
#'   new_rows, new_cols, filters) if data_format='channels_last'.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_local_conv2d <- function(inputs, kernel, kernel_size, strides, output_shape, data_format = NULL) {
  keras$backend$local_conv2d(
    inputs = inputs,
    kernel = kernel,
    kernel_size = as.integer(kernel_size),
    strides = as.integer(strides),
    output_shape = as.integer(output_shape),
    data_format = data_format
  )
}


#' Element-wise log.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_log <- function(x) {
  keras$backend$log(
    x = x
  )
}


#' Computes log(sum(exp(elements across dimensions of a tensor))).
#'
#' This function is more numerically stable than log(sum(exp(x))). It avoids
#' overflows caused by taking the exp of large inputs and underflows caused by
#' taking the log of small inputs.
#'
#' @param x A tensor or variable.
#' @param axis An integer, the axis to reduce over (axis indexes are 1-based).
#' @param keepdims A boolean, whether to keep the dimensions or not. If
#'   `keepdims` is `FALSE`, the rank of the tensor is reduced by 1. If
#'   `keepdims` is `TRUE`, the reduced dimension is retained with length 1.
#'
#' @return The reduced tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_logsumexp <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$logsumexp(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Sets the manual variable initialization flag.
#'
#' This boolean flag determines whether variables should be initialized as they
#' are instantiated (default), or if the user should handle the initialization
#' (e.g. via `tf$initialize_all_variables()`).
#'
#' @param value Logical
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_manual_variable_initialization <- function(value) {
  keras$backend$manual_variable_initialization(
    value = value
  )
}


#' Map the function fn over the elements elems and return the outputs.
#' 
#' @param fn Function that will be called upon each element in elems
#' @param elems tensor
#' @param name A string name for the map node in the graph
#' @param dtype Output data type.
#' 
#' @return Tensor with dtype `dtype`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_map_fn <- function(fn, elems, name = NULL, dtype = NULL) {
  keras$backend$map_fn(
    fn = fn,
    elems = elems,
    name = name,
    dtype = dtype
  )
}


#' Maximum value in a tensor.
#'
#' @param x A tensor or variable.
#' @param axis An integer, the axis to find maximum values (axis indexes are
#'   1-based).
#' @param keepdims A boolean, whether to keep the dimensions or not. If
#'   `keepdims` is `FALSE`, the rank of the tensor is reduced by 1. If
#'   `keepdims` is `TRUE`, the reduced dimension is retained with length 1.
#'
#' @return A tensor with maximum values of `x`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_max <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$max(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Element-wise maximum of two tensors.
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_maximum <- function(x, y) {
  keras$backend$maximum(
    x = x,
    y = y
  )
}


#' Mean of a tensor, alongside the specified axis.
#'
#' @param x A tensor or variable.
#' @param axis A list of axes to compute the mean over (axis indexes are
#'   1-based).
#' @param keepdims A boolean, whether to keep the dimensions or not. If
#'   `keepdims` is `FALSE`, the rank of the tensor is reduced by 1 for each
#'   entry in `axis`. If `keep_dims` is `TRUE`, the reduced dimensions are
#'   retained with length 1.
#'
#' @return A tensor with the mean of elements of `x`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_mean <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$mean(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Minimum value in a tensor.
#'
#' @param x A tensor or variable.
#' @param axis An integer, axis to find minimum values (axis indexes are
#'   1-based).
#' @param keepdims A boolean, whether to keep the dimensions or not. If
#'   `keepdims` is `FALSE`, the rank of the tensor is reduced by 1. If
#'   `keepdims` is `TRUE`, the reduced dimension is retained with length 1.
#'
#' @return A tensor with miminum values of `x`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_min <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$min(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Element-wise minimum of two tensors.
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_minimum <- function(x, y) {
  keras$backend$minimum(
    x = x,
    y = y
  )
}


#' Compute the moving average of a variable.
#' 
#' @param x A `Variable`.
#' @param value A tensor with the same shape as `x`.
#' @param momentum The moving average momentum.
#' 
#' @return An operation to update the variable.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_moving_average_update <- function(x, value, momentum) {
  keras$backend$moving_average_update(
    x = x,
    value = value,
    momentum = momentum
  )
}



#' Returns the number of axes in a tensor, as an integer.
#' 
#' @param x Tensor or variable.
#' 
#' @return Integer (scalar), number of axes.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_ndim <- function(x) {
  keras$backend$ndim(
    x = x
  )
}


#' Computes mean and std for batch then apply batch_normalization on batch.
#' 
#' @param x Input tensor or variable.
#' @param gamma Tensor by which to scale the input.
#' @param beta Tensor with which to center the input.
#' @param reduction_axes iterable of integers, axes over which to normalize.
#' @param epsilon Fuzz factor.
#' 
#' @return A list length of 3, `(normalized_tensor, mean, variance)`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_normalize_batch_in_training <- function(x, gamma, beta, reduction_axes, epsilon = 0.001) {
  keras$backend$normalize_batch_in_training(
    x = x,
    gamma = gamma,
    beta = beta,
    reduction_axes = as_integer_tuple(reduction_axes),
    epsilon = epsilon
  )
}


#' Element-wise inequality between two tensors.
#' 
#' @param x Tensor or variable.
#' @param y Tensor or variable.
#' 
#' @return A bool tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_not_equal <- function(x, y) {
  keras$backend$not_equal(
    x = x,
    y = y
  )
}


#' Computes the one-hot representation of an integer tensor.
#'
#' @param indices nD integer tensor of shape `(batch_size, dim1, dim2, ...
#'   dim(n-1))`
#' @param num_classes Integer, number of classes to consider.
#'
#' @return (n + 1)D one hot representation of the input with shape
#'   `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_one_hot <- function(indices, num_classes) {
  keras$backend$one_hot(
    indices = indices,
    num_classes = as.integer(num_classes)
  )
}


#' Instantiates an all-ones tensor variable and returns it.
#' 
#' @param shape Tuple of integers, shape of returned Keras variable.
#' @param dtype String, data type of returned Keras variable.
#' @param name String, name of returned Keras variable.
#' 
#' @return A Keras variable, filled with `1.0`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_ones <- function(shape, dtype = NULL, name = NULL) {
  keras$backend$ones(
    shape = as_integer_tuple(shape),
    dtype = dtype,
    name = name
  )
}


#' Instantiates an all-ones variable of the same shape as another tensor.
#'
#' @param x Keras variable or tensor.
#' @param dtype String, dtype of returned Keras variable. NULL uses the dtype
#'   of x.
#' @param name String, name for the variable to create.
#'
#' @return A Keras variable with the shape of x filled with ones.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_ones_like <- function(x, dtype = NULL, name = NULL) {
  keras$backend$ones_like(
    x = x,
    dtype = dtype,
    name = name
  )
}


#' Permutes axes in a tensor.
#'
#' @param x Tensor or variable.
#' @param pattern A list of dimension indices, e.g. `(1, 3, 2)`. Dimension
#'   indices are 1-based.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_permute_dimensions <- function(x, pattern) {
  keras$backend$permute_dimensions(
    x = x,
    pattern = as_axis(pattern)
  )
}


#' Instantiates a placeholder tensor and returns it.
#'
#' @param shape Shape of the placeholder (integer list, may include `NULL`
#'   entries).
#' @param ndim Number of axes of the tensor. At least one of {`shape`, `ndim`}
#'   must be specified. If both are specified, `shape` is used.
#' @param dtype Placeholder type.
#' @param sparse Logical, whether the placeholder should have a sparse type.
#' @param name Optional name string for the placeholder.
#'
#' @return Tensor instance (with Keras metadata included).
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_placeholder <- function(shape = NULL, ndim = NULL, dtype = NULL, sparse = FALSE, name = NULL) {
  keras$backend$placeholder(
    shape = backend_normalize_shape(shape),
    ndim = as.integer(ndim),
    dtype = dtype,
    sparse = sparse,
    name = name
  )
}


#' 2D Pooling.
#' 
#' @param x Tensor or variable.
#' @param pool_size list of 2 integers.
#' @param strides list of 2 integers.
#' @param padding string, `"same"` or `"valid"`.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' @param pool_mode string, `"max"` or `"avg"`.
#' 
#' @return A tensor, result of 2D pooling.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_pool2d <- function(x, pool_size, strides = c(1, 1), padding = "valid", data_format = NULL, pool_mode = "max") {
  keras$backend$pool2d(
    x = x,
    pool_size = as.integer(pool_size),
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    pool_mode = pool_mode
  )
}


#' 3D Pooling.
#' 
#' @param x Tensor or variable.
#' @param pool_size list of 3 integers.
#' @param strides list of 3 integers.
#' @param padding string, `"same"` or `"valid"`.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' @param pool_mode string, `"max"` or `"avg"`.
#' 
#' @return A tensor, result of 3D pooling.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_pool3d <- function(x, pool_size, strides = c(1, 1, 1), padding = "valid", 
                     data_format = NULL, pool_mode = "max") {
  keras$backend$pool3d(
    x = x,
    pool_size = as.integer(pool_size),
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    pool_mode = pool_mode
  )
}


#' Element-wise exponentiation.
#' 
#' @param x Tensor or variable.
#' @param a R integer.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_pow <- function(x, a) {
  keras$backend$pow(
    x = x,
    a = as.integer(a)
  )
}



#' Prints `message` and the tensor value when evaluated.
#'
#' Note that `print_tensor` returns a new tensor identical to `x` which should
#' be used in the following code. Otherwise the print operation is not taken
#' into account during evaluation.
#'
#' @param x Tensor to print.
#' @param message Message to print jointly with the tensor.
#'
#' @return The same tensor `x`, unchanged.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_print_tensor <- function(x, message = "") {
  keras$backend$print_tensor(
    x = x,
    message = message
  )
}


#' Multiplies the values in a tensor, alongside the specified axis.
#'
#' @param x A tensor or variable.
#' @param axis An integer, axis to compute the product over (axis indexes are
#'   1-based).
#' @param keepdims A boolean, whether to keep the dimensions or not. If
#'   `keepdims` is `FALSE`, the rank of the tensor is reduced by 1. If
#'   `keepdims` is `TRUE`, the reduced dimension is retained with length 1.
#'
#' @return A tensor with the product of elements of `x`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_prod <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$prod(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Returns a tensor with random binomial distribution of values.
#' 
#' @param shape A list of integers, the shape of tensor to create.
#' @param p A float, `0. <= p <= 1`, probability of binomial distribution.
#' @param dtype String, dtype of returned tensor.
#' @param seed Integer, random seed.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_random_binomial <- function(shape, p = 0.0, dtype = NULL, seed = NULL) {
  keras$backend$random_binomial(
    shape = backend_normalize_shape(shape),
    p = p,
    dtype = dtype,
    seed = as_nullable_integer(seed)
  )
}


#' Returns a tensor with normal distribution of values.
#'
#' @param shape A list of integers, the shape of tensor to create.
#' @param mean A float, mean of the normal distribution to draw samples.
#' @param stddev A float, standard deviation of the normal distribution to draw
#'   samples.
#' @param dtype String, dtype of returned tensor.
#' @param seed Integer, random seed.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_random_normal <- function(shape, mean = 0.0, stddev = 1.0, dtype = NULL, seed = NULL) {
  keras$backend$random_normal(
    shape = backend_normalize_shape(shape),
    mean = mean,
    stddev = stddev,
    dtype = dtype,
    seed = as_nullable_integer(seed)
  )
}


#' Instantiates a variable with values drawn from a normal distribution.
#'
#' @param shape Tuple of integers, shape of returned Keras variable.
#' @param mean Float, mean of the normal distribution.
#' @param scale Float, standard deviation of the normal distribution.
#' @param dtype String, dtype of returned Keras variable.
#' @param name String, name of returned Keras variable.
#' @param seed Integer, random seed.
#'
#' @return A Keras variable, filled with drawn samples.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_random_normal_variable <- function(shape, mean, scale, dtype = NULL, name = NULL, seed = NULL) {
  keras$backend$random_normal_variable(
    shape = backend_normalize_shape(shape),
    mean = mean,
    scale = scale,
    dtype = dtype,
    name = name,
    seed = as_nullable_integer(seed)
  )
}


#' Returns a tensor with uniform distribution of values.
#' 
#' @param shape A list of integers, the shape of tensor to create.
#' @param minval A float, lower boundary of the uniform distribution to draw samples.
#' @param maxval A float, upper boundary of the uniform distribution to draw samples.
#' @param dtype String, dtype of returned tensor.
#' @param seed Integer, random seed.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_random_uniform <- function(shape, minval = 0.0, maxval = 1.0, dtype = NULL, seed = NULL) {
  keras$backend$random_uniform(
    shape = backend_normalize_shape(shape),
    minval = minval,
    maxval = maxval,
    dtype = dtype,
    seed = as_nullable_integer(seed)
  )
}


#' Instantiates a variable with values drawn from a uniform distribution.
#' 
#' @param shape Tuple of integers, shape of returned Keras variable.
#' @param low Float, lower boundary of the output interval.
#' @param high Float, upper boundary of the output interval.
#' @param dtype String, dtype of returned Keras variable.
#' @param name String, name of returned Keras variable.
#' @param seed Integer, random seed.
#' 
#' @return A Keras variable, filled with drawn samples.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_random_uniform_variable <- function(shape, low, high, dtype = NULL, name = NULL, seed = NULL) {
  keras$backend$random_uniform_variable(
    shape = backend_normalize_shape(shape),
    low = low,
    high = high,
    dtype = dtype,
    name = name,
    seed = as_nullable_integer(seed)
  )
}


#' Rectified linear unit.
#' 
#' With default values, it returns element-wise `max(x, 0)`.
#' 
#' @param x A tensor or variable.
#' @param alpha A scalar, slope of negative section (default=`0.`).
#' @param max_value Saturation threshold.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_relu <- function(x, alpha = 0.0, max_value = NULL) {
  keras$backend$relu(
    x = x,
    alpha = alpha,
    max_value = max_value
  )
}


#' Repeats a 2D tensor.
#'
#' If x has shape (samples, dim) and n is 2, the output will have shape
#' (samples, 2, dim).
#'
#' @param x Tensor or variable.
#' @param n Integer, number of times to repeat.
#'
#' @return A tensor
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_repeat <- function(x, n) {
  keras$backend$`repeat`(
    x = x,
    n = as.integer(n)
  )
}


#' Repeats the elements of a tensor along an axis.
#' 
#' If `x` has shape `(s1, s2, s3)` and `axis` is `2`, the output
#' will have shape `(s1, s2 * rep, s3)`.
#' 
#' @param x Tensor or variable.
#' @param rep Integer, number of times to repeat.
#' @param axis Axis along which to repeat (axis indexes are 1-based)
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_repeat_elements <- function(x, rep, axis) {
  keras$backend$repeat_elements(
    x = x,
    rep = as.integer(rep),
    axis = as_axis(axis)
  )
}


#' Reset graph identifiers.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_reset_uids <- function() {
  keras$backend$reset_uids(
  )
}


#' Reshapes a tensor to the specified shape.
#' 
#' @param x Tensor or variable.
#' @param shape Target shape list.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_reshape <- function(x, shape) {
  keras$backend$reshape(
    x = x,
    shape = backend_normalize_shape(shape)
  )
}


#' Resizes the images contained in a 4D tensor.
#' 
#' @param x Tensor or variable to resize.
#' @param height_factor Positive integer.
#' @param width_factor Positive integer.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_resize_images <- function(x, height_factor, width_factor, data_format) {
  keras$backend$resize_images(
    x = x,
    height_factor = as.integer(height_factor),
    width_factor = as.integer(width_factor),
    data_format = data_format
  )
}


#' Resizes the volume contained in a 5D tensor.
#' 
#' @param x Tensor or variable to resize.
#' @param depth_factor Positive integer.
#' @param height_factor Positive integer.
#' @param width_factor Positive integer.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_resize_volumes <- function(x, depth_factor, height_factor, width_factor, data_format) {
  keras$backend$resize_volumes(
    x = x,
    depth_factor = as.integer(depth_factor),
    height_factor = as.integer(height_factor),
    width_factor = as.integer(width_factor),
    data_format = data_format
  )
}


#' Reverse a tensor along the specified axes.
#'
#' @param x Tensor to reverse.
#' @param axes Integer or list of integers of axes to reverse (axis indexes are
#'   1-based).
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_reverse <- function(x, axes) {
  keras$backend$reverse(
    x = x,
    axes = as_axis(axes)
  )
}

#' Iterates over the time dimension of a tensor
#'
#' @param step_function RNN step function.
#' @param inputs Tensor with shape (samples, ...) (no time dimension),
#'   representing input for the batch of samples at a certain time step.
#' @param initial_states Tensor with shape (samples, output_dim) (no time
#'   dimension), containing the initial values for the states used in the step
#'   function.
#' @param go_backwards Logical If `TRUE`, do the iteration over the time
#'   dimension in reverse order and return the reversed sequence.
#' @param mask Binary tensor with shape (samples, time, 1), with a zero for
#'   every element that is masked.
#' @param constants A list of constant values passed at each step.
#' @param unroll Whether to unroll the RNN or to use a symbolic loop
#'   (while_loop or scan depending on backend).
#' @param input_length Not relevant in the TensorFlow implementation. Must be
#'   specified if using unrolling with Theano.
#'
#' @return A list with:
#'
#'   - `last_output`: the latest output of the rnn, of shape (samples, ...) 
#'   - `outputs`: tensor with shape (samples, time, ...) where each entry
#'      `outputs[s, t]` is the output of the step function at time t for sample s. 
#'   - `new_states`: list of tensors, latest states returned by the step
#'      function, of shape (samples, ...).
#'   
#' @template roxlate-keras-backend  
#'
#' @export
k_rnn <- function(step_function, inputs, initial_states, go_backwards = FALSE, 
                  mask = NULL, constants = NULL, unroll = FALSE, 
                  input_length = NULL) {
  keras$backend$rnn(
    step_function = step_function,
    inputs = inputs,
    initial_states = initial_states,
    go_backwards = go_backwards,
    mask = mask,
    constants = constants,
    unroll = unroll,
    input_length = as.integer(input_length)
  )
}


#' Element-wise rounding to the closest integer.
#' 
#' In case of tie, the rounding mode used is "half to even".
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#'
#' @template roxlate-keras-backend  
#'   
#' @export
k_round <- function(x) {
  keras$backend$round(
    x = x
  )
}


#' 2D convolution with separable filters.
#' 
#' @param x input tensor
#' @param depthwise_kernel convolution kernel for the depthwise convolution.
#' @param pointwise_kernel kernel for the 1x1 convolution.
#' @param strides strides list (length 2).
#' @param padding string, `"same"` or `"valid"`.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' @param dilation_rate list of integers, dilation rates for the separable convolution.
#' 
#' @return Output tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_separable_conv2d <- function(x, depthwise_kernel, pointwise_kernel, strides = c(1, 1), 
                               padding = "valid", data_format = NULL, dilation_rate = c(1, 1)) {
  keras$backend$separable_conv2d(
    x = x,
    depthwise_kernel = depthwise_kernel,
    pointwise_kernel = pointwise_kernel,
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate)
  )
}


#' Sets the learning phase to a fixed value.
#' 
#' @param value Learning phase value, either 0 or 1 (integers).
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_set_learning_phase <- function(value) {
  keras$backend$set_learning_phase(
    value = as.integer(value)
  )
}


#' Sets the value of a variable, from an R array.
#' 
#' @param x Tensor to set to a new value.
#' @param value Value to set the tensor to, as an R array (of the same shape).
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_set_value <- function(x, value) {
  keras$backend$set_value(
    x = x,
    value = value
  )
}


#' Returns the symbolic shape of a tensor or variable.
#'
#' @param x A tensor or variable.
#'
#' @return A symbolic shape (which is itself a tensor).
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_shape <- function(x) {
  keras$backend$shape(
    x = x
  )
}


#' Element-wise sigmoid.
#' 
#' @param x A tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_sigmoid <- function(x) {
  keras$backend$sigmoid(
    x = x
  )
}


#' Element-wise sign.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_sign <- function(x) {
  keras$backend$sign(
    x = x
  )
}


#' Computes sin of x element-wise.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_sin <- function(x) {
  keras$backend$sin(
    x = x
  )
}


#' Softmax of a tensor.
#' 
#' @param x A tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_softmax <- function(x) {
  keras$backend$softmax(
    x = x
  )
}


#' Softplus of a tensor.
#' 
#' @param x A tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_softplus <- function(x) {
  keras$backend$softplus(
    x = x
  )
}


#' Softsign of a tensor.
#' 
#' @param x A tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_softsign <- function(x) {
  keras$backend$softsign(
    x = x
  )
}


#' Categorical crossentropy with integer targets.
#' 
#' @param target An integer tensor.
#' @param output A tensor resulting from a softmax (unless `from_logits` is TRUE, in which case `output` is expected to be the logits).
#' @param from_logits Boolean, whether `output` is the result of a softmax, or is a tensor of logits.
#' 
#' @return Output tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_sparse_categorical_crossentropy <- function(target, output, from_logits = FALSE) {
  keras$backend$sparse_categorical_crossentropy(
    target = target,
    output = output,
    from_logits = from_logits
  )
}


#' Pads the 2nd and 3rd dimensions of a 4D tensor.
#' 
#' @param x Tensor or variable.
#' @param padding Tuple of 2 lists, padding pattern.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#' 
#' @return A padded 4D tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_spatial_2d_padding <- function(x, padding = list(list(1, 1), list(1, 1)), data_format = NULL) {
  keras$backend$spatial_2d_padding(
    x = x,
    padding = padding,
    data_format = data_format
  )
}


#' Pads 5D tensor with zeros along the depth, height, width dimensions.
#'
#' Pads these dimensions with respectively `padding[[1]]`, `padding[[2]]`, and
#' `padding[[3]]` zeros left and right. For 'channels_last' data_format, the
#' 2nd, 3rd and 4th dimension will be padded. For 'channels_first' data_format,
#' the 3rd, 4th and 5th dimension will be padded.
#'
#' @param x Tensor or variable.
#' @param padding List of 3 lists, padding pattern.
#' @param data_format string, `"channels_last"` or `"channels_first"`.
#'
#' @return A padded 5D tensor.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_spatial_3d_padding <- function(x, 
                                 padding = list(list(1, 1), list(1, 1), list(1, 1)), 
                                 data_format = NULL) {
  keras$backend$spatial_3d_padding(
    x = x,
    padding = padding,
    data_format = data_format
  )
}


#' Element-wise square root.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_sqrt <- function(x) {
  keras$backend$sqrt(
    x = x
  )
}


#' Element-wise square.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_square <- function(x) {
  keras$backend$square(
    x = x
  )
}


#' Removes a 1-dimension from the tensor at index `axis`.
#'
#' @param x A tensor or variable.
#' @param axis Axis to drop (axis indexes are 1-based).
#'
#' @return A tensor with the same data as `x` but reduced dimensions.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_squeeze <- function(x, axis) {
  keras$backend$squeeze(
    x = x,
    axis = as_axis(axis)
  )
}


#' Stacks a list of rank `R` tensors into a rank `R+1` tensor.
#'
#' @param x List of tensors.
#' @param axis Axis along which to perform stacking (axis indexes are 1-based).
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_stack <- function(x, axis = 1) {
  keras$backend$stack(
    x = x,
    axis = as_axis(axis)
  )
}


#' Standard deviation of a tensor, alongside the specified axis.
#'
#' @param x A tensor or variable.
#' @param axis An integer, the axis to compute the standard deviation over
#'   (axis indexes are 1-based).
#' @param keepdims A boolean, whether to keep the dimensions or not. If
#'   `keepdims` is `FALSE`, the rank of the tensor is reduced by 1. If
#'   `keepdims` is `TRUE`, the reduced dimension is retained with length 1.
#'
#' @return A tensor with the standard deviation of elements of `x`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_std <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$std(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Returns `variables` but with zero gradient w.r.t. every other variable.
#'
#' @param variables tensor or list of tensors to consider constant with respect
#'   to any other variable.
#'
#' @return A single tensor or a list of tensors (depending on the passed
#'   argument) that has constant gradient with respect to any other variable.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_stop_gradient <- function(variables) {
  keras$backend$stop_gradient(
    variables = variables
  )
}


#' Sum of the values in a tensor, alongside the specified axis.
#'
#' @param x A tensor or variable.
#' @param axis An integer, the axis to sum over (axis indexes are 1-based).
#' @param keepdims A boolean, whether to keep the dimensions or not. If
#'   `keepdims` is `FALSE`, the rank of the tensor is reduced by 1. If
#'   `keepdims` is `TRUE`, the reduced dimension is retained with length 1.
#'
#' @return A tensor with sum of `x`.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_sum <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$sum(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Switches between two operations depending on a scalar value.
#' 
#' Note that both `then_expression` and `else_expression`
#' should be symbolic tensors of the *same shape*.
#' 
#' @param condition tensor (`int` or `bool`).
#' @param then_expression either a tensor, or a function that returns a tensor.
#' @param else_expression either a tensor, or a function that returns a tensor.
#' 
#' @return The selected tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_switch <- function(condition, then_expression, else_expression) {
  keras$backend$switch(
    condition = condition,
    then_expression = then_expression,
    else_expression = else_expression
  )
}



#' Element-wise tanh.
#' 
#' @param x A tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_tanh <- function(x) {
  keras$backend$tanh(
    x = x
  )
}


#' Pads the middle dimension of a 3D tensor.
#' 
#' @param x Tensor or variable.
#' @param padding List of 2 integers, how many zeros to add at the start and end of dim 1.
#' 
#' @return A padded 3D tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_temporal_padding <- function(x, padding = c(1, 1)) {
  keras$backend$temporal_padding(
    x = x,
    padding = as_integer_tuple(padding, force_tuple = TRUE)
  )
}

#' Creates a tensor by tiling `x` by `n`.
#' 
#' @param x A tensor or variable
#' @param n A list of integers. The length must be the same as the number of dimensions in `x`.
#' 
#' @return A tiled tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_tile <- function(x, n) {
  keras$backend$tile(
    x = x,
    n = list(as.integer(n))
  )
}


#' Converts a sparse tensor into a dense tensor and returns it.
#'
#' @param tensor A tensor instance (potentially sparse).
#'
#' @return A dense tensor.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_to_dense <- function(tensor) {
  keras$backend$to_dense(
    tensor = tensor
  )
}


#' Transposes a tensor and returns it.
#' 
#' @param x Tensor or variable.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_transpose <- function(x) {
  keras$backend$transpose(
    x = x
  )
}


#' Returns a tensor with truncated random normal distribution of values.
#' 
#' The generated values follow a normal distribution
#' with specified mean and standard deviation,
#' except that values whose magnitude is more than
#' two standard deviations from the mean are dropped and re-picked.
#' 
#' @param shape A list of integers, the shape of tensor to create.
#' @param mean Mean of the values.
#' @param stddev Standard deviation of the values.
#' @param dtype String, dtype of returned tensor.
#' @param seed Integer, random seed.
#' 
#' @return A tensor.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_truncated_normal <- function(shape, mean = 0.0, stddev = 1.0, dtype = NULL, seed = NULL) {
  keras$backend$truncated_normal(
    shape = backend_normalize_shape(shape),
    mean = mean,
    stddev = stddev,
    dtype = dtype,
    seed = as_nullable_integer(seed)
  )
}


#' Update the value of `x` to `new_x`.
#' 
#' @param x A `Variable`.
#' @param new_x A tensor of same shape as `x`.
#' 
#' @return The variable `x` updated.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_update <- function(x, new_x) {
  keras$backend$update(
    x = x,
    new_x = new_x
  )
}


#' Update the value of `x` by adding `increment`.
#' 
#' @param x A `Variable`.
#' @param increment A tensor of same shape as `x`.
#' 
#' @return The variable `x` updated.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_update_add <- function(x, increment) {
  keras$backend$update_add(
    x = x,
    increment = increment
  )
}


#' Update the value of `x` by subtracting `decrement`.
#' 
#' @param x A `Variable`.
#' @param decrement A tensor of same shape as `x`.
#' 
#' @return The variable `x` updated.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_update_sub <- function(x, decrement) {
  keras$backend$update_sub(
    x = x,
    decrement = decrement
  )
}


#' Variance of a tensor, alongside the specified axis.
#'
#' @param x A tensor or variable.
#' @param axis An integer, the axis to compute the variance over (axis indexes
#'   are 1-based).
#' @param keepdims A boolean, whether to keep the dimensions or not. If
#'   `keepdims` is `FALSE`, the rank of the tensor is reduced by 1. If
#'   `keepdims` is `TRUE`, the reduced dimension is retained with length 1.
#'
#' @return A tensor with the variance of elements of `x`.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_var <- function(x, axis = NULL, keepdims = FALSE) {
  keras$backend$var(
    x = x,
    axis = as_axis(axis),
    keepdims = keepdims
  )
}


#' Instantiates a variable and returns it.
#' 
#' @param value Numpy array, initial value of the tensor.
#' @param dtype Tensor type.
#' @param name Optional name string for the tensor.
#' @param constraint Optional projection function to be applied to the variable after an optimizer update.
#' 
#' @return A variable instance (with Keras metadata included).
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_variable <- function(value, dtype = NULL, name = NULL, constraint = NULL) {
  keras$backend$variable(
    value = value,
    dtype = dtype,
    name = name,
    constraint = constraint
  )
}


#' Instantiates an all-zeros variable and returns it.
#' 
#' @param shape Tuple of integers, shape of returned Keras variable
#' @param dtype String, data type of returned Keras variable
#' @param name String, name of returned Keras variable
#' 
#' @return A variable (including Keras metadata), filled with `0.0`.
#' 
#' @template roxlate-keras-backend  
#' 
#' @export
k_zeros <- function(shape, dtype = NULL, name = NULL) {
  keras$backend$zeros(
    shape = backend_normalize_shape(shape),
    dtype = dtype,
    name = name
  )
}


#' Instantiates an all-zeros variable of the same shape as another tensor.
#'
#' @param x Keras variable or Keras tensor.
#' @param dtype String, dtype of returned Keras variable. NULL uses the dtype
#'   of x.
#' @param name String, name for the variable to create.
#'
#' @return A Keras variable with the shape of x filled with zeros.
#'
#' @template roxlate-keras-backend  
#'
#' @export
k_zeros_like <- function(x, dtype = NULL, name = NULL) {
  keras$backend$zeros_like(
    x = x,
    dtype = dtype,
    name = name
  )
}

as_axis <- function(axis) {
  if (length(axis) > 1) {
    sapply(axis, as_axis)
  } else {
    axis <- as_nullable_integer(axis)
    if (is.null(axis))
      axis
    else if (axis == -1L)
      axis
    else
      axis - 1L
  }
}


backend_normalize_shape <- function(shape) {
  
  # if it's a Python object or a list with python objects then leave it alone
  if (inherits(shape, "python.builtin.object"))
    return(shape)
  
  if (is.list(shape)) {
    if (any(sapply(unlist(shape), function(x) inherits(x, "python.builtin.object"))))
      return(shape)
  }
  
  normalize_shape(shape)
}
