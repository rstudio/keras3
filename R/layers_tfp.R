


#' Pass-through layer that adds a KL divergence penalty to the model loss
#'
#' @inheritParams layer_dense
#'
#' @param distribution Distribution instance corresponding to `b` as in  `KL[a, b]`.
#'  The previous layer's output is presumed to be a `Distribution` instance and is `a`.
#' @param use_exact_kl Logical indicating if KL divergence should be
#'  calculated exactly via `tfp$distributions$kl_divergence` or via Monte Carlo approximation.
#'  Default value: `FALSE`.
#' @param test_points_reduce_axis Integer vector or scalar representing dimensions
#'  over which to `reduce_mean` while calculating the Monte Carlo approximation of the KL divergence.
#'  As is with all `tf$reduce_*` ops, NULL means reduce over all dimensions;
#'  `()` means reduce over none of them. Default value: `()` (i.e., no reduction).
#' @param test_points_fn A callable taking a `tfp$distribution` instance and returning a tensor
#'  used for random test points to approximate the KL divergence.
#'  Default value: `tf$convert_to_tensor`.
#' @param weight Multiplier applied to the calculated KL divergence for each Keras batch member.
#' Default value: NULL (i.e., do not weight each batch member).
#'
#' @family Probabilistic layers (require TensorFlow probability)
#'
#' @export
layer_kl_divergence_add_loss <- function(object,
                                         distribution,
                                         use_exact_kl = NULL,
                                         test_points_reduce_axis = NULL,
                                         test_points_fn = NULL,
                                         weight = NULL,
                                         input_shape = NULL,
                                         batch_input_shape = NULL,
                                         batch_size = NULL,
                                         dtype = NULL,
                                         name = NULL,
                                         trainable = NULL,
                                         weights = NULL) {
  if (backend()$backend() != "tensorflow")
    stop("TensorFlow probability layers can only be used with the TensorFlow backend.")
  
  args <- list(
    distribution_b = distribution,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
  
  if (!is.null(use_exact_kl))
    args$use_exact_kl = use_exact_kl
  if (!is.null(test_points_reduce_axis))
    args$test_points_reduce_axis = test_points_reduce_axis
  if (!is.null(test_points_fn))
    args$test_points_fn = test_points_fn
  if (!is.null(weight))
    args$weight = weight
  
  tensorflow_probability <- import("tensorflow_probability")
  create_layer(
    tensorflow_probability$python$layers$distribution_layer$KLDivergenceAddLoss,
    object,
    args
  )
  
}


#' A `d`-variate MVNTriL Keras layer from `d+d*(d+1)/ 2` params.
#'
#' @inheritParams layer_dense
#'
#' @param event_shape Integer vector tensor representing the shape of single draw from this distribution.
#' @param convert_to_tensor_fn A callable that takes a `tfd$Distribution` instance and returns a
#'  `tf$Tensor`-like object. Default value: `tfd$Distribution$sample`.
#' @param validate_args  Logical, default `FALSE`. When `TRUE` distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When `FALSE` invalid inputs may
#'  silently render incorrect outputs. Default value: `FALSE`.
#'
#' @family Probabilistic layers (require TensorFlow probability)
#'
#' @export
layer_multivariate_normal_tril <- function(object,
                                           event_size,
                                           convert_to_tensor_fn = NULL,
                                           validate_args = NULL,
                                           batch_input_shape = NULL,
                                           input_shape = NULL,
                                           batch_size = NULL,
                                           dtype = NULL,
                                           name = NULL,
                                           trainable = NULL,
                                           weights = NULL) {
  if (backend()$backend() != "tensorflow")
    stop("TensorFlow probability layers can only be used with the TensorFlow backend.")
  
  args <- list(
    event_size = as.integer(event_size),
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
  
  if (!is.null(convert_to_tensor_fn))
    args$convert_to_tensor_fn = convert_to_tensor_fn
  if (!is.null(validate_args))
    args$validate_args = validate_args
  
  tensorflow_probability <- import("tensorflow_probability")
  create_layer(
    tensorflow_probability$python$layers$distribution_layer$MultivariateNormalTriL,
    object,
    args
  )
  
}

#' An Independent-Bernoulli Keras layer from `prod(event_shape)` params
#'
#' @inheritParams layer_dense
#'
#' @param event_size Scalar integer representing the size of single draw from this distribution.
#' @param convert_to_tensor_fn A callable that takes a `tfd$Distribution` instance and returns a
#'  `tf$Tensor`-like object. Default value: `tfd$Distribution$sample`.
#' @param sample_dtype `dtype` of samples produced by this distribution.
#'  Default value: `NULL` (i.e., previous layer's `dtype`).
#' @param validate_args  Logical, default `FALSE`. When `TRUE` distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When `FALSE` invalid inputs may
#'  silently render incorrect outputs. Default value: `FALSE`.
#'
#' @family Probabilistic layers (require TensorFlow probability)
#'
#' @export
layer_independent_bernoulli <- function(object,
                                        event_shape,
                                        convert_to_tensor_fn = NULL,
                                        sample_dtype = NULL,
                                        validate_args = NULL,
                                        batch_input_shape = NULL,
                                        input_shape = NULL,
                                        batch_size = NULL,
                                        dtype = NULL,
                                        name = NULL,
                                        trainable = NULL,
                                        weights = NULL) {
  if (backend()$backend() != "tensorflow")
    stop("TensorFlow probability layers can only be used with the TensorFlow backend.")
  
  args <- list(
    event_shape = as.integer(event_shape),
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )
  
  if (!is.null(convert_to_tensor_fn))
    args$convert_to_tensor_fn = convert_to_tensor_fn
  if (!is.null(sample_dtype))
    args$sample_dtype = sample_dtype
  if (!is.null(validate_args))
    args$validate_args = validate_args
  
  tensorflow_probability <- import("tensorflow_probability")
  create_layer(
    tensorflow_probability$python$layers$distribution_layer$IndependentBernoulli,
    object,
    args
  )
  
}
