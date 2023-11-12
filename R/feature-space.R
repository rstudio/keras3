#' @export
#' @rdname layer_feature_space
feature_cross <- function(feature_names, crossing_dim, output_mode = "one_hot") {
  args <- capture_args2(list(crossing_dim = as_integer))
  keras$utils$FeatureSpace$cross(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_custom <-
function(dtype, preprocessor, output_mode) {
  args <- capture_args2()
  keras$utils$FeatureSpace$feature(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_float <-
function(name = NULL) {
  args <- capture_args2()
  keras$utils$FeatureSpace$float(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_float_rescaled <-
function(scale = 1.0,
         offset = 0.0,
         name = NULL) {
  args <- capture_args2()
  keras$utils$FeatureSpace$float_rescaled(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_float_normalized <-
function(name = NULL) {
  args <- capture_args2()
  keras$utils$FeatureSpace$float_normalized(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_float_discretized <-
function(num_bins,
         bin_boundaries = NULL,
         output_mode = "one_hot",
         name = NULL) {
  args <- capture_args2(list(num_bins = as_integer))
  keras$utils$FeatureSpace$float_discretized(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_integer_categorical <-
function(max_tokens = NULL,
         num_oov_indices = 1,
         output_mode = "one_hot",
         name = NULL) {
  args <- capture_args2(list(max_tokens = as_integer, num_oov_indices = as_integer))
  keras$utils$FeatureSpace$integer_categorical(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_string_categorical <-
function(max_tokens = NULL,
         num_oov_indices = 1,
         output_mode = "one_hot",
         name = NULL) {
  args <- capture_args2(list(max_tokens = as_integer, num_oov_indices = as_integer))
  keras$utils$FeatureSpace$string_categorical(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_string_hashed <-
function(num_bins,
         output_mode = "one_hot",
         name = NULL) {
  args <- capture_args2(list(num_bins = as_integer))
  keras$utils$FeatureSpace$string_hashed(!!!args)
}

#' @export
#' @rdname layer_feature_space
feature_integer_hashed <-
function(num_bins,
         output_mode = "one_hot",
         name = NULL) {
  args <- capture_args2(list(num_bins = as_integer))
  keras$utils$FeatureSpace$integer_hashed(!!!args)
}
