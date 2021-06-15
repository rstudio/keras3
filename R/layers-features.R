#' Constructs a DenseFeatures.
#'
#' A layer that produces a dense Tensor based on given feature_columns.
#'
#' @inheritParams layer_dense
#'
#' @param feature_columns An iterable containing the FeatureColumns to use as
#'    inputs to your model. All items should be instances of classes derived from
#'    `DenseColumn` such as `numeric_column`, `embedding_column`, `bucketized_column`,
#'    `indicator_column`. If you have categorical features, you can wrap them with an
#'    `embedding_column` or `indicator_column`. See `tfestimators::feature_columns()`.
#'
#' @family core layers
#'
#' @export
layer_dense_features <- function(object, feature_columns, name = NULL,
                                 trainable = NULL, input_shape = NULL,
                                 batch_input_shape = NULL, batch_size = NULL, dtype = NULL,
                                 weights = NULL) {

  if (!is_tensorflow_implementation() || !tensorflow::tf_version() >= "1.14")
    stop("layer_dense_features requires TensorFlow implementation and version >= 1.14")

  # feature_columns must be unamed otherwise they are converted to a dict
  names(feature_columns) <- NULL

  create_layer(keras$layers$DenseFeatures, object, list(
    feature_columns = feature_columns,
    name = name,
    trainable = trainable,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    weights = weights
  ))
}
