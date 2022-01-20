

#' @title Loss functions
#' @rdname loss-functions
#' @name loss-functions
#'
#' @param y_true Ground truth values. shape = `[batch_size, d1, .. dN]`.
#' @param y_pred The predicted values. shape = `[batch_size, d1, .. dN]`.
#'   (Tensor of the same shape as `y_true`)
#'
#' @param axis The axis along which to compute crossentropy (the features axis).
#'   Axis is 1-based (e.g, first axis is `axis=1`). Defaults to `-1` (the last axis).
#'
#' @param ... Additional arguments passed on to the Python callable (for forward
#'   and backwards compatibility).
#'
#' @param reduction Only applicable if `y_true` and `y_pred` are missing. Type
#'   of `keras$losses$Reduction` to apply to loss. Default value is `AUTO`.
#'   `AUTO` indicates that the reduction option will be determined by the usage
#'   context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When
#'   used with `tf$distribute$Strategy`, outside of built-in training loops such
#'   as `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an
#'   error. Please see this custom training [tutorial](
#'   https://www.tensorflow.org/tutorials/distribute/custom_training) for more
#'   details.
#'
#' @param name Only applicable if `y_true` and `y_pred` are missing. Optional
#'   name for the Loss instance.
#'
#' @details Loss functions for model training. These are typically supplied in
#'   the `loss` parameter of the [compile.keras.engine.training.Model()]
#'   function.
#'
#' @returns If called with `y_true` and `y_pred`, then the corresponding loss is
#'   evaluated and the result returned (as a tensor). Alternatively, if `y_true`
#'   and `y_pred` are missing, then a callable is returned that will compute the
#'   loss function and, by default, reduce the loss to a scalar tensor; see the
#'   `reduction` parameter for details. (The callable is a typically a class
#'   instance that inherits from `keras$losses$Loss`).
#'
#' @seealso [compile.keras.engine.training.Model()],
#'   [loss_binary_crossentropy()]
#'
NULL



#' @section binary_crossentropy:
#'
#'   Computes the binary crossentropy loss.
#'
#'   `label_smoothing` details: Float in `[0, 1]`. If `> 0` then smooth the labels
#'   by squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
#'   for the target class and `0.5 * label_smoothing` for the non-target class.
#'
#' @param from_logits Whether `y_pred` is expected to be a logits tensor. By
#'   default, we assume that `y_pred` encodes a probability distribution.
#'
#' @rdname loss-functions
#' @aliases "binary_crossentropy", "BinaryCrossentropy"
#' @export
loss_binary_crossentropy <-
  function(y_true, y_pred,
           from_logits = FALSE, label_smoothing = 0, axis = -1L,
           ..., reduction = "auto", name = "binary_crossentropy") {
    args <- capture_args(match.call(), list(axis = as_axis))
    py_callable <- if (missing(y_true) && missing(y_pred))
      keras$losses$BinaryCrossentropy
    else
      keras$losses$binary_crossentropy
    do.call(py_callable, args)
  }
attr(loss_binary_crossentropy, "py_function_name") <- "binary_crossentropy"



#' @section categorical_crossentropy:
#'
#'   Computes the categorical crossentropy loss.
#'
#'   When using the categorical_crossentropy loss, your targets should be in
#'   categorical format (e.g. if you have 10 classes, the target for each sample
#'   should be a 10-dimensional vector that is all-zeros except for a 1 at the
#'   index corresponding to the class of the sample). In order to convert
#'   integer targets into categorical targets, you can use the Keras utility
#'   function [to_categorical()]:
#'
#'   `categorical_labels <- to_categorical(int_labels, num_classes = NULL)`
#'
#' @param from_logits Whether `y_pred` is expected to be a logits tensor. By
#'   default we assume that `y_pred` encodes a probability distribution.
#' @param label_smoothing Float in `[0, 1]`. If `> 0` then smooth the labels.
#'   For example, if `0.1`, use `0.1 / num_classes` for non-target labels and
#'   `0.9 + 0.1 / num_classes` for target labels.
#'
#' @rdname loss-functions
#' @export
loss_categorical_crossentropy <-
  function(y_true, y_pred,
           from_logits = FALSE, label_smoothing = 0L, axis = -1L,
           ..., reduction = "auto", name = "categorical_crossentropy") {
  args <- capture_args(match.call(), list(axis = as_axis))
  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$CategoricalCrossentropy
  else
    keras$losses$categorical_crossentropy
  do.call(py_callable, args)
}
attr(loss_categorical_crossentropy, "py_function_name") <- "categorical_crossentropy"
c("categorical_crossentropy", "CategoricalCrossentropy")



#' @rdname loss-functions
#' @export
loss_categorical_hinge <-
  function(y_true, y_pred,
           ..., reduction = "auto", name = "categorical_hinge") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$CategoricalHinge
  else
    keras$losses$categorical_hinge
  do.call(py_callable, args)
}
attr(loss_categorical_hinge, "py_function_name") <- "categorical_hinge"
c("categorical_hinge", "CategoricalHinge")

# LossCategoricalHinge
# keras$losses$CategoricalHinge()


#' @rdname loss-functions
#' @export
loss_cosine_similarity <- function(y_true, y_pred, axis = -1L,
                                   ..., reduction = "auto", name = "cosine_similarity") {
  args <- capture_args(match.call(), list(axis = as_axis))
  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$CosineSimilarity
  else
    keras$losses$cosine_similarity
  do.call(py_callable, args)
}
attr(loss_cosine_similarity, "py_function_name") <- "cosine_similarity"
c("cosine_similarity", "CosineSimilarity")

#' @rdname loss-functions
#' @export
loss_hinge <- function(y_true, y_pred, ..., reduction = "auto", name = "hinge") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$Hinge
  else
    keras$losses$hinge
  do.call(py_callable, args)
}
attr(loss_hinge, "py_function_name") <- "hinge"
c("hinge", "Hinge")


#' @section huber:
#'
#' Computes Huber loss value.
#' For each value x in `error = y_true - y_pred`:
#' ```
#' loss = 0.5 * x^2                  if |x| <= d
#' loss = d * |x| - 0.5 * d^2        if |x| > d
#' ```
#' where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
#'
#' @param delta A float, the point where the Huber loss function changes from a
#'   quadratic to linear.
#'
#' @rdname loss-functions
#' @export
loss_huber <- function(y_true, y_pred, delta = 1, ..., reduction = "auto", name = "huber_loss") {
  args <- capture_args(match.call())
  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$Huber
  else
    keras$losses$huber
  do.call(py_callable, args)
}
attr(loss_huber, "py_function_name") <- "huber"
c("huber", "Huber")



#' @rdname loss-functions
#' @export
loss_kullback_leibler_divergence <-
  function(y_true, y_pred,
           ..., reduction = "auto", name = "kl_divergence") {
    args <- capture_args(match.call())

    py_callable <- if (missing(y_true) && missing(y_pred))
      keras$losses$KLDivergence
    else
      keras$losses$kullback_leibler_divergence
    do.call(py_callable, args)
  }
attr(loss_kullback_leibler_divergence, "py_function_name") <- "kullback_leibler_divergence"
c("kl_divergence", "kld", "KLD", "KLDivergence", "kullback_leibler_divergence")

#' @rdname loss-functions
#' @export
loss_kl_divergence <- loss_kullback_leibler_divergence

#' @section log_cosh:
#'
#' Logarithm of the hyperbolic cosine of the prediction error.
#'
#'   `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
#'   to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
#'   like the mean squared error, but will not be so strongly affected by the
#'   occasional wildly incorrect prediction. However, it may return NaNs if the
#'   intermediate value `cosh(y_pred - y_true)` is too large to be represented
#'   in the chosen precision.
#'
#' @rdname loss-functions
#' @export
loss_logcosh <- function(y_true, y_pred, ..., reduction = "auto", name = "log_cosh") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$LogCosh
  else
    keras$losses$logcosh
  do.call(py_callable, args)
}
attr(loss_logcosh, "py_function_name") <- "log_cosh"
c("log_cosh", "logcosh", "LogCosh")


#' @rdname loss-functions
#' @export
loss_mean_absolute_error <-
  function(y_true, y_pred,
           ..., reduction = "auto", name = "mean_absolute_error") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$MeanAbsoluteError
  else
    keras$losses$mean_absolute_error
  do.call(py_callable, args)
}
attr(loss_mean_absolute_error, "py_function_name") <- "mean_absolute_error"
c("mae", "MAE", "mean_absolute_error", "MeanAbsoluteError")





#' @rdname loss-functions
#' @export
loss_mean_absolute_percentage_error <-
  function(y_true, y_pred, ..., reduction = "auto", name = "mean_absolute_percentage_error") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$MeanAbsolutePercentageError
  else
    keras$losses$mean_absolute_percentage_error
  do.call(py_callable, args)
}
attr(loss_mean_absolute_percentage_error, "py_function_name") <- "mean_absolute_percentage_error"
c("mape", "MAPE", "mean_absolute_percentage_error", "MeanAbsolutePercentageError")

#' @rdname loss-functions
#' @export
loss_mean_squared_error <- function(y_true, y_pred,
                                    ..., reduction = "auto", name = "mean_squared_error") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$MeanSquaredError
  else
    keras$losses$mean_squared_error
  do.call(py_callable, args)
}
attr(loss_mean_squared_error, "py_function_name") <- "mean_squared_error"
c("mse", "MSE", "mean_squared_error", "MeanSquaredError")



#' @rdname loss-functions
#' @export
loss_mean_squared_logarithmic_error <-
  function(y_true, y_pred, ...,
           reduction = "auto", name = "mean_squared_logarithmic_error") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$MeanSquaredLogarithmicError
  else
    keras$losses$mean_squared_logarithmic_error
  do.call(py_callable, args)
}
attr(loss_mean_squared_logarithmic_error, "py_function_name") <- "mean_squared_logarithmic_error"
c("msle", "MSLE", "mean_squared_logarithmic_error", "MeanSquaredLogarithmicError")



#' @rdname loss-functions
#' @export
loss_poisson <- function(y_true, y_pred, ..., reduction = "auto", name = "poisson") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$Poisson
  else
    keras$losses$poisson
  do.call(py_callable, args)
}
attr(loss_poisson, "py_function_name") <- "poisson"
c("poisson", "Poisson")


#' @rdname loss-functions
#' @export
loss_sparse_categorical_crossentropy <-
  function(y_true, y_pred, from_logits = FALSE, axis = -1L,
           ..., reduction = "auto", name = "sparse_categorical_crossentropy") {
  args <- capture_args(match.call(), list(axis = as_axis))

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$SparseCategoricalCrossentropy
  else
    keras$losses$sparse_categorical_crossentropy
  do.call(py_callable, args)
}
attr(loss_sparse_categorical_crossentropy, "py_function_name") <- "sparse_categorical_crossentropy"
c("sparse_categorical_crossentropy", "SparseCategoricalCrossentropy")



#' @rdname loss-functions
#' @export
loss_squared_hinge <- function(y_true, y_pred, ..., reduction = "auto", name = "squared_hinge") {
  args <- capture_args(match.call())

  py_callable <- if (missing(y_true) && missing(y_pred))
    keras$losses$SquaredHinge
  else
    keras$losses$squared_hinge
  do.call(py_callable, args)
}
attr(loss_squared_hinge, "py_function_name") <- "squared_hinge"
c("squared_hinge", "SquaredHinge")









#' (Deprecated) loss_cosine_proximity
#'
#' `loss_cosine_proximity` is deprecated and will be removed in a future
#' version. It has been renamed to `loss_cosine_similarity`().
#'
#' @param ... passed on to [loss_cosine_similarity()]
#'
#' @keywords internal
#' @export
loss_cosine_proximity <- function(...) {
  warning("loss_cosine_proximity is deprecated and will be removed in a future version.",
          " Please use loss_cosine_similarity instead.")
  loss_cosine_similarity(...)
}
attr(loss_cosine_proximity, "py_function_name") <- "cosine_proximity"
