


#' Layer/Model configuration
#'
#' A layer config is an object returned from `get_config()` that contains the
#' configuration of a layer or model. The same layer or model can be
#' reinstantiated later (without its trained weights) from this configuration
#' using `from_config()`. The config does not include connectivity information,
#' nor the class name (those are handled externally).
#'
#' @param object Layer or model object
#' @param config Object with layer or model configuration
#' @param custom_objects list of custom objects needed to instantiate the layer,
#'   e.g., custom layers defined by `new_layer_class()` or similar.
#'
#' @returns `get_config()` returns an object with the configuration,
#'   `from_config()` returns a re-instantiation of the object.
#'
#' @note Objects returned from `get_config()` are not serializable via RDS. If
#'   you want to save and restore a model across sessions, you can use
#'   [`save_model_config()`] (for model configuration only, not weights)
#'   or [`save_model()`] to save the model configuration and weights
#'   to the filesystem.
#'
#' @family model functions
#' @family layer methods
#'
#' @export
get_config <- function(object) {
  config <- object$get_config()
  attr(config, "__class__") <- object$`__class__`
  config
}

#' @rdname get_config
#' @export
from_config <- function(config, custom_objects = NULL) {
  class <- attr(config, "__class__", TRUE) #%||% keras$Model
  class <- resolve_py_obj(class, env = parent.frame())
  if(is.null(class) || reticulate::py_is_null_xptr(class))
    stop(glue::trim('
       attr(config, "__class__") is an invalid pointer from a previous R session.
       The output of `get_config()` is not serializable via RDS.'))

  args <- list(config)
  args$custom_objects <- normalize_custom_objects(custom_objects)
  do.call(class$from_config, args)
}


# TODO: we might be able to make get_config() output serializable via saveRDS,
# if we replace __class__ with a module address, like
# `__class__`$`__module__` and `__module__`$`__name__`, but we'd need checks
# to make sure it's builtin/ importable python module.
#
# attr(config, "__class__.__module__") <- `__class__`$`__module__`
# attr(config, "__class__.__name__") <- `__class__`$`__name__`

# OR: make it serializable only for models:
# `__class__` <- object$`__class__`
# if (!py_is(`__class__`, keras$Model))
#   attr(config, "__class__") <- `__class__`
# Then in from_config(): class <- attr(...) %||% keras$Model

# @param class The Keras class to restore. This can be:
# You can update with `attr(config, "__class__") <- <__class__>`, where <__class__> can be
# - An R function like `layer_dense` or a custom `Layer()` class.
# - An R language object like `quote(layer_dense)` (will be evaluated in the calling frame)
# - A Python class object, like `reticulate::import("keras")$layers$Dense`'))

# class <- keras$Model
# class <- attr(config, "__class__", TRUE)
# if(is.null(class) || reticulate::py_is_null_xptr(class)) {
#   stop("`attr(config, '__class__'` is a null pointer from an external session",
#        "If you know the original config class, you can provide it as an R object (e.g., class = layer_dense)")
#   class <- import(attr(config, "__class__.__module__", TRUE))[[attr(config, "__class__.__name__")]]
# }


#' Layer/Model weights as R arrays
#'
#' @param object Layer or model object
#' @param trainable if `NA` (the default), all weights are returned. If `TRUE`,
#'   only weights of trainable variables are returned. If `FALSE`, only weights
#'   of non-trainable variables are returned.
#' @param weights Weights as R array
#'
#' @note You can access the Layer/Model as `KerasVariables` (which are also
#'   backend-native tensors like `tf.Variable`) at `object$weights`,
#'   `object$trainable_weights`, or `object$non_trainable_weights`
#'
#' @family model persistence
#' @family layer methods
#'
#' @returns A list of R arrays.
#' @export
get_weights <- function(object, trainable = NA) {
  if(is.na(trainable))
    x <- object$get_weights()
  else if(isTRUE(trainable))
    x <- lapply(object$trainable_weights, function(x) x$numpy())
  else if (isFALSE(trainable))
    x <- lapply(object$non_trainable_weights, function(x) x$numpy())
  else stop("`trainable` must be NA, TRUE, or FALSE")
  lapply(x, as_r_value)
}

#' @rdname get_weights
#' @export
set_weights <- function(object, weights) {
  object$set_weights(weights)
  invisible(object)
}




#' Count the total number of scalars composing the weights.
#'
#' @param object Layer or model object
#'
#' @returns An integer count
#'
#' @family layer methods
#'
#' @export
count_params <- function(object) {
  object$count_params()
}



#' Reset the state for a model, layer or metric.
#'
#' @param object Model, Layer, or Metric instance
#'
#' Not all Layers have resettable state (E.g., `adapt()`-able preprocessing
#' layers and rnn layers have resettable state, but a `layer_dense()` does not).
#' Calling this on a Layer instance without any resettable-state will error.
#'
#' @family layer methods
#  @family preprocessing layers
#  @family metrics
#  @family rnn layers
#'
#' @returns `object`, invisibly.
#' @export
reset_state <- function(object) {
  object$reset_state()
  invisible(object)
}
