



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
#' @return `get_config()` returns an object with the configuration,
#'   `from_config()` returns a re-instantiation of the object.
#'
#' @note Objects returned from `get_config()` are not serializable. Therefore,
#'   if you want to save and restore a model across sessions, you can use the
#'   `model_to_json()` function (for model configuration only, not weights) or
#'   the `save_model_tf()` function to save the model configuration and weights
#'   to the filesystem.
#'
#' @family model functions
#' @family layer methods
#'
#' @export
get_config <- function(object) {

  # call using lower level reticulate functions to prevent conversion to list
  # (the object will remain a python dictionary for full fidelity)
  # get_fn <- py_get_attr(object, "get_config")
  # config <- py_call(get_fn)
  config <- object$get_config()

  # set attribute indicating class
  attr(config, "config_class") <- object$`__class__`
  config
}

#' @rdname get_config
#' @export
from_config <- function(config,
                        custom_objects = NULL,
                        class = attr(config, "config_class", TRUE) %||% keras$Model) {
  args <- list(config)
  if(length(custom_objects))
    args[[2L]] <- objects_with_py_function_names(custom_objects)
  do.call(class$from_config, args)
}

#' Layer/Model weights as R arrays
#'
#' @param object Layer or model object
#' @param trainable if `NA` (the default), all weights are returned. If `TRUE, `
#' @param weights Weights as R array
#'
#' @note You can access the Layer/Model as `tf.Tensors` or `tf.Variables` at
#'   `object$weights`, `object$trainable_weights`, or
#'   `object$non_trainable_weights`
#'
#' @family model persistence
#' @family layer methods
#'
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
#' @return An integer count
#'
#' @family layer methods
#'
#' @export
count_params <- function(object) {
  object$count_params()
}



#' Reset the states for a layer
#'
#' @param object Model or layer object
#'
#' @family layer methods
#'
#' @export
reset_states <- function(object) {
  object$reset_states()
  invisible(object)
}


as_node_index <- function(node_index) {
  as.integer(node_index-1)
}

