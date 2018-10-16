
#' Create a Keras custom model
#' 
#' @param model_fn Function that returns an R custom model
#' @param name Optional name for model
#' 
#' @return A Keras model
#' 
#' @details For documentation on using custom models, 
#'   see <https://keras.rstudio.com/articles/custom_models.html>.
#' 
#' @export
keras_model_custom <- function(model_fn, name = NULL) {
  
  # verify version
  if (is_tensorflow_implementation() && keras_version() < "2.1.6")
    stop("Custom models require TensorFlow v1.9 or higher")
  else if (!is_tensorflow_implementation() && keras_version() < "2.2.0")
    stop("Custom models require Keras v2.2 or higher")
  
  # create the python subclass 
  python_path <- system.file("python", package = "keras")
  tools <- import_from_path("kerastools", path = python_path)
  model <- tools$model$RModel(name = name)
  
  # call the R model function
  r_model_call <- model_fn(model)
  
  # set the _r_call for delegation
  model$`_r_call` <- r_model_call
  
  # return model
  model
}

#' @export
print.kerastools.model.RModel <- function(x, ...) {
  if (!x$built) {
    cat("Custom Keras model: not yet fitted")
    return(invisible(x))
  }
  NextMethod()
}

#' @export
summary.kerastools.model.RModel <- function(object, ...) {
  if (!object$built) {
    cat("This custom model has not yet been built. To see a summary, compile and fit with some data.")
    return(invisible(NULL))
  }
  NextMethod()
}