#' Constructs an `Estimator` instance from given keras model.
#' 
#' @param keras_model Keras model in memory.
#' @param keras_model_path Directory to a keras model on disk.
#' @param custom_objects Dictionary for custom objects.
#' @param model_dir Directory to save Estimator model parameters, graph and etc.
#' @param config Configuration object.
#' 
#' @return An Estimator from given keras model.
#' 
#' @section Raises:
#' * ValueError: if both keras_model and keras_model_path was given. 
#' * ValueError: if the keras_model_path is a GCS URI. 
#' * ValueError: if keras_model has not been compiled.
#' 
#' @export
model_to_estimator <- function(
  keras_model = NULL,
  keras_model_path = NULL,
  custom_objects = NULL,
  model_dir = NULL,
  config = NULL) {
  if (is_backend("tensorflow")) {
    tf$keras$estimator$model_to_estimator(
      keras_model = keras_model,
      keras_model_path = keras_model_path,
      custom_objects = custom_objects,
      model_dir = model_dir,
      config = config
    ) 
  } else {
    stop("You must use tensorflow backend in order to use model_to_estimator()")
  }
}
