
#' Save/Load models using HDF5 files
#' 
#' @param object Model object to save
#' @param filepath File path 
#' @param compile Whether to compile the model after loading.
#' @param overwrite Overwrite existing file if necessary
#' @param include_optimizer If `TRUE`, save optimizer's state.
#' @param custom_objects Mapping class names (or function names) of custom 
#'   (non-Keras) objects to class/functions
#' 
#' @details The following components of the model are saved: 
#' 
#'   - The model architecture, allowing to re-instantiate the model. 
#'   - The model weights. 
#'   - The state of the optimizer, allowing to resume training exactly where you
#'     left off.
#' This allows you to save the entirety of the state of a model
#' in a single file.
#' 
#' Saved models can be reinstantiated via `load_model()`. The model returned by
#' `load_model()` is a compiled model ready to be used (unless the saved model
#' was never compiled in the first place or `compile = FALSE` is specified.
#' 
#' @family model persistence
#' 
#' @export
save_model_hdf5 <- function(object, filepath, overwrite = TRUE, include_optimizer = TRUE) {
  
  if (!have_h5py())
    stop("The h5py Python package is required to save and load models")
  
  filepath <- normalize_path(filepath)
  if (confirm_overwrite(filepath, overwrite)) {
    keras$models$save_model(model = object, 
                            filepath = filepath, 
                            overwrite = overwrite,
                            include_optimizer = include_optimizer)
    invisible(TRUE) 
  } else {
    invisible(FALSE)
  }
}

#' @rdname save_model_hdf5
#' @export
load_model_hdf5 <- function(filepath, custom_objects = NULL, compile = TRUE) {
  
  if (!have_h5py())
    stop("The h5py Python package is required to save and load models")
  
  # build args dynamically so we can only pass `compile` if it's supported
  # (compile requires keras 2.0.4 / tensorflow 1.3)
  args <- list(
    filepath = normalize_path(filepath), 
    custom_objects = custom_objects
  )
  if (keras_version() >= "2.0.4")
    args$compile <- compile
  
  do.call(keras$models$load_model, args)
}


#' Save/Load model weights using HDF5 files
#' 
#' @param object Model object to save/load
#' @param filepath Path to the file 
#' @param overwrite Whether to silently overwrite any existing
#'   file at the target location
#' @param by_name Whether to load weights by name or by topological order.
#' 
#' @details The weight file has:
#'   - `layer_names` (attribute), a list of strings (ordered names of model layers).
#'   - For every layer, a `group` named `layer.name`
#'   - For every such layer group, a group attribute `weight_names`, a list of strings
#'     (ordered names of weights tensor of the layer).
#'  - For every weight in the layer, a dataset storing the weight value, named after 
#'    the weight tensor.
#'    
#'   For `load_model_weights()`, if `by_name` is `FALSE` (default) weights are 
#'   loaded based on the network's topology, meaning the architecture should be 
#'   the same as when the weights were saved. Note that layers that don't have 
#'   weights are not taken into account in the topological ordering, so adding
#'   or removing layers is fine as long as they don't have weights.
#'   
#'   If `by_name` is `TRUE`, weights are loaded into layers only if they share
#'   the same name. This is useful for fine-tuning or transfer-learning models
#'   where some of the layers have changed.
#'   

#' @family model persistence
#' 
#' @export
save_model_weights_hdf5 <- function(object, filepath, overwrite = TRUE) {
  
  if (!have_h5py())
    stop("The h5py Python package is required to save and load model weights")
  filepath <- normalize_path(filepath)
  if (confirm_overwrite(filepath, overwrite)) {
    object$save_weights(filepath = filepath, overwrite = overwrite)
    invisible(TRUE)
  } else {
    invisible(FALSE)
  }
}


#' @rdname save_model_weights_hdf5
#' @export
load_model_weights_hdf5 <- function(object, filepath, by_name = FALSE) {
  if (!have_h5py())
    stop("The h5py Python package is required to save and load model weights")
  invisible(object$load_weights(filepath = normalize_path(filepath), by_name = by_name))
}


#' Model configuration as JSON
#' 
#' Save and re-load models configurations as JSON. Note that the representation
#' does not include the weights, only the architecture.
#' 
#' @param object Model object to save
#' @param custom_objects Optional named list mapping names to custom classes or 
#'   functions to be considered during deserialization.
#' @param json JSON with model configuration
#'   
#' @family model persistence
#'   
#' @export
model_to_json <- function(object) {
  object$to_json()
}


#' @rdname model_to_json
#' @export
model_from_json <- function(json, custom_objects = NULL) {
  keras$models$model_from_json(json, custom_objects)
}


#' Model configuration as YAML
#' 
#' Save and re-load models configurations as YAML Note that the representation
#' does not include the weights, only the architecture.
#
#' @inheritParams model_to_json
#' 
#' @param yaml YAML with model configuration
#' 
#' @family model persistence
#' 
#' @export
model_to_yaml <- function(object) {
  
  if (!have_pyyaml())
    stop("The pyyaml Python package is required to save and load models as YAML")
  
  object$to_yaml()  
}

#' @rdname model_to_yaml
#' @export
model_from_yaml <- function(yaml, custom_objects = NULL) {
  
  if (!have_pyyaml())
    stop("The pyyaml Python package is required to save and load models as YAML")
  
  keras$models$model_from_yaml(yaml, custom_objects)
}



