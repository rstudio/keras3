
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
#' Saved models can be reinstantiated via `load_model_hdf5()`. The model returned by
#' `load_model_hdf5()` is a compiled model ready to be used (unless the saved model
#' was never compiled in the first place or `compile = FALSE` is specified.
#'
#' @note The [serialize_model()] function enables saving Keras models to
#' R objects that can be persisted across R sessions.
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
    mirror_to_run_dir(filepath)
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
    mirror_to_run_dir(filepath)
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

#' Serialize a model to an R object
#'
#' Model objects are external references to Keras objects which cannot be saved
#' and restored across R sessions. The `serialize_model()` and
#' `unserialize_model()` functions provide facilities to convert Keras models to
#' R objects for persistence within R data files.
#'
#' @note The [save_model_hdf5()] function enables saving Keras models to
#' external hdf5 files.
#'
#' @inheritParams save_model_hdf5
#' @param model Keras model or R "raw" object containing serialized Keras model.
#'
#' @return `serialize_model()` returns an R "raw" object containing an hdf5
#'   version of the Keras model. `unserialize_model()` returns a Keras model.
#'
#' @family model persistence
#'
#' @export
serialize_model <- function(model, include_optimizer = TRUE) {
  
  if (!inherits(model, "keras.engine.training.Model"))
    stop("You must pass a Keras model object to serialize_model")
  
  # write hdf5 file to temp file
  tmp <- tempfile(pattern = "keras_model", fileext = ".h5")  
  on.exit(unlink(tmp), add = TRUE)
  save_model_hdf5(model, tmp, include_optimizer = include_optimizer)
  
  # read it back into a raw vector
  readBin(tmp, what = "raw", n = file.size(tmp))
}

#' @rdname serialize_model
#' @export
unserialize_model <- function(model, custom_objects = NULL, compile = TRUE) {
  
  # write raw hdf5 bytes to temp file 
  tmp <- tempfile(pattern = "keras_model", fileext = ".h5")  
  on.exit(unlink(tmp), add = TRUE)
  writeBin(model, tmp)
  
  # read in from hdf5
  load_model_hdf5(tmp, custom_objects = custom_objects, compile = compile)
}


# utility function to mirror saved models/weights into the run_dir
# whenever a training_run is active
mirror_to_run_dir <- function(filepath) {
  
  if (tfruns::is_run_active()) {
    
    # get the full path to the saved file and the working dir
    saved_filepath <- normalizePath(filepath, mustWork = FALSE, winslash = "/")
    wd <- normalizePath(getwd(), mustWork = FALSE, winslash = "/")
  
    # if the saved file is within the working dir then mirror it  
    if (grepl(paste0("^", wd), saved_filepath)) {
      
      # compute target path within run_dir
      relative_filepath <- relative_to(wd, saved_filepath)
      copy_filepath <- file.path(run_dir(), relative_filepath)
      
      # create directory if needed
      copy_dir <- dirname(copy_filepath)
      if (!utils::file_test("-d", copy_dir))
        dir.create(copy_dir, recursive = TRUE)
      
      # perform the copy
      file.copy(saved_filepath, copy_filepath, overwrite = TRUE)
    }
  }
  
}





