
#' Save/Load models using HDF5 files
#' 
#' @param object Model object to save
#' @param filepath File path 
#' @param compile Whether to compile the model after loading.
#' @param overwrite Overwrite existing file if necessary
#' @param include_optimizer If `TRUE`, save optimizer's state.
#' @param custom_objects Mapping class names (or function names) of custom 
#'   (non-Keras) objects to class/functions (for example, custom metrics
#'   or custom loss functions).
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
#' was never compiled in the first place or `compile = FALSE` is specified).
#'
#' As an alternative to providing the `custom_objects` argument, you can 
#' execute the definition and persistence of your model using the 
#' [with_custom_object_scope()] function.
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
  
  # prepare custom objects
  custom_objects <- objects_with_py_function_names(custom_objects)
  
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
#' @param skip_mismatch Logical, whether to skip loading of layers
#'   where there is a mismatch in the number of weights, or a mismatch in the
#'   shape of the weight (only valid when `by_name = FALSE`).
#' @param reshape Reshape weights to fit the layer when the correct number
#'   of values are present but the shape does not match.
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
load_model_weights_hdf5 <- function(object, filepath, by_name = FALSE,
                                    skip_mismatch = FALSE, reshape = FALSE) {
  if (!have_h5py())
    stop("The h5py Python package is required to save and load model weights")
  
  args <- list(
    filepath = normalize_path(filepath), 
    by_name = by_name
  )
  
  if (keras_version() >= "2.1.4")
    args$skip_mismatch <- skip_mismatch
  
  if (keras_version() >= "2.1.4")
    args$reshape <- reshape
  
  do.call(object$load_weights, args)
  
  invisible(object)
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

reload_model <- function(object) {
  old_config <- object$get_config()
  old_weights <- object$get_weights()
  
  models <- import("keras.models")
  if ("keras.models.Sequential" %in% class(object)) {
    new_model <- models$Sequential$from_config(old_config)
  }
  else {
    new_model <- models$Model$from_config(old_config)
  }
  
  new_model$set_weights(old_weights)
  
  new_model
}

#' Export a Saved Model
#'
#' Serialize a model to disk.
#'
#' @param object An \R object.
#' @param export_dir_base A string containing a directory in which to export the
#'   SavedModel.
#' @param overwrite Should the \code{export_dir_base} directory be overwritten?
#' @param versioned Should the model be exported under a versioned subdirectory?
#' @param remove_learning_phase Should the learning phase be removed by saving
#'   and reloading the model? Defaults to \code{TRUE}.
#' @param as_text Whether to write the SavedModel in text format.
#' @param ... Unused
#' 
#' @return The path to the exported directory, as a string.
#'
#' @export
export_savedmodel.keras.engine.training.Model <- function(
  object,
  export_dir_base,
  overwrite = TRUE,
  versioned = !overwrite,
  remove_learning_phase = TRUE,
  as_text = FALSE,
  ...) {
  if (!is_backend("tensorflow"))
    stop("'export_savedmodel' is only supported in the TensorFlow backend.")
  
  if (versioned) {
    export_dir_base <- file.path(export_dir_base, format(Sys.time(), "%Y%m%d%H%M%OS", tz = "GMT"))
  }
  
  if (is_tensorflow_implementation()) {
    stop(
      "'export_savedmodel()' is currently unsupported under the TensorFlow Keras ",
      "implementation, consider using 'tfestimators::keras_model_to_estimator()'."
    )
  } else if (identical(remove_learning_phase, TRUE)) {
    k_set_learning_phase(0)
    message("Keras learning phase set to 0 for export (restart R session before doing additional training)")
    object <- reload_model(object)
  }
  
  sess <- backend()$get_session()

  input_info <- lapply(object$inputs, function(e) {
    tensorflow::tf$saved_model$utils$build_tensor_info(e)
  })
  
  output_info <- lapply(object$outputs, function(e) {
    tensorflow::tf$saved_model$utils$build_tensor_info(e)
  })
  
  names(input_info) <- lapply(object$input_names, function(e) e)
  names(output_info) <- lapply(object$output_names, function(e) e)
  
  if (overwrite && file.exists(export_dir_base))
    unlink(export_dir_base, recursive = TRUE)

  builder <- tensorflow::tf$saved_model$builder$SavedModelBuilder(export_dir_base)
  builder$add_meta_graph_and_variables(
    sess,
    list(
      tensorflow::tf$python$saved_model$tag_constants$SERVING
    ),
    signature_def_map = list(
      serving_default = tensorflow::tf$saved_model$signature_def_utils$build_signature_def(
        inputs = input_info,
        outputs = output_info,
        method_name = tensorflow::tf$saved_model$signature_constants$PREDICT_METHOD_NAME
      )
    )
  )
  
  builder$save(as_text = as_text)
  
  invisible(export_dir_base)
}

