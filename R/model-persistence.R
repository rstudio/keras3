
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
  
  args <- list(
    model = object, 
    filepath = filepath, 
    overwrite = overwrite,
    include_optimizer = include_optimizer
  )
  
  if (tensorflow::tf_version() >= "1.14.0") {
    args[["save_format"]] <- "h5"
  }
  
  if (confirm_overwrite(filepath, overwrite)) {
    do.call(keras$models$save_model, args)
    invisible(TRUE) 
  } else {
    invisible(FALSE)
  }
}

#' Save/Load models using SavedModel format
#' 
#' @inheritParams save_model_hdf5
#' @param signatures Signatures to save with the SavedModel. Please see the signatures 
#'  argument in `tf$saved_model$save` for details.
#' @param options Optional `tf$saved_model$SaveOptions` object that specifies options
#'  for saving to SavedModel
#'  
#' @family model persistence
#' 
#' @export
save_model_tf <- function(object, filepath, overwrite = TRUE, include_optimizer = TRUE, 
                          signatures = NULL, options = NULL) {
  
  if (tensorflow::tf_version() < "2.0.0")
    stop("save_model_tf only works with TF >= 2.0.0", call.=FALSE)
  
  filepath <- normalize_path(filepath)
  
  args <- list(
    model = object, 
    filepath = filepath, 
    overwrite = overwrite,
    include_optimizer = include_optimizer,
    signatures = signatures,
    options = options,
    save_format = "tf"
  )
  
  
  if (confirm_overwrite(filepath, overwrite)) {
    do.call(keras$models$save_model, args)
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
  
  load_model(filepath, custom_objects, compile)
}

#' @rdname save_model_tf
#' @export
load_model_tf <- function(filepath, custom_objects = NULL, compile = TRUE) {
  
  if (tensorflow::tf_version() < "2.0.0")
    stop("TensorFlow version >= 2.0.0 is requires to load models in the SavedModel format.", 
         call. = FALSE)
  
  load_model(filepath, custom_objects, compile)
}

load_model <- function(filepath, custom_objects = NULL, compile = TRUE) {
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
    object$save_weights(filepath = filepath, overwrite = overwrite, 
                        save_format = "h5")
    invisible(TRUE)
  } else {
    invisible(FALSE)
  }
}

#' Save model weights in the SavedModel format
#' 
#' @inheritParams save_model_weights_hdf5
#' 
#' @details 
#' When saving in TensorFlow format, all objects referenced by the network 
#' are saved in the same format as `tf.train.Checkpoint`, including any Layer instances 
#' or Optimizer instances assigned to object attributes. For networks constructed from 
#' inputs and outputs using `tf.keras.Model(inputs, outputs)`, Layer instances used by 
#' the network are tracked/saved automatically. For user-defined classes which inherit 
#' from `tf.keras.Model`, Layer instances must be assigned to object attributes, 
#' typically in the constructor. 
#' 
#' See the documentation of `tf.train.Checkpoint` and `tf.keras.Model` for details.
#' 
#' @export
save_model_weights_tf <- function(object, filepath, overwrite = TRUE) {
  
  if (!is_tensorflow_implementation())
    stop("Save weights to the SavedModel format requires the TensorFlow implementation.")
  
  if (!tensorflow::tf_version() >= "2.0")
    stop("Save weights to the SavedModel format requires TensorFlow version >= 2.0")
  
  filepath <- normalize_path(filepath)
  if (confirm_overwrite(filepath, overwrite)) {
    object$save_weights(filepath = filepath, overwrite = overwrite, 
                        save_format = "tf")
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
  
  if (keras_version() >= "2.1.4" && !is_tensorflow_implementation()) {
    args$skip_mismatch <- skip_mismatch
    args$reshape <- reshape
  }
   
  do.call(object$load_weights, args)
  
  invisible(object)
}

#' @inheritParams load_model_weights_hdf5
#' @rdname save_model_weights_tf
#' @export
load_model_weights_tf <- function(object, filepath, by_name = FALSE,
                                    skip_mismatch = FALSE, reshape = FALSE) {
  
  args <- list(
    filepath = normalize_path(filepath), 
    by_name = by_name
  )
  
  if (keras_version() >= "2.1.4" && !is_tensorflow_implementation()) {
    args$skip_mismatch <- skip_mismatch
    args$reshape <- reshape
  }
  
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
  old_config <- get_config(object)
  old_weights <- get_weights(object)
  
  new_model <- from_config(old_config)
  set_weights(new_model, old_weights)
  
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
#' @param ... Other arguments passed to tf.saved_model.save. (Used only if 
#'   TensorFlow version >= 2.0)
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
  
  export_dir_base <- normalize_path(export_dir_base)
  
  if (!is_backend("tensorflow"))
    stop("'export_savedmodel' is only supported in the TensorFlow backend.")
  
  if (versioned) {
    export_dir_base <- file.path(export_dir_base, format(Sys.time(), "%Y%m%d%H%M%OS", tz = "GMT"))
  }
  
  if (identical(remove_learning_phase, TRUE)) {
    k_set_learning_phase(0)
    message("Keras learning phase set to 0 for export (restart R session before doing additional training)")
    object <- reload_model(object)
  }
  
  if (tensorflow::tf_version() >= "1.14") {
    
    if (overwrite && file.exists(export_dir_base))
      unlink(export_dir_base, recursive = TRUE)
    
    if (as_text)
      warning("as_text is ignored in TensorFlow 1.14")
    
    tensorflow::tf$saved_model$save(
      obj = object, 
      export_dir = export_dir_base, 
      ...
    )
    
  } else {
    
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
    
  }
  
  invisible(export_dir_base)
}

#' Export to Saved Model format
#' 
#' @param model A Keras model to be saved. If the model is subclassed, the flag 
#'   `serving_only` must be set to `TRUE`.
#' @param saved_model_path a string specifying the path to the SavedModel directory.
#' @param custom_objects Optional dictionary mapping string names to custom classes 
#'   or functions (e.g. custom loss functions).
#' @param as_text  bool, `FALSE` by default. Whether to write the SavedModel proto in text 
#'   format. Currently unavailable in serving-only mode.
#' @param input_signature A possibly nested sequence of `tf.TensorSpec` objects, used to 
#'   specify the expected model inputs. See tf.function for more details.
#' @param serving_only bool, `FALSE` by default. When this is true, only the 
#'   prediction graph is saved.
#'   
#' @note This functionality is experimental and only works with TensorFlow 
#'   version >= "2.0".
#'   
#' @return Invisibly returns the `saved_model_path`.
#' @family saved_model
#' 
#' @export
model_to_saved_model <- function(model, saved_model_path, custom_objects = NULL, 
                                 as_text = FALSE, input_signature = NULL, 
                                 serving_only = FALSE) {
  
  if (!is_tensorflow_implementation())
    stop("TensorFlow implementation is required.")
  
  if (!tensorflow::tf_version() >= "1.14")
    stop("TensorFlow version >= 1.14 is required. Use export_savedmodel ",
         "if you need to export to saved model format in older versions.")
  
  
  saved_model_path <- path.expand(saved_model_path)
  
  tensorflow::tf$keras$experimental$export_saved_model(
    model = model,
    saved_model_path = saved_model_path,
    custom_objects = custom_objects,
    as_text = as_text,
    input_signature = input_signature,
    serving_only = serving_only
  )
  
  invisible(saved_model_path)
}

#' Load a Keras model from the Saved Model format
#'
#' @inheritParams model_to_saved_model
#' 
#' @return a Keras model.
#' @family saved_model
#' 
#' @note This functionality is experimental and only works with TensorFlow 
#'   version >= "2.0".
#' 
#' @export
model_from_saved_model <- function(saved_model_path, custom_objects = NULL) {
  
  if (!is_tensorflow_implementation())
    stop("TensorFlow implementation is required.")
  
  if (!tensorflow::tf_version() >= "1.14")
    stop("TensorFlow version >= 1.14 is required. Use export_savedmodel ",
         "if you need to export to saved model format in older versions.")
  
  saved_model_path <- path.expand(saved_model_path)
  tensorflow::tf$keras$experimental$load_from_saved_model(
    saved_model_path = saved_model_path,
    custom_objects = custom_objects
  )
}


