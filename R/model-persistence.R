
#' Saves a model as a `.keras` file.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' model <- keras_model_sequential(input_shape = c(3)) |>
#'   layer_dense(5) |>
#'   layer_activation_softmax()
#'
#' model |> save_model("model.keras")
#' loaded_model <- load_model("model.keras")
#'
#' x <- random_uniform(c(10, 3))
#' stopifnot(all.equal(
#'   model |> predict(x),
#'   loaded_model |> predict(x)
#' ))
#' ```
#'
#' The saved `.keras` file contains:
#'
#' - The model's configuration (architecture)
#' - The model's weights
#' - The model's optimizer's state (if any)
#'
#' Thus models can be reinstantiated in the exact same state.
#'
#' ```{r}
#' zip::zip_list("model.keras")[, "filename"]
#' ```
#'
#' ```{r, include = FALSE}
#' unlink("model.keras")
#' ```
#'
#' @param filepath
#' string,
#' Path where to save the model. Must end in `.keras`.
#'
#' @param overwrite
#' Whether we should overwrite any existing model
#' at the target location, or instead ask the user
#' via an interactive prompt.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @tether keras.saving.save_model
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/models/Model/save>
save_model <-
function (model, filepath, overwrite = FALSE, ...)
{
  args <- capture_args2(NULL, ignore = "model")
  args$overwrite <- confirm_overwrite(filepath, overwrite)
  keras$saving$save_model(model, !!!args)
  invisible(filepath)
}




confirm_overwrite <- function(filepath, overwrite) {
  if (isTRUE(overwrite))
    return(TRUE)

  if (!file.exists(filepath))
    return(overwrite)

  if (interactive())
    overwrite <-
      askYesNo(sprintf("File '%s' already exists - overwrite?", filepath),
               default = FALSE)
  if (!isTRUE(overwrite))
    stop("File '", filepath, "' already exists (pass overwrite = TRUE to force save).",
         call. = FALSE)

  TRUE
}


#' Loads a model saved via `model.save()`.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' model <- keras_model_sequential(input_shape = c(3)) |>
#'   layer_dense(5) |>
#'   layer_activation_softmax()
#'
#' model |> save_model("model.keras")
#' loaded_model <- load_model("model.keras")
#'
#' x <- random_uniform(c(10, 3))
#' stopifnot(all.equal(
#'   model |> predict(x),
#'   loaded_model |> predict(x)
#' ))
#' ```
#'
#' ```{r, include = FALSE}
#' unlink("model.keras")
#' ```
#'
#' Note that the model variables may have different name values
#' (`var$name` property, e.g. `"dense_1/kernel:0"`) after being reloaded.
#' It is recommended that you use layer attributes to
#' access specific variables, e.g. `model |> get_layer("dense_1") |> _$kernel`.
#'
#' @returns
#' A Keras model instance. If the original model was compiled,
#' and the argument `compile = TRUE` is set, then the returned model
#' will be compiled. Otherwise, the model will be left uncompiled.
#'
#' @param filepath
#' string, path to the saved model file.
#'
#' @param custom_objects
#' Optional named list mapping names
#' to custom classes or functions to be
#' considered during deserialization.
#'
#' @param compile
#' Boolean, whether to compile the model after loading.
#'
#' @param safe_mode
#' Boolean, whether to disallow unsafe `lambda` deserialization.
#' When `safe_mode=FALSE`, loading an object has the potential to
#' trigger arbitrary code execution. This argument is only
#' applicable to the Keras v3 model format. Defaults to `TRUE`.
#'
#' @export
#' @tether keras.saving.load_model
#' @family saving
#' @seealso
#' + <https:/keras.io/keras_core/api/models/model_saving_apis/model_saving_and_loading#loadmodel-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/saving/load_model>
load_model <-
function (filepath, custom_objects = NULL, compile = TRUE, safe_mode = TRUE)
{
  args <- capture_args2(list(custom_objects = objects_with_py_function_names))
  do.call(keras$saving$load_model, args)
}


#' Saves all layer weights to a `.weights.h5` file.
#'
#' @param filepath
#' string.
#' Path where to save the model. Must end in `.weights.h5`.
#'
#' @param overwrite
#' Whether we should overwrite any existing model
#' at the target location, or instead ask the user
#' via an interactive prompt.
#'
#' @export
#' @tether keras.Model.save_weights
#' @seealso
#' + <https:/keras.io/api/models/model_saving_apis/weights_saving_and_loading#saveweights-method>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/save_weights>
save_model_weights <-
function (self, filepath, overwrite = FALSE)
{
    args <- capture_args2(NULL)
    args$overwrite <- confirm_overwrite(filepath, overwrite)
    do.call(keras$Model$save_weights, args)
    invisible(filepath)
}


#' Load weights from a file saved via `save_model_weights()`.
#'
#' @description
#' Weights are loaded based on the network's
#' topology. This means the architecture should be the same as when the
#' weights were saved. Note that layers that don't have weights are not
#' taken into account in the topological ordering, so adding or removing
#' layers is fine as long as they don't have weights.
#'
#' **Partial weight loading**
#'
#' If you have modified your model, for instance by adding a new layer
#' (with weights) or by changing the shape of the weights of a layer,
#' you can choose to ignore errors and continue loading
#' by setting `skip_mismatch=TRUE`. In this case any layer with
#' mismatching weights will be skipped. A warning will be displayed
#' for each skipped layer.
#'
#' @param filepath
#' String, path to the weights file to load.
#' It can either be a `.weights.h5` file
#' or a legacy `.h5` weights file.
#'
#' @param skip_mismatch
#' Boolean, whether to skip loading of layers where
#' there is a mismatch in the number of weights, or a mismatch in
#' the shape of the weights.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @tether keras.Model.load_weights
#' @seealso
#' + <https:/keras.io/api/models/model_saving_apis/weights_saving_and_loading#loadweights-method>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/load_weights>
load_model_weights <-
function (self, filepath, skip_mismatch = FALSE, ...)
{
  args <- capture_args2(NULL)
  do.call(keras$Model$load_weights, args)
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


#' Serialize a model to an R object
#'
#' Model objects are external references to Keras objects which cannot be saved
#' and restored across R sessions. The `serialize_model()` and
#' `unserialize_model()` functions provide facilities to convert Keras models to
#' R objects for persistence within R data files.
#'
#' @note The [save_model()] function enables saving Keras models to
#' external hdf5 files.
#'
#' @param model Keras model or R "raw" object containing serialized Keras model.
#'
#' @return `serialize_model()` returns an R "raw" object containing an hdf5
#'   version of the Keras model. `unserialize_model()` returns a Keras model.
#'
#' @family model persistence
#'
#' @export
serialize_model <- function(model, ...) {

  if (!inherits(model, "keras.models.model.Model"))
    stop("You must pass a Keras model object to serialize_model()")

  file <- tempfile(pattern = "keras_model-", fileext = ".keras")
  on.exit(unlink(file), add = TRUE)
  save_model(model, file, ...)

  # read it back into a raw vector
  readBin(tmp, what = "raw", n = file.size(tmp))
}

#' @rdname serialize_model
#' @export
unserialize_model <- function(model, custom_objects = NULL, compile = TRUE) {

  # write raw model '.keras' bytes to temp file
  file <- tempfile(pattern = "keras_model-", fileext = ".keras")
  on.exit(unlink(file), add = TRUE)
  writeBin(model, file)

  load_model(file, custom_objects = custom_objects, compile = compile)
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
export_savedmodel.keras.models.model.Model <- function(
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
    # k_set_learning_phase(0) # TODO: fate of this symbol in keras 3?
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

#' (Deprecated) Export to Saved Model format
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
#' @keywords internal
#' @export
model_to_saved_model <- function(model, saved_model_path, custom_objects = NULL,
                                 as_text = FALSE, input_signature = NULL,
                                 serving_only = FALSE) {


  if (!is_tensorflow_implementation())
    stop("TensorFlow implementation is required.")

  if (tensorflow::tf_version() > "2.0")
    stop("This function is deprecated as of TF version 2.1")

  if (tensorflow::tf_version() > "1.14")
    warning("This function is experimental and will be deprecated in TF 2.1")

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

  if (tensorflow::tf_version() > "2.0")
    stop("This function is deprecated as of TF version 2.1")

  if (tensorflow::tf_version() > "1.14")
    warning("This function is experimental and will be deprecated in TF 2.1")

  if (!tensorflow::tf_version() >= "1.14")
    stop("TensorFlow version >= 1.14 is required. Use export_savedmodel ",
         "if you need to export to saved model format in older versions.")

  saved_model_path <- path.expand(saved_model_path)
  tensorflow::tf$keras$experimental$load_from_saved_model(
    saved_model_path = saved_model_path,
    custom_objects = custom_objects
  )
}




#' Provide a scope with mappings of names to custom objects
#'
#' @param objects Named list of objects
#' @param expr Expression to evaluate
#'
#' @details
#' There are many elements of Keras models that can be customized with
#' user objects (e.g. losses, metrics, regularizers, etc.). When
#' loading saved models that use these functions you typically
#' need to explicitily map names to user objects via the `custom_objects`
#' parmaeter.
#'
#' The `with_custom_object_scope()` function provides an alternative that
#' lets you create a named alias for a user object that applies to an entire
#' block of code, and is automatically recognized when loading saved models.
#'
#' @examples \dontrun{
#' # define custom metric
#' metric_top_3_categorical_accuracy <-
#'   custom_metric("top_3_categorical_accuracy", function(y_true, y_pred) {
#'     metric_top_k_categorical_accuracy(y_true, y_pred, k = 3)
#'   })
#'
#' with_custom_object_scope(c(top_k_acc = sparse_top_k_cat_acc), {
#'
#'   # ...define model...
#'
#'   # compile model (refer to "top_k_acc" by name)
#'   model |> compile(
#'     loss = "binary_crossentropy",
#'     optimizer = optimizer_nadam(),
#'     metrics = c("top_k_acc")
#'   )
#'
#'   # save the model
#'   model |> save_model("my_model.keras")
#'
#'   # loading the model within the custom object scope doesn't
#'   # require explicitly providing the custom_object
#'   reloaded_model <- load_model("my_model.keras")
#' })
#' }
#'
#' @export
with_custom_object_scope <- function(objects, expr) {
  objects <- objects_with_py_function_names(objects)
  with(keras$utils$custom_object_scope(objects), expr)
}

#' @importFrom rlang names2
objects_with_py_function_names <- function(objects) {
  if(is.null(objects))
    return(NULL)

  if(!is.list(objects))
    objects <- list(objects)

  object_names <- rlang::names2(objects)

  # try to infer missing names or raise an error
  for (i in seq_along(objects)) {
    name <- object_names[[i]]
    o <- objects[[i]]

    if (name == "") {
      if (inherits(o, "keras_layer_wrapper"))
        o <- attr(o, "Layer")

      if (inherits(o, "python.builtin.object"))
        name <- o$`__name__`
      else if (inherits(o, "R6ClassGenerator"))
        name <- o$classname
      else if (is.character(attr(o, "py_function_name", TRUE)))
        name <- attr(o, "py_function_name", TRUE)
      else
        stop("object name could not be infered; please supply a named list",
             call. = FALSE)

      object_names[[i]] <- name
    }
  }

  # add a `py_function_name` attr for bare R functions, if it's missing
  objects <- lapply(1:length(objects), function(i) {
    object <- objects[[i]]
    if (is.function(object) &&
        !inherits(object, "python.builtin.object") &&
        is.null(attr(object, "py_function_name", TRUE)))
      attr(object, "py_function_name") <- object_names[[i]]
    object
  })

  names(objects) <- object_names
  objects
}
