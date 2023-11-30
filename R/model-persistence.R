
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
#' ```
#' ```{r, results = 'hide'}
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
function (model, filepath = NULL, overwrite = FALSE, ...)
{
  if(is.null(filepath) -> return_serialized) {
    filepath <- tempfile(pattern = "keras_model-", fileext = ".keras")
    on.exit(unlink(filepath), add = TRUE)
  }

  overwrite <- confirm_overwrite(filepath, overwrite)
  keras$saving$save_model(model, filepath, overwrite = overwrite)

  if(return_serialized)
    readBin(filepath, what = "raw", n = file.size(filepath))
  else
    invisible(model)
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
#' ```
#' ```{r, results = 'hide'}
#' x <- random_uniform(c(10, 3))
#' stopifnot(all.equal(
#'   model |> predict(x),
#'   loaded_model |> predict(x)
#' ))
#' ```
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
#' @param model
#' string, path to the saved model file,
#' or a raw vector, as returned by `save_model(filepath = NULL)`
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
function (model, custom_objects = NULL, compile = TRUE, safe_mode = TRUE)
{
  args <- capture_args2(list(custom_objects = objects_with_py_function_names),
                        ignore = "model")
  if (is.raw(model)) {
    serialized_model <- model
    filepath <- tempfile(pattern = "keras_model-", fileext = ".keras")
    on.exit(unlink(filepath), add = TRUE)
    writeBin(serialized_model, filepath)
  } else {
    filepath <- model
  }
  keras$saving$load_model(filepath, !!!args)
}


#' Saves all layer weights to a `.weights.h5` file.
#'
#' @param model A keras Model object
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
function (model, filepath, overwrite = FALSE)
{
    overwrite <- confirm_overwrite(filepath, overwrite)
    keras$Model$save_weights(model, filepath, overwrite = overwrite)
    invisible(model)
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



#' Save Model configuration as JSON
#'
#' Save and re-load models configurations as JSON. Note that the representation
#' does not include the weights, only the architecture.
#'
#' @param model Model object to save
#' @param custom_objects Optional named list mapping names to custom classes or
#'   functions to be considered during deserialization.
#' @param filepath path to json file with the model config.
#'
#' @family model persistence
#' @tether keras.Model.to_json
#' @export
save_model_config <- function(model, filepath, overwrite = FALSE, ...)
{
  confirm_overwrite(filepath, overwrite)
  writeLines(model$to_json(), filepath)
  invisible(model)
}


#' @rdname save_model_config
#' @export
#' @tether keras.models.model_from_json
load_model_config <- function(filepath, custom_objects = NULL)
{
  json <- paste0(readLines(filepath), collapse = "\n")
  keras$models$model_from_json(json, custom_objects)
}



#' Registers an object with the Keras serialization framework.
#'
#' @description
#' This decorator injects the decorated class or function into the Keras custom
#' object registry, so that it can be serialized and deserialized without
#' needing an entry in the user-provided `custom_objects` argument. It also injects a
#' function that Keras will call to get the object's serializable string key.
#'
#' Note that to be serialized and deserialized, classes must implement the
#' `get_config()` method. Functions do not have this requirement.
#'
#' The object will be registered under the key `'package>name'` where `name`,
#' defaults to the object name if not passed.
#'
#' # Examples
#' ```{r}
#' # Note that `'my_package'` is used as the `package` argument here, and since
#' # the `name` argument is not provided, `'MyDense'` is used as the `name`.
#' layer_my_dense <- Layer("MyDense")
#' register_custom_object(layer_my_dense, package = "my_package")
#'
#' MyDense <- environment(layer_my_dense)$Layer # the python class obj
#' stopifnot(exprs = {
#'   get_registered_object('my_package>MyDense') == MyDense
#'   get_registered_name(MyDense) == 'my_package>MyDense'
#' })
#' ```
#'
#' @param package
#' The package that this class belongs to. This is used for the
#' `key` (which is `"package>name"`) to identify the class.
#' Defaults to the current package name, or `"Custom"` outside of a package.
#'
#' @param name
#' The name to serialize this class under in this package.
#'
#' @export
#' @family object registration saving
#' @family saving
#' @tether keras.saving.register_keras_serializable
register_custom_object <-
function (object, name = NULL, package = NULL)
{
  object_in <- object

  # maybe unwrap `object` to get the pyobj, resolve `name` if needed
  if (inherits(object, "keras_layer_wrapper"))
    object <- environment(object)$Layer
  else if (inherits(object, "R6ClassGenerator"))
    object <- r_to_py.R6ClassGenerator(object)

  name <- name %||%
    as_r_value(py_get_attr(object, "__name__", TRUE)) %||%
    attr(object, "py_function_name", TRUE) %||%
    deparse1(substitute(object))

  if (!inherits(object, "python.builtin.object") &&
      is.function(object) &&
      is.null(attr(object, "py_function_name", TRUE)))
    object <- py_func2(object, TRUE, name = name)

  if (is.null(package)) {
    topenvname <- eval(quote(environmentName(topenv())), parent.frame())
    package <- if (topenvname %in% c("", "R_GlobalEnv"))
      "Custom" else topenvname
  }

  keras$saving$register_keras_serializable(package, name)(object)
  invisible(object_in)
}

#' @export
clear_registered_custom_objects <- function() {
  py_eval("lambda keras: keras.saving.get_custom_objects().clear()")(keras)
}


#' Retrieves a live reference to the global list of custom objects.
#'
#' @description
#' Custom objects set using using `custom_object_scope()` are not added to the
#' global list of custom objects, and will not appear in the returned
#' list.
#'
#' # Examples
#' ```{r, eval = FALSE}
#' get_custom_objects()$clear()
#' get_custom_objects()$update(list(MyObject = MyObject))
#' ```
#'
#' @returns
#' A named list, the global dictionary mapping registered class names to classes.
#'
#' @export
#' @family object registration saving
#' @family saving
#' @family utils
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_custom_objects>
#' @tether keras.utils.get_custom_objects
get_custom_objects <-
function ()
{
  py_call(r_to_py(keras$utils$get_custom_objects))
}


#' Returns the name registered to an object within the Keras framework.
#'
#' @description
#' This function is part of the Keras serialization and deserialization
#' framework. It maps objects to the string names associated with those objects
#' for serialization/deserialization.
#'
#' @returns
#' The name associated with the object, or the default Python name if the
#' object is not registered.
#'
#' @param obj
#' The object to look up.
#'
#' @export
#' @family object registration saving
#' @family saving
#' @family utils
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_registered_name>
#' @tether keras.utils.get_registered_name
get_registered_name <-
function (obj)
{
  args <- capture_args2(NULL)
  do.call(keras$utils$get_registered_name, args)
}


#' Returns the class associated with `name` if it is registered with Keras.
#'
#' @description
#' This function is part of the Keras serialization and deserialization
#' framework. It maps strings to the objects associated with them for
#' serialization/deserialization.
#'
#' # Examples
#' ```{r, eval = FALSE}
#' from_config <- function(cls, config, custom_objects = NULL) {
#'   if ('my_custom_object_name' %in% names(config)) {
#'     config$hidden_cls <- get_registered_object(
#'       config$my_custom_object_name,
#'       custom_objects = custom_objects)
#'   }
#' }
#' ```
#'
#' @returns
#' An instantiable class associated with `name`, or `NULL` if no such class
#' exists.
#'
#' @param name
#' The name to look up.
#'
#' @param custom_objects
#' A named list of custom objects to look the name up in.
#' Generally, custom_objects is provided by the user.
#'
#' @param module_objects
#' A named list of custom objects to look the name up in.
#' Generally, module_objects is provided by midlevel library
#' implementers.
#'
#' @export
#' @family object registration saving
#' @family saving
#' @family utils
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_registered_object>
#' @tether keras.utils.get_registered_object
get_registered_object <-
function (name, custom_objects = NULL, module_objects = NULL)
{
  args <- capture_args2(NULL)
  obj <- do.call(keras$utils$get_registered_object, args)
  # if(inherits(obj, keras$layers$Layer))
    # obj <- create_layer_wrapper(obj)
  obj
}



reload_model <- function(object) {
  old_config <- get_config(object)
  old_weights <- get_weights(object)

  new_model <- from_config(old_config)
  set_weights(new_model, old_weights)

  new_model
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

normalize_custom_objects <- function(objects) {
  if(is.null(objects))
    return(NULL)

  if(!is.list(objects))
    objects <- list(objects)

  objects <- do.call(c, .mapply(function(object, name) {
    # unwrap or convert as needed to get the python object
    # try to infer missing names or raise an error
    # return a named list (to convert to a dict)

    if (inherits(object, "keras_layer_wrapper"))
      object <- environment(object)$Layer

    else if (inherits(object, "R6ClassGenerator"))
      object <- r_to_py.R6ClassGenerator(object)

    else if (!inherits(object, "python.builtin.object") &&
             is.function(object) &&
             is.null(attr(object, "py_function_name", TRUE))) {
      if(name == "")
        stop("A name must be provided for the custom object")
      object <- py_func2(object, TRUE, name = name)
    }

    if (name == "") {
      if (inherits(object, "python.builtin.object"))
        name <- object$`__name__`
      else if (is.character(name <- attr(o, "py_function_name", TRUE))) {}
      else
        stop("object name could not be infered; please supply a named list",
             call. = FALSE)
    }
    setNames(list(object), name)
  }, list(objects, rlang::names2(objects)), NULL))

  objects
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
