
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
#' @param model a keras model.
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
#' @param model A keras model.
#'
#' @export
#' @seealso [load_model()]
#' @family saving and loading functions
#' @tether keras.saving.save_model
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/models/Model/save>
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





#' Loads a model saved via `save_model()`.
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
#' @family saving and loading functions
#' @seealso
#' + <https://keras.io/api/models/model_saving_apis/model_saving_and_loading#loadmodel-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/saving/load_model>
load_model <-
function (model, custom_objects = NULL, compile = TRUE, safe_mode = TRUE)
{
  args <- capture_args(list(custom_objects = normalize_custom_objects),
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
#' @family saving and loading functions
#' @tether keras.Model.save_weights
#' @seealso
#' + <https://keras.io/api/models/model_saving_apis/weights_saving_and_loading#saveweights-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/save_weights>
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
#' @param model A keras model.
#'
#' @export
#' @family saving and loading functions
#' @tether keras.Model.load_weights
#' @seealso
#' + <https://keras.io/api/models/model_saving_apis/weights_saving_and_loading#loadweights-method>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/load_weights>
load_model_weights <-
function (model, filepath, skip_mismatch = FALSE, ...)
{
  args <- capture_args(ignore = "model")
  do.call(model$load_weights, args)
}

#' Save and load model configuration as JSON
#'
#' Save and re-load models configurations as JSON. Note that the representation
#' does not include the weights, only the architecture.
#'
#' Note: `save_model_config()` serializes the model to JSON using
#' `serialize_keras_object()`, not `get_config()`. `serialize_keras_object()`
#' returns a superset of `get_config()`, with additional information needed to
#' create the class object needed to restore the model. See example for how to
#' extract the `get_config()` value from a saved model.
#'
#' ```{r}
#' model <- keras_model_sequential(input_shape = 10) |> layer_dense(10)
#' file <- tempfile("model-config-", fileext = ".json")
#' save_model_config(model, file)
#'
#' # load a new model instance with the same architecture but different weights
#' model2 <- load_model_config(file)
#'
#' stopifnot(exprs = {
#'   all.equal(get_config(model), get_config(model2))
#'
#'   # To extract the `get_config()` value from a saved model config:
#'   all.equal(
#'       get_config(model),
#'       structure(jsonlite::read_json(file)$config,
#'                 "__class__" = keras_model_sequential()$`__class__`)
#'   )
#' })
#' ```
#'
#' @param model Model object to save
#' @param custom_objects Optional named list mapping names to custom classes or
#'   functions to be considered during deserialization.
#' @param filepath path to json file with the model config.
#' @param overwrite
#' Whether we should overwrite any existing model configuration json
#' at `filepath`, or instead ask the user
#' via an interactive prompt.
#'
#' @family saving and loading functions
#' @tether keras.Model.to_json
#' @export
save_model_config <- function(model, filepath = NULL, overwrite = FALSE)
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
  keras$models$model_from_json(json, normalize_custom_objects(custom_objects))
}


#' \[TF backend only] Create a TF SavedModel artifact for inference
#'
#' @description
#' (e.g. via TF-Serving).
#'
#' **Note:** This can currently only be used with the TF backend.
#'
#' This method lets you export a model to a lightweight SavedModel artifact
#' that contains the model's forward pass only (its `call()` method)
#' and can be served via e.g. TF-Serving. The forward pass is registered
#' under the name `serve()` (see example below).
#'
#' The original code of the model (including any custom layers you may
#' have used) is *no longer* necessary to reload the artifact -- it is
#' entirely standalone.
#'
#' # Examples
#' ```r
#' # Create the artifact
#' model |> tensorflow::export_savedmodel("path/to/location")
#'
#' # Later, in a different process / environment...
#' library(tensorflow)
#' reloaded_artifact <- tf$saved_model$load("path/to/location")
#' predictions <- reloaded_artifact$serve(input_data)
#'
#' # see tfdeploy::serve_savedmodel() for serving a model over a local web api.
#' ```
#'
# If you would like to customize your serving endpoints, you can
# use the lower-level `import("keras").export.ExportArchive` class. The
# `export()` method relies on `ExportArchive` internally.
#'
#' @param export_dir_base
#' string, file path where to save
#' the artifact.
#'
#' @param ... For forward/backward compatability.
#'
#' @param object A keras model.
#'
#' @exportS3Method tensorflow::export_savedmodel
#' @tether keras.Model.export
#' @family saving and loading functions
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/Model/export>
export_savedmodel.keras.src.models.model.Model <- function(object, export_dir_base, ...) {
  object$export(export_dir_base, ...)
}




#' Reload a Keras model/layer that was saved via `export_savedmodel()`.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' model <- keras_model_sequential(input_shape = c(784)) |> layer_dense(10)
#' model |> export_savedmodel("path/to/artifact")
#' reloaded_layer <- layer_tfsm(filepath = "path/to/artifact")
#' input <- random_normal(c(2, 784))
#' output <- reloaded_layer(input)
#' stopifnot(all.equal(as.array(output), as.array(model(input))))
#' ```
#' ```{r, include = FALSE}
#' unlink("path", recursive = TRUE)
#' ```
#'
#' The reloaded object can be used like a regular Keras layer, and supports
#' training/fine-tuning of its trainable weights. Note that the reloaded
#' object retains none of the internal structure or custom methods of the
#' original object -- it's a brand new layer created around the saved
#' function.
#'
#' **Limitations:**
#'
#' * Only call endpoints with a single `inputs` tensor argument
#' (which may optionally be a named list/list of tensors) are supported.
#' For endpoints with multiple separate input tensor arguments, consider
#' subclassing `layer_tfsm` and implementing a `call()` method with a
#' custom signature.
#' * If you need training-time behavior to differ from inference-time behavior
#' (i.e. if you need the reloaded object to support a `training=TRUE` argument
#' in `__call__()`), make sure that the training-time call function is
#' saved as a standalone endpoint in the artifact, and provide its name
#' to the `layer_tfsm` via the `call_training_endpoint` argument.
#'
#' @param filepath
#' string, the path to the SavedModel.
#'
#' @param call_endpoint
#' Name of the endpoint to use as the `call()` method
#' of the reloaded layer. If the SavedModel was created
#' via `export_savedmodel()`,
#' then the default endpoint name is `'serve'`. In other cases
#' it may be named `'serving_default'`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param dtype
#' datatype (e.g., `"float32"`).
#'
#' @param call_training_endpoint
#' see description
#'
#' @param trainable
#' see description
#'
#' @export
#' @family layers
#' @family saving and loading functions
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TFSMLayer>
#'
#' @tether keras.layers.TFSMLayer
layer_tfsm <-
function (object, filepath, call_endpoint = "serve", call_training_endpoint = NULL,
    trainable = TRUE, name = NULL, dtype = NULL)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$TFSMLayer, object, args)
}



#' Registers a custom object with the Keras serialization framework.
#'
#' @description
#' This function registers a custom class or function with the Keras custom
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
#' register_keras_serializable(layer_my_dense, package = "my_package")
#'
#' MyDense <- environment(layer_my_dense)$`__class__` # the python class obj
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
#' @param object
#' A keras object.
#'
#'
#' @export
#' @family saving and loading functions
#' @family serialization utilities
#' @tether keras.saving.register_keras_serializable
register_keras_serializable <-
function (object, name = NULL, package = NULL)
{

  py_object <- resolve_py_obj(
    object,
    default_name = name %||% deparse1(substitute(object))
  )

  package <- package %||%
    replace_val(environmentName(topenv(parent.frame())),
                c("", "base", "R_GlobalEnv"), "Custom")

  keras$saving$register_keras_serializable(package, name)(py_object)
  invisible(object)
}



#' Get/set the currently registered custom objects.
#'
#' @description
#' Custom objects set using using `custom_object_scope()` are not added to the
#' global list of custom objects, and will not appear in the returned
#' list.
#'
#' # Examples
#' ```{r, eval = FALSE}
#' get_custom_objects()
#' ```
#'
#' You can use `set_custom_objects()` to restore a previous registry state.
#' ```{r, eval = FALSE}
#' # within a function, if you want to temporarily modify the registry,
#' orig_objects <- set_custom_objects(clear = TRUE)
#' on.exit(set_custom_objects(orig_objects))
#'
#' ## temporarily modify the global registry
#' # register_keras_serializable(....)
#' # ....  <do work>
#' # on.exit(), the previous registry state is restored.
#' ```
#'
#' @note
#' `register_keras_serializable()` is preferred over `set_custom_objects()` for
#' registering new objects.
#'
#' @returns
#' An R named list mapping registered names to registered objects.
#' `set_custom_objects()` returns the registry values before updating, invisibly.
#'
#' @export
#' @family serialization utilities
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_custom_objects>
#' @tether keras.saving.get_custom_objects
get_custom_objects <-
function ()
{
  keras$saving$get_custom_objects()
}

#' @rdname get_custom_objects
#' @param objects A named list of custom objects, as returned by
#'   `get_custom_objects()` and `set_custom_objects()`.
#' @param clear bool, whether to clear the custom object registry before
#'   populating it with `objects`.
#' @export
set_custom_objects <- function(objects = named_list(), clear = TRUE) {
  # This doesn't use `get_custom_objects.update()` directly because there is a
  # bug upstream: modifying the global custom objects dict does not update the
  # global custom names dict, and there are no consistency checks between the
  # two dicts. They can get out-of-sync if you modify the global custom objects
  # dict directly without updating the custom names dict. The only safe way to
  # modify the global dict using the official (exported) api is to call
  # register_keras_serializable().
  # o <- py_call(r_to_py(keras$saving$get_custom_objects)); o$clear()
  m <- import(keras$saving$get_custom_objects$`__module__`, convert = FALSE)
  out <- invisible(py_to_r(m$GLOBAL_CUSTOM_OBJECTS))

  if(clear) {
    m$GLOBAL_CUSTOM_NAMES$clear()
    m$GLOBAL_CUSTOM_OBJECTS$clear()
  }

  if(length(objects)) {
    objects <- normalize_custom_objects(objects)
    m$GLOBAL_CUSTOM_OBJECTS$update(objects)
    m$GLOBAL_CUSTOM_NAMES$clear()
    py_eval("lambda m: m.GLOBAL_CUSTOM_NAMES.update(
      {obj: name for name, obj in m.GLOBAL_CUSTOM_OBJECTS.items()})")(m)
  }

  out
}


#' Returns the name registered to an object within the Keras framework.
#'
#' @description
#' This function is part of the Keras serialization and deserialization
#' framework. It maps objects to the string names associated with those objects
#' for serialization/deserialization.
#'
#' @returns
#' The name associated with the object, or the default name if the
#' object is not registered.
#'
#' @param obj
#' The object to look up.
#'
#' @export
#' @family serialization utilities
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_registered_name>
#' @tether keras.saving.get_registered_name
get_registered_name <-
function (obj)
{
  py_obj <- resolve_py_obj(obj, default_name = stop("Object must have a `name` attribute"))
  keras$saving$get_registered_name(py_obj)
}


#' Returns the class associated with `name` if it is registered with Keras.
#'
#' @description
#' This function is part of the Keras serialization and deserialization
#' framework. It maps strings to the objects associated with them for
#' serialization/deserialization.
#'
#' # Examples
#' ```r
#' from_config <- function(cls, config, custom_objects = NULL) {
#'   if ('my_custom_object_name' \%in\% names(config)) {
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
#' Generally, `module_objects` is provided by midlevel library
#' implementers.
#'
#' @export
#' @family serialization utilities
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_registered_object>
#' @tether keras.saving.get_registered_object
get_registered_object <-
function (name, custom_objects = NULL, module_objects = NULL)
{
  args <- capture_args(list(
    custom_objects = normalize_custom_objects,
    module_objects = normalize_custom_objects
  ))
  obj <- do.call(keras$saving$get_registered_object, args)
  # if(inherits(obj, keras$layers$Layer))
    # obj <- create_layer_wrapper(obj)
  obj
}


#' Retrieve the full config by serializing the Keras object.
#'
#' @description
#' `serialize_keras_object()` serializes a Keras object to a named list
#' that represents the object, and is a reciprocal function of
#' `deserialize_keras_object()`. See `deserialize_keras_object()` for more
#' information about the full config format.
#'
#' @returns
#' A named list that represents the object config.
#' The config is expected to contain simple types only, and
#' can be saved as json.
#' The object can be
#' deserialized from the config via `deserialize_keras_object()`.
#'
#' @param obj
#' the Keras object to serialize.
#'
#' @export
#' @family serialization utilities
#' @seealso
#' + <https://keras.io/api/models/model_saving_apis/serialization_utils#serializekerasobject-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/saving/serialize_keras_object>
serialize_keras_object <-
function (obj)
{
  keras$saving$serialize_keras_object(obj)
}


#' Retrieve the object by deserializing the config dict.
#'
#' @description
#' The config dict is a Python dictionary that consists of a set of key-value
#' pairs, and represents a Keras object, such as an `Optimizer`, `Layer`,
#' `Metrics`, etc. The saving and loading library uses the following keys to
#' record information of a Keras object:
#'
#' - `class_name`: String. This is the name of the class,
#'   as exactly defined in the source
#'   code, such as "LossesContainer".
#' - `config`: Named List. Library-defined or user-defined key-value pairs that store
#'   the configuration of the object, as obtained by `object$get_config()`.
#' - `module`: String. The path of the python module. Built-in Keras classes
#'   expect to have prefix `keras`.
#' - `registered_name`: String. The key the class is registered under via
#'   `register_keras_serializable(package, name)` API. The
#'   key has the format of `'{package}>{name}'`, where `package` and `name` are
#'   the arguments passed to `register_keras_serializable()`. If `name` is not
#'   provided, it uses the class name. If `registered_name` successfully
#'   resolves to a class (that was registered), the `class_name` and `config`
#'   values in the config dict will not be used. `registered_name` is only used for
#'   non-built-in classes.
#'
#' For example, the following config list represents the built-in Adam optimizer
#' with the relevant config:
#'
#' ```{r}
#' config <- list(
#'   class_name = "Adam",
#'   config = list(
#'     amsgrad = FALSE,
#'     beta_1 = 0.8999999761581421,
#'     beta_2 = 0.9990000128746033,
#'     epsilon = 1e-07,
#'     learning_rate = 0.0010000000474974513,
#'     name = "Adam"
#'   ),
#'   module = "keras.optimizers",
#'   registered_name = NULL
#' )
#' # Returns an `Adam` instance identical to the original one.
#' deserialize_keras_object(config)
#' ```
#'
#' If the class does not have an exported Keras namespace, the library tracks
#' it by its `module` and `class_name`. For example:
#'
#' ```r
#' config <- list(
#'   class_name = "MetricsList",
#'   config =  list(
#'     ...
#'   ),
#'   module = "keras.trainers.compile_utils",
#'   registered_name = "MetricsList"
#' )
#'
#' # Returns a `MetricsList` instance identical to the original one.
#' deserialize_keras_object(config)
#' ```
#'
#' And the following config represents a user-customized `MeanSquaredError`
#' loss:
#'
#' ```{r, include = FALSE}
#' # setup for example
#' o_registered <- set_custom_objects(clear = TRUE)
#' ```
#' ```{r}
#' # define a custom object
#' loss_modified_mse <- Loss(
#'   "ModifiedMeanSquaredError",
#'   inherit = loss_mean_squared_error)
#'
#' # register the custom object
#' register_keras_serializable(loss_modified_mse)
#'
#' # confirm object is registered
#' get_custom_objects()
#' get_registered_name(loss_modified_mse)
#'
#' # now custom object instances can be serialized
#' full_config <- serialize_keras_object(loss_modified_mse())
#'
#' # the `config` arguments will be passed to loss_modified_mse()
#' str(full_config)
#'
#' # and custom object instances can be deserialized
#' deserialize_keras_object(full_config)
#' # Returns the `ModifiedMeanSquaredError` object
#' ```
#' ```{r, include = FALSE}
#' # cleanup from example
#' set_custom_objects(o_registered, clear = TRUE)
#' ```
#'
#' @returns
#' The object described by the `config` dictionary.
#'
#' @param config
#' Named list describing the object.
#'
#' @param custom_objects
#' Named list containing a mapping between custom
#' object names the corresponding classes or functions.
#'
#' @param safe_mode
#' Boolean, whether to disallow unsafe `lambda` deserialization.
#' When `safe_mode=FALSE`, loading an object has the potential to
#' trigger arbitrary code execution. This argument is only
#' applicable to the Keras v3 model format. Defaults to `TRUE`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family serialization utilities
#' @seealso
#' + <https://keras.io/api/models/model_saving_apis/serialization_utils#deserializekerasobject-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/saving/deserialize_keras_object>
deserialize_keras_object <-
function (config, custom_objects = NULL, safe_mode = TRUE, ...)
{
    args <- capture_args(list(custom_objects = normalize_custom_objects))
    do.call(keras$saving$deserialize_keras_object, args)
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
#' need to explicitly map names to user objects via the `custom_objects`
#' parameter.
#'
#' The `with_custom_object_scope()` function provides an alternative that
#' lets you create a named alias for a user object that applies to an entire
#' block of code, and is automatically recognized when loading saved models.
#'
#' # Examples
#' ```r
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
#' ```
#'
#' @family saving and loading functions
#' @family serialization utilities
#' @export
with_custom_object_scope <- function(objects, expr) {
  objects <- normalize_custom_objects(objects)
  with(keras$saving$CustomObjectScope(objects), expr)
}


# ---- internal utilities ----

normalize_custom_objects <- function(objects) {

  objects <- as_list(objects)
  if(!length(objects))
    return(NULL)

  objects <- do.call(c, .mapply(function(object, name) {
    # unwrap or convert as needed to get the python object
    # try to infer correct names or raise an error
    # return a named list (to convert to a dict), or NULL

    if (inherits(object, "R6ClassGenerator"))
      object <- r_to_py.R6ClassGenerator(object)

    object <- resolve_py_obj(
      object, default_name = name %""%
        stop("object name could not be infered; please supply a named list"))

    out <- list(object)
    names(out) <- as_r_value(name %""% object$`__name__`)
    out
  }, list(objects, rlang::names2(objects)), NULL))

  objects
}


confirm_overwrite <- function(filepath, overwrite) {
  if (isTRUE(overwrite))
    return(TRUE)

  if (!file.exists(filepath))
    return(overwrite)

  if (interactive())
    overwrite <- utils::askYesNo(
      sprintf("File '%s' already exists - overwrite?", filepath),
      default = FALSE)
  if (!isTRUE(overwrite))
    stop("File '", filepath, "' already exists (pass overwrite = TRUE to force save).",
         call. = FALSE)

  TRUE
}
