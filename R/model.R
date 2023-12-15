
#' Keras Model
#'
#' A model is a directed acyclic graph of layers.
#'
#' @param inputs Input layer
#' @param outputs Output layer
#' @param ... Any additional arguments
#' @family model functions
#'
#' @examples
#' \dontrun{
#' library(keras3)
#'
#' # input layer
#' inputs <- layer_input(shape = c(784))
#'
#' # outputs compose input + dense layers
#' predictions <- inputs %>%
#'   layer_dense(units = 64, activation = 'relu') %>%
#'   layer_dense(units = 64, activation = 'relu') %>%
#'   layer_dense(units = 10, activation = 'softmax')
#'
#' # create and compile model
#' model <- keras_model(inputs = inputs, outputs = predictions)
#' model %>% compile(
#'   optimizer = 'rmsprop',
#'   loss = 'categorical_crossentropy',
#'   metrics = c('accuracy')
#' )
#' }
#' @export
keras_model <- function(inputs, outputs = NULL, ...) {
  keras$models$Model(inputs = inputs, outputs = outputs, ...)
}


#' Keras Model composed of a linear stack of layers
#'
#' @param layers List of layers to add to the model
#' @param name Name of model
#' @inheritDotParams sequential_model_input_layer
#'
#' @note
#'
#' If any arguments are provided to `...`, then the sequential model is
#' initialized with a `InputLayer` instance. If not, then the first layer passed
#' to a Sequential model should have a defined input shape. What that means is
#' that it should have received an `input_shape` or `batch_input_shape`
#' argument, or for some type of layers (recurrent, Dense...) an `input_dim`
#' argument.
#'
#' @family model functions
#'
#' @examples
#' \dontrun{
#'
#' library(keras3)
#'
#' model <- keras_model_sequential()
#' model %>%
#'   layer_dense(units = 32, input_shape = c(784)) %>%
#'   layer_activation('relu') %>%
#'   layer_dense(units = 10) %>%
#'   layer_activation('softmax')
#'
#' model %>% compile(
#'   optimizer = 'rmsprop',
#'   loss = 'categorical_crossentropy',
#'   metrics = c('accuracy')
#' )
#'
#' # alternative way to provide input shape
#' model <- keras_model_sequential(input_shape = c(784)) %>%
#'   layer_dense(units = 32) %>%
#'   layer_activation('relu') %>%
#'   layer_dense(units = 10) %>%
#'   layer_activation('softmax')
#'
#' }
#' @export
keras_model_sequential <- function(layers = NULL, name = NULL, ...) {

  if (length(list(...)))
    layers <- c(sequential_model_input_layer(...), layers)

  if(!is.null(layers) && !is.list(layers))
    layers <- list(layers)

  keras$models$Sequential(layers = layers, name = name)
}




#' sequential_model_input_layer
#'
#' @param input_shape an integer vector of dimensions (not including the batch
#'   axis), or a `tf$TensorShape` instance (also not including the batch axis).
#' @param batch_size  Optional input batch size (integer or NULL).
#' @param dtype Optional datatype of the input. When not provided, the Keras
#'   default float type will be used.
#' @param input_tensor Optional tensor to use as layer input. If set, the layer
#'   will use the `tf$TypeSpec` of this tensor rather than creating a new
#'   placeholder tensor.
#' @param sparse Boolean, whether the placeholder created is meant to be sparse.
#'   Default to `FALSE`.
#' @param ragged Boolean, whether the placeholder created is meant to be ragged.
#'   In this case, values of 'NULL' in the 'shape' argument represent ragged
#'   dimensions. For more information about `RaggedTensors`, see this
#'   [guide](https://www.tensorflow.org/guide/ragged_tensor). Default to
#'   `FALSE`.
#' @param type_spec A `tf$TypeSpec` object to create Input from. This
#'   `tf$TypeSpec` represents the entire batch. When provided, all other args
#'   except name must be `NULL`.
#' @param ... additional arguments passed on to `keras$layers$InputLayer`.
#' @param input_layer_name,name  Optional name of the input layer (string).
#'
#' @keywords internal
sequential_model_input_layer <- function(input_shape = NULL,
                                         batch_size = NULL,
                                         dtype = NULL,
                                         input_tensor = NULL,
                                         sparse = NULL,
                                         name = NULL,
                                         ragged = NULL,
                                         type_spec = NULL,
                                         ...,
                                         input_layer_name = NULL) {
  # keras$layers$Input can't be used with a Sequential Model, have to use
  # keras$layers$LayerInput instead.
  args <- capture_args(match.call(),
                       list(input_shape = normalize_shape,
                            batch_size = as_nullable_integer))

  if ("input_layer_name" %in% names(args)) {
    # a bare `name` arg would normally belong to the model, not the input layer
    if (!is.null(args[["input_layer_name"]]))
      args[["name"]] <- args[["input_layer_name"]]

    args[["input_layer_name"]] <- NULL
  }

  # renamed in v3
  args$shape <- args$input_shape
  args$input_shape <- NULL

  do.call(keras$layers$InputLayer, args)
}


#' @importFrom reticulate py_to_r_wrapper
#' @export
py_to_r_wrapper.keras.models.model.Model <- function(x) {
  force(x)
  function(object, ...) {
    compose_layer(object, x, ...)
  }
}

#' @export
py_to_r_wrapper.kerastools.model.RModel <- function(x) {
  force(x)
  function(...) {
    x$call(...)
  }
}



#  py_to_r_wrapper.keras.engine.base_layer.Layer <- function(x) {
#    force(x)
#    function(...) {
#      if(!missing(..1) && inherits(..1, "keras.engine.sequential.Sequential")) {
#        if(length(list(...)) > 1)
#          warning("Other arguments to ... are ignored because layer instance already created")
#        model <- ..1
#        model$add(x)
#        model
#      } else
#        x(...)
#    }
#  }


#' Clone a model instance.
#'
#' Model cloning is similar to calling a model on new inputs, except that it
#' creates new layers (and thus new weights) instead of sharing the weights of
#' the existing layers.
#'
#' @param model Instance of Keras model (could be a functional model or a
#'   Sequential model).
#' @param input_tensors Optional list of input tensors to build the model upon.
#'   If not provided, placeholders will be created.
#' @param clone_function Callable to be used to clone each layer in the target
#'   model (except `InputLayer` instances). It takes as argument the layer
#'   instance to be cloned, and returns the corresponding layer instance to be
#'   used in the model copy. If unspecified, this callable defaults to the
#'   following serialization/deserialization function:
#'
#'   ```function(layer) layer$`__class__`$from_config(layer$get_config())```
#'
#'   By passing a custom callable, you can customize your copy of the model,
#'   e.g. by wrapping certain layers of interest (you might want to replace all
#'   LSTM instances with equivalent `Bidirectional(LSTM(...))` instances, for
#'   example).
#'
#' @export
clone_model <- function(model, input_tensors = NULL, clone_function = NULL) {
  args <- capture_args(match.call())
  do.call(keras$models$clone_model, args)
}




resolve_callbacks <- function(args, callbacks) {
  args <- append(args, list(callbacks = normalize_callbacks(callbacks)))
  args
}

as_model_verbose_arg <- function(x) {
  if(!identical(x, "auto"))
    return(as.integer(x))
  # x == auto
  if(isTRUE(getOption('knitr.in.progress')))
    return(2L)
  x # "auto"
}






#' Retrieves a layer based on either its name (unique) or index.
#'
#' Indices are based on order of horizontal graph traversal (bottom-up) and are
#' 1-based. If `name` and `index` are both provided, `index` will take
#' precedence.
#'
#' @param object Keras model object
#' @param name String, name of layer.
#' @param index Integer, index of layer (1-based). Also valid are negative
#'   values, which count from the end of model.
#'
#' @return A layer instance.
#'
#' @family model functions
#'
#' @export
get_layer <- function(object, name = NULL, index = NULL) {
  object$get_layer(
    name = name,
    index = as_layer_index(index)
  )
}


#' Remove the last layer in a model
#'
#' @param object Keras model object
#'
#' @family model functions
#'
#' @export
pop_layer <- function(object) {
  object$pop()
}


#' Print a summary of a Keras model
#'
#' @param object,x Keras model instance
#' @param line_length Total length of printed lines
#' @param positions Relative or absolute positions of log elements in each line.
#'   If not provided, defaults to `c(0.33, 0.55, 0.67, 1.0)`.
#' @param expand_nested Whether to expand the nested models. If not provided,
#'   defaults to `FALSE`.
#' @param show_trainable Whether to show if a layer is trainable. If not
#'   provided, defaults to `FALSE`.
#' @param compact Whether to remove white-space only lines from the model
#'   summary. (Default `TRUE`)
#' @param ... for `summary()` and `print()`, passed on to `format()`. For
#'   `format()`, passed on to `model$summary()`.
#'
#' @family model functions
#'
#' @return `format()` returns a length 1 character vector. `print()` returns the
#'   model object invisibly. `summary()` returns the output of `format()`
#'   invisibly after printing it.
#'
#' @export
summary.keras.models.model.Model <- function(object, ...) {
  writeLines(f <- format.keras.models.model.Model(object, ...))
  invisible(f)
}

with_rich_config <- function(expr) {
  os <- reticulate::import("os")

  # set the number of columns to the current width.
  if (is.null(os$environ$get("COLUMNS"))) {
    os$environ["COLUMNS"] = as.character(getOption("width"))
    on.exit({
      os$environ$pop("COLUMNS")
    }, add = TRUE)
  }

  # allow colored output if the terminal supports it
  if (cli::num_ansi_colors() >= 256 && is.null(os$environ$get("FORCE_COLOR"))) {
    os$environ["FORCE_COLOR"] = "yes"
    os$environ["COLORTERM"] = "truecolor"
    on.exit({
      os$environ$pop("FORCE_COLOR")
      os$environ$pop("COLORTERM")
    }, add = TRUE)
  }

  force(expr)
}

#' @rdname summary.keras.models.model.Model
#' @export
format.keras.models.model.Model <-
function(x,
         line_length = width - (11L * show_trainable),
         positions = NULL,
         expand_nested = FALSE,
         show_trainable = x$built && as.logical(length(x$non_trainable_weights)),
         ...,
         compact = TRUE) {

  if (py_is_null_xptr(x))
    return("<pointer: 0x0>")

  args <- capture_args(match.call(), ignore = c("x", "compact"))

  with_rich_config(
    out <- trimws(py_capture_output(do.call(x$summary, args)))
  )

  if(compact) {
    # strip empty lines
    out <- gsub("(\\n\\s*\\n)", "\n", out, perl = TRUE)
    if(expand_nested)
      out <- gsub("\\n\\|\\s+\\|\\n", "\n", out)
  }

  out
}

#
#' @rdname summary.keras.models.model.Model
#' @export
print.keras.models.model.Model <- function(x, ...) {
  writeLines(format.keras.models.model.Model(x, ...))
  invisible(x)
}

#' @importFrom reticulate py_str
#' @export
py_str.keras.models.model.Model <- function(object, line_length = getOption("width"), positions = NULL, ...) {
  # still invoked by utils::str()
  # warning("`py_str()` generic is deprecated")
  format.keras.models.model.Model(object, line_length = line_length, positions = positions, ...)
}


# determine whether to view metrics or not
resolve_view_metrics <- function(verbose, epochs, metrics) {
  (epochs > 1)          &&            # more than 1 epoch
  (verbose > 0) &&                    # verbose mode is on
  !is.null(getOption("viewer")) &&    # have an internal viewer available
  nzchar(Sys.getenv("RSTUDIO"))       # running under RStudio
}


write_history_metadata <- function(history) {
  properties <- list()
  properties$validation_samples <- history$params$validation_samples
  tfruns::write_run_metadata("properties", properties)
}


as_class_weight <- function(class_weight) {
  # convert class weights to python dict
  if (!is.null(class_weight)) {
    if (is.list(class_weight))
      class_weight <- dict(class_weight)
    else
      stop("class_weight must be a named list of weights")
  }
}

have_module <- function(module) {
  tryCatch({ import(module); TRUE; }, error = function(e) FALSE)
}

have_h5py <- function() {
  have_module("h5py")
}

have_pyyaml <- function() {
  have_module("yaml")
}

have_requests <- function() {
  have_module("requests")
}

have_pillow <- function() {
  have_module("PIL") # aka Pillow
}
