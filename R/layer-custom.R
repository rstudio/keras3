

#' (Deprecated) Base R6 class for Keras layers
#'
#' Custom R6 layers can now inherit directly from `keras$layers$Layer` or other layers.
#'
#' @docType class
#'
#' @format An [R6Class] generator object
#' @section Methods: \describe{ \item{\code{build(input_shape)}}{Creates the
#'   layer weights (must be implemented by all layers that have weights)}
#'   \item{\code{call(inputs,mask)}}{Call the layer on an input tensor.}
#'   \item{\code{compute_output_shape(input_shape)}}{Compute the output shape
#'   for the layer.}
#'   \item{\code{add_loss(losses, inputs)}}{Add losses to the layer.}
#'   \item{\code{add_weight(name,shape,dtype,initializer,regularizer,trainable,constraint)}}{Adds
#'   a weight variable to the layer.} }
#'
#' @return [KerasLayer].
#'
#' @keywords internal
#'
#' @export
KerasLayer <- R6Class("KerasLayer",

  public = list(

    # Create the layer weights.
    build = function(input_shape) {

    },

    # Call the layer on an input tensor.
    call = function(inputs, mask = NULL) {
      stop("Keras custom layers must implement the call function")
    },

    # Compute the output shape for the layer.
    compute_output_shape = function(input_shape) {
      input_shape
    },

    # Add losses to the layer
    add_loss = function(losses, inputs = NULL) {
      args <- list()
      args$losses <- losses
      args$inputs <- inputs
      do.call(private$wrapper$add_loss, args)
    },

    # Adds a weight variable to the layer.
    add_weight = function(name, shape, dtype = NULL, initializer = NULL,
                          regularizer = NULL, trainable = TRUE, constraint = NULL) {

      args <- list()
      args$name <- name
      args$shape <- shape
      args$dtype <- dtype
      args$initializer <- initializer
      args$regularizer <- regularizer
      args$trainable <- trainable
      args$constraint <- constraint

      do.call(private$wrapper$add_weight, args)
    },

    # back reference to python layer that wraps us
    .set_wrapper = function(wrapper) {
      private$wrapper <- wrapper
    },

    python_layer = function() {
      private$wrapper
    }
  ),

  active = list(
    input = function(value) {
      if (missing(value)) return(private$wrapper$input)
      else private$wrapper$input <- value
    },
    output = function(value) {
      if (missing(value)) return(private$wrapper$output)
      else private$wrapper$output <- value
    }
  ),

  private = list(
    wrapper = NULL
  )
)


compat_custom_KerasLayer_handler <- function(layer_class, args) {
    # common layer parameters (e.g. "input_shape") need to be passed to the
    # Python Layer constructor rather than the R6 constructor. Here we
    # extract and set aside any of those arguments we find and set them to
    # NULL within the args list which will be passed to the R6 layer
    common_arg_names <- c("input_shape", "batch_input_shape", "batch_size",
                          "dtype", "name", "trainable", "weights")

    py_wrapper_args <- args[common_arg_names]
    py_wrapper_args[sapply(py_wrapper_args, is.null)] <- NULL
    for (arg in names(py_wrapper_args))
      args[[arg]] <- NULL

    # create the R6 layer
    r6_layer <- do.call(layer_class$new, args)

    # create the python wrapper (passing the extracted py_wrapper_args)
    python_path <- system.file("python", package = "keras")
    tools <- import_from_path("kerastools", path = python_path)
    py_wrapper_args$r_build <- r6_layer$build
    py_wrapper_args$r_call <-  reticulate::py_func(r6_layer$call)
    py_wrapper_args$r_compute_output_shape <- r6_layer$compute_output_shape
    layer <- do.call(tools$layer$RLayer, py_wrapper_args)

    # set back reference in R layer
    r6_layer$.set_wrapper(layer)
    list(layer, args)
}




py_formals <- function(py_obj) {
  # returns python fn formals as a list
  # like base::formals(), but for py functions/methods
  inspect <- reticulate::import("inspect")
  sig <- if (inspect$isclass(py_obj)) {
    inspect$signature(py_obj$`__init__`)
  } else
    inspect$signature(py_obj)

  args <- pairlist()
  it <- sig$parameters$items()$`__iter__`()
  repeat {
    x <- reticulate::iter_next(it)
    if (is.null(x))
      break

    name <- x[[1]]
    param <- x[[2]]

    if (param$kind == inspect$Parameter$VAR_KEYWORD ||
        param$kind == inspect$Parameter$VAR_POSITIONAL) {
      args[["..."]] <- quote(expr = )
      next
    }

    default <- param$default

    if (inherits(default, "python.builtin.object")) {
      if (default != inspect$Parameter$empty)
        # must be something complex that failed to convert
        warning(glue::glue(
          "Failed to convert default arg {param} for {name} in {py_obj_expr}"
        ))
      args[name] <- list(quote(expr = ))
      next
    }

    args[name] <- list(default) # default can be NULL
  }
  args
}




#' Create a Keras Layer wrapper
#'
#' @param Layer A R6 or Python class generator that inherits from
#'   `keras$layers$Layer`
#' @param modifiers A named list of functions to modify to user-supplied
#'   arguments before they are passed on to the class constructor. (e.g.,
#'   `list(units = as.integer)`)
#' @param convert Boolean, whether the Python class and its methods should by
#'   default convert python objects to R objects.
#'
#' See guide 'making_new_layers_and_models_via_subclassing.Rmd' for example usage.
#'
#' @return An R function that behaves similarly to the builtin keras `layer_*`
#'   functions. When called, it will create the class instance, and also
#'   optionally call it on a supplied argument `object` if it is present. This
#'   enables keras layers to compose nicely with the pipe (`%>%`).
#'
#'   The R function will arguments taken from the `initialize` (or `__init__`)
#'   method of the Layer.
#'
#'   If Layer is an R6 object, this will delay initializing the python
#'   session, so it is safe to use in an R package.
#'
#' @export
#' @importFrom rlang %||%
create_layer_wrapper <- function(Layer, modifiers = NULL, convert = TRUE) {

  force(Layer)
  modifiers <- utils::modifyList(
    list(
      # include helpers for standard layer args by default,
      # but leave an escape hatch allowing users to override/opt-out.
      input_shape = as_tf_shape,
      batch_input_shape = as_tf_shape,
      batch_size = as.integer
    ),
    as.list(modifiers)
  )

  wrapper <- function(object) {
    args <- capture_args(match.call(), modifiers, ignore = "object")
    create_layer(Layer, object, args)
  }

  formals(wrapper) <- local({

    if(inherits(Layer, "py_R6ClassGenerator"))
      Layer <- attr(Layer, "r6_class")

    if (inherits(Layer, "python.builtin.type")) {
      f <- py_formals(Layer)
    } else if (inherits(Layer, "R6ClassGenerator")) {
      m <- Layer$public_methods
      init <- m$initialize %||% m$`__init__` %||% function(){}
      f <- formals(init)
    } else
      stop('Unrecognized type passed `create_layer_wrapper()`.',
           ' class() must be an "R6ClassGenerator" or a "python.builtin.type"')
    f$self <- NULL
    c(formals(wrapper), f)
  })

  class(wrapper) <- c("keras_layer_wrapper", "function")
  attr(wrapper, "Layer") <- Layer

  # create_layer() will call r_to_py() as needed, but we create a promise here
  # to avoid creating the class constructor from scratch every time a class
  # instance is created.
  if (!inherits(Layer, "python.builtin.type"))
    delayedAssign("Layer", r_to_py(attr(wrapper, "Layer", TRUE), convert))

  wrapper
}


#' @export
r_to_py.keras_layer_wrapper <- function(x, convert = FALSE) {
  layer <- attr(x, "Layer", TRUE)
  if (!inherits(layer, "python.builtin.type"))
    layer <- r_to_py(layer, convert)
  layer
}


as_tf_shape <- function (x) {
  if (inherits(x, "tensorflow.python.framework.tensor_shape.TensorShape"))
    x
  else
    shape(dims = x)
}
