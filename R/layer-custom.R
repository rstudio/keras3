
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
#' @param convert ignored.
#'
#' See guide 'making_new_layers_and_models_via_subclassing.Rmd' for example usage.
#'
#' @return An R function that behaves similarly to the builtin keras `layer_*`
#'   functions. When called, it will create the class instance, and also
#'   optionally call it on a supplied argument `object` if it is present. This
#'   enables keras layers to compose nicely with the pipe (`|>`).
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

  if(!isTRUE(convert))
    warning("convert argument is ignored")

  out <- as.function.default(
    c(alist(object = ), formals(Layer),
      bquote({
        args <- capture_args2(.(modifiers), ignore = "object")
        create_layer(Layer, object, args)
      })),
    envir = list2env(list(Layer = Layer),
                     parent = parent.env(environment()))

  )

  class(out) <- c("keras_Layer_wrapper",
                  "keras_layer_wrapper",
                  "function")
  out
}
