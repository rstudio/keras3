# closet with future and past utils not currently in use

#' @importFrom rlang names2
named_list <- function(...) {
    exprs <- eval(substitute(alist(...)))
    vals <- list(...)
    nms <- names2(vals)

    missing <- nms == ""
    if (all(!missing))
      return(vals)

    deparse2 <- function(x) paste(deparse(x, 500L), collapse = "")
    defaults <- vapply(exprs[missing], deparse2, "", USE.NAMES = FALSE)
    names(vals)[missing] <- defaults
    vals
}

drop_nulls <- function(x, i = NULL) {
  if(is.null(i))
    x[!vapply(x, is.null, FALSE, USE.NAMES = FALSE)]
  else {
    drop <- rep(FALSE, length(x))
    drop[i] <- vapply(x[i], is.null, FALSE, USE.NAMES = FALSE)
    x[drop]
  }
}

# conflict with utils::zip, maybe another name?
zip <- function(..., simplify = TRUE)
  .mapply(if(simplify) c else list, list(...), NULL)


inspect <- reticulate::import("inspect")

py_formals <- function(py_obj) {
  # returns python fn formals as a list (formals(),
  # but for py functions/methods
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
    c(name, param) %<-% x

    # we generally don't need to supply self in R
    # though arguably this might be better somewhere else
    if (name == 'self')
      next

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
      args[[name]] <- quote(expr = )
      next
    }

    args[[name]] <- default
  }
  args

}


docstring_parser <- reticulate::import("docstring_parser")
# reticulate::py_install("docstring_parser", pip = TRUE)


get_doc <- function(py_obj) {
  docstring_parser$parse(
    inspect$getdoc(py_obj))
    # style = docstring_parser$DocstringStyle$GOOGLE)
    # ## not all doc strings successfully parse google style,
    # ## some default to REST style
}


py_str.docstring_parser.common.Docstring <- function(x) {
  cat(docstring_parser$compose(x))
}

py_str.docstring_parser.common.DocstringParam <- function(x) {
  d <- x$description
  if(!is.null(x$default))
    d <- paste("\nDefault: ", x$default)

  d <- stringi::stri_split_lines1(d)
  d[1] %<>%  paste("#' @param", x$arg_name, .)
  if (length(d) > 1)
    d[-1] %<>% paste("#'   ", .)
  writeLines(d)
}


maybe_rename <- function(x, ...) {
  spec <- list(...)
  i <- names(x) %in% spec
  names(x)[i] <- names(spec)[spec == names(x)[i]]
  x
}




# tf$reshape() doesn't accept a tf.TensorShape object
# normalize_shape <-
function (x) {
  # reflect NULL back
  if (is.null(x))
    return(x)
  else
    as_shape(x)
}


# tf$reshape() doesn't accept a tf.TensorShape object
# as_shape <-
function (x) {

  if (inherits(x, "tensorflow.python.framework.tensor_shape.TensorShape"))
    return(x)

  if (is.null(x))
    dims <- NULL
  else
    dims <- lapply(x, function(d) {
      if (is.null(d) || isTRUE(is.na(d)))
        NULL
      else
        as.integer(d)
    })

  tensorflow::tf$TensorShape(dims)
}
