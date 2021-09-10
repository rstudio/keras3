
library(tidyverse)
library(tensorflow)
library(keras)

stopifnot(interactive())
inspect <- reticulate::import("inspect")

docstring_parser <- tryCatch(
  reticulate::import("docstring_parser"),
  error = function(e) {
    reticulate::py_install("docstring_parser", pip = TRUE)
    reticulate::import("docstring_parser")
  }
)


get_doc <- function(py_obj) {
  doc <- docstring_parser$parse(
    inspect$getdoc(py_obj))
  doc$object <- py_obj
  doc
    # style = docstring_parser$DocstringStyle$GOOGLE)
    # ## not all doc strings successfully parse google style,
    # ## some default to REST style
    #
  # TODO: Bug: this lumps class attributes with __init__ args
}


py_str.docstring_parser.common.Docstring <- function(x) {
  cat(docstring_parser$compose(x))
}


cleanup_description <- function(x) {

    # remove leading and trailing whitespace
    x <- gsub("^\\s+|\\s+$", "", x)

    # convert 2+ whitespace to 1 ws
    # x <- gsub("(\\s\\s+)", " ", x)

    # convert literals
    x <- gsub("None", "NULL", x, fixed=TRUE)
    x <- gsub("True", "TRUE", x, fixed=TRUE)
    x <- gsub("False", "FALSE", x, fixed=TRUE)

    # convert tuple to list
    x <- gsub("tuple", "list", x, fixed=TRUE)
    x <- gsub("list/list", "list", x, fixed=TRUE)

    x
}

r_doc_from_py_fn <- function(py_fn, name = NULL) {
  con <- textConnection("r-doc", "w")
  on.exit(close(con))
  cat <- function(...,  file = con)
    base::cat(..., "\n", file = file)

  x <- get_doc(py_fn)


  # first sentence is taken as title
  # 2nd paragraph is taken as @description
  # 3rd paragraph + is taken as @details

  title <- cleanup_description(x$short_description)
  # title should have no trailing '.'
  if (str_sub(title, -1) == ".")
    title <- str_sub(title, end = -2)

  # cat("@title ", title)
  cat(title)

  desc <- cleanup_description(x$long_description)
  cat()

  # avoid splitting across @description and @details,
  # so put everything in @details
  if (length(desc) != 0 && str_detect(desc, "\n")) {
    # cat("@description") # description can't be empty
    cat("@details")
  }
  cat(desc)

  for (p in x$params) {
    if (p$arg_name %in% c("name", "dtype")) next
    cat("\n@param", p$arg_name, cleanup_description(p$description))
  }

  cat("@param ... standard layer arguments.")
  # TODO: @inheritDotParams keras.layers.Layer

  cat()

  py_full_name <- paste0(py_fn$`__module__`, ".", py_fn$`__name__`)
  cat("@seealso")
  cat(sprintf("  +  <%s>", reticulate:::.module_help_handlers$tensorflow(py_full_name)))
  # TODO: add tests for all the F1 url pages to find+fix broken links
  cat("  +  <https://keras.io/api/layers>")

  cat("@export")

  x <- textConnectionValue(con)
  x <- stringr::str_flatten(x, "\n")
  x <- gsub("\n", "\n#' ", x)
  x <- str_c("#' ", x, "\n", name)
  x
}

# source is the object
# topic is the character string of obj name

new_layer_wrapper <- function(py_obj) {
  # py_obj_expr <- substitute(py_obj)

  transformers <- NULL
  frmls <- keras:::py_formals(py_obj)
  for(i in seq_along(frmls)) {
    key <- names(frmls)[i]
    if(identical(unname(frmls[i]), list(quote(expr = ))))
      next
    val <- frmls[[i]]

    if(is.integer(val))
      transformers[[key]] <- quote(as.integer)
    if(key == "axis")
       transformers[[key]] <- quote(as_axis)
  }

  py_obj_expr <- substitute(keras$layers$NAME, list(NAME=as.name(py_obj$`__name__`)))
  fn_body <- bquote({
    args <- capture_args(match.call(), .(transformers), ignore = "object")
    create_layer(.(py_obj_expr), object, args)
  })

  frmls$self <- NULL
  fn <- as.function(c(alist(object=), frmls, fn_body))

  fn_string <- deparse(fn)

  # deparse adds a space for some reason
  fn_string <- sub("function (", "function(", fn_string, fixed = TRUE)

  r_wrapper_name <- sprintf("layer_%s <- ", snakecase::to_snake_case(py_obj$`__name__`))
  fn_string <- str_flatten(c(r_wrapper_name, fn_string), "\n")
  docs <- r_doc_from_py_fn(py_obj)
  out <- str_flatten(c(docs, fn_string), "")
  class(out) <-  "r_py_wrapper2"
  out
}


print.r_py_wrapper2 <- function(x, ...) {
  clipr::write_clip(x)
  cat(x)
}

## example usage:
# new_layer_wrapper(keras$layers$TextVectorization)
