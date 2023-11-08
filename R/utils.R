
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
#'   model %>% compile(
#'     loss = "binary_crossentropy",
#'     optimizer = optimizer_nadam(),
#'     metrics = c("top_k_acc")
#'   )
#'
#'   # save the model
#'   save_model_hdf5("my_model.h5")
#'
#'   # loading the model within the custom object scope doesn't
#'   # require explicitly providing the custom_object
#'   load_model_hdf5("my_model.h5")
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

#' Keras array object
#'
#' Convert an R vector, matrix, or array object to an array that has the optimal
#' in-memory layout and floating point data type for the current Keras backend.
#'
#' Keras does frequent row-oriented access to arrays (for shuffling and drawing
#' batches) so the order of arrays created by this function is always
#' row-oriented ("C" as opposed to "Fortran" ordering, which is the default for
#' R arrays).
#'
#' If the passed array is already a NumPy array with the desired `dtype` and "C"
#' order then it is returned unmodified (no additional copies are made).
#'
#' @param x Object or list of objects to convert
#' @param dtype NumPy data type (e.g. float32, float64). If this is unspecified
#'   then R doubles will be converted to the default floating point type for the
#'   current Keras backend.
#'
#' @return NumPy array with the specified `dtype` (or list of NumPy arrays if a
#'   list was passed for `x`).
#'
#' @export
keras_array <- function(x, dtype = NULL) {

  # reflect NULL
  if (is.null(x))
    return(x)

  # reflect HDF5
  if (inherits(x, "keras.utils.io_utils.HDF5Matrix"))
    return(x)

  # reflect tensor for keras v2.2 or TF implementation >= 1.12
  if (is_tensorflow_implementation()) {
    if (
      tf_version() >= "1.12" &&
      (
        is_keras_tensor(x) || is.list(x) && all(vapply(x, is_keras_tensor, logical(1)))
      )
    ) {
      return(x)
    }
  } else {
    if ((keras_version() >= "2.2.0") && is_keras_tensor(x)) {
      return(x)
    }
  }

  # error for data frames
  if (is.data.frame(x)) {
    x <- as.list(x)
  }

  # allow passing things like pandas.Series(), for workarounds like
  # https://github.com/rstudio/keras/issues/1341
  if(inherits(x, "python.builtin.object"))
    return(x)

  # recurse for lists
  if (is.list(x))
    return(lapply(x, keras_array))

  # convert to numpy
  if (!inherits(x, "numpy.ndarray")) {

    # establish the target datatype - if we are converting a double from R
    # into numpy then use the default floatx for the current backend
    if (is.null(dtype) && is.double(x))
      dtype <- k_floatx()

    # convert non-array to array
    if (!is.array(x))
      x <- as.array(x)

    # do the conversion (will result in Fortran column ordering)
    x <- r_to_py(x)
  }

  # if we don't yet have a dtype then use the converted type
  if (is.null(dtype))
    dtype <- x$dtype

  # ensure we use C column ordering (won't create a new array if the array
  # is already using C ordering)
  x$astype(dtype = dtype, order = "C", copy = FALSE)
}


#' Return the default float type, as a string.
#'
#' @description
#' E.g. `'float16'`, `'float32'`, `'float64'`.
#'
#' # Returns
#' String, the current default float type.
#'
#' # Examples
#' ```python
#' keras.config.floatx()
#' # 'float32'
#' ```
#'
#' @export
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/config/floatx>
k_floatx <- function() {
  keras$config$floatx()
}


function(x) {
  # k_config_floatx?
  if(missing(x))
    keras$config$floatx()
  else
    keras$config$set_floatx(x)
}


#' Check if Keras is Available
#'
#' Probe to see whether the Keras Python package is available in the current
#' system environment.
#'
#' @param version Minimum required version of Keras (defaults to `NULL`, no
#'   required version).
#'
#' @return Logical indicating whether Keras (or the specified minimum version of
#'   Keras) is available.
#'
#' @examples
#' \dontrun{
#' # testthat utilty for skipping tests when Keras isn't available
#' skip_if_no_keras <- function(version = NULL) {
#'   if (!is_keras_available(version))
#'     skip("Required keras version not available for testing")
#' }
#'
#' # use the function within a test
#' test_that("keras function works correctly", {
#'   skip_if_no_keras()
#'   # test code here
#' })
#' }
#'
#' @export
is_keras_available <- function(version = NULL) {
  implementation_module <- resolve_implementation_module()
  if (reticulate::py_module_available(implementation_module)) {
    if (!is.null(version))
      keras_version() >= version
    else
      TRUE
  } else {
    FALSE
  }
}


#' Keras implementation
#'
#' Obtain a reference to the Python module used for the implementation of Keras.
#'
#' These are the available Python modules which implement Keras:
#'
#' - keras
#' - tensorflow.keras ("tensorflow")
#' - keras_core ("core")
#'
#' This function returns a reference to the implementation being currently
#' used by the keras package. The default implementation is "keras".
#' You can override this by setting the `KERAS_IMPLEMENTATION` environment
#' variable to "tensorflow".
#'
#' @return Reference to the Python module used for the implementation of Keras.
#'
#' @export
implementation <- function() {
  keras
}


is_backend <- function(name) {
  identical(keras$config$backend(), name)
}

is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}

is_osx <- function() {
  Sys.info()["sysname"] == "Darwin"
}

is_layer <- function(object) {
  inherits(object, "keras.engine.topology.Layer")
}

relative_to <- function(dir, file) {

  # normalize paths
  dir <- normalizePath(dir, mustWork = FALSE, winslash = "/")
  file <- normalizePath(file, mustWork = FALSE, winslash = "/")

  # ensure directory ends with a /
  if (!identical(substr(dir, nchar(dir), nchar(dir)), "/")) {
    dir <- paste(dir, "/", sep="")
  }

  # if the file is prefixed with the directory, return a relative path
  if (identical(substr(file, 1, nchar(dir)), dir))
    file <- substr(file, nchar(dir) + 1, nchar(file))

  # simplify ./
  if (identical(substr(file, 1, 2), "./"))
    file <- substr(file, 3, nchar(file))

  file
}


is_keras_tensor <- function(x) {
  if (is_tensorflow_implementation()) {
    if (tensorflow::tf_version() >= "2.0") tensorflow::tf$is_tensor(x) else tensorflow::tf$contrib$framework$is_tensor(x)
  } else {
    k_is_tensor(x)
  }
}


split_dots_named_unnamed <- function(dots) {
  nms <- names(dots)
  if (is.null(nms))
    return(list(unnamed = dots, named = list()))
  named <- nzchar(nms)
  list(unnamed = dots[!named], named = dots[named])
}


assert_all_dots_named <- function(envir = parent.frame(), cl) {

  x <- evalq(eval(substitute(alist(...))), envir)
  if (!length(x)) return()

  if (identical(x[length(x)], list(quote(expr =))))
    x[[length(x)]] <- NULL

  if (!length(x)) return()

  x <- names(x)
  if (is.character(x) && !anyNA(x) && all(x != ""))
    return()

  stop("All arguments provided to `...` must be named.\n",
       "Call with unnamed arguments in dots:\n  ",
       paste(deparse(cl, 500L), collapse = "\n"))
}

# TODO: should there be some default modifiers in capture_args() for standard layer args
# like, input_shape, batch_input_shape, etc.



capture_args <- function(cl, modifiers = NULL, ignore = NULL,
                         envir = parent.frame(), fn = sys.function(-1)) {

  ## bug: match.call() resolves incorrectly if dots are from not the default sys.parent()
  ## e.g, this fails if dots originate from the callers caller:
  #    cl <- eval(quote(match.call()), parent.frame())
  ## workaround: caller must call match.call() from the correct frame

  ## note: capture_args() must always be called at the top level of the intended function body.
  ## sys.function(-1) resolves to the incorrect function if the  capture_args()
  ## call is itself a promise in another call. E.g.,:
  ##   do.call(foo, capture_args(match.call())) fails because fn resolves to do.call()

  fn_arg_nms <- names(formals(fn))
  known_args <- intersect(names(cl), fn_arg_nms)
  known_args <- setdiff(known_args, ignore)
  names(known_args) <- known_args
  cl2 <- c(quote(list), lapply(known_args, as.symbol))

  if("..." %in% fn_arg_nms && !"..." %in% ignore) {
    assert_all_dots_named(envir, cl)
    # this might reorder args by assuming ... are last, but it doesn't matter
    # since everything is supplied as a keyword arg to the Python side anyway
    cl2 <- c(cl2, quote(...))
  }

  args <- eval(as.call(cl2), envir)

  # check `ignore` again, since arg might have been in `...`
  for(nm in intersect(names(args), ignore))
    args[[nm]] <- NULL

  nms_to_modify <- intersect(names(args), names(modifiers))
  for (nm in nms_to_modify)
    args[nm] <- list(modifiers[[nm]](args[[nm]]))
   # list() so if modifier returns NULL, don't remove the arg

  args
}


#' @importFrom rlang quos eval_tidy
# capture_args2 <-
  function(modifiers = NULL, ignore = NULL) {
  cl0 <- cl <- sys.call(-1L)
  envir <- parent.frame(2L)
  fn <- sys.function(-1L)

  # first defuse rlang !!! and := in calls
  # cl[[1L]] <- quote(rlang::quos)
  cl[[1L]] <- quos
  cl_exprs <- eval(cl, envir)

  # build up a call to base::list() using the exprs
  cl <- as.call(c(list, cl_exprs))

  # match.call()
  cl <- match.call(fn, cl,
                   expand.dots = !"..." %in% ignore,
                   envir = envir)

  # filter out args to ignore
  for(ig in intersect(names(cl), ignore))
    cl[[ig]] <- NULL

  # eval and capture args
  args <- eval_tidy(cl, env = envir)

  # apply modifier functions. e.g., as_nullable_integer
  nms_to_modify <- intersect(names(args), names(modifiers))
  for (nm in nms_to_modify) {

    # escape hatch: user supplied python objects pass through untransformed
    if (inherits(args[[nm]] -> val, "python.builtin.object"))
      next

    # list() so if modifier returns NULL, don't remove the arg
    args[nm] <- list(modifiers[[nm]](val))
  }

  args
}


#' @importFrom rlang quos eval_tidy
capture_args2 <- function(modifiers = NULL, ignore = NULL, force = NULL) {
  cl <- sys.call(-1L)
  envir <- parent.frame(1L)
  fn <- sys.function(-1L)

  if (identical(cl[length(cl)], as.call(list(quote(expr = )))))
    cl[[length(cl)]] <- NULL
  cl <- match.call(fn, cl, expand.dots = TRUE, envir = parent.frame(2))

  fn_arg_nms <- names(formals(fn))
  known_args <- intersect(names(cl), fn_arg_nms)
  known_args <- setdiff(known_args, ignore)
  known_args <- union(known_args, force)
  names(known_args) <- known_args
  cl2 <- c(quote(list), lapply(known_args, as.symbol))

  if("..." %in% fn_arg_nms && !"..." %in% ignore) {
    assert_all_dots_named(envir, cl)
    # this might reorder args by assuming ... are last, but it doesn't matter
    # since everything is supplied as a keyword arg to the Python side anyway
    cl2 <- c(cl2, quote(...))
    # use list2 to accept trailing empty `,`
    cl2[[1]] <- quote(rlang::list2)
  }

  args <- eval(as.call(cl2), envir)

  # filter out args to ignore
  for(ig in intersect(names(cl), ignore))
    cl[[ig]] <- NULL

  # apply modifier functions. e.g., as_nullable_integer
  nms_to_modify <- intersect(names(args), names(modifiers))
  for (nm in nms_to_modify) {

    # escape hatch: user supplied python objects pass through untransformed
    if (inherits(args[[nm]] -> val, "python.builtin.object"))
      next

    # list() so if modifier returns NULL, don't remove the arg
    args[nm] <- list(modifiers[[nm]](val))
  }

  args
}


is_scalar <- function(x) identical(length(x), 1L)

is_mac_arm64 <- function() {
  sys_info <- Sys.info()
  sys_info[["sysname"]] == "Darwin" &&
  sys_info[["machine"]] == "arm64"
}


#' Plot a Keras model
#'
#' @param x A Keras model instance
#' @param to_file File name of the plot image. If `NULL` (the default), the
#'   model is drawn on the default graphics device. Otherwise, a file is saved.
#' @param show_shapes whether to display shape information.
#' @param show_dtype whether to display layer dtypes.
#' @param show_layer_names whether to display layer names.
#' @param ... passed on to `keras$utils$plot_model()`. Used for forward and
#'   backward compatibility.
#' @param rankdir a string specifying the format of the plot: `'TB'` creates a
#'   vertical plot; `'LR'` creates a horizontal plot. (argument passed to PyDot)
#' @param expand_nested Whether to expand nested models into clusters.
#' @param dpi Dots per inch. Increase this value if the image text appears
#'   excessively pixelated.
#' @param layer_range `list` containing two character strings, which is the
#'   starting layer name and ending layer name (both inclusive) indicating the
#'   range of layers for which the plot will be generated. It also accepts regex
#'   patterns instead of exact name. In such case, start predicate will be the
#'   first element it matches to `layer_range[1]` and the end predicate will be
#'   the last element it matches to `layer_range[2]`. By default `NULL` which
#'   considers all layers of model. Note that you must pass range such that the
#'   resultant subgraph must be complete.
#' @param show_layer_activations Display layer activations (only for layers that
#'   have an `activation` property).
#'
#' @return Nothing, called for it's side effects.
#'
#' @section Raises: ValueError: if `plot_model` is called before the model is
#'   built, unless a `input_shape = ` argument was supplied to
#'   `keras_model_sequential()`.
#'
#' @section Requirements:
#'   This function requires pydot and graphviz.
#'   `pydot` is by default installed by `install_keras()`, but if you installed
#'   tensorflow by other means, you can install pydot directly with :
#'   ````
#'   reticulate::py_install("pydot", pip = TRUE)
#'   ````
#'   In a conda environment, you can install graphviz with:
#'   ```
#'   reticulate::conda_install(packages = "graphviz")
#'   # Restart the R session after install.
#'   ```
#'   Otherwise you can install graphviz from here:
#'   <https://graphviz.gitlab.io/download/>
#'
#' @export
plot.keras.models.model.Model <-
function(x,
         show_shapes = FALSE,
         show_dtype = FALSE,
         show_layer_names = TRUE,
         ...,
         rankdir = "TB",
         expand_nested = FALSE,
         dpi = 96,
         layer_range = NULL,
         show_layer_activations = FALSE,
         to_file = NULL) {
  args <- capture_args(match.call(), ignore = "x")
  args$model <- x
  if (is.null(to_file)) {
    args$to_file <-
      tempfile(paste0("keras_", x$name), fileext = ".png")
    on.exit(unlink(args$to_file), add = TRUE)
  }

  if(is_windows() && identical(.Platform$GUI, "RStudio")) {
    # need to patch subprocess.Popen to avoid an OSError exception.
    # https://github.com/rstudio/keras/issues/1337
    # https://stackoverflow.com/questions/43593348/winerror-6-the-handle-is-invalid-from-python-check-output-spawn-in-electron-app/43606682#43606682
    .patched_subprocess_Popen <-
      import("kerastools.utils")$`_patched_subprocess_Popen`()
    .patched_subprocess_Popen$`__enter__`()
    on.exit(.patched_subprocess_Popen$`__exit__`(NULL, NULL, NULL),
            add = TRUE)
  }

  tryCatch(
    do.call(keras$utils$plot_model, args),
    error = function(e) {
      message("See ?keras::plot.keras.models.model.Model for ",
              " instructions on how to install graphviz and pydot")
      e$call <- sys.call(1)
      stop(e)
    }
  )
  if(!is.null(to_file))
    return(invisible())

  img <- png::readPNG(args$to_file, native = TRUE)
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, ncol(img)), ylim = c(0, nrow(img)),
                        asp = 1, yaxs = "i", xaxs = "i")
  graphics::rasterImage(img, 0, 0, ncol(img), nrow(img), interpolate = FALSE)
  invisible()
}


#' zip lists
#'
#' This is conceptually similar to `zip()` in Python, or R functions
#' `purrr::transpose()` and `data.table::transpose()` (albeit, accepting
#' elements in `...` instead of a single list), with one crucial difference: if
#' the provided objects are named, then matching is done by names, not
#' positions.
#'
#' All arguments supplied must be of the same length. If positional matching is
#' required, then all arguments provided must be unnamed. If matching by names,
#' then all arguments must have the same set of names, but they can be in
#' different orders.
#'
#' @param ... R lists or atomic vectors, optionally named.
#'
#' @return A inverted list
#' @export
#'
#' @examples
#' gradients <- list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3")
#' weights <- list("weight_1", "weight_2", "weight_3")
#' str(zip_lists(gradients, weights))
#' str(zip_lists(gradient = gradients, weight = weights))
#'
#' names(gradients) <- names(weights) <- paste0("layer_", 1:3)
#' str(zip_lists(gradients, weights[c(3, 1, 2)]))
#'
#' names(gradients) <- paste0("gradient_", 1:3)
#' try(zip_lists(gradients, weights)) # error, names don't match
#' # call unname directly for positional matching
#' str(zip_lists(unname(gradients), unname(weights)))
zip_lists <- function(...) {
  dots <- list(...)
  if(length(dots) == 1)
    dots <- dots[[1L]]

  nms1 <- names(dots[[1L]])

  if (is.character(nms1))
    if (!anyDuplicated(nms1) && !anyNA(nms1) && !all(nzchar(nms1)))
      stop("All names must be unique. Call `unname()` if you want positional matching")

  for(i in seq_along(dots)[-1L]) {
    d_nms <- names(dots[[i]])
    if(identical(nms1, d_nms))
       next
    if(setequal(nms1, d_nms)) {
      dots[[i]] <- dots[[i]][nms1]
      next
    }
    stop("All names of arguments provided to `zip_lists()` must match.",
         " Call `unname()` on each argument if you want positional matching")
  }

  ans <- .mapply(list, dots, NULL)
  names(ans) <- nms1
  ans
}


drop_nulls <- function(x, i = NULL) {
  if(is.null(i))
    return(x[!vapply(x, is.null, FALSE, USE.NAMES = FALSE)])

  drop <- logical(length(x))
  names(drop) <- names(x)
  drop[i] <- vapply(x[i], is.null, FALSE, USE.NAMES = FALSE)
  x[!drop]
}

#' @export
as.array.keras.backend.common.variables.KerasVariable <- function(x, ...) {
  as_r_value(k_convert_to_numpy(x))
}

#' @export
as.numeric.keras.backend.common.variables.KerasVariable <- function(x, ...) {
  as.numeric(as_r_value(k_convert_to_numpy(x)))
}

#' @export
as.double.keras.backend.common.variables.KerasVariable <- function(x, ...) {
  as.double(as_r_value(k_convert_to_numpy(x)))
}

#' @export
as.integer.keras.backend.common.variables.KerasVariable <- function(x, ...) {
  as.integer(as_r_value(k_convert_to_numpy(x)))
}

as_r_value <- function (x) {
  if (inherits(x, "python.builtin.object"))
    py_to_r(x)
  else x
}


# internal `[` method that ensures functions in this namespace use one-based
# indexing in case user has a global option set for zero-based indexing.
`[.tensorflow.tensor` <-
  getS3method("[", "tensorflow.tensor", envir = asNamespace("tensorflow"))
formals(`[.tensorflow.tensor`)$style <- "R"
formals(`[.tensorflow.tensor`)$options <-
  tensorflow::tf_extract_opts(
    one_based = TRUE,
    inclusive_stop = TRUE,
    disallow_out_of_bounds = TRUE,
    warn_tensors_passed_asis = FALSE,
    warn_negatives_pythonic = FALSE
  )



standard_layer_arg_modifiers <- list(
    input_shape = normalize_shape,
    batch_input_shape = normalize_shape,
    batch_size = as_nullable_integer,
    seed = as_nullable_integer
  )

#' @importFrom rlang dots_list
named_list <- function(...)
  dots_list(...,
            .named = TRUE,
            .ignore_empty = "trailing",
            .preserve_empty = FALSE,
            .homonyms = "error",
            .check_assign = FALSE)


knit_vignette <- function(input, ..., output_dir) {
  # print(sys.call())
  # stop()
  input <- normalizePath(input)
  render_dir <- dirname(input)
  if(getwd() != render_dir) {
    message("Changing wd to ", render_dir)
    owd <- setwd(render_dir)
    on.exit(setwd(owd))
  }


  # ~/github/rstudio/keras/vignettes/writing_your_own_callbacks.Rmd
  name <- basename(render_dir)
  output_file <- normalizePath(sprintf("../../vignettes/%s.Rmd", name),
                               mustWork = FALSE)
  message("output_file: ", output_file)

  set.seed(1)
  keras::set_random_seed(1)
  knitr::opts_chunk$set(
    collapse = TRUE,
    comment = "##" #>
  )
  rmarkdown::render(
    input,
    output_format = rmarkdown::github_document(preserve_yaml = TRUE),
    # output_format = rmarkdown::md_document( preserve_yaml = TRUE, ext = "Rmd"), #
    # output_format = rmarkdown::md_document(preserve_yaml = FALSE), # , ext = "Rmd"
    output_file = output_file,
    envir = new.env(parent = globalenv()),
    ...
  )
  x <- readLines(output_file)
  # if(length(grep("^knit: keras:::knit_vignette", x) -> i))
    # x <- x[-i]
  end_fm_i <- which(x == "---")[2]
  x_fm <- x[2:(end_fm_i-1)]
  yaml.load <- getExportedValue("yaml", "yaml.load")
  as.yaml <- getExportedValue("yaml", "as.yaml")
  fm <- yaml.load(x_fm)

  fm$knit <- NULL
  fm$output <- "rmarkdown::html_vignette"
  fm$accelerator <- NULL
  last_modified_date <- reticulate:::system2t("git", c("log -1 --pretty=format:'%ad'",
                                         "--date=format:'%Y-%m-%d'",
                                         "--", shQuote(input)), stdout = TRUE)
  # message("Last modified: ", last_modified_date)
  fm$date <- sprintf("Last Modified: %s; Last Rendered: %s",
                     last_modified_date, format(Sys.Date()))
  # TODO: fm$date <- Last compiled on `r format(Sys.time(), '%d %B, %Y')`, last updated on `r system(git `
  # fm$date <- format(Sys.Date())
  vignette <- glue::glue_data(list(title = fm$title), .trim = FALSE,
                              .open = "<<", .close = ">>",
"vignette: >
  %\\VignetteIndexEntry{<<title>>}
  %\\VignetteEngine{knitr::rmarkdown}
  %\\VignetteEncoding{UTF-8}")

  # dumping vignette via as.yaml breaks downstream, the rd entry needs to be a block
  fm <- as.yaml(fm) # has a trailing \n
  fm <- paste0(fm, vignette)

  x <- c("---", fm, "---", x[-(1:end_fm_i)])
  writeLines(x, output_file)
}


# Generate a Random Array
#
# This function generates an array with random numbers.
# The dimensions of the array are specified by the user.
# The generation function for the random numbers can also be customized.
#
# @param ... Dimensions for the array as separate integers or as a single vector.
# @param gen A function for generating random numbers, defaulting to `runif`.
#
# @return Returns an array with the specified dimensions filled with random numbers.
#
# @examples
# # Create a 3x3 matrix with random numbers from uniform distribution
# random_array(3, 3)
#
# # Create a 2x2x2 array with random numbers from normal distribution
# random_array(2, 2, 2, gen = rnorm)
#
# # Create a 2x2 array with a sequence of integers.
# random_array(2, 2, gen = seq)
#
# @export
random_array <- function(..., gen = stats::runif) {
  dim <- unlist(c(...), use.names = FALSE)
  array(gen(prod(dim)), dim = dim)
}
