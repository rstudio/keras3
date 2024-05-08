# ---- general utils ----

is_backend <- function(name) {
  identical(keras$config$backend(), name)
}

is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}

is_osx <- function() {
  Sys.info()["sysname"] == "Darwin"
}

is_mac_arm64 <- function() {
  sys_info <- Sys.info()
  sys_info[["sysname"]] == "Darwin" &&
    sys_info[["machine"]] == "arm64"
}

is_scalar <- function(x) identical(length(x), 1L)

# is_py_object <- function(x) is_py_object(x)

split_dots_named_unnamed <- function(dots) {
  nms <- names(dots)
  if (is.null(nms))
    return(list(unnamed = dots, named = list()))
  named <- nzchar(nms)
  list(unnamed = dots[!named], named = dots[named])
}

drop_nulls <- function(x, i = NULL) {
  if(is.null(i))
    return(x[!vapply(x, is.null, FALSE, USE.NAMES = FALSE)])

  drop <- logical(length(x))
  names(drop) <- names(x)
  drop[i] <- vapply(x[i], is.null, FALSE, USE.NAMES = FALSE)
  x[!drop]
}

#' @importFrom rlang dots_list
# identical to rlang::list2(), except .named = TRUE
named_list <- function(...)
  dots_list(...,
            .named = TRUE,
            # not the default
            .ignore_empty = "trailing",
            .preserve_empty = FALSE,
            .homonyms = "error",
            .check_assign = FALSE)

`append1<-` <- function(x, value) {
  x[[length(x) + 1L]] <- value
  x
}

`append<-` <- function(x, value) c(x, value)

`prepend<-` <- function(x, value) c(value, x) # c(x[integer()], value, x)

replace_val <- function(x, old, new) {
  if (!is_scalar(new))
    stop("Unexpected length of replacement value in replace_val().\n",
         "`new` must be length 1, not ", length(new))
  x[x %in% old] <- new
  x
}

imap <- function(.x, .f, ...) {
  out <- .mapply(.f, list(.x, names(.x) %||% seq_along(.x)), list(...))
  names(out) <- names(.x)
  out
}

map2 <- function(.x, .y, .f, ...) {
  out <- .mapply(.f, list(.x, .y), list(...))
  if(length(.x) == length(out))
    names(out) <- names(.x)
  out
}

map_chr <- function(.x, .f, ...) {
  out <- vapply(X = .x, FUN = .f, FUN.VALUE = "", ..., USE.NAMES = FALSE)
  names(out) <- names(.x)
  out
}

map_lgl <- function(.x, .f, ...) {
  out <- vapply(X = .x, FUN = .f, FUN.VALUE = TRUE, ..., USE.NAMES = FALSE)
  names(out) <- names(.x)
  out
}

map_int <- function(.x, .f, ...) {
  out <- vapply(X = .x, FUN = .f, FUN.VALUE = 0L, ..., USE.NAMES = FALSE)
  names(out) <- names(.x)
  out
}

last <- function(x) x[[length(x)]]

second_to_last <- function(x)
  if((lx <- length(x)) > 1) x[[lx-1L]]

rename <- function(x, ..., .skip_existing = TRUE) {
  dots <- list(...)
  nms <- names(x)
  for(i in seq_along(dots)) {
    newname <- names(dots)[[i]]
    oldname <- dots[[i]]
    if(.skip_existing && newname %in% nms)
      next
    nms[match(oldname, nms)] <- newname
  }
  names(x) <- nms
  x
}

`%""%` <- function (x, y) {
  if(!is.character(x))
    stop("x must be character")
  not_empty <- nzchar(x)
  if(all(not_empty))
    return(x)
  if(!is.character(y))
    stop("y must be character")
  # don't force `y` unless needed
  if (!identical(length(y), length(x))) {
    stopifnot(identical(length(y), 1L))
    y <- rep(y, length(x))
  }
  empty <- !not_empty
  x[empty] <- y[empty]
  x
}


# ---- arg checkers ----

check_bool <- function(x) {
  if (identical(x, TRUE) || identical(x, FALSE))
    x
  else
    stop(sprintf("`%s` arg must be `TRUE` or `FALSE`",
                 deparse1(substitute(x))))
}



# ---- arg transformers ----

as_array <- function(x)
  if(is.null(x) || is_py_object(x) || is.array(x))
    x else as.array(x)

as_py_array <- function(x)
  if(is.null(x) || is_py_object(x))
    x else np_array(x)

as_r_value <- function (x)
  if (is_py_object(x))
    py_to_r(x) else x

as_axis <- function(axis) {
  if (is.null(axis))
    return(NULL)

  if (length(axis) > 1)
    return(lapply(axis, as_axis))

  axis <- as.integer(axis)

  if (axis == 0L)
    stop("`axis` argument is 1 based, received 0")

  if (axis > 0L) axis - 1L
  else axis
}


# Helper function to coerce shape arguments to tuple
# tf$reshape()/k_reshape() doesn't accept a tf.TensorShape object
normalize_shape <- function(shape) {

  # reflect NULL back
  if (is.null(shape))
    return(shape)

  # already fixed up
  if (inherits(shape, "keras_shape"))
    return(shape)

  # if it's a list or a numeric vector then convert to integer
  # NA's in are accepted as NULL
  # also accept c(NA), as if it was a numeric
  if (is.list(shape) || is.numeric(shape) ||
      (is.logical(shape) && all(is.na(shape)))) {

    shape <- lapply(shape, function(value) {
      # Pass through python objects unmodified, only coerce R objects
      # supplied shapes, e.g., to tf$random$normal, can be a list that's a mix
      # of scalar integer tensors and regular integers
      if (is_py_object(value))
        return(value)

      # accept NA,NA_integer_,NA_real_ as NULL
      if ((is_scalar(value) && is.na(value)))
        return(NULL)

      if (!is.null(value))
        as.integer(value)
      else
        NULL
    })
  }

  if (inherits(shape, "tensorflow.python.framework.tensor_shape.TensorShape"))
    shape <- as.list(shape$as_list()) # unpack for tuple()

  # coerce to tuple so it's iterable
  tuple(shape)
}

as_integer <- function(x) {
  if (is.numeric(x))
    as.integer(x)
  else
    x
}

as_integer_array <- function(x) {
  if(is.atomic(x))
    x <- as.array(x)
  if(is.array(x) && storage.mode(x) != "integer")
    storage.mode(x) <- "integer"
  x
}

as_integer_tuple <- function(x, force_tuple = FALSE) {
  if (is.null(x))
    x
  else if (is.list(x) || force_tuple)
    tuple(as.list(as.integer(x)))
  else
    as.integer(x)
}

as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

as_layer_index <- function(x) {
  if (is.null(x))
    return(x)

  x <- as.integer(x)

  if (x == 0L)
    stop("`index` for get_layer() is 1-based (0 was passed as the index)")

  if (x > 0L)
    x - 1L
  else
    x
}



as_node_index <- function(node_index) {
  as.integer(node_index-1)
}


# Helper function to normalize paths
normalize_path <- function(path) {
  if (is.null(path))
    NULL
  else
    normalizePath(path.expand(path), mustWork = FALSE)
}


# unused
as_index <- function(x) {
  if(storage.mode(x) == "double")
    storage.mode(x) <- "integer"
  # k_array() pass through here...
  # TODO: implement an efficient way to check for negative slices
  x - 1L
}


# Sketch for an alternative approach to offsetting indexes,
# so that they are 1 based in the R runtime, but convert into python
#  as 0 based. Alternative implementaiton for Callback() epochs,
#  LearningRateSchedule(), and similar.
#
# as_r_index <- function(x) {
#   if(is.double(x))
#     x <- as.integer(x)
#   class(x) <- c("r_index", class(x))
#   x
# }
#
# r_to_py.r_index <- function(x) {
#   if (x > 0L) x - 1L else x
# }
#
# zero_to_one_index <- function(x) x + 1L


# ---- resolve_py_obj ----



resolve_wrapper_py_obj_expr <- function(x, prefer_class = TRUE) {
  if (!identical(class(x), "function"))
    return()

  ns <- environment(sys.function()) # keras3 namespace
  xe <- environment(x)

  if (identical(xe, emptyenv()))
    return()

  # only inspect pkg functions, or pkg wrapped functions

  ## is a wrapper returned by new_wrapped_py_class(), like Layer()
  if (identical(parent.env(xe), ns))
    return(quote(`__class__`))

  ## is a pkg exported function
  if (!(identical(xe, ns)))
    return()

  # standard builtin wrapper, like layer_dense, loss_*
  # (or Layer(), though that's handled above)
  last_cl <- last(body(x))
  if (is.call(last_cl) &&
      (identical(last_cl[[1L]], quote(do.call)) ||
       identical(last_cl[[1L]], quote(create_layer)))) {
    expr <- last_cl[[2L]]
    if (identical(expr, quote(callable))) {
      # loss_ or metric_ wrapper
      if (prefer_class)
        expr <- second_to_last(body(x))[[c(3, 3)]]
      else
        expr <- second_to_last(body(x))[[c(3, 4)]]
    }
    return(expr)
  }

  # application wrapper
  if (is.call(last_cl) &&
      identical(last_cl[[1L]], quote(set_preprocessing_attributes)) &&
      is.call(last_cl2 <- as.list(body(x))[[length(body(x)) - 1L]]) &&
      (identical(last_cl2[[c(3L, 1L)]], quote(do.call))))
    return(last_cl2[[c(3L, 2L)]])

  # bare builtin op_wrapper, like
  # op_add <- function(x1, x2) keras$ops$add(x1, x2)
  if (is.call(cl <- body(x)) &&
      (is.call(cl0 <- cl1 <- cl[[1L]]) ||
       (
         identical(cl0, quote(`{`)) &&
         length(cl1 <- as.list(cl[-1])) == 1 &&
         is.call(cl <- cl1[[1L]]) &&
         is.call(cl0 <- cl1 <- cl[[1L]])
       )))
  {
    while (is.call(cl0) && identical(cl0[[1L]], quote(`$`)))
      cl0 <- cl0[[2L]]

    if (identical(cl0, quote(keras)))
      return(cl1)
  }

  NULL
}

resolve_py_obj <- function(x, default_name = "anonymous_R_function",
                           env = asNamespace("keras3"),
                           prefer_class = TRUE,
                           convert = TRUE) {
  # this function is used:
  # - to resolve `inherit` args in the keras subclassing API
  #    (e.g., if `inherit` arg is a wrapper like `layer_dense`, or
  #    `layer_custom` returned by a Layer("Custom", ...))
  # - to resolve args that can come in as callables to `compile()`
  #    (e.g., loss, metrics)
  # - to resolve args that can come in as callables passed to layer_* constructors.
  #    (e.g., activations, initializers)
  # - to resolve custom_objects supplied to the saving & serialization API,
  #    (e.g., with_custom_object_scope(), load_model(), ...)

  # - `x` can come in as a language object, enabling lazy evaluation /
  #   delayed initialization python
  # - If `x` is a package exported wrapper, like `layer_dense` or similar,
  #   this will return the py callable object, like `keras$layers$Dense`
  #   This should work with *all* exported wrappers
  #   (loss_, activation_, layer_, op_*, etc.)
  # - Otherwise, If `x` is a bare R function, it will be coerced to
  #   a python function with `py_func2()`, which is similar to the default
  #   r_to_py(<func>) except:
  #     - the constructed python wrapper has an accurate signature that
  #       matches the R func (needed in some places where keras inspects the
  #       callable signature)
  #     - We work harder/better to resolve an appropriate __name__ (accepting
  #       R attributes "name", "__name__" and "py_function_name", and give an
  #       opportunity for us to provide a better default like "custom_metric"
  #       from methods like `compile()`)
  # - Otherwise, we return `x` unmodified (assuming it will be coerced via
  #   r_to_py() downstream). If `convert = FALSE`, we eagerly call `r_to_py(x)`.

  if (is.language(x))
    x <- eval(x, env)

  if (is.null(x) || is_py_object(x))
    return(x)

  if (is_bare_r_function(x)) {

    py_obj_expr <- resolve_wrapper_py_obj_expr(x, prefer_class = prefer_class)
    if (!is.null(py_obj_expr)) {
      # eval in environment(x): wrapper env, where we might find `__class__`.
      py_obj <- tryCatch(eval(py_obj_expr, environment(x)),
                         error = function(e) NULL)

      if (is_py_object(py_obj))
        return(py_obj)
    }

    return(as_py_function(x, default_name = default_name))
  }

  if (convert) x else r_to_py(x)
}

is_bare_r_function <- function(x) {
  identical(class(x), "function")
}

as_py_name <- function(x) {
  # sanitize a deparsed R expression into valid python symbol string
  if(is.language(x))
    x <- deparse(x, width.cutoff = 500L)[1]
  x <- make.names(as.character(x))
  x <- gsub(".", "_", x, fixed = TRUE)
  x
}

as_py_function <- function(fn, default_name = "r_func") {
  if(is_py_object(fn))
    return(fn)

  name <-
    attr(fn, "py_function_name", TRUE) %||%
    attr(fn, "__name__", TRUE) %||%
    attr(fn, "name", TRUE) %||%
    default_name

  # TODO: try to generate a pretty name using deparse(substitute(x)) would need
  # to update capture_args() to construct calls to transformers so that
  # substitute will work here.
  # if(is.null(name)) { name <- as_py_name(deparse1(substitute(x)))}
  py_func2(fn, convert = TRUE, name = name)
}

get_function_name <- function(fn) {
  if (is_py_object(fn))
    return(py_to_r(py_get_attr(fn, "__name__")))

  attr(fn, "py_function_name", TRUE) %||%
  attr(fn, "__name__", TRUE) %||%
  attr(fn, "name", TRUE)
}



# if(FALSE) {
#   # TODO: use this to generate a static list for populating
#   # a reverse lookup hashtable
# x <- lapply(asNamespace("keras3"), resolve_wrapper_py_obj_expr) |>
#   purrr::map_chr(\(expr) if(is.null(expr)) "" else deparse1(expr))
# df <- tibble::enframe(x, value = "expr")
# df <- df[order(df$name),]
# success <- df$expr != ""
#
#
# df[success, ] |> print(n = Inf)
# df[!success, ] |> print(n = Inf)
#
# # prefer_class = FALSE
# x <- lapply(asNamespace("keras3"), resolve_wrapper_py_obj_expr,
#             prefer_class = FALSE) |>
#   purrr::map_chr(\(expr) if(is.null(expr)) "" else deparse1(expr))
# df <- tibble::enframe(x, value = "expr")
# df <- df[order(df$name),]
# success <- df$expr != ""
# df[success, ] |> print(n = Inf)
# df[!success, ] |> print(n = Inf)a
# }


# as_activation <- NULL

# on_load_make_as_activation <- function() {
#   if (getRversion() < "4.2") {
#     as_activation <<- .as_activation
#   } else {
#     as_activation <<- local({
#       # make a hashtab to do reverse look ups, converting exported closures like
#       # `activation_elu` to a builtin activation name string "elu". The
#       # motivation is to avoid needlessly popping out to an R closure if we're
#       # using a bultin. We have to do this at runtime since the hastab
#       # needs the closure object address.
#       delayedAssign("h", local({
#         nms <- grep("^activation_", getNamespaceExports("keras3"), value = TRUE)
#         h <- utils::hashtab("address", length(nms))
#         ns <- asNamespace("keras3")
#         for (name in nms)
#           utils::sethash(h, getExportedValue(ns, name),
#                          substr(name, 12L, 999L))
#         h
#       }))
#
#       function(x) utils::gethash(h, x) %||% .as_activation(x)
#     })
#   }
# }
#
# .as_activation <- function(x) {
#   if (is.null(x) || is_py_object(x))
#     return(x)
#
#   name <- attr(x, "py_function_name", TRUE)
#   if (is_string(name) && identical(x, get0(
#     paste0("activation_", name),
#     envir = environment(sys.function()),
#     inherits = FALSE
#   )))
#     # it's a builtin; the name string will be resolved upstream via
#     # keras.activations.get(name)
#     return(name)
#
#   if (is.function(x))
#     return(as_py_function(x, default_name = "custom_activation"))
#   x
# }
#



# ---- capture_args ----
# capture_args_v1 <-
function(cl, modifiers = NULL, ignore = NULL,
         envir = parent.frame(), fn = sys.function(-1)) {

  ## bug: match.call() resolves incorrectly if dots are from not the default sys.parent()
  ## e.g, this fails if dots originate from the callers caller:
  #    cl <- eval(quote(match.call()), parent.frame())
  ## workaround: caller must call match.call() from the correct frame

  ## note: capture_args_v1() must always be called at the top level of the intended function body.
  ## sys.function(-1) resolves to the incorrect function if the  capture_args()
  ## call is itself a promise in another call. E.g.,:
  ##  do.call(foo, capture_args_v1(match.call())) fails because fn resolves to do.call()

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


#' @importFrom rlang list2
capture_args <- function(modifiers = NULL, ignore = NULL, force = NULL,
                         enforce_all_dots_named = TRUE) {
  call <- sys.call(-1L)
  envir <- parent.frame(1L)
  fn <- sys.function(-1L)
  # if("capture_args" %in% all.names(call, unique = TRUE))
  #   stop("incorrect usage of capture_args(), must be evaluated as ",
  #        "a standard expression, not as not a promise (i.e., not as part ",
  #         "of a call of another function")

  # match.call() automatically omits missing() args in the returned call. These
  # user calls all standardize to the same thing:
  # - layer_dense(, 10)
  # - layer_dense(object = , 10)
  # - layer_dense(object = , 10, )
  # - layer_dense(, 10, )
  # all standardize to:
  # - layer_dense(units = 10)
  call <- match.call(fn, call, expand.dots = TRUE, envir = parent.frame(2))

  # message("call: ", deparse1(call))

  fn_arg_nms <- names(formals(fn))
  known_args <- intersect(names(call), fn_arg_nms)
  if (length(ignore) && !is.character(ignore)) {
    # e.g., ignore = c("object", \(nms) startsWith(nms, "."))
    ignore <- as.character(unlist(lapply(ignore, function(ig) {
      if (is.character(ig)) return(ig)
      stopifnot(is.function(ig))
      ig <- ig(known_args) # ignore fn can return either lgl or int for [
      if (!is.character(ig))
        ig <- known_args[ig]
      ig
    }), use.names = FALSE))
  }
  known_args <- setdiff(known_args, ignore)
  known_args <- union(known_args, force)
  names(known_args) <- known_args

  if ("..." %in% fn_arg_nms && !"..." %in% ignore) {
    if (enforce_all_dots_named)
      assert_all_dots_named(envir, call)
    # match.call already drops missing args that match to known args, but it
    # doesn't protect from missing args that matched into ...
    # use list2() to allow dropping a trailing missing arg in ... also
    dots <- quote(...)
    list_sym <- quote(list2)
  } else {
    dots <- NULL
    list_sym <- quote(list)
  }

  # this might reorder args by assuming ... are last, but it doesn't matter
  # since everything is supplied as a keyword arg to the Python side anyway
  call <- as.call(c(list_sym, lapply(known_args, as.symbol), dots))
  args <- eval(call, envir)

  # filter out ignore again, in case any were in ...
  # we could probably enhance the `call` constructed above to use, e.g.,
  # ..1, ..2, ..4, to skip ignores, and avoid forcing them.
  if (length(ignores_in_dots <- intersect(names(call), ignore)))
    args[ignores_in_dots] <- NULL

  # apply modifier functions. e.g., as_nullable_integer()
  if (length(names_to_modify <-
             intersect(names(args), names(modifiers))))
    args[names_to_modify] <-
    map2(modifiers[names_to_modify], args[names_to_modify],
         function(modifier, arg) modifier(arg))

  args
}


capture_args3 <-
  function(modifiers = NULL, ignore = NULL) {
    # currently unused
    # like capture_args(), but will also unpack `!!!args`
    # e.g.,
    # constraints <- list(kernel_constraint = constraint_unitnorm(),
    #                     bias_constraint = constraint_unitnorm())
    # layer_dense(units = 2, !!!constraints)
    cl0 <- cl <- sys.call(-1L)
    envir <- parent.frame(2L)
    fn <- sys.function(-1L)

    # first defuse rlang !!! and := in calls
    cl[[1L]] <- rlang::quos
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
    args <- rlang::eval_tidy(cl, env = envir)

    # apply modifier functions. e.g., as_nullable_integer
    nms_to_modify <- intersect(names(args), names(modifiers))
    for (name in nms_to_modify)
      # list() so if modifier returns NULL, don't remove the arg
      args[name] <- list(modifiers[[name]](args[[name]]))

    args
  }


modify_intersection <- function(x, modifiers) {
  for (name in intersect(names(x), names(modifiers))) {
    x[[name]] <- modifiers[[name]](x[[name]])
  }
  x
}


assert_all_dots_named <- function(envir = parent.frame(), cl) {

  x <- evalq(eval(substitute(alist(...))), envir)
  if (!length(x)) return()

  # ignore trailing missing arg
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

















# ---- py helpers ----

py_is <- function(x, y) {
  is_py_object(x) &&
  is_py_object(y) &&
  identical(py_id(x), py_id(y))
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











# ---- unused / dead ----


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


# internal `[` method that ensures functions in this namespace use one-based
# indexing in case user has a global option set for zero-based indexing.

if (FALSE) {
  # roxygen2 now wants this exported.
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
}



standard_layer_arg_modifiers <- list(
  input_shape = normalize_shape,
  batch_input_shape = normalize_shape,
  batch_size = as_nullable_integer,
  seed = as_nullable_integer
)


if (getRversion() < "4.0")
  activeBindingFunction <- function(nm, env) {
    as.list.environment(env, all.names = TRUE)[[nm]]
  }


# don't dispatch to as.list(), just wrap in list()
as_list <- function(x) if (is.null(x) || is.list(x)) x else list(x)
