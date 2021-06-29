#' @importFrom reticulate r_to_py import_builtins py_eval py_dict
#' @export
r_to_py.R6ClassGenerator <- function(x, convert = FALSE) {

  if(!is.null(x$private_fields) || !is.null(x$private_methods))
    stop("Python classes do not support private attributes")

  inherit <- resolve_py_type_inherits(x$get_inherit(), convert)

  env <- new.env(parent = x$parent_env)
  methods <- as_py_methods(x$public_methods, env)
  active <- as_py_methods(x$active, env)

  # having convert=FALSE here means py callables are not wrapped in R functions
  # so build everything with convert=TRUE, then maybe fixup `convert` in the
  # final return object
  builtins <- import_builtins()

  py_property <- builtins$property
  active <- lapply(active, function(fn) py_property(fn, fn))

  namespace <- c(x$public_fields, methods, active)

  # we need a __module__ because python-keras introspects to see if a layer is
  # subclassed by consulting layer.__module__
  # (not sure why builtins.issubclass() doesn't work over there)
  if(!"__module__" %in% names(namespace))
    namespace$`__module__` <-  paste0("<R6type>", x$classname, sep=".")
    # sprintf("<R6type.%s.%s>", format(x$parent_env), x$classname)

  exec_body <- py_eval("lambda ns_entries: (lambda ns: ns.update(ns_entries))", convert=convert)(
    py_dict(names(namespace), unname(namespace), convert = convert))

  py_cls <- import("types")$new_class(
    name = x$classname,
    bases = inherit$bases,
    kwds = inherit$keywords,
    exec_body = exec_body
  )

  assign("convert", convert, as.environment(py_cls))

  env$`__class__` <- py_cls
  env[[x$classname]] <- py_cls

  evalq({
    super <- base::structure(function(type = `__class__`,
                                      object = base::get("self", parent.frame()))
      reticulate::import_builtins()$super(type, object),
      class = "python_class_super")
  }, env)

  py_cls
}



resolve_py_type_inherits <- function(inherit, convert=FALSE) {

  # inherits can be
  # a) NULL %||% list()
  # b) a python.builtin.type or R6ClassGenerator
  # c) a list or tuple of python.builtin.types and/or R6ClassGenerators
  # d) a list, with keyword args meant to be passed to builtin.type()
  #
  # returns: list(tuple_of_'python.builtin.type's, r_named_list_of_kwds)
  # (both potentially of length 0)

  if(!length(inherit))
    return(bases = tuple(), keywords = list())

  bases <-
    if (inherits(inherit, "python.builtin.tuple")) as.list(inherit)
  else if (is.list(inherit)) inherit
  else list(inherit)


  # split out keyword args (e.g., metaclass=)
  keywords <- list()
  for (nm in names(bases)) {
    if(is.na(nm) || !nzchar(nm)) next
    keywords[[nm]] <- bases[[nm]]
    bases[[nm]] <- NULL
  }
  names(bases) <- NULL

  bases <- lapply(bases, function(cls) {
    if (inherits(cls, "R6ClassGenerator"))
      return(r_to_py.R6ClassGenerator(cls, convert))

    if (!inherits(cls, "python.builtin.object"))
      tryCatch(
        cls <- r_to_py(cls),
        error = function(e)
          stop(e, "Supplied superclasses must be python objects, not: ",
               paste(class(cls), collapse = ", "))
      )

    cls
  })

  bases <- do.call(tuple, bases)

  list(bases = bases, keywords = keywords)
}


as_py_methods <- function(x, env) {
  out <- list()
  for (name in names(x)) {
    fn <- x[[name]]
    name <- switch(name,
                   initialize = "__init__",
                   finalize = "__del__",
                   name)
    out[[name]]  <- as_py_method(fn, name, env)
  }
  out
}

#' @importFrom reticulate py_func py_clear_last_error
as_py_method <- function(fn, name, env) {

    # if user did conversion, they're responsible for ensuring it is right.
    if (inherits(fn, "python.builtin.object"))
      return(fn)

    if (!is.function(fn))
      stop("Cannot coerce non-function to a python class method")

    environment(fn) <- env

    if (!identical(formals(fn)[1], alist(self =)))
      formals(fn) <- c(alist(self =), formals(fn))

    # __init__ must return NULL
    if (name == "__init__")
      body(fn)[[length(body(fn)) + 1L]] <- quote(invisible(NULL))

    fn <- tryCatch({
      # python tensorflow does quite a bit of introspection on user-supplied
      # functions e.g., as part of determining which of the optional arguments
      # should be passed to layer.call(,training=,mask=). Here, we try to make
      # user supplied R function present to python tensorflow introspection
      # tools as faithfully as possible, but with a silent fallback.
      #
      # TODO: reticulate::py_func() pollutes __main__ with 'wrap_fn', doesn't
      # call py_clear_last_error(), doesn't assign __name__
      fn <- py_func(fn)
      fn$`__name__` <- name
      fn
    },

    error = function(e) {
      # TODO: if py_func conversion fails, we can maybe try harder and attach a
      # __signature__ attribute constructed using inspect.Signature().
      py_clear_last_error()
      attr(fn, "py_function_name") <- name
      fn
    })

    fn
}



# TODO: (maybe?) factor out a py_class() function,
# funnel r_to_py.R6ClassGenerator() and %py_class%() to go through py_class()
# export py_class()
# differences from reticulate::PyClass would be:
# *) python objects (including callables) pass through unmodified
# *) all R functions are forced to share the same parent/mask
# *) R functions are maybe modified to ensure their first formal is `quote(self=)`
# *) make the converted functions present to python introspection tools better
# *) `super` can be accessed in both R6 style using `$`, and python-style as a callable
# *) `super()` can resolve `self` properly when called from a nested scope
# *) method calls respect user-supplied `convert` values for all args
#
