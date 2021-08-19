#' @importFrom reticulate r_to_py import_builtins py_eval py_dict py_call
#' @export
r_to_py.R6ClassGenerator <- function(x, convert = FALSE) {

  if(!is.null(x$private_fields) || !is.null(x$private_methods))
    stop("Python classes do not support private attributes")

  inherit <- resolve_py_type_inherits(x$get_inherit(), convert)

  env <- new.env(parent = x$parent_env) # mask that will contain `__class__`

  # R6 by default includes this in public methods list, not applicable here.
  methods <- x$public_methods
  methods$clone <- NULL

  methods <- as_py_methods(methods, env, convert)
  active <- as_py_methods(x$active, env, convert)

  # having convert=FALSE here means py callables are not wrapped in R functions
  # https://github.com/rstudio/reticulate/issues/1024
  builtins <- import_builtins(convert)

  py_property <- builtins$property
  active <- lapply(active, function(fn) py_call(py_property, fn, fn))

  namespace <- c(x$public_fields, methods, active)

  # we need a __module__ because python-keras introspects to see if a layer is
  # subclassed by consulting layer.__module__
  # (not sure why builtins.issubclass() doesn't work over there)
  if(!"__module__" %in% names(namespace))
    namespace$`__module__` <- "<R6type>"
    # sprintf("<R6type.%s.%s>", format(x$parent_env), x$classname)


  new_exec_body <- py_eval("lambda ns_entries: (lambda ns: ns.update(ns_entries))",
                           convert=convert)
  exec_body <- py_call(new_exec_body,
                       py_dict(names(namespace), unname(namespace), convert))

  py_cls <- py_call(import("types", convert=convert)$new_class,
    name = x$classname,
    bases = inherit$bases,
    kwds = inherit$keywords,
    exec_body = exec_body
  )

  # https://github.com/rstudio/reticulate/issues/1024
  py_cls <- py_to_r(py_cls)
  assign("convert", convert, as.environment(py_cls))

  env$`__class__` <- py_cls
  env[[x$classname]] <- py_cls

  eval(quote({
    super <- base::structure(
      function(type = get("__class__"),
               object = base::get("self", parent.frame())) {
        convert <- get("convert", envir = as.environment(object))
        bt <- reticulate::import_builtins(convert)
        reticulate::py_call(bt$super, type, object)
      },
      class = "python_class_super")
  }), env)

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
    return(list(bases = tuple(), keywords = list()))

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


as_py_methods <- function(x, env, convert) {
  out <- list()

  if ("initialize" %in% names(x) && "__init__" %in% names(x))
    stop("You should not specify both `__init__` and `initialize` methods.")

  if ("finalize" %in% names(x) && "__del__" %in% names(x))
    stop("You should not specify both `__del__` and `finalize` methods.")

  for (name in names(x)) {
    fn <- x[[name]]
    name <- switch(name,
                   initialize = "__init__",
                   finalize = "__del__",
                   name)
    out[[name]]  <- as_py_method(fn, name, env, convert)
  }
  out
}

#' @importFrom reticulate py_func py_clear_last_error
as_py_method <- function(fn, name, env, convert) {

    # if user did conversion, they're responsible for ensuring it is right.
    if (inherits(fn, "python.builtin.object")) {
      #assign("convert", convert, as.environment(fn))
      return(fn)
    }

    if (!is.function(fn))
      stop("Cannot coerce non-function to a python class method")

    environment(fn) <- env

    if (!identical(formals(fn)[1], alist(self =)))
      formals(fn) <- c(alist(self =), formals(fn))

    doc <- NULL
    if (body(fn)[[1]] == quote(`{`) &&
        typeof(body(fn)[[2]]) == "character") {
      doc <- glue::trim(body(fn)[[2]])
      body(fn)[[2]] <- NULL
    }

    # __init__ must return NULL
    if (name == "__init__")
      body(fn)[[length(body(fn)) + 1L]] <- quote(invisible(NULL))

    # python tensorflow does quite a bit of introspection on user-supplied
    # functions e.g., as part of determining which of the optional arguments
    # should be passed to layer.call(,training=,mask=). Here, we try to make
    # user supplied R function present to python tensorflow introspection
    # tools as faithfully as possible, but with a silent fallback.
    #
    # TODO: reticulate::py_func() pollutes __main__ with 'wrap_fn', doesn't
    # call py_clear_last_error(), doesn't assign __name__, doesn't accept `convert`

    # Can't use py_func here because it doesn't accept a `convert` argument

    py_sig <- tryCatch(r_formals_to_py__signature__(fn),
                       error = function(e) NULL)

    attr(fn, "py_function_name") <- name

    # https://github.com/rstudio/reticulate/issues/1024
    fn <- py_to_r(r_to_py(fn, convert))
    assign("convert", convert, as.environment(fn))

    if(!is.null(py_sig))
      fn$`__signature__` <- py_sig

    if(!is.null(doc))
      fn$`__doc__` <- doc

    fn
}

r_formals_to_py__signature__ <- function(fn) {
  inspect <- import("inspect", convert = FALSE)
  py_repr <- import_builtins(FALSE)$repr
  params <- py_eval("[]", convert = FALSE)
  Param <- inspect$Parameter

  frmls <- formals(fn)
  kind <- Param$POSITIONAL_OR_KEYWORD
  for (nm in names(frmls)) {
    if(nm == "...") {
      params$extend(list(
        Param("_R_dots_positional_args", Param$VAR_POSITIONAL),
        Param("_R_dots_keyword_args", Param$VAR_KEYWORD)
      ))
      kind <- Param$KEYWORD_ONLY
      next
    }

    if(identical(frmls[[nm]], quote(expr=))) {
      params$append(
        inspect$Parameter(nm, kind)
      )
      next
    }

    default <- r_to_py(eval(frmls[[nm]], environment(fn)))
    params$append(
      inspect$Parameter(nm, kind, default=default)
    )
  }
  inspect$Signature(params)
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


#' Make a python class constructor
#'
#' @param spec a bare symbol `MyClassName`, or a call `MyClassName(SuperClass)`
#' @param body an expression that can be evaluated to construct the class
#'   methods.
#'
#' @return The python class constructor, invisibly. Note, the same constructor is
#'   also assigned in the parent frame.
#' @export
#' @aliases py_class
#'
#' @examples
#' \dontrun{
#' MyClass %py_class% {
#'   initialize <- function(x) {
#'     print("Hi from MyClass$initialize()!")
#'     self$x <- x
#'   }
#'   my_method <- function() {
#'     self$x
#'   }
#' }
#'
#' my_class_instance <- MyClass(42)
#' my_class_instance$my_method()
#'
#' MyClass2(MyClass) %py_class% {
#'   "This will be a __doc__ string for MyClass2"
#'
#'   initialize <- function(...) {
#'     "This will be the __doc__ string for the MyClass2.__init__() method"
#'     print("Hi from MyClass2$initialize()!")
#'     super$initialize(...)
#'   }
#' }
#'
#' my_class_instance2 <- MyClass2(42)
#' my_class_instance2$my_method()
#'
#' reticulate::py_help(MyClass2) # see the __doc__ strings and more!
#' }
`%py_class%` <- function(spec, body) {
  spec <- substitute(spec)
  body <- substitute(body)
  parent_env <- parent.frame()

  inherit <- NULL
  convert <- TRUE

  if (is.call(spec)) {
    classname <- as.character(spec[[1L]])

    # `convert` keyword argument is intercepted here, all other keyword args are
    # passed on to __builtin__.type() (e.g, metaclass=)
    if(!is.null(spec$convert)) {
      convert <- eval(spec$convert, parent_env)
      spec$convert <- NULL
    }

    if(length(spec) <= 2) {
      spec <- spec[[length(spec)]]
    } else {
      spec[[1]] <- quote(base::list)
    }

    inherit <- spec # R6Class wants an expression for this

  } else {
    stopifnot(is.symbol(spec))
    classname <- as.character(spec)
  }

  env <- new.env(parent = parent_env)
  eval(body, env)
  if (!"__doc__" %in% names(env) &&
      body[[1]] == quote(`{`) &&
      typeof(body[[2]]) == "character")
    env$`__doc__` <- glue::trim(body[[2]])

  public <- active <- list()
  for(nm in names(env)) {
    if(bindingIsActive(nm, env)) {
      # requires R >= 4.0
      active[[nm]] <- activeBindingFunction(nm, env)
    } else
      public[[nm]] <- env[[nm]]
  }

  # R6Class() calls substitute() on inherit;
  r_cls <- eval(as.call(list(
    quote(R6::R6Class),
    classname = classname,
    public = public,
    active = active,
    inherit = inherit,
    cloneable = FALSE,
    parent_env = parent_env
  )))

  # TODO: think about best practice for how to delay creating this
  # if defined in a package.
  py_cls <- r_to_py.R6ClassGenerator(r_cls, convert)
  assign(classname, py_cls, envir = parent_env)
  invisible(py_cls)
}

if (getRversion() < "4.0")
  activeBindingFunction <- function(nm, env) {
    as.list.environment(env, all.names = TRUE)[[nm]]
  }
