

# TODO: use this in register_keras_serializable()?
generate_module_name <- function(env) {
  while((name <- environmentName(env)) == "")
    env <- parent.env(env)
  if(isNamespace(env))
    name <- paste0("namespace:", name)
  else if (name == "R_GlobalEnv")
    name <- "globalenv"
  sprintf("<r-%s>", name)
}

new_py_class <- function(classname,
                         members = list(),
                         inherit = NULL,
                         parent_env = parent.frame(),
                         inherit_expr,
                         convert = TRUE) {
  if (!missing(inherit_expr))
    inherit <- eval(inherit_expr, parent_env)
  new_py_type(
    classname,
    members = members,
    inherit = inherit,
    parent_env = parent_env
  )
}


new_wrapped_py_class <-
function(classname,
         members = list(),
         inherit = NULL,
         parent_env = parent.frame(),
         private = list(),
         modifiers = quote(expr =),
         default_formals = \(...) {})
{
  # force all new_py_type() args
  classname; members; inherit; parent_env; private;

  delayedAssign(classname,
    new_py_type(
      classname = classname,
      members = members,
      inherit = resolve_py_obj(inherit, env = parent_env),
      parent_env = parent_env,
      private = private
    )
  )
  delayedAssign("__class__", get(classname))

  if (is_keras_loaded()) {
    # force promise, get actual frmls
    frmls <- formals(`__class__`)
  } else {
    # try to infer frmls
    frmls <- formals(members$`__init__ ` %||%
                     members$initialize %||%
                     default_formals)
  }
  frmls$self <- NULL

  bdy <- bquote({
    args <- capture_args(.(modifiers), enforce_all_dots_named = FALSE)
    do.call(.(as.name(classname)), args)
  })
  rm(modifiers, default_formals) # free memory

  as.function.default(c(frmls, bdy))
}

new_py_type <-
function(classname,
         members = list(),
         inherit = NULL,
         parent_env = parent.frame(),
         private = list())
{

  if (is.language(inherit))
    inherit <- eval(inherit, parent_env)

  convert <- TRUE
  inherit <- resolve_py_type_inherits(inherit, convert)
  mask_env <- new.env(parent = parent_env)
  # common-mask-env: `super`, `__class__`, classname

  members <- normalize_py_type_members(members, mask_env, convert, classname)

  # we need a __module__ because python-keras introspects to see if a layer is
  # subclassed by consulting layer.__module__
  # (not sure why builtins.issubclass() doesn't work over there)
  # `__module__` is used to construct the S3 class() of py_class instances,
  # it needs to be stable (e.g, can't use format(x$parent_env))
  if (!"__module__" %in% names(members))
    members$`__module__` <- generate_module_name(parent_env)

  exec_body <- py_eval(
    "lambda ns_entries: (lambda ns: ns.update(ns_entries))")(members)

  py_class <- import("types")$new_class(
    name = classname,
    bases = inherit$bases,
    kwds = inherit$keywords,
    exec_body = exec_body
  )

  mask_env$`__class__` <- py_class
  mask_env[[classname]] <- py_class
  if (!is.null(private)) {
    attr(mask_env, "get_private") <-
      new_get_private(private, shared_mask_env = mask_env)
  }

  eval(envir = mask_env, quote({
    super <- function(
      type = `__class__`,
      object_or_type = base::get("self", envir = base::parent.frame()))
      {
        convert <- base::get("convert", envir = base::as.environment(object_or_type))
        py_builtins <- reticulate::import_builtins(convert)
        reticulate::py_call(py_builtins$super, type, object_or_type)
      }
    class(super) <- "python_builtin_super_getter"
  }))


  py_class
}

# S3 methods for nice access from class methods like
# - super$initialize()
# - super()$initialize()
# - super(Classname, self)$initialize()
#' @export
`$.python_builtin_super_getter` <- function(x, name) {
  super <- do.call(x, list(), envir = parent.frame()) # call super()
  name <- switch(name, initialize = "__init__", finalize = "__del__", name)
  out <- py_get_attr(super, name)
  convert <- get0("convert", as.environment(out), inherits = FALSE,
                  ifnotfound = TRUE)
  if (convert) py_to_r(out) else out
}

#' @export
`[[.python_builtin_super_getter` <- `$.python_builtin_super_getter`

# @importFrom utils .DollarNames
# @export
# .DollarNames.python_builtin_super_getter <- function(x, pattern) {
# ## commended out because the python.builtin.super object doesn't
# ## have populated attributes itself, only a dynamic `__getattr__` method
# ## that resolves dynamically.
#   for(envir in rev(sys.frames()))
#     if(identical(parent.env(envir), environment(x)))
#       break
# ## not the user frame. So we can't (reliably) resolve `self` by looking at the stack.
#   super <- do.call(x, list(), envir = envir) # call super()
#
#   # dispatches to reticulate:::.DollarNames.python.builtin.object()
#   names <- .DollarNames(super, pattern)
#   types <- attr(names, "types", TRUE) %||% integer()
#
#   py_attr_names <- py_list_attributes(super)
#   if("__init__" %in% py_attr_names) {
#     append(names) <- "initialize"
#     append(types) <- 6L
#   }
#   if("__del__" %in% py_attr_names) {
#     append(names) <- "finalize"
#     append(types) <- 6L
#   }
#
#   idx <- grepl(pattern, names)
#   names <- names[idx]
#   types <- types[idx]
#
#   if (length(names) > 0) {
#     # set types
#     oidx <- order(names)
#     names <- names[oidx]
#     attr(names, "types") <- types[oidx]
#     attr(names, "helpHandler") <- "reticulate:::help_handler"
#   }
#
#   names
# }


#' @importFrom reticulate r_to_py import_builtins py_eval py_dict py_call
#' @export
r_to_py.R6ClassGenerator <- function(x, convert = TRUE) {
  new_py_type(
    classname = x$classname,
    inherit = x$get_inherit(),
    members = c(x$public_fields,
                x$public_methods,
                lapply(x$active, active_property)),
    private = c(x$private_fields,
                x$private_methods),
    parent_env = x$parent_env
  )
}


normalize_py_type_members <- function(members, env, convert, classname) {

  if (all(c("initialize", "__init__") %in% names(members)))
    stop("You should not specify both `__init__` and `initialize` methods.")

  if (all(c("finalize", "__del__") %in% names(members)))
    stop("You should not specify both `__del__` and `finalize` methods.")

  names(members) <- names(members) |>
    replace_val("initialize", "__init__") |>
    replace_val("finalize", "__del__")

  members <- imap(members, function(x, name) {
    if (!is.function(x))
      return(x)
    as_py_method(x, name, env, convert,
                 label = sprintf("%s$%s", classname, name))
  })

  members
}



#' @importFrom reticulate py_get_item py_del_item import
new_get_private <- function(members, shared_mask_env) {
  force(members); force(shared_mask_env)

  # python should never see privates.
  # also, avoid invoking __hash__ on the py obj, which
  # might error or return non-unique values.
  delayedAssign("class_privates", fastmap::fastmap())

  new_instance_private <- function(self) {
    private <- new.env(parent = emptyenv())
    class_privates$set(py_id(self), private)

    import("weakref")$finalize(
      self, del_instance_private, self)

    instance_mask_env <- new.env(parent = shared_mask_env)
    # TODO: is this `self` assignment a circular reference that prevents the
    # object from being collected? should it be a weakref?
    # add tests to make sure that the object is collected when it should be.
    instance_mask_env$self <- self
    instance_mask_env$private <- private
    members <- lapply(members, function(member) {
      if (is.function(member) && !inherits(member, "python.builtin.object"))
        environment(member) <- instance_mask_env
      member
    })
    active <- map_lgl(members, is_marked_active)
    list2env(members[!active], envir = private)
    imap(members[active], function(fn, name) {
      makeActiveBinding(name, fn, private)
    })
    private
  }

  del_instance_private <- function(self) {
    class_privates$remove(py_id(self))
  }

  function(self) {
    class_privates$get(py_id(self)) %||%
      new_instance_private(self)
  }
}


#' @importFrom reticulate tuple dict
resolve_py_type_inherits <- function(inherit, convert=FALSE) {

  # inherits can be
  # a) NULL %||% list()
  # b) a python.builtin.type or R6ClassGenerator
  # c) a list or tuple of python.builtin.types and/or R6ClassGenerators
  # d) a list, with keyword args meant to be passed to builtin.type()
  #
  # returns: list(tuple_of_'python.builtin.type's, r_named_list_of_kwds)
  # (both potentially of length 0)

  if(is.null(inherit) || identical(inherit, list()))
    return(list(bases = tuple(), keywords = dict()))

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



#' @importFrom reticulate py_func py_clear_last_error
as_py_method <- function(fn, name, env, convert, label) {

    # if user did conversion, they're responsible for ensuring it is right.
    if (inherits(fn, "python.builtin.object")) {
      #assign("convert", convert, as.environment(fn))
      return(fn)
    }

    srcref <- attr(fn, "srcref")

    if (!is.function(fn))
      stop("Cannot coerce non-function to a python class method")

    environment(fn) <- env

    decorators <- attr(fn, "py_decorators", TRUE)
    # if(is_marked_active(fn))

    if ("staticmethod" %in% decorators) {
      # do nothing
    } else if ("classmethod" %in% decorators) {
      fn <- fn |> ensure_first_arg_is(cls = )
    } else {
      # standard pathway, ensure the method receives 'self' as first arg
      fn <- fn |> ensure_first_arg_is(self = )
    }

    doc <- NULL
    if (is.call(body(fn)) &&
        body(fn)[[1]] == quote(`{`) &&
        length(body(fn)) > 1 &&
        typeof(body(fn)[[2]]) == "character") {
      doc <- glue::trim(body(fn)[[2]])
      body(fn)[[2]] <- NULL
    }

    # __init__ must return NULL
    if (name == "__init__") {
      body(fn) <- substitute({
        body
        invisible(NULL)
      }, list(body = body(fn)))
    }

    if (!"private" %in% names(formals(fn)) &&
        "private" %in% all.names(body(fn))) {
      body(fn) <- substitute({
        delayedAssign("private", attr(parent.env(environment()), "get_private", TRUE)(self))
        body
      }, list(body = body(fn)))
    }

    # python tensorflow does quite a bit of introspection on user-supplied
    # functions e.g., as part of determining which of the optional arguments
    # should be passed to layer.call(,training=,mask=). Here, we try to make
    # user supplied R function present to python tensorflow introspection
    # tools as faithfully as possible, but with a silent fallback.
    #
    # TODO: reticulate::py_func() pollutes __main__ with 'wrap_fn', doesn't
    # call py_clear_last_error(), doesn't assign __name__, doesn't accept `convert`

    # Can't use py_func here because it doesn't accept a `convert` argument

   # Can't use __signature__ to communicate w/ the python side anymore
   # because binding of 'self' for instance methods doesn't update __signature__,
   # resulting in errors for checks in keras_core for 'build()' method arg names.

    # attr(fn, "py_function_name") <- name
    attr(fn, "pillar") <- list(label = label) # for print method of rlang::trace_back()

    fn <- py_func2(fn, convert, name = name)
    # https://github.com/rstudio/reticulate/issues/1024
    # fn <- py_to_r(r_to_py(fn, convert))
    # assign("convert", convert, as.environment(fn))

    if(!is.null(doc))
      fn$`__doc__` <- doc

    attr(fn, "srcref") <- srcref
    # TODO, maybe also copy over "wholeSrcref". See `removeSource()` as a starting point.
    # This is used to generate clickable links in rlang traceback printouts.
    bt <- import_builtins()
    for (dec in decorators) {
      if (identical(dec, "property") && length(formals(fn)) > 1) {
        fn <- bt$property(fn, fn) # getter and setter
        next
      }
      if (is_string(dec)) {
        dec <- bt[[dec]]
      }
      fn <- dec(fn)
    }
    fn
}

#' @importFrom rlang is_string
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
        Param("_R_dots_positional_args", Param$VAR_POSITIONAL)
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
  if("..." %in% names(frmls))
    # need to make sure that `**kwarg` is last in signature,
    # in case there are args after R `...`, we need to reorder
    # so the py sig looks like `(foo, *args, bar, **kwargs)`
    params$extend(list(
      Param("_R_dots_keyword_args", Param$VAR_KEYWORD)
    ))

  inspect$Signature(params)
}


py_func2 <- function(fn, convert, name = deparse(substitute(fn))) {
  # TODO: wrap this all in a tryCatch() that gives a nice error message
  # about unsupported signatures
  sig <- r_formals_to_py__signature__(fn) |> py_to_r()
  inspect <- import("inspect")
  pass_sig <- iterate(sig$parameters$values(), function(p) {
    if(p$kind == inspect$Parameter$POSITIONAL_ONLY)
      p$name
    else if (p$kind == inspect$Parameter$VAR_POSITIONAL)
     paste0("*", p$name)
    else if (p$kind == inspect$Parameter$VAR_KEYWORD)
     paste0("**", p$name)
    else
     paste0(p$name, "=", p$name)
  })
  pass_sig <- paste0(pass_sig, collapse = ", ")
  code <- glue::glue(r"---(
def wrap_fn(_fn):
  def {name}{py_str(sig)}:
    return _fn({pass_sig})
  return {name}
  )---")
  util <- reticulate::py_run_string(code, local = TRUE, convert = convert)
  util$wrap_fn(fn)
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


# @seealso <https://tensorflow.rstudio.com/articles/new-guides/python_subclasses.html>


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
#'
#' # In addition to `self`, there is also `private` available.
#' # This is an R environment unique to each class instance, where you can
#' # store objects that you don't want converted to Python, but still want
#' # available from methods. You can also assign methods to private, and
#' # `self` and `private` will be available in private methods.
#'
#' MyClass %py_class% {
#'
#'   initialize <- function(x) {
#'     print("Hi from MyClass$initialize()!")
#'     private$y <- paste("A Private field:", x)
#'   }
#'
#'   get_private_field <- function() {
#'     private$y
#'   }
#'
#'   private$a_private_method <- function() {
#'     cat("a_private_method() was called.\n")
#'     cat("private$y is ", sQuote(private$y), "\n")
#'   }
#'
#'   call_private_method <- function()
#'     private$a_private_method()
#'
#'   # equivalent of @property decorator in python
#'   an_active_property %<-active% function(x = NULL) {
#'     if(!is.null(x)) {
#'       cat("`an_active_property` was assigned", x, "\n")
#'       return(x)
#'     } else {
#'       cat("`an_active_property` was accessed\n")
#'       return(42)
#'     }
#'   }
#' }
#'
#' inst1 <- MyClass(1)
#' inst2 <- MyClass(2)
#' inst1$get_private_field()
#' inst2$get_private_field()
#' inst1$call_private_method()
#' inst2$call_private_method()
#' inst1$an_active_property
#' inst1$an_active_property <- 11
#' }
`%py_class%` <- function(spec, body) {
  spec <- substitute(spec)
  body <- substitute(body)
  parent_env <- parent.frame()

  inherit <- NULL
  convert <- TRUE
  delay_load <- !identical(topenv(parent_env), globalenv()) # likely in a package

  if (is.call(spec)) {
    classname <- as.character(spec[[1L]])

    # `convert` keyword argument is intercepted here
    if(!is.null(spec$convert)) {
      convert <- eval(spec$convert, parent_env)
      spec$convert <- NULL
    }

    # `delay_load` keyword argument is intercepted here
    if(!is.null(spec$delay_load)) {
      delay_load <- eval(spec$delay_load, parent_env)
      spec$delay_load <- NULL
    }

    # all other keyword args are passed on to __builtin__.type() (e.g, metaclass=)
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
  env$private <- new.env(parent = emptyenv())

  eval(body, env)

  if (!"__doc__" %in% names(env) &&
      body[[1]] == quote(`{`) &&
      typeof(body[[2]]) == "character")
    env$`__doc__` <- glue::trim(body[[2]])

  private <- as.list.environment(env$private, all.names = TRUE)
  rm(list = "private", envir = env)

  public <- active <- list()
  for (nm in names(env)) {
    if (bindingIsActive(nm, env))
      active[[nm]] <- activeBindingFunction(nm, env)
    else if (is_marked_active(env[[nm]]))
      active[[nm]] <- env[[nm]]
    else
      public[[nm]] <- env[[nm]]
  }

  # TODO: re-enable delayed pyclasses.
  # if (delay_load)
  #   py_class <- delayed_r_to_py_R6ClassGenerator(r6_class, convert)
  # else
  #   py_class <- r_to_py.R6ClassGenerator(r6_class, convert)

  inherit <- eval(inherit, parent_env)
  active <- lapply(active, active_property)

  py_class <-  new_py_type(
    classname = classname,
    inherit = inherit,
    members = c(public, active),
    private = private,
    parent_env = parent_env
  )

  # attr(py_class, "r6_class") <- r6_class

  assign(classname, py_class, envir = parent_env)
  invisible(py_class)
}


#' @importFrom reticulate py_call py_to_r
py_callable_as_function2 <- function(callable, convert) {
  force(callable)
  force(convert)

  function(...) {
    result <- py_call(callable, ...)

    if (convert)
      result <- py_to_r(result)

    if (is.null(result))
      invisible(result)
    else
      result
  }
}


delayed_r_to_py_R6ClassGenerator <- function(r6_class, convert) {
  force(r6_class)
  force(convert)

  py_object <- new.env(parent = emptyenv())
  py_object$delayed <- TRUE
  attr(py_object, "class") <- c("py_R6ClassGenerator",
                                "python.builtin.type",
                                "python.builtin.object")
  attr(py_object, "r6_class") <- r6_class
  py_object_real <- NULL
  # keep a reference alive here, since this object
  # has the C finalizer registered
  force_py_object <- function(nm) {
    if (exists("delayed", envir = py_object, inherits = FALSE)) {
      py_object_real <<-
        attr(r_to_py.R6ClassGenerator(r6_class, convert), "py_object")
      list2env(as.list.environment(py_object_real, all.names = TRUE),
               py_object)
      rm(list = "delayed", envir = py_object)
    }

    if(missing(nm))
      py_object
    else
      get(nm, envir = py_object)
  }

  delayedAssign("pyobj", force_py_object("pyobj"), assign.env = py_object)
  delayedAssign("convert", force_py_object("convert"), assign.env = py_object)

  fn <- py_callable_as_function2(NULL, convert)
  attributes(fn) <- attributes(py_object)
  attr(fn, "py_object") <- py_object

  delayedAssign("callable", force_py_object(), assign.env = environment(fn))

  fn
}

# @export
# print.py_R6ClassGenerator <-
function(x, ...) {
  r6_class <- attr(x, "r6_class")
  if (isTRUE(get0("delayed", attr(x, "py_object"))))
    cat(sprintf("<R6type.%s> (delayed)\n", r6_class$classname))
  else
    NextMethod()

  print(r6_class)
}

# @export
# `$.py_R6ClassGenerator` <-
function(x, name) {
  if (identical(name, "new"))
    return(x)
  NextMethod()
}

# @exportS3Method pillar::type_sum
# @rawNamespace S3method(pillar::type_sum,py_R6ClassGenerator)
# type_sum.py_R6ClassGenerator <-
function(x) {
  cl <- class(x)[[1L]]
  if(startsWith(cl, "R6type."))
    cl <- substr(cl, 8L, 2147483647L)
  cl
}


#' Make an Active Binding
#'
#' @param sym symbol to bind
#' @param value A function to call when the value of `sym` is accessed.
#'
#' @return `value`, invisibly
#' @export
#'
#' @details Active bindings defined in a [`%py_class%`] are converted to
#'   `@property` decorated methods.
#'
#' @seealso [`makeActiveBinding()`]
#'
#' @examples
#' set.seed(1234)
#' x %<-active% function(value) {
#'   message("Evaluating function of active binding")
#'   if(missing(value))
#'     runif(1)
#'   else
#'    message("Received: ", value)
#' }
#' x
#' x
#' x <- "foo"
#' x <- "foo"
#' x
#' rm(x) # cleanup
`%<-active%` <- function(sym, value) {
  makeActiveBinding(substitute(sym), value, parent.frame())
  invisible(value)
}



maybe_delayed_r_to_py_R6ClassGenerator <-
  function(x, convert = FALSE,
           parent_env = parent.frame()) {
    if (identical(topenv(parent_env), globalenv()))
      # not in a package
      r_to_py.R6ClassGenerator(x, convert)
    else
      delayed_r_to_py_R6ClassGenerator(x, convert)
  }

ensure_first_arg_is <- function(fn, ...) {
  frmls <- formals(fn)
  arg <- eval(substitute(alist(...)))
  if (!identical(frmls[1], arg))
    formals(fn) <- c(arg, frmls)
  fn
}



#' Create an active property class method
#'
#' @param fn An R function
#'
#' @description
#'
#' # Example
#' ```r
#' layer_foo <- Model("Foo", ...,
#'   metrics = active_property(function() {
#'     list(self$d_loss_metric,
#'          self$g_loss_metric)
#'   }))
#' ```
#'
#' @export
active_property <- function(fn) {
  if(!is.function(fn))
    stop("Only functions can be active properties")
  append1(attr(fn, "py_decorators")) <- "property"
  fn
}

decorate_method <- function(fn, decorator) {
  append1(attr(fn, "py_decorators")) <- decorator
  fn
}

drop_null_defaults <- function(args, fn = sys.function(-1L)) {
  null_default_args <- names(which(vapply(formals(fn), is.null, TRUE)))
  drop_nulls(args, null_default_args)
}

is_marked_active <- function(x) {
  for (dec in attr(x, "py_decorators", TRUE))
    if (identical(dec, "property"))
      return (TRUE)
  FALSE
}
