

# experimental proof of concept:
#  - make the wrappers mere skeletons that only pass through the call,
#  - easily decorate select supplied arguments,
#  - preserve descent auto completion in R,
#  - supporting python's **kwargs using ...

# Downsides: A little more magical than reticulate::py_function_wrapper()

# Upsides: - less massaging of arguments needed to maintain compat across
# versions. This way, keras package can be built against the latest version of
# TF, but on earlier versions, backwards compatible calls still just get passed
# through to python unmodified (tf typically does a good job of maintaining
# backwards compat in the official api).
#
# e.g, this approach would avoid needing fixes like lr -> learning_rate in
# optimizers

options("keras.debug" = TRUE)

inspect <- reticulate::import("inspect")
substitute_call <- function(x, decorate = NULL,
                            cl = sys.call(-1)) {
  cl[[1]] <- substitute(x)

  for (nm in intersect(names(decorate), names(cl))) {
    cl[[nm]] <- substitute(decorator(arg),
                           list(decorator = decorate[[nm]],
                                arg = cl[[nm]]))
  }

  cl
}

dummy_wrapper <- function(a, b, ...) {
  substitute_call(foo)
}

dummy_wrapper()
dummy_wrapper(b = 3)
dummy_wrapper(z = 3)


make_r_wrapper <-
  function(py_obj,
           decorate = NULL,
           envir = parent.frame()) {
    py_obj_expr <- substitute(py_obj)
    sig <- inspect$signature(py_obj)

    args <- pairlist()
    it <- sig$parameters$items()$`__iter__`()
    repeat {
      x <- reticulate::iter_next(it)
      if (is.null(x))
        break
      c(name, param) %<-% x

      if (param$kind == inspect$Parameter$VAR_KEYWORD)
        name <- "..."

      default <- param$default

      if (inherits(default, "python.builtin.object")) {
        if (default != inspect$Parameter$empty)
          # must be something complex that failed to convert
          warning(glue::glue(
            "Failed to convert default arg {default} for {name} in {py_obj_expr}"
          ))
        args[[name]] <- quote(expr = )
      } else
        args[[name]] <- default
    }
    body <- substitute({
      cl <- substitute_call(py_obj_expr, decorate = decorate)
      if(getOption("keras.debug")) print(cl)
      eval.parent(cl)
    })

    as.function.default(c(args, body), envir)
}

keras$optimizers %>% names()
keras$optimizers %>% names() %>% grep("^[A-Z]", ., value = T)


"Adadelta"
optomizer_adadelta <- make_r_wrapper(keras$optimizers$Adadelta)

"Adagrad"
optomizer_adagrad <- make_r_wrapper(keras$optimizers$Adagrad)

"Adam"
optomizer_adam <- make_r_wrapper(keras$optimizers$Adam)

"Adamax"
optomizer_adamax <- make_r_wrapper(keras$optimizers$Adamax)

"Ftrl"
optomizer_ftrl <- make_r_wrapper(keras$optimizers$Ftrl)

"Nadam"
optomizer_nadam <- make_r_wrapper(keras$optimizers$Nadam)

"RMSprop"
optomizer_rmsprop <- make_r_wrapper(keras$optimizers$RMSprop)

"SGD"
optomizer_sgd <- make_r_wrapper(keras$optimizers$SGD)


optomizer_adam
# function (learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
#     epsilon = 1e-07, amsgrad = FALSE, name = "Adam", ...)
# {
#     cl <- substitute_call(keras$optimizers$Adam, decorate = NULL)
#     if (getOption("keras.debug"))
#         print(cl)
#     eval.parent(cl)
# }

optomizer_adam()
# keras$optimizers$Adam()
# <tensorflow.python.keras.optimizer_v2.adam.Adam>

optomizer_adam(lr = .1)
# keras$optimizers$Adam(lr = 0.1)
# <tensorflow.python.keras.optimizer_v2.adam.Adam>
# /home/tomasz/.local/share/r-miniconda/envs/r-reticulate/lib/python3.7/site-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:375: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
#   "The `lr` argument is deprecated, use `learning_rate` instead.")

