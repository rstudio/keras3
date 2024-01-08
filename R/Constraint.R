#' Define a custom Constraint class.
#'
#' @description
#' Base class for weight constraints.
#'
#' A `Constraint()` instance works like a stateless function.
#' Users who subclass the `Constraint` class should override
#' the `call()` method, which takes a single
#' weight parameter and return a projected version of that parameter
#' (e.g. normalized or clipped). Constraints can be used with various Keras
#' layers via the `kernel_constraint` or `bias_constraint` arguments.
#'
#' Here's a simple example of a non-negative weight constraint:
#' ```{r}
#' constraint_nonnegative <- Constraint("NonNegative",
#'   call = function(w) {
#'     w * op_cast(w >= 0, dtype = w$dtype)
#'   }
#' )
#' weight <- op_convert_to_tensor(c(-1, 1))
#' constraint_nonnegative()(weight)
#' ```
#'
#' Usage in a layer:
#' ```{r, output = FALSE}
#' layer_dense(units = 4, kernel_constraint = constraint_nonnegative())
#' ```
#'
#' @param
#' call
#' ```r
#' \(w)
#' ```
#' Applies the constraint to the input weight variable.
#'
#' By default, the inputs weight variable is not modified.
#' Users should override this method to implement their own projection
#' function.
#'
#' Args:
#' * `w`: Input weight variable.
#'
#' Returns:
#' Projected variable (by default, returns unmodified inputs).
#'
#' @param
#' get_config
#' ```r
#' \()
#' ```
#' Function that returns a named list of the object config.
#'
#' A constraint config is a named list (JSON-serializable) that can
#' be used to reinstantiate the same object
#' (via `do.call(<constraint_class>, <config>)`).
#'
#' @tether keras.constraints.Constraint
#' @export
#' @family constraints
Constraint <- function(classname, call = NULL, get_config = NULL,
                       ...,
                       public = list(),
                       private = list(),
                       inherit = NULL,
                       parent_env = parent.frame()) {

  members <- Reduce(function(x, y) modifyList(x, y, keep.null = TRUE),
                    list(drop_nulls(named_list(call, get_config)),
                         list2(...),
                         public))

  if(!"__call__" %in% names(members) &&
     "call" %in% names(members))
    members <- rename(members, "__call__" = "call")

  members <- modify_intersection(members, list(
    from_config = function(x) decorate_method(x, "classmethod")
  ))

  inherit <- substitute(inherit) %||%
    quote(keras3:::keras$constraints$Constraint)

  new_wrapped_py_class(
    classname = classname,
    members = members,
    inherit = inherit,
    parent_env = parent_env,
    private = private
  )
}

