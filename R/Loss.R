
#' Define a custom loss class
#'
#' @param call
#' ```r
#' function(y_true, y_pred)
#' ```
#' Method to be implemented by subclasses:
#' Function that contains the logic for loss calculation using
#' `y_true`, `y_pred`.
#'
#' @details
#'
#' Example subclass implementation:
#'
#' ```{r}
#' loss_custom_mse <- Loss(
#'   classname = "CustomMeanSquaredError",
#'   call = function(y_true, y_pred) {
#'     op_mean(op_square(y_pred - y_true), axis = -1)
#'   }
#' )
#'
#' # Usage in compile()
#' model <- keras_model_sequential(input_shape = 10) |> layer_dense(10)
#' model |> compile(loss = loss_custom_mse())
#'
#' # Standalone usage
#' mse <- loss_custom_mse(name = "my_custom_mse_instance")
#'
#' y_true <- op_ones(c(5, 5))
#' y_pred <- y_true - .5
#' mse(y_true, y_pred)
#' ```
#
#' # Methods defined by base `Loss` class:
#'
#' * ```r
#'   initialize(name=NULL, reduction="sum_over_batch_size", dtype=NULL)
#'   ```
#'
#' * ```r
#'   get_config()
#'   ```
#'
#' @inheritSection Layer Symbols in scope
#' @inheritParams Layer
#' @export
#' @family losses
#' @tether keras.losses.Loss
Loss <-
function(classname, call = NULL,
         ...,
         public = list(),
         private = list(),
         inherit = NULL,
         parent_env = parent.frame()) {

  members <- drop_nulls(named_list(call))
  members <- modifyList(members, list2(...), keep.null = TRUE)
  members <- modifyList(members, public, keep.null = TRUE)

  members <- rename(members, "__call__" = "call",
                    .skip_existing = TRUE)

  members <- modify_intersection(members, list(
    from_config = function(x) decorate_method(x, "classmethod")
  ))

  inherit <- substitute(inherit) %||%
    quote(base::asNamespace("keras3")$keras$losses$Loss)

  new_wrapped_py_class(
    classname = classname,
    members = members,
    inherit = inherit,
    parent_env = parent_env,
    private = private
  )
}
