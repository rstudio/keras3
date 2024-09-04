
#' Subclass the base `Loss` class
#'
#' Use this to define a custom loss class. Note, in most cases you do not need
#' to subclass `Loss` to define a custom loss: you can also pass a bare R
#' function, or a named R function defined with [`custom_metric()`], as a loss
#' function to `compile()`.
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
#' y_true <- op_arange(20) |> op_reshape(c(4, 5))
#' y_pred <- op_arange(20) |> op_reshape(c(4, 5)) * 2
#' (loss <- mse(y_true, y_pred))
#'
#' loss2 <- (y_pred - y_true)^2 |>
#'   op_mean(axis = -1) |>
#'   op_mean()
#'
#' stopifnot(all.equal(as.array(loss), as.array(loss2)))
#'
#' sample_weight <-array(c(.25, .25, 1, 1))
#' (weighted_loss <- mse(y_true, y_pred, sample_weight = sample_weight))
#'
#' weighted_loss2 <- (y_true - y_pred)^2 |>
#'   op_mean(axis = -1) |>
#'   op_multiply(sample_weight) |>
#'   op_mean()
#'
#' stopifnot(all.equal(as.array(weighted_loss),
#'                     as.array(weighted_loss2)))
#' ```
#
#' # Methods defined by base `Loss` class:
#'
#' * ```r
#'   initialize(name=NULL, reduction="sum_over_batch_size", dtype=NULL)
#'   ```
#'   Args:
#'   * `name`: Optional name for the loss instance.
#'   * `reduction`: Type of reduction to apply to the loss. In almost all cases
#'       this should be `"sum_over_batch_size"`.
#'       Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.
#'   * `dtype`: The dtype of the loss's computations. Defaults to `NULL`, which
#'       means using [`config_floatx()`]. `config_floatx()` is a
#'       `"float32"` unless set to different value
#'       (via [`config_set_floatx()`]). If a `keras$DTypePolicy` is
#'       provided, then the `compute_dtype` will be utilized.
#'
#' * ```
#'   __call__(y_true, y_pred, sample_weight=NULL)
#'   ```
#'   Call the loss instance as a function, optionally with `sample_weight`.
#'
#' * ```r
#'   get_config()
#'   ```
#'
#' # Readonly properties:
#'
#' * ```r
#'   dtype
#'   ```
#'
#' @returns A function that returns `Loss` instances, similar to the
#'   builtin loss functions.
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

  members <- modify_intersection(members, list(
    from_config = function(x) decorate_method(x, "classmethod")
  ))

  inherit <- substitute(inherit) %||%
    quote(base::asNamespace("keras3")$keras$Loss)

  new_wrapped_py_class(
    classname = classname,
    members = members,
    inherit = inherit,
    parent_env = parent_env,
    private = private,
    default_formals = function(name=NULL, reduction="sum_over_batch_size", dtype=NULL) {}
  )
}
