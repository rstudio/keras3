#' For loop implementation.
#'
#' @description
#'
#' # Returns
#' The final state after the loop.
#'
#' # Examples
#' ```python
#' lower = 0
#' upper = 10
#' body_fun = lambda i, s: (i + 1, s + i)
#' init_val = 0
#' keras.ops.fori_loop(lower, upper, body_fun, init_val)
#' # 45
#' ```
#'
#' @param lower The initial value of the loop variable.
#' @param upper The upper bound of the loop variable.
#' @param body_fun A callable that represents the loop body. Must take two
#'     arguments: the loop variable and the loop state. The loop state
#'     should be updated and returned by this function.
#' @param init_val The initial value of the loop state.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/fori_loop>
k_fori_loop <-
function (lower, upper, body_fun, init_val)
keras$ops$fori_loop(lower, upper, body_fun, init_val)
