#' Evenly round to the given number of decimals.
#'
#' @description
#'
#' # Returns
#'     Output tensor.
#'
#' @param x Input tensor.
#' @param decimals Number of decimal places to round to. Defaults to `0`.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/round>
k_round <-
function (x, decimals = 0L)
{
    args <- capture_args2(list(decimals = as_integer))
    do.call(keras$ops$round, args)
}
