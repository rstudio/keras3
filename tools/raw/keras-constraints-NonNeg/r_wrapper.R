#' Constrains the weights to be non-negative.
#'
#' @description
#'
#' @export
#' @family constraint
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/constraints/NonNeg>
constraint_nonneg <-
function ()
{
    args <- capture_args2(NULL)
    do.call(keras$constraints$NonNeg, args)
}
