#' Callback that terminates training when a NaN loss is encountered.
#'
#' @description
#'
#' @export
#' @family callback
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TerminateOnNaN>
callback_terminate_on_nan <-
function ()
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$TerminateOnNaN, args)
}
