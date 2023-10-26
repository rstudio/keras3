#' Callback that prints metrics to stdout.
#'
#' @description
#'
#' # Raises
#'     ValueError: In case of invalid `count_mode`.
#'
#' @param count_mode One of `"steps"` or `"samples"`.
#' Whether the progress bar should
#' count samples seen or steps (batches) seen.
#'
#' @export
#' @family callback
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ProgbarLogger>
callback_progbar_logger <-
function (count_mode = NULL)
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$ProgbarLogger, args)
}
