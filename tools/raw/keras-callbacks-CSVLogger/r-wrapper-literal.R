#' Callback that streams epoch results to a CSV file.
#'
#' @description
#' Supports all values that can be represented as a string,
#' including 1D iterables such as `np.ndarray`.
#'
#' # Examples
#' ```python
#' csv_logger = CSVLogger('training.log')
#' model.fit(X_train, Y_train, callbacks=[csv_logger])
#' ```
#'
#' @param filename Filename of the CSV file, e.g. `'run/log.csv'`.
#' @param separator String used to separate elements in the CSV file.
#' @param append Boolean. True: append if file exists (useful for continuing
#'     training). False: overwrite existing file.
#'
#' @export
#' @family callback
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger>
callback_csv_logger <-
function (filename, separator = ",", append = FALSE)
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$CSVLogger, args)
}
