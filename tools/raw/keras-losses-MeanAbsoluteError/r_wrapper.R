#' Computes the mean of absolute difference between labels and predictions.
#'
#' @description
#' ```python
#' loss = mean(abs(y_true - y_pred), axis=-1)
#' ```
#' Formula:
#'
#' ```python
#' loss = mean(abs(y_true - y_pred))
#' ```
#'
#' # Returns
#' Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' # Examples
#' ```python
#' y_true = np.random.randint(0, 2, size=(2, 3))
#' y_pred = np.random.random(size=(2, 3))
#' loss = keras.losses.mean_absolute_error(y_true, y_pred)
#' ```
#'
#' @param reduction Type of reduction to apply to the loss. In almost all cases
#'     this should be `"sum_over_batch_size"`.
#'     Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
#' @param name Optional name for the loss instance.
#' @param y_true Ground truth values with shape = `[batch_size, d0, .. dN]`.
#' @param y_pred The predicted values with shape = `[batch_size, d0, .. dN]`.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family loss
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError>
loss_mean_absolute_error <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "mean_absolute_error")
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$MeanAbsoluteError
    else keras$losses$mean_absolute_error
    do.call(callable, args)
}, py_function_name = "mean_absolute_error")
