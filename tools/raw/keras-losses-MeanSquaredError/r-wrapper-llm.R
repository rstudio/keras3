#' Computes the mean of squares of errors between labels and predictions.
#'
#' @description
#' Formula:
#'
#' ```python
#' loss = mean(square(y_true - y_pred), axis=-1)
#' ```
#' Formula:
#'
#' ```python
#' loss = mean(square(y_true - y_pred))
#' ```
#'
#' # Examples
#' ```python
#' y_true = np.random.randint(0, 2, size=(2, 3))
#' y_pred = np.random.random(size=(2, 3))
#' loss = keras.losses.mean_squared_error(y_true, y_pred)
#' ```
#'
#' # Returns
#'     Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.
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
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError>
loss_mean_squared_error <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "mean_squared_error")
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$MeanSquaredError
    else keras$losses$mean_squared_error
    do.call(callable, args)
}, py_function_name = "mean_squared_error")
