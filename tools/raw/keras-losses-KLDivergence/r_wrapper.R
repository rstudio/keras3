#' Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.
#'
#' @description
#' Formula:
#'
#' ```python
#' loss = y_true * log(y_true / y_pred)
#' ```
#' Formula:
#'
#' ```python
#' loss = y_true * log(y_true / y_pred)
#' ```
#'
#' # Returns
#' KL Divergence loss values with shape = `[batch_size, d0, .. dN-1]`.
#'
#' # Examples
#' ```python
#' y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
#' y_pred = np.random.random(size=(2, 3))
#' loss = keras.losses.kl_divergence(y_true, y_pred)
#' assert loss.shape == (2,)
#' y_true = ops.clip(y_true, 1e-7, 1)
#' y_pred = ops.clip(y_pred, 1e-7, 1)
#' assert np.array_equal(
#'     loss, np.sum(y_true * np.log(y_true / y_pred), axis=-1))
#' ```
#'
#' @param reduction Type of reduction to apply to the loss. In almost all cases
#'     this should be `"sum_over_batch_size"`.
#'     Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
#' @param name Optional name for the loss instance.
#' @param y_true Tensor of true targets.
#' @param y_pred Tensor of predicted targets.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family loss
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence>
loss_kl_divergence <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size",
    name = "kl_divergence")
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred))
        keras$losses$KLDivergence
    else keras$losses$kl_divergence
    do.call(callable, args)
}, py_function_name = "kl_divergence")
