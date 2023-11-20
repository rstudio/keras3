#' Computes the cosine similarity between `y_true` & `y_pred`.
#'
#' @description
#' Formula:
#' ```python
#' loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
#' ```
#'
#' Note that it is a number between -1 and 1. When it is a negative number
#' between -1 and 0, 0 indicates orthogonality and values closer to -1
#' indicate greater similarity. This makes it usable as a loss function in a
#' setting where you try to maximize the proximity between predictions and
#' targets. If either `y_true` or `y_pred` is a zero vector, cosine
#' similarity will be 0 regardless of the proximity between predictions
#' and targets.
#'
#' Note that it is a number between -1 and 1. When it is a negative number
#' between -1 and 0, 0 indicates orthogonality and values closer to -1
#' indicate greater similarity. This makes it usable as a loss function in a
#' setting where you try to maximize the proximity between predictions and
#' targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity
#' will be 0 regardless of the proximity between predictions and targets.
#'
#' Formula:
#'
#' ```python
#' loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
#' ```
#'
#' # Examples
#' ```python
#' y_true = [[0., 1.], [1., 1.], [1., 1.]]
#' y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
#' loss = keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
#' # [-0., -0.99999994, 0.99999994]
#' ```
#'
#' @returns
#' Cosine similarity tensor.
#'
#' @param axis
#' The axis along which the cosine similarity is computed
#' (the features axis). Defaults to `-1`.
#'
#' @param reduction
#' Type of reduction to apply to the loss. In almost all cases
#' this should be `"sum_over_batch_size"`.
#' Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
#'
#' @param name
#' Optional name for the loss instance.
#'
#' @param y_true
#' Tensor of true targets.
#'
#' @param y_pred
#' Tensor of predicted targets.
#'
#' @param ...
#' Passed on to the Python callable
#'
#' @export
#' @family losses
#' @seealso
#' + <https:/keras.io/api/losses/regression_losses#cosinesimilarity-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity>
loss_cosine_similarity <-
function (y_true, y_pred, axis = -1L, ..., reduction = "sum_over_batch_size",
    name = "cosine_similarity")
{
}
