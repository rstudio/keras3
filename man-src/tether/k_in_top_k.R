#' Checks if the targets are in the top-k predictions.
#'
#' @description
#'
#' # Examples
#' ```python
#' targets = keras.ops.convert_to_tensor([2, 5, 3])
#' predictions = keras.ops.convert_to_tensor(
#' [[0.1, 0.4, 0.6, 0.9, 0.5],
#'  [0.1, 0.7, 0.9, 0.8, 0.3],
#'  [0.1, 0.6, 0.9, 0.9, 0.5]])
#' in_top_k(targets, predictions, k=3)
#' # array([ True False  True], shape=(3,), dtype=bool)
#' ```
#'
#' @returns
#' A boolean tensor of the same shape as `targets`, where each element
#' indicates whether the corresponding target is in the top-k predictions.
#'
#' @param targets
#' A tensor of true labels.
#'
#' @param predictions
#' A tensor of predicted labels.
#'
#' @param k
#' An integer representing the number of predictions to consider.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https:/keras.io/keras_core/api/ops/core#intopk-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/in_top_k>
k_in_top_k <-
function (targets, predictions, k)
{
}
