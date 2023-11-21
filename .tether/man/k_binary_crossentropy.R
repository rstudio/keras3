#' Computes binary cross-entropy loss between target and output tensor.
#'
#' @description
#' The binary cross-entropy loss is commonly used in binary
#' classification tasks where each input sample belongs to one
#' of the two classes. It measures the dissimilarity between the
#' target and output probabilities or logits.
#'
#' # Examples
#' ```python
#' target = keras.ops.convert_to_tensor([0, 1, 1, 0])
#' output = keras.ops.convert_to_tensor([0.1, 0.9, 0.8, 0.2])
#' binary_crossentropy(target, output)
#' # array([0.10536054 0.10536054 0.22314355 0.22314355],
#' #       shape=(4,), dtype=float32)
#' ```
#'
#' @returns
#' Integer tensor: The computed binary cross-entropy loss between
#' `target` and `output`.
#'
#' @param target
#' The target tensor representing the true binary labels.
#' Its shape should match the shape of the `output` tensor.
#'
#' @param output
#' The output tensor representing the predicted probabilities
#' or logits. Its shape should match the shape of the
#' `target` tensor.
#'
#' @param from_logits
#' (optional) Whether `output` is a tensor of logits or
#' probabilities.
#' Set it to `True` if `output` represents logits; otherwise,
#' set it to `False` if `output` represents probabilities.
#' Defaults to`False`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https:/keras.io/keras_core/api/ops/nn#binarycrossentropy-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/binary_crossentropy>
k_binary_crossentropy <-
function (target, output, from_logits = FALSE)
{
}
