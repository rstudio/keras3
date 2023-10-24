#' Layer that applies an update to the cost function based input activity.
#'
#' @description
#'
#' # Input Shape
#' Arbitrary. Use the keyword argument `input_shape`
#' (tuple of integers, does not include the samples axis)
#' when using this layer as the first layer in a model.
#'
#' # Output Shape
#'     Same shape as input.
#'
#' @param l1 L1 regularization factor (positive float).
#' @param l2 L2 regularization factor (positive float).
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family regularization layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ActivityRegularization>
layer_activity_regularization <-
function (object, l1 = 0, l2 = 0, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$ActivityRegularization, object,
        args)
}
