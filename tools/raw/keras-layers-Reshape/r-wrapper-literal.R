#' Layer that reshapes inputs into the given shape.
#'
#' @description
#'
#' # Input Shape
#' Arbitrary, although all dimensions in the input shape must be
#' known/fixed. Use the keyword argument `input_shape` (tuple of integers,
#' does not include the samples/batch size axis) when using this layer as
#' the first layer in a model.
#'
#' # Output Shape
#' `(batch_size, *target_shape)`
#'
#' # Examples
#' ```python
#' x = keras.Input(shape=(12,))
#' y = keras.layers.Reshape((3, 4))(x)
#' y.shape
#' # (None, 3, 4)
#' ```
#'
#' ```python
#' # also supports shape inference using `-1` as dimension
#' y = keras.layers.Reshape((-1, 2, 2))(x)
#' y.shape
#' # (None, 3, 2, 2)
#' ```
#'
#' @param target_shape Target shape. Tuple of integers, does not include the
#' samples dimension (batch size).
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family reshaping layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape>
layer_reshape <-
function (object, target_shape, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape,
        target_shape = as_integer), ignore = "object")
    create_layer(keras$layers$Reshape, object, args)
}
