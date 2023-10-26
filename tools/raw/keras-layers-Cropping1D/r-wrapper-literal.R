#' Cropping layer for 1D input (e.g. temporal sequence).
#'
#' @description
#' It crops along the time dimension (axis 1).
#'
#' # Examples
#' ```python
#' input_shape = (2, 3, 2)
#' x = np.arange(np.prod(input_shape)).reshape(input_shape)
#' x
#' # [[[ 0  1]
#' #   [ 2  3]
#' #   [ 4  5]]
#' #  [[ 6  7]
#' #   [ 8  9]
#' #   [10 11]]]
#' y = keras.layers.Cropping1D(cropping=1)(x)
#' y
#' # [[[2 3]]
#' #  [[8 9]]]
#' ```
#'
#' # Input Shape
#' 3D tensor with shape `(batch_size, axis_to_crop, features)`
#'
#' # Output Shape
#'     3D tensor with shape `(batch_size, cropped_axis, features)`
#'
#' @param cropping Int, or tuple of int (length 2), or dictionary.
#' - If int: how many units should be trimmed off at the beginning and
#'   end of the cropping dimension (axis 1).
#' - If tuple of 2 ints: how many units should be trimmed off at the
#'   beginning and end of the cropping dimension
#'   (`(left_crop, right_crop)`).
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family reshaping layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping1D>
layer_cropping_1d <-
function (object, cropping = list(1L, 1L), ...)
{
    args <- capture_args2(list(cropping = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Cropping1D, object, args)
}
