#' Leaky version of a Rectified Linear Unit activation layer.
#'
#' @description
#' This layer allows a small gradient when the unit is not active.
#'
#' Formula:
#'
#' ``` python
#' f(x) = alpha * x if x < 0
#' f(x) = x if x >= 0
#' ```
#'
#' # Examples
#' ``` python
#' leaky_relu_layer = LeakyReLU(negative_slope=0.5)
#' input = np.array([-10, -5, 0.0, 5, 10])
#' result = leaky_relu_layer(input)
#' # result = [-5. , -2.5,  0. ,  5. , 10.]
#' ```
#'
#' @param negative_slope Float >= 0.0. Negative slope coefficient.
#'   Defaults to `0.3`.
#' @param ... Base layer keyword arguments, such as
#'     `name` and `dtype`.
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @export
#' @family activations layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU>
layer_activation_leaky_relu <-
function (object, negative_slope = 0.3, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$LeakyReLU, object, args)
}
