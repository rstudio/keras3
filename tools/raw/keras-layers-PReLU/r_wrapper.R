#' Parametric Rectified Linear Unit activation layer.
#'
#' @description
#' Formula:
#' ``` python
#' f(x) = alpha * x for x < 0
#' f(x) = x for x >= 0
#' ```
#' where `alpha` is a learned array with the same shape as x.
#'
#' @param alpha_initializer Initializer function for the weights.
#' @param alpha_regularizer Regularizer for the weights.
#' @param alpha_constraint Constraint for the weights.
#' @param shared_axes The axes along which to share learnable parameters for the
#'     activation function. For example, if the incoming feature maps are
#'     from a 2D convolution with output shape
#'     `(batch, height, width, channels)`, and you wish to share parameters
#'     across space so that each filter only has one set of parameters,
#'     set `shared_axes=[1, 2]`.
#' @param ... Base layer keyword arguments, such as `name` and `dtype`.
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @export
#' @family activations layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/PReLU>
layer_activation_parametric_relu <-
function (object, alpha_initializer = "Zeros", alpha_regularizer = NULL,
    alpha_constraint = NULL, shared_axes = NULL, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape,
        shared_axes = function (x)
        lapply(x, as_integer)), ignore = "object")
    create_layer(keras$layers$PReLU, object, args)
}
