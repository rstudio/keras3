#' Just your regular densely-connected NN layer.
#'
#' @description
#' `Dense` implements the operation:
#' `output = activation(dot(input, kernel) + bias)`
#' where `activation` is the element-wise activation function
#' passed as the `activation` argument, `kernel` is a weights matrix
#' created by the layer, and `bias` is a bias vector created by the layer
#' (only applicable if `use_bias` is `True`).
#'
#' # Note
#' If the input to the layer has a rank greater than 2, `Dense`
#' computes the dot product between the `inputs` and the `kernel` along the
#' last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
#' For example, if input has dimensions `(batch_size, d0, d1)`, then we create
#' a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
#' of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
#' `batch_size * d0` such sub-tensors). The output in this case will have
#' shape `(batch_size, d0, units)`.
#'
#' # Input Shape
#' N-D tensor with shape: `(batch_size, ..., input_dim)`.
#' The most common situation would be
#' a 2D input with shape `(batch_size, input_dim)`.
#'
#' # Output Shape
#' N-D tensor with shape: `(batch_size, ..., units)`.
#' For instance, for a 2D input with shape `(batch_size, input_dim)`,
#' the output would have shape `(batch_size, units)`.
#'
#' @param units Positive integer, dimensionality of the output space.
#' @param activation Activation function to use.
#'     If you don't specify anything, no activation is applied
#'     (ie. "linear" activation: `a(x) = x`).
#' @param use_bias Boolean, whether the layer uses a bias vector.
#' @param kernel_initializer Initializer for the `kernel` weights matrix.
#' @param bias_initializer Initializer for the bias vector.
#' @param kernel_regularizer Regularizer function applied to
#'     the `kernel` weights matrix.
#' @param bias_regularizer Regularizer function applied to the bias vector.
#' @param activity_regularizer Regularizer function applied to
#'     the output of the layer (its "activation").
#' @param kernel_constraint Constraint function applied to
#'     the `kernel` weights matrix.
#' @param bias_constraint Constraint function applied to the bias vector.
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family core layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>
layer_dense <-
function (object, units, activation = NULL, use_bias = TRUE,
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args2(list(units = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Dense, object, args)
}
