#' A layer that uses `einsum` as the backing computation.
#'
#' @description
#' This layer can perform einsum calculations of arbitrary dimensionality.
#'
#' # Examples
#' **Biased dense layer with einsums**
#'
#' This example shows how to instantiate a standard Keras dense layer using
#' einsum operations. This example is equivalent to
#' `keras.layers.Dense(64, use_bias=True)`.
#'
#' ```python
#' layer = keras.layers.EinsumDense("ab,bc->ac",
#'                                       output_shape=64,
#'                                       bias_axes="c")
#' input_tensor = keras.Input(shape=[32])
#' output_tensor = layer(input_tensor)
#' output_tensor.shape
#' # (None, 64)
#' ```
#'
#' **Applying a dense layer to a sequence**
#'
#' This example shows how to instantiate a layer that applies the same dense
#' operation to every element in a sequence. Here, the `output_shape` has two
#' values (since there are two non-batch dimensions in the output); the first
#' dimension in the `output_shape` is `None`, because the sequence dimension
#' `b` has an unknown shape.
#'
#' ```python
#' layer = keras.layers.EinsumDense("abc,cd->abd",
#'                                       output_shape=(None, 64),
#'                                       bias_axes="d")
#' input_tensor = keras.Input(shape=[32, 128])
#' output_tensor = layer(input_tensor)
#' output_tensor.shape
#' # (None, 32, 64)
#' ```
#'
#' **Applying a dense layer to a sequence using ellipses**
#'
#' This example shows how to instantiate a layer that applies the same dense
#' operation to every element in a sequence, but uses the ellipsis notation
#' instead of specifying the batch and sequence dimensions.
#'
#' Because we are using ellipsis notation and have specified only one axis, the
#' `output_shape` arg is a single value. When instantiated in this way, the
#' layer can handle any number of sequence dimensions - including the case
#' where no sequence dimension exists.
#'
#' ```python
#' layer = keras.layers.EinsumDense("...x,xy->...y",
#'                                       output_shape=64,
#'                                       bias_axes="y")
#' input_tensor = keras.Input(shape=[32, 128])
#' output_tensor = layer(input_tensor)
#' output_tensor.shape
#' # (None, 32, 64)
#' ```
#'
#' @param equation An equation describing the einsum to perform.
#'     This equation must be a valid einsum string of the form
#'     `ab,bc->ac`, `...ab,bc->...ac`, or
#'     `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
#'     axis expression sequence.
#' @param output_shape The expected shape of the output tensor
#'     (excluding the batch dimension and any dimensions
#'     represented by ellipses). You can specify `None` for any dimension
#'     that is unknown or can be inferred from the input shape.
#' @param activation Activation function to use. If you don't specify anything,
#'     no activation is applied
#'     (that is, a "linear" activation: `a(x) = x`).
#' @param bias_axes A string containing the output dimension(s)
#'     to apply a bias to. Each character in the `bias_axes` string
#'     should correspond to a character in the output portion
#'     of the `equation` string.
#' @param kernel_initializer Initializer for the `kernel` weights matrix.
#' @param bias_initializer Initializer for the bias vector.
#' @param kernel_regularizer Regularizer function applied to the `kernel` weights
#'     matrix.
#' @param bias_regularizer Regularizer function applied to the bias vector.
#' @param kernel_constraint Constraint function applied to the `kernel` weights
#'     matrix.
#' @param bias_constraint Constraint function applied to the bias vector.
#' @param ... Base layer keyword arguments, such as `name` and `dtype`.
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @export
#' @family core layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/EinsumDense>
layer_einsum_dense <-
function (object, equation, output_shape, activation = NULL,
    bias_axes = NULL, kernel_initializer = "glorot_uniform",
    bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$EinsumDense, object, args)
}
