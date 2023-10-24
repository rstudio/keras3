#' Performs elementwise addition operation.
#'
#' @description
#' It takes as input a list of tensors, all of the same shape,
#' and returns a single tensor (also of the same shape).
#'
#' # Examples
#' ```python
#' input_shape = (2, 3, 4)
#' x1 = np.random.rand(*input_shape)
#' x2 = np.random.rand(*input_shape)
#' y = keras.layers.Add()([x1, x2])
#' ```
#'
#' Usage in a Keras model:
#'
#' ```python
#' input1 = keras.layers.Input(shape=(16,))
#' x1 = keras.layers.Dense(8, activation='relu')(input1)
#' input2 = keras.layers.Input(shape=(32,))
#' x2 = keras.layers.Dense(8, activation='relu')(input2)
#' # equivalent to `added = keras.layers.add([x1, x2])`
#' added = keras.layers.Add()([x1, x2])
#' out = keras.layers.Dense(4)(added)
#' model = keras.models.Model(inputs=[input1, input2], outputs=out)
#' ```
#'
#' @param ... Passed on to the Python callable
#' @param inputs layers to combine
#'
#' @export
#' @family merging layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add>
layer_add <-
function (inputs, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- do.call(keras$layers$Add, args)
    if (length(inputs))
        layer(inputs)
    else layer
}
