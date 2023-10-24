#' Concatenates a list of inputs.
#'
#' @description
#' It takes as input a list of tensors, all of the same shape except
#' for the concatenation axis, and returns a single tensor that is the
#' concatenation of all inputs.
#'
#' # Examples
#' ```python
#' x = np.arange(20).reshape(2, 2, 5)
#' y = np.arange(20, 30).reshape(2, 1, 5)
#' keras.layers.Concatenate(axis=1)([x, y])
#' ```
#'
#' Usage in a Keras model:
#'
#' ```python
#' x1 = keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
#' x2 = keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
#' y = keras.layers.Concatenate()([x1, x2])
#' ```
#'
#' # Returns
#'     A tensor, the concatenation of the inputs alongside axis `axis`.
#'
#' @param axis Axis along which to concatenate.
#' @param ... Standard layer keyword arguments.
#' @param inputs layers to combine
#'
#' @export
#' @family merging layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate>
layer_concatenate <-
function (inputs, ..., axis = -1L)
{
    args <- capture_args2(list(axis = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- do.call(keras$layers$Concatenate, args)
    if (length(inputs))
        layer(inputs)
    else layer
}
