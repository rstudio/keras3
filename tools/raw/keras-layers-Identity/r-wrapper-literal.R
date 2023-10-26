#' Identity layer.
#'
#' @description
#' This layer should be used as a placeholder when no operation is to be
#' performed. The layer just returns its `inputs` argument as output.
#'
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family core layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Identity>
layer_identity <-
function (object, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Identity, object, args)
}
