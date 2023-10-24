#' Split a tensor into chunks.
#'
#' @description
#'
#' # Note
#' A split does not have to result in equal division when using
#' Torch backend.
#'
#' # Returns
#'     A list of tensors.
#'
#' @param x Input tensor.
#' @param indices_or_sections Either an integer indicating the number of
#'     sections along `axis` or a list of integers indicating the indices
#'     along `axis` at which the tensor is split.
#' If an integer, N, the tensor will be split into N
#'     equal sections along `axis`. If a 1-D array of sorted integers,
#'     the entries indicate indices at which the tensor will be split
#'     along `axis`.
#' @param axis Axis along which to split. Defaults to `0`.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/split>
k_split <-
function (x, indices_or_sections, axis = 0L)
{
    args <- capture_args2(list(indices_or_sections = as_integer,
        axis = as_axis))
    do.call(keras$ops$split, args)
}
