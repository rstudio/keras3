#' Pad a tensor.
#'
#' @description
#'
#' # Note
#' Torch backend only supports modes `"constant"`, `"reflect"`,
#'     `"symmetric"` and `"circular"`.
#'     Only Torch backend supports `"circular"` mode.
#'
#' Note:
#'     Tensorflow backend only supports modes `"constant"`, `"reflect"`
#'     and `"symmetric"`.
#'
#' # Returns
#'     Padded tensor.
#'
#' @param x Tensor to pad.
#' @param pad_width Number of values padded to the edges of each axis.
#'     `((before_1, after_1), ...(before_N, after_N))` unique pad
#'     widths for each axis.
#'     `((before, after),)` yields same before and after pad for
#'     each axis.
#'     `(pad,)` or `int` is a shortcut for `before = after = pad`
#'     width for all axes.
#' @param mode One of `"constant"`, `"edge"`, `"linear_ramp"`,
#'     `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
#'     `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`,
#'     `"circular"`. Defaults to`"constant"`.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/pad>
k_pad <-
function (x, pad_width, mode = "constant")
{
    args <- capture_args2(list(pad_width = as_integer))
    do.call(keras$ops$pad, args)
}
