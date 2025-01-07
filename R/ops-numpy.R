#' Safe element-wise division which returns 0 where the denominator is 0.
#'
#' @returns
#' The quotient `x1/x2`, element-wise, with zero where x2 is zero.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.divide_no_nan
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/divide_no_nan>
op_divide_no_nan <-
function (x1, x2)
keras$ops$divide_no_nan(x1, x2)


#' Performs an indirect partition along the given axis.
#'
#' @description
#' It returns an array
#' of indices of the same shape as `x` that index data along the given axis
#' in partitioned order.
#'
#' @returns
#' Array of indices that partition `x` along the specified `axis`.
#'
#' @param x
#' Array to sort.
#'
#' @param kth
#' Element index to partition by.
#' The k-th element will be in its final sorted position and all
#' smaller elements will be moved before it and all larger elements
#' behind it. The order of all elements in the partitions is undefined.
#' If provided with a sequence of k-th it will partition all of them
#' into their sorted position at once.
#'
#' @param axis
#' Axis along which to sort. The default is `-1` (the last axis).
#' If `NULL`, the flattened array is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.argpartition
op_argpartition <-
function (x, kth, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$argpartition, args)
}


#' Compute the bit-wise AND of two arrays element-wise.
#'
#' @description
#' Computes the bit-wise AND of the underlying binary representation of the
#' integers in the input arrays. This ufunc implements the C/Python operator
#' `&`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @param y
#' Input integer tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.bitwise_and
op_bitwise_and <-
function (x, y)
{
    args <- capture_args(list(x = as_integer, y = as_integer))
    do.call(keras$ops$bitwise_and, args)
}


#' Compute bit-wise inversion, or bit-wise NOT, element-wise.
#'
#' @description
#' Computes the bit-wise NOT of the underlying binary representation of the
#' integers in the input arrays. This ufunc implements the C/Python operator
#' `~`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @family numpy ops
#' @family ops
#' @export
#' @tether keras.ops.bitwise_invert
op_bitwise_invert <-
function (x)
{
    keras$ops$bitwise_invert(as_integer(x))
}


#' Shift the bits of an integer to the left.
#'
#' @description
#' Bits are shifted to the left by appending `y` 0s at the right of `x`.
#' Since the internal representation of numbers is in binary format, this
#' operation is equivalent to multiplying `x` by `2**y`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @param y
#' Input integer tensor.
#'
#' @export
#' @tether keras.ops.bitwise_left_shift
#' @family numpy ops
#' @family ops
op_bitwise_left_shift <-
function (x, y)
{
    args <- capture_args(list(x = as_integer, y = as_integer))
    do.call(keras$ops$bitwise_left_shift, args)
}


#' Compute bit-wise inversion, or bit-wise NOT, element-wise.
#'
#' @description
#' Computes the bit-wise NOT of the underlying binary representation of the
#' integers in the input arrays. This ufunc implements the C/Python operator
#' `~`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @export
#' @tether keras.ops.bitwise_not
#' @family numpy ops
#' @family ops
op_bitwise_not <-
function (x)
{
    keras$ops$bitwise_not(as_integer(x))
}


#' Compute the bit-wise OR of two arrays element-wise.
#'
#' @description
#' Computes the bit-wise OR of the underlying binary representation of the
#' integers in the input arrays. This ufunc implements the C/Python operator
#' `|`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @param y
#' Input integer tensor.
#'
#' @export
#' @tether keras.ops.bitwise_or
#' @family numpy ops
#' @family ops
op_bitwise_or <-
function (x, y)
{
    args <- capture_args(list(x = as_integer, y = as_integer))
    do.call(keras$ops$bitwise_or, args)
}


#' Shift the bits of an integer to the right.
#'
#' @description
#' Bits are shifted to the right `y`. Because the internal representation of
#' numbers is in binary format, this operation is equivalent to dividing `x` by
#' `2**y`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @param y
#' Input integer tensor.
#'
#' @export
#' @tether keras.ops.bitwise_right_shift
#' @family numpy ops
#' @family ops
op_bitwise_right_shift <-
function (x, y)
{
    args <- capture_args(list(x = as_integer, y = as_integer))
    do.call(keras$ops$bitwise_right_shift, args)
}


#' Compute the bit-wise XOR of two arrays element-wise.
#'
#' @description
#' Computes the bit-wise XOR of the underlying binary representation of the
#' integers in the input arrays. This ufunc implements the C/Python operator
#' `^`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @param y
#' Input integer tensor.
#'
#' @export
#' @tether keras.ops.bitwise_xor
#' @family numpy ops
#' @family ops
op_bitwise_xor <-
function (x, y)
{
    args <- capture_args(list(x = as_integer, y = as_integer))
    do.call(keras$ops$bitwise_xor, args)
}


#' Computes a histogram of the data tensor `x`.
#'
#' @description
#'
#' # Examples
#' ```{r, comment = "#>", strip.white = FALSE}
#' input_tensor <- random_uniform(8)
#' c(counts, edges) %<-% op_histogram(input_tensor)
#'
#' counts
#' edges
#' ```
#'
#' @returns
#' A list of two tensors containing:
#' - A tensor representing the counts of elements in each bin.
#' - A tensor representing the bin edges.
#'
#' @param x
#' Input tensor.
#'
#' @param bins
#' An integer representing the number of histogram bins.
#' Defaults to 10.
#'
#' @param range
#' A pair of numbers representing the lower and upper range of the bins.
#' If not specified, it will use the min and max of `x`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.histogram
op_histogram <-
function (x, bins = 10L, range = NULL)
{
    args <- capture_args(list(bins = as_integer, range = as_tuple))
    do.call(keras$ops$histogram, args)
}


#' Shift the bits of an integer to the left.
#'
#' @description
#' Bits are shifted to the left by appending `y` 0s at the right of `x`.
#' Since the internal representation of numbers is in binary format, this
#' operation is equivalent to multiplying `x` by `2^y`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @param y
#' Input integer tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.left_shift
op_left_shift <-
function (x, y)
{
    args <- capture_args(list(x = as_integer, y = as_integer))
    do.call(keras$ops$left_shift, args)
}


#' Shift the bits of an integer to the right.
#'
#' @description
#' Bits are shifted to the right `y`. Because the internal representation of
#' numbers is in binary format, this operation is equivalent to dividing `x` by
#' `2^y`.
#'
#' @returns
#' Result tensor.
#'
#' @param x
#' Input integer tensor.
#'
#' @param y
#' Input integer tensor.
#'
#' @export
#' @tether keras.ops.right_shift
#' @family numpy ops
#' @family ops
op_right_shift <-
function (x, y)
{
    args <- capture_args(list(x = as_integer, y = as_integer))
    do.call(keras$ops$right_shift, args)
}


#' Computes log of the determinant of a hermitian positive definite matrix.
#'
#' @returns
#' The natural log of the determinant of matrix.
#'
#' @param x
#' Input matrix. It must 2D and square.
#'
#' @export
#' @tether keras.ops.logdet
#' @family numpy ops
#' @family ops
op_logdet <-
function (x)
keras$ops$logdet(x)


#' Performs a safe saturating cast to the desired dtype.
#'
#' @description
#' Saturating cast prevents data type overflow when casting to `dtype` with
#' smaller values range. E.g.
#' `op_cast(c(-1, 256), "float32") |> op_cast("uint8")` returns `c(255, 0)`,
#' but `op_cast(c(-1, 256), "float32") |> op_saturate_cast("uint8")` returns
#' `c(0, 255)`.
#'
#' # Examples
#' Image resizing with bicubic interpolation may produce values outside
#' original range.
#' ```{r}
#' image2x2 <- np_array(as.integer(c(0, 1, 254, 255)), "uint8") |>
#'   array_reshape(c(1, 2, 2, 1))
#' image4x4 <- image2x2 |>
#'   tensorflow::tf$image$resize(shape(4, 4), method="bicubic")
#' image4x4 |> as.array() |> drop()
#' ```
#'
#' Casting this resized image back to `uint8` will cause overflow.
#' ```{r}
#' image4x4_casted <- op_cast(image4x4, "uint8")
#' image4x4_casted |> as.array() |> drop()
#' ```
#'
#' Saturate casting to `uint8` will clip values to `uint8` range before
#' casting and will not cause overflow.
#' ```{r}
#' image4x4_saturate_casted <- image4x4 |> op_saturate_cast("uint8")
#' image4x4_saturate_casted |> as.array() |> drop()
#' ```
#'
#' @returns
#' A safely casted tensor of the specified `dtype`.
#'
#' @param x
#' A tensor or variable.
#'
#' @param dtype
#' The target type.
#'
#' @export
#' @tether keras.ops.saturate_cast
#' @family numpy ops
#' @family ops
op_saturate_cast <-
function (x, dtype)
keras$ops$saturate_cast(x, dtype)


#' Return the truncated value of the input, element-wise.
#'
#' @description
#' The truncated value of the scalar `x` is the nearest integer `i` which is
#' closer to zero than `x` is. In short, the fractional part of the signed
#' number `x` is discarded.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0))
#' op_trunc(x)
#' ```
#'
#' @returns
#' The truncated value of each element in `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.trunc
op_trunc <-
function (x)
keras$ops$trunc(x)


#' Calculate the base-2 exponential of all elements in the input tensor.
#'
#' @returns
#' Output tensor, element-wise base-2 exponential of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.exp2
op_exp2 <-
function (x)
keras$ops$exp2(x)


#' Return the inner product of two tensors.
#'
#' @description
#' Ordinary inner product of vectors for 1-D tensors
#' (without complex conjugation), in higher dimensions
#' a sum product over the last axes.
#'
#' Multidimensional arrays are treated as vectors by flattening
#' all but their last axes. The resulting dot product is performed
#' over their last axes.
#'
#' @returns
#' Output tensor. The shape of the output is determined by
#' broadcasting the shapes of `x1` and `x2` after removing
#' their last axes.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor. The last dimension of `x1` and `x2`
#' must match.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.inner
op_inner <-
function (x1, x2)
keras$ops$inner(x1, x2)
