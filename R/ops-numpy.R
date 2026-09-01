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
ops$divide_no_nan(x1, x2)


#' Performs an indirect partition along the given axis.
#'
#' @description
#' It returns an array
#' of indices of the same shape as `x` that index data along the given axis
#' in partitioned order.
#'
#' ```{r}
#' x <- op_convert_to_tensor(c(9, 3, 6, 2, 8, 5, 7, 1, 10, 4))
#' x@r[op_argpartition(x, 3)]
#' x@r[op_argpartition(x, 5)]
#' x@r[op_argpartition(x, 7)]
#' ```
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
#' @param zero_indexed
#' If `TRUE`, the returned indices are zero-based (`0` encodes to first
#' position); if `FALSE` (default), the returned indices are one-based (`1`
#' encodes to first position).
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.argpartition
op_argpartition <-
function (x, kth, axis = -1L, zero_indexed = FALSE)
{
    args <- capture_args(list(x = as_array, axis = as_axis, kth = as_py_index),
                         ignore = "zero_indexed")
    result <- do.call(ops$argpartition, args)
    if (zero_indexed) result else result + 1L
}


#' Compute the Pearson correlation coefficient matrix.
#'
#'
#' # Examples
#' ```{r}
#' x <- op_array(matrix(c(1, 2, 3,
#'                        2, 3, 4), nrow = 2, byrow = TRUE))
#' op_corrcoef(x)
#' ```
#'
#' @param x
#' A 2D tensor of shape `(N, D)`, where `N` is the number of variables
#' and `D` is the number of observations.
#'
#' @returns
#' A tensor of shape `(N, N)` representing the correlation matrix.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.corrcoef
op_corrcoef <-
function (x)
ops$corrcoef(x)


#' Computes the cube root of the input tensor, element-wise.
#'
#' @description
#' Returns the real-valued cube root of `x`, handling negative inputs in the
#' real domain.
#'
#' # Examples
#' ```{r}
#' op_cbrt(c(-8, 0, 8))
#' ```
#'
#' @param x
#' Input tensor.
#'
#' @returns
#' A tensor containing the cube root of each element in `x`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.cbrt
op_cbrt <-
function (x)
ops$cbrt(x)


#' Convert angles from degrees to radians.
#'
#' @description
#' The conversion is defined as:
#' `rad = deg * (pi / 180)`.
#'
#' # Examples
#' ```{r}
#' op_deg2rad(c(0, 90, 180))
#' ```
#'
#' @returns
#' A tensor containing angles converted to radians.
#'
#' @param x
#' Input tensor of angles in degrees.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.deg2rad
op_deg2rad <-
function (x)
ops$deg2rad(x)


#' Bartlett window function.
#'
#' @description
#' The Bartlett window is a triangular window that rises then falls linearly.
#'
#' # Examples
#' ```{r}
#' op_bartlett(5)
#' ```
#'
#' @returns
#' A 1D tensor containing the window values.
#'
#' @param x
#' Length of the window. Must be a positive integer.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.bartlett
op_bartlett <-
function (x)
ops$bartlett(as_integer(x))


#' Blackman window function.
#'
#' @description
#' The Blackman window is a taper formed by using a weighted cosine.
#'
#' @param x
#' Length of the window. Must be a positive integer.
#'
#' @returns
#' A 1D tensor containing the window values.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.blackman
op_blackman <-
function (x)
ops$blackman(as_integer(x))


#' Hamming window function.
#'
#' @description
#' The Hamming window is defined as:
#' `w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))` for `0 <= n <= N - 1`.
#'
#' # Examples
#' ```{r}
#' op_hamming(5)
#' ```
#'
#' @returns
#' A 1D tensor containing the window values.
#'
#' @param x
#' Length of the window. Must be a positive integer.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.hamming
op_hamming <-
function (x)
ops$hamming(as_integer(x))


#' Hanning window function.
#'
#' @description
#' The Hanning window is defined as:
#' `w[n] = 0.5 - 0.5 * cos(2 * pi * n / (N - 1))` for `0 <= n <= N - 1`.
#'
#' # Examples
#' ```{r}
#' op_hanning(5)
#' ```
#'
#' @returns
#' A 1D tensor containing the window values.
#'
#' @param x
#' Length of the window. Must be a positive integer.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.hanning
op_hanning <-
function (x)
ops$hanning(as_integer(x))


#' Heaviside step function.
#'
#' @description
#' The Heaviside step function is defined as:
#' `heaviside(x1, x2) = 0` if `x1 < 0`,
#' `heaviside(x1, x2) = 1` if `x1 > 0`, and
#' `heaviside(x1, x2) = x2` if `x1 == 0`.
#'
#' # Examples
#' ```{r}
#' x1 <- op_array(c(-2, 0, 3))
#' op_heaviside(x1, 0.5)
#' ```
#'
#' @returns
#' A tensor broadcast from `x1` and `x2` containing `0`, `1`, or `x2`.
#'
#' @param x1
#' Tensor input.
#'
#' @param x2
#' Value to use when `x1 == 0`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.heaviside
op_heaviside <-
function (x1, x2)
ops$heaviside(x1, x2)


#' Kaiser window function.
#'
#' @description
#' The Kaiser window is defined as:
#' `w[n] = I0(beta * sqrt(1 - (2 * n / (N - 1) - 1)^2)) / I0(beta)` where
#' `I0` is the modified zeroth-order Bessel function of the first kind.
#'
#' # Examples
#' ```{r}
#' op_kaiser(5, beta = 14)
#' ```
#'
#' @returns
#' A 1D tensor containing the window values.
#'
#' @param x
#' Length of the window. Must be a positive integer.
#'
#' @param beta
#' Shape parameter for the window.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.kaiser
op_kaiser <-
function (x, beta)
ops$kaiser(as_integer(x), beta)


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
    do.call(ops$bitwise_and, args)
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
    ops$bitwise_invert(as_integer(x))
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
    do.call(ops$bitwise_left_shift, args)
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
    ops$bitwise_not(as_integer(x))
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
    do.call(ops$bitwise_or, args)
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
    do.call(ops$bitwise_right_shift, args)
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
    do.call(ops$bitwise_xor, args)
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
    do.call(ops$histogram, args)
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
    do.call(ops$left_shift, args)
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
    do.call(ops$right_shift, args)
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
ops$logdet(x)


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
ops$saturate_cast(x, dtype)


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
ops$trunc(x)


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
ops$exp2(x)


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
ops$inner(x1, x2)


#' Create a two-dimensional array with the flattened input diagonal.
#'
#' @description
#' the k-th diagonal.
#'
#' @returns
#' A 2-D tensor with the flattened input on the specified diagonal.
#'
#' @param x
#' Input tensor to be flattened and placed on the diagonal.
#'
#' @param k
#' The diagonal to place the flattened input. Defaults to `0`.
#' Use `k > 0` for diagonals above the main diagonal,
#' and `k < 0` for diagonals below the main diagonal.
#'
#' @export
#' @tether keras.ops.diagflat
#' @family numpy ops
#' @family ops
op_diagflat <-
function (x, k = 0L)
{
    args <- capture_args(list(k = as_integer))
    do.call(ops$diagflat, args)
}


#' Rotate an array by 90 degrees in the plane specified by axes.
#'
#' @description
#' This function rotates an array counterclockwise
#' by 90 degrees `k` times in the plane specified by `axes`.
#' Supports arrays of two or more dimensions.
#'
#' # Examples
#'
#' ```{r}
#' m <- 1:4 |> op_reshape(c(2, 2))
#' m
#' op_rot90(m)
#' ```
#'
#' ```{r}
#' m <- 1:8 |> op_reshape(c(2, 2, 2))
#' m
#' op_rot90(m, k = 1, axes = c(2, 3))
#' ```
#'
#' @returns
#' Rotated array.
#'
#' @param array
#' Input array to rotate.
#'
#' @param k
#' Number of times the array is rotated by 90 degrees.
#'
#' @param axes
#' A tuple of two integers specifying the
#' plane of rotation (defaults to `(1, 2)`).
#'
#' @export
#' @tether keras.ops.rot90
#' @family numpy ops
#' @family ops
op_rot90 <-
function (array, k = 1L, axes = list(1L, 2L))
{
    args <- capture_args(list(k = as_integer, axes = as_axis))
    do.call(keras$ops$rot90, args)
}

#' Return the sign bit of the elements of `x`.
#'
#' @description
#' The output boolean tensor contains `TRUE` where the sign of `x` is negative,
#' and `FALSE` otherwise.
#'
#' @returns
#' Output boolean tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @tether keras.ops.signbit
#' @family numpy ops
#' @family ops
op_signbit <-
function (x)
keras$ops$signbit(x)


#' Return indices of maximum values along an axis, ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' x <- op_array(matrix(c(1, NaN, 3, NaN, 2, 0),
#'                      nrow = 2, byrow = TRUE))
#' op_nanargmax(x, axis = 2)
#' ```
#'
#' @returns
#' A tensor of indices, with `axis` removed unless `keepdims = TRUE`.
#' Indices are one-based unless `zero_indexed = TRUE`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis along which to find maximum values. By default, the index is into the
#' flattened tensor.
#'
#' @param keepdims
#' Whether to retain reduced axes as dimensions of size one.
#'
#' @param zero_indexed
#' Whether to return zero-based indices. Defaults to `FALSE`, returning
#' one-based indices suitable for R indexing.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanargmax
op_nanargmax <-
function (x, axis = NULL, keepdims = FALSE, zero_indexed = FALSE)
{
    args <- capture_args(list(axis = as_axis), ignore = "zero_indexed")
    result <- do.call(ops$nanargmax, args)
    if (zero_indexed) result else result + 1L
}


#' Return indices of minimum values along an axis, ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' x <- op_array(matrix(c(1, NaN, 3, NaN, 2, 0),
#'                      nrow = 2, byrow = TRUE))
#' op_nanargmin(x, axis = 2)
#' ```
#'
#' @returns
#' A tensor of indices, with `axis` removed unless `keepdims = TRUE`.
#' Indices are one-based unless `zero_indexed = TRUE`.
#'
#' @inheritParams op_nanargmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanargmin
op_nanargmin <-
function (x, axis = NULL, keepdims = FALSE, zero_indexed = FALSE)
{
    args <- capture_args(list(axis = as_axis), ignore = "zero_indexed")
    result <- do.call(ops$nanargmin, args)
    if (zero_indexed) result else result + 1L
}


#' Return cumulative products, treating `NaN`s as one.
#'
#' # Examples
#' ```{r}
#' x <- op_array(matrix(c(1, NaN, 3, NaN, 2, 1),
#'                      nrow = 2, byrow = TRUE))
#' op_nancumprod(x, axis = 2)
#' ```
#'
#' @returns
#' An output tensor with the same shape as `x` when `axis` is supplied, or a
#' flattened tensor otherwise.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis along which to compute cumulative products. By default, the input is
#' flattened.
#'
#' @param dtype
#' Data type of the returned tensor. Defaults to the data type of `x`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nancumprod
op_nancumprod <-
function (x, axis = NULL, dtype = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nancumprod, args)
}


#' Return cumulative sums, treating `NaN`s as zero.
#'
#' # Examples
#' ```{r}
#' x <- op_array(matrix(c(1, NaN, 3, NaN, 2, 1),
#'                      nrow = 2, byrow = TRUE))
#' op_nancumsum(x, axis = 2)
#' ```
#'
#' @returns
#' An output tensor with the same shape as `x` when `axis` is supplied, or a
#' flattened tensor otherwise.
#'
#' @inheritParams op_nancumprod
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nancumsum
op_nancumsum <-
function (x, axis = NULL, dtype = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nancumsum, args)
}


#' Compute a maximum while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' x <- op_array(matrix(c(1, NaN, 3, NaN, 2, 1),
#'                      nrow = 2, byrow = TRUE))
#' op_nanmax(x, axis = 2)
#' ```
#'
#' @returns
#' A tensor containing maximum values. If all values along a reduced axis are
#' `NaN`, the corresponding result is `NaN`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which to reduce. By default, the input is flattened.
#'
#' @param keepdims
#' Whether to retain reduced axes as dimensions of size one.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanmax
op_nanmax <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanmax, args)
}


#' Compute a mean while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nanmean(c(1, NaN, 3))
#' ```
#'
#' @returns
#' A tensor containing mean values. If all values along a reduced axis are
#' `NaN`, the corresponding result is `NaN`.
#'
#' @inheritParams op_nanmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanmean
op_nanmean <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanmean, args)
}


#' Compute a median while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nanmedian(c(1, NaN, 3))
#' ```
#'
#' @returns
#' A tensor containing median values. If all values along a reduced axis are
#' `NaN`, the corresponding result is `NaN`.
#'
#' @inheritParams op_nanmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanmedian
op_nanmedian <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanmedian, args)
}


#' Compute a minimum while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nanmin(c(1, NaN, 3))
#' ```
#'
#' @returns
#' A tensor containing minimum values. If all values along a reduced axis are
#' `NaN`, the corresponding result is `NaN`.
#'
#' @inheritParams op_nanmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanmin
op_nanmin <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanmin, args)
}


#' Compute percentiles while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nanpercentile(c(1, 2, NaN, 4), 50)
#' ```
#'
#' @returns
#' A tensor containing the requested percentiles, with `NaN` values omitted.
#'
#' @param x
#' Input tensor.
#'
#' @param q
#' Percentile or sequence of percentiles between 0 and 100, inclusive.
#'
#' @param axis
#' Axis or axes along which to compute percentiles. By default, the input is
#' flattened.
#'
#' @param method
#' Method used when the requested percentile lies between data points `i < j`.
#' `"linear"` returns `i + (j - i) * fraction`; `"lower"` returns `i`;
#' `"higher"` returns `j`; `"midpoint"` returns `(i + j) / 2`; and `"nearest"`
#' returns whichever of `i` or `j` is nearest. Defaults to `"linear"`.
#'
#' @param keepdims
#' Whether to retain reduced axes as dimensions of size one.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanpercentile
op_nanpercentile <-
function (x, q, axis = NULL, method = "linear", keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanpercentile, args)
}


#' Compute a product while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nanprod(c(1, NaN, 3))
#' ```
#'
#' @returns
#' A tensor containing products, with `NaN` values treated as one.
#'
#' @inheritParams op_nanmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanprod
op_nanprod <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanprod, args)
}


#' Compute quantiles while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nanquantile(c(1, 2, NaN, 4), 0.5)
#' ```
#'
#' @returns
#' A tensor containing the requested quantiles, with `NaN` values omitted.
#'
#' @inheritParams op_nanpercentile
#' @param q
#' Probability or sequence of probabilities between 0 and 1, inclusive.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanquantile
op_nanquantile <-
function (x, q, axis = NULL, method = "linear", keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanquantile, args)
}


#' Compute a standard deviation while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nanstd(c(1, NaN, 3))
#' ```
#'
#' @returns
#' A tensor containing population standard deviations, with `NaN` values
#' omitted.
#'
#' @inheritParams op_nanmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanstd
op_nanstd <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanstd, args)
}


#' Compute a sum while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nansum(c(1, NaN, 3))
#' ```
#'
#' @returns
#' A tensor containing sums, with `NaN` values treated as zero.
#'
#' @inheritParams op_nanmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nansum
op_nansum <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nansum, args)
}


#' Compute a variance while ignoring `NaN`s.
#'
#' # Examples
#' ```{r}
#' op_nanvar(c(1, NaN, 3))
#' ```
#'
#' @returns
#' A tensor containing population variances, with `NaN` values omitted.
#'
#' @inheritParams op_nanmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nanvar
op_nanvar <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$nanvar, args)
}


#' Return the next floating-point values from `x1` toward `x2`.
#'
#' # Examples
#' ```{r}
#' op_nextafter(c(1, 1), c(2, 0))
#' ```
#'
#' @returns
#' A tensor containing the next representable values, element-wise.
#'
#' @param x1
#' Input tensor whose values will be moved.
#'
#' @param x2
#' Input tensor indicating the direction in which to move each value of `x1`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.nextafter
op_nextafter <-
function (x1, x2)
{
    args <- capture_args(list(x1 = as_array, x2 = as_array))
    do.call(ops$nextafter, args)
}


#' Compute percentiles along an axis.
#'
#' # Examples
#' ```{r}
#' op_percentile(c(1, 2, 3, 4), c(25, 75))
#' ```
#'
#' @returns
#' If `q` is a single percentile and `axis = NULL`, a scalar tensor. If `q`
#' contains multiple percentiles, the first result axis corresponds to `q`;
#' the other axes are those remaining after reduction of `x`.
#'
#' @inheritParams op_nanpercentile
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.percentile
op_percentile <-
function (x, q, axis = NULL, method = "linear", keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$percentile, args)
}


#' Compute the Moore-Penrose pseudoinverse of a matrix.
#'
#' The pseudoinverse is computed using singular value decomposition. Singular
#' values less than or equal to `rcond` times the largest singular value for
#' each matrix are treated as zero.
#'
#' # Examples
#' ```{r}
#' x <- op_array(matrix(1:6, nrow = 3), dtype = "float32")
#' op_matmul(op_pinv(x), x)
#' ```
#'
#' @returns
#' A tensor of shape `(..., N, M)` containing the pseudoinverse of each input
#' matrix of shape `(..., M, N)`.
#'
#' @param x
#' Input matrix or batch of matrices.
#'
#' @param rcond
#' Cutoff ratio for small singular values. Values less than or equal to
#' `rcond * largest_singular_value` are treated as zero. If `NULL`, the backend
#' default is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.pinv
op_pinv <-
function (x, rcond = NULL)
ops$pinv(x, rcond)


#' Compute the peak-to-peak range along an axis.
#'
#' The peak-to-peak range is the maximum minus the minimum.
#'
#' # Examples
#' ```{r}
#' x <- op_array(matrix(c(1, 3, 2, 4, 0, 5),
#'                      nrow = 2, byrow = TRUE))
#' op_ptp(x, axis = 2)
#' ```
#'
#' @returns
#' A tensor containing peak-to-peak ranges.
#'
#' @inheritParams op_nanmax
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.ptp
op_ptp <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$ptp, args)
}


#' Convert angles from radians to degrees.
#'
#' The conversion is `degrees = radians * (180 / pi)`.
#'
#' # Examples
#' ```{r}
#' op_rad2deg(c(0, pi / 2, pi))
#' ```
#'
#' @returns
#' A tensor containing angles in degrees.
#'
#' @param x
#' Input tensor containing angles in radians.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.rad2deg
op_rad2deg <-
function (x)
ops$rad2deg(x)


#' Compute the normalized sinc function.
#'
#' The normalized sinc function is `sin(pi * x) / (pi * x)` when `x != 0`
#' and one when `x == 0`. The value at zero is defined by this limit, making
#' the function continuous and infinitely differentiable.
#'
#' # Examples
#' ```{r}
#' op_sinc(c(0, 1, 2))
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.sinc
op_sinc <-
function (x)
ops$sinc(as_array(x))


#' Integrate using the composite trapezoidal rule.
#'
#' # Examples
#' ```{r}
#' y <- op_array(matrix(1:6, nrow = 2, byrow = TRUE))
#' op_trapezoid(y, axis = 2)
#' ```
#'
#' @returns
#' A tensor containing the approximate integral along `axis`.
#'
#' @param y
#' Input tensor.
#'
#' @param x
#' Optional sample points corresponding to `y`. If `NULL`, samples are spaced
#' by `dx`.
#'
#' @param dx
#' Spacing between sample points when `x` is `NULL`.
#'
#' @param axis
#' Axis along which to integrate. Defaults to the last axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.trapezoid
op_trapezoid <-
function (y, x = NULL, dx = 1, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(ops$trapezoid, args)
}


#' Find unique elements of a tensor.
#'
#' When `size` is supplied, the output shape is fixed, making the operation
#' compatible with JIT compilation such as JAX and TensorFlow graph mode.
#'
#' # Examples
#' ```{r}
#' op_unique(c(3L, 1L, 2L, 1L, 3L, 2L))
#'
#' c(values, first, inverse, counts) %<-% op_unique(
#'   c(3L, 1L, 2L, 1L, 3L, 2L),
#'   return_index = TRUE,
#'   return_inverse = TRUE,
#'   return_counts = TRUE
#' )
#' ```
#'
#' @returns
#' A tensor of unique values. If any `return_*` argument is `TRUE`, returns a
#' list containing the values followed, in argument order, by the requested
#' first-occurrence indices, inverse indices, and counts. Indices are one-based
#' unless `zero_indexed = TRUE`. When `size` is supplied and `axis = NULL`, the
#' unique-values tensor has shape `(size)`.
#'
#' @param x
#' Input tensor.
#'
#' @param sorted
#' Whether to sort unique elements in ascending order. Defaults to `TRUE`.
#'
#' @param return_index
#' Whether to return indices of the first occurrence of each unique element.
#' Defaults to `FALSE`.
#'
#' @param return_inverse
#' Whether to return indices of the unique values, or unique subarrays when
#' `axis` is supplied, that reconstruct `x`. Defaults to `FALSE`.
#'
#' @param return_counts
#' Whether to return the number of occurrences of each unique element. Defaults
#' to `FALSE`.
#'
#' @param axis
#' Axis along which subarrays are treated as elements. If `NULL`, `x` is
#' flattened. Defaults to `NULL`.
#'
#' @param size
#' Optional fixed number of unique values to return. If fewer values are found,
#' the output is padded with `fill_value`; if more are found, it is truncated.
#'
#' @param fill_value
#' Value used to pad results when `size` is larger than the number of unique
#' values. `NULL` uses the upstream default of zero.
#'
#' @param zero_indexed
#' Whether returned first-occurrence and inverse indices are zero-based.
#' Defaults to `FALSE`, returning indices suitable for R indexing.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.unique
op_unique <-
function (x, sorted = TRUE, return_index = FALSE, return_inverse = FALSE,
    return_counts = FALSE, axis = NULL, size = NULL, fill_value = NULL,
    zero_indexed = FALSE)
{
    args <- capture_args(
      list(axis = as_axis, size = as_integer),
      ignore = "zero_indexed"
    )
    result <- do.call(ops$unique, args)

    if (zero_indexed || (!return_index && !return_inverse))
      return(result)

    position <- 2L
    if (return_index) {
      result[[position]] <- result[[position]] + 1L
      position <- position + 1L
    }
    if (return_inverse)
      result[[position]] <- result[[position]] + 1L

    result
}


#' Generate a Vandermonde matrix.
#'
#' # Examples
#' ```{r}
#' op_vander(c(1, 2, 3), N = 3, increasing = TRUE)
#' ```
#'
#' @returns
#' A Vandermonde matrix with `length(x)` rows and `N` columns.
#'
#' @param x
#' One-dimensional input tensor.
#'
#' @param N
#' Number of columns. If `NULL`, defaults to `length(x)`.
#'
#' @param increasing
#' Whether powers increase from left to right.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.vander
op_vander <-
function (x, N = NULL, increasing = FALSE)
{
    args <- capture_args(list(x = as_array, N = as_integer))
    do.call(ops$vander, args)
}


#' Create a bitwise view of tensor data with another data type.
#'
#' Unlike [op_cast()], this function reinterprets the existing bytes rather
#' than converting values.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(1L, 2L), dtype = "int32")
#' op_view(x, "float32")
#' ```
#'
#' @returns
#' A view of `x` with data type `dtype`.
#'
#' @param x
#' Input tensor.
#'
#' @param dtype
#' Data type of the returned view. If `NULL`, returns a view with the input
#' data type.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.view
op_view <-
function (x, dtype = NULL)
ops$view(x, dtype)


#' Split an array vertically.
#'
#' # Examples
#' ```{r}
#' x <- op_reshape(op_arange(12), c(4, 3))
#' op_vsplit(x, 2)
#' op_vsplit(x, array(c(2L, 4L)))
#' ```
#'
#' @returns
#' A list of tensors split along the first axis.
#'
#' @inheritParams op_split
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.vsplit
op_vsplit <-
function (x, indices_or_sections)
{
    args <- capture_args(list(
        indices_or_sections = as_split_indices_or_sections))
    do.call(ops$vsplit, args)
}


#' Test whether two tensors are equal within a tolerance.
#'
#' @description
#' The absolute difference between `x1` and `x2` is compared with
#' `atol + rtol * abs(x2)`.
#'
#' # Examples
#' ```{r}
#' op_allclose(c(1, 2), c(1, 2 + 1e-6))
#' ```
#'
#' @returns A scalar boolean tensor.
#'
#' @param x1,x2 Input tensors.
#' @param rtol Positive relative tolerance, typically a small value.
#' @param atol Positive absolute tolerance, typically a small value.
#' @param equal_nan Whether `NaN` values in the same positions compare equal.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.allclose
op_allclose <-
function (x1, x2, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE)
ops$allclose(x1, x2, rtol, atol, equal_nan)


as_split_indices_or_sections <- function(x) {
  if (is.atomic(x)) {
    if (length(x) > 1L || is.array(x))
      return(as_py_index(x))
    return(as.integer(x))
  }

  if (inherits(x, "numpy.ndarray")) {
    if (as_r_value(x$ndim) > 0L)
      return(as_py_index(x))
    return(as.integer(x))
  }

  if (op_is_tensor(x)) {
    if (op_ndim(x) > 0L)
      return(as_py_index(x))
    return(x)
  }

  stop("`indices_or_sections` must be a scalar or 1-D tensor or array")
}


#' Split a tensor into possibly uneven chunks.
#'
#' @description
#' When the requested number of sections does not divide the selected axis
#' evenly, earlier sections contain one more element than later sections.
#'
#' # Examples
#' ```{r}
#' op_array_split(op_arange(10), 3)
#' ```
#'
#' @returns A list of tensors.
#'
#' @param x Input tensor.
#' @param indices_or_sections Number of sections to create.
#' @param axis Axis along which to split. Defaults to `1`, the first axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.array_split
op_array_split <-
function (x, indices_or_sections, axis = 1L)
{
    args <- capture_args(list(
        indices_or_sections = as_integer,
        axis = as_axis))
    do.call(ops$array_split, args)
}


#' Compute pairwise Euclidean distances between vectors.
#'
#' # Examples
#' ```{r}
#' x <- op_array(rbind(c(0, 0), c(3, 4)), dtype = "float32")
#' y <- op_array(rbind(c(0, 0)), dtype = "float32")
#' op_cdist(x, y)
#' ```
#'
#' @returns A tensor of shape `(..., m, n)` containing pairwise distances.
#'
#' @param x A tensor of shape `(..., m, d)`.
#' @param y A tensor of shape `(..., n, d)`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.cdist
op_cdist <-
function (x, y)
ops$cdist(x, y)


#' Compute the inverse of a symmetric positive-definite matrix from its
#' Cholesky factor.
#'
#' # Examples
#' ```{r}
#' factor <- op_array(diag(c(2, 3)), dtype = "float32")
#' op_cholesky_inverse(factor)
#' ```
#'
#' @returns A tensor of shape `(..., M, M)` containing the inverse of the
#'   matrix represented by `x`.
#'
#' @param x Lower- or upper-triangular Cholesky factor with shape
#'   `(..., M, M)`.
#' @param upper Whether `x` is upper triangular. If `FALSE`, `x` is treated as
#'   lower triangular.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.cholesky_inverse
op_cholesky_inverse <-
function (x, upper = FALSE)
ops$cholesky_inverse(x, upper)


#' Split a tensor depth-wise.
#'
#' # Examples
#' ```{r}
#' x <- op_reshape(op_arange(24), c(2, 3, 4))
#' op_dsplit(x, 2)
#' ```
#'
#' @returns A list of tensors.
#'
#' @param x Input tensor with at least three dimensions.
#' @param indices_or_sections If an integer `N`, split into `N` equal sections
#'   along the third axis. If a one-dimensional vector of sorted integers, its
#'   entries give the 1-based positions at which to split the third axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.dsplit
op_dsplit <-
function (x, indices_or_sections)
{
    args <- capture_args(list(
        indices_or_sections = as_split_indices_or_sections))
    do.call(ops$dsplit, args)
}


#' Stack tensors depth-wise along the third axis.
#'
#' This is equivalent to concatenation along the third axis after
#' two-dimensional tensors of shape `(M, N)` are reshaped to `(M, N, 1)` and
#' one-dimensional tensors of shape `(N)` are reshaped to `(1, N, 1)`.
#'
#' # Examples
#' ```{r}
#' op_dstack(list(op_array(1:3), op_array(4:6)))
#' ```
#'
#' @returns A tensor formed by stacking `xs` along the third axis.
#'
#' @param xs A list of tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.dstack
op_dstack <-
function (xs)
ops$dstack(xs)


#' Create an uninitialized tensor matching another tensor.
#'
#' # Examples
#' ```{r}
#' x <- op_ones(c(2, 3), dtype = "float32")
#' op_empty_like(x)
#' ```
#'
#' @returns A tensor with the shape and requested dtype of `x`. Its contents
#'   are arbitrary.
#'
#' @param x Input tensor whose shape and dtype are used.
#' @param dtype Optional output dtype. If `NULL`, the dtype of `x` is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.empty_like
op_empty_like <-
function (x, dtype = NULL)
ops$empty_like(x, dtype)


#' Compute the complementary error function element-wise.
#'
#' # Examples
#' ```{r}
#' op_erfc(c(-1, 0, 1))
#' ```
#'
#' @returns A tensor with the same dtype as `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.erfc
op_erfc <-
function (x)
ops$erfc(x)


#' Compute floating-point absolute values element-wise.
#'
#' @description
#' Integer and boolean inputs are promoted to the default floating-point dtype.
#' Complex values are not supported.
#'
#' # Examples
#' ```{r}
#' op_fabs(c(-1L, 2L))
#' ```
#'
#' @returns A tensor containing the absolute value of each element in `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.fabs
op_fabs <-
function (x)
ops$fabs(x)


#' Reverse the columns of a tensor.
#'
#' For a two-dimensional tensor, this reverses the entries in each row from
#' left to right. Columns are preserved but appear in reverse order.
#'
#' # Examples
#' ```{r}
#' op_fliplr(op_reshape(op_arange(6), c(2, 3)))
#' ```
#'
#' @returns A tensor with its columns reversed.
#'
#' @param x Input tensor with at least two dimensions.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.fliplr
op_fliplr <-
function (x)
ops$fliplr(x)


#' Reverse the rows of a tensor.
#'
#' For a two-dimensional tensor, this reverses the entries in each column from
#' top to bottom. Rows are preserved but appear in reverse order.
#'
#' # Examples
#' ```{r}
#' op_flipud(op_reshape(op_arange(6), c(2, 3)))
#' ```
#'
#' @returns A tensor with its rows reversed.
#'
#' @param x Input tensor with at least one dimension.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.flipud
op_flipud <-
function (x)
ops$flipud(x)


#' Compute the element-wise maximum while ignoring `NaN` values.
#'
#' If one compared element is `NaN`, the non-`NaN` element is returned. If both
#' elements are `NaN`, `NaN` is returned.
#'
#' # Examples
#' ```{r}
#' op_fmax(c(2, NaN), c(1, 4))
#' ```
#'
#' @returns A tensor containing the element-wise maximum of `x1` and `x2`.
#'
#' @param x1,x2 Input tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.fmax
op_fmax <-
function (x1, x2)
ops$fmax(x1, x2)


#' Compute the element-wise minimum while ignoring `NaN` values.
#'
#' If one compared element is `NaN`, the non-`NaN` element is returned. If both
#' elements are `NaN`, `NaN` is returned.
#'
#' # Examples
#' ```{r}
#' op_fmin(c(2, NaN), c(1, 4))
#' ```
#'
#' @returns A tensor containing the element-wise minimum of `x1` and `x2`.
#'
#' @param x1,x2 Input tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.fmin
op_fmin <-
function (x1, x2)
ops$fmin(x1, x2)


#' Compute the element-wise remainder using truncated division.
#'
#' @description
#' The result has the same sign as the dividend `x1`, unlike `op_mod()`, whose
#' result has the same sign as the divisor `x2`.
#'
#' # Examples
#' ```{r}
#' op_fmod(c(-5, 5), 3)
#' ```
#'
#' @returns A tensor containing the element-wise remainder.
#'
#' @param x1 Dividend tensor.
#' @param x2 Divisor tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.fmod
op_fmod <-
function (x1, x2)
ops$fmod(x1, x2)


#' Compute the greatest common divisor element-wise.
#'
#' # Examples
#' ```{r}
#' op_gcd(c(12L, 18L), c(8L, 12L))
#' ```
#'
#' @returns An integer tensor containing element-wise greatest common divisors.
#'
#' @param x1,x2 Integer input tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.gcd
op_gcd <-
function (x1, x2)
{
    args <- capture_args(list(x1 = as_integer, x2 = as_integer))
    do.call(ops$gcd, args)
}


#' Return values spaced evenly on a logarithmic scale.
#'
#' @description
#' The endpoints are specified directly. Each output sample is a constant
#' multiple of the previous sample.
#'
#' # Examples
#' ```{r}
#' op_geomspace(1, 1000, num = 4)
#' ```
#'
#' @returns A tensor containing `num` samples on a logarithmic scale.
#'
#' @param start Starting value.
#' @param stop Final value when `endpoint = TRUE`. When `endpoint = FALSE`,
#'   `num + 1` values are spaced over the interval in log space and the final
#'   value is omitted.
#' @param num Number of samples. Defaults to `50`.
#' @param endpoint Whether `stop` is included. Defaults to `TRUE`.
#' @param dtype Optional output dtype.
#' @param axis Axis in the result along which samples are stored. This matters
#'   only when `start` or `stop` is array-like. Defaults to `1`, the first axis.
#'   The Torch backend does not support this argument.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.geomspace
op_geomspace <-
function (start, stop, num = 50L, endpoint = TRUE, dtype = NULL,
    axis = 1L)
{
    args <- capture_args(list(num = as_integer, axis = as_axis))
    do.call(ops$geomspace, args)
}


#' Split a tensor horizontally.
#'
#' # Examples
#' ```{r}
#' x <- op_reshape(op_arange(16), c(4, 4))
#' op_hsplit(x, 2)
#' ```
#'
#' @returns A list of tensors.
#'
#' @param x Input tensor.
#' @param indices_or_sections If an integer `N`, split into `N` equal sections
#'   along the second axis, or the first axis for a one-dimensional tensor. If
#'   a one-dimensional vector of sorted integers, its entries give the 1-based
#'   positions at which to split that axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.hsplit
op_hsplit <-
function (x, indices_or_sections)
{
    args <- capture_args(list(
        indices_or_sections = as_split_indices_or_sections))
    do.call(ops$hsplit, args)
}


#' Compute hypotenuses element-wise.
#'
#' @description
#' This is equivalent to `sqrt(x1^2 + x2^2)`, with broadcasting.
#'
#' # Examples
#' ```{r}
#' op_hypot(c(3, 5), c(4, 12))
#' ```
#'
#' @returns A tensor with shape determined by broadcasting `x1` and `x2`.
#'
#' @param x1,x2 Input tensors representing the legs of right triangles.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.hypot
op_hypot <-
function (x1, x2)
ops$hypot(x1, x2)


#' Compute the modified Bessel function of the first kind, order zero.
#'
#' # Examples
#' ```{r}
#' op_i0(c(0, 1))
#' ```
#'
#' @returns A tensor containing the element-wise result.
#'
#' @param x Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.i0
op_i0 <-
function (x)
ops$i0(x)


#' Test whether elements of one tensor occur in another.
#'
#' # Examples
#' ```{r}
#' op_isin(c(0L, 1L, 2L), c(0L, 2L))
#' ```
#'
#' @returns A boolean tensor with the same shape as `x1`.
#'
#' @param x1 Values to test.
#' @param x2 Values against which to test each element of `x1`. May be a tensor,
#'   list, or scalar.
#' @param assume_unique Whether both inputs can be assumed to contain unique
#'   elements. `TRUE` can improve performance; `FALSE` handles duplicates
#'   correctly. Defaults to `FALSE`.
#' @param invert Whether to invert the result, returning `TRUE` where elements
#'   of `x1` are absent from `x2`. Defaults to `FALSE`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.isin
op_isin <-
function (x1, x2, assume_unique = FALSE, invert = FALSE)
ops$isin(x1, x2, assume_unique, invert)


#' Test element-wise for negative infinity.
#'
#' # Examples
#' ```{r}
#' op_isneginf(c(-Inf, Inf, 0))
#' ```
#'
#' @returns A boolean tensor with the same shape as `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.isneginf
op_isneginf <-
function (x)
ops$isneginf(x)


#' Test element-wise for positive infinity.
#'
#' # Examples
#' ```{r}
#' op_isposinf(c(-Inf, Inf, 0))
#' ```
#'
#' @returns A boolean tensor with the same shape as `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.isposinf
op_isposinf <-
function (x)
ops$isposinf(x)


#' Test element-wise for real numbers.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(1 + 0i, 1 + 1i), dtype = "complex64")
#' op_isreal(x)
#' ```
#'
#' @returns A boolean tensor with the same shape as `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.isreal
op_isreal <-
function (x)
ops$isreal(x)


#' Compute the Kronecker product of two tensors.
#'
#' If `x1` has shape `(a1, ..., an)` and `x2` has shape `(b1, ..., bn)`, the
#' result has shape `(a1 * b1, ..., an * bn)`.
#'
#' # Examples
#' ```{r}
#' op_kron(c(1, 2), c(3, 4))
#' ```
#'
#' @returns A tensor containing the Kronecker product of `x1` and `x2`.
#'
#' @param x1,x2 Input tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.kron
op_kron <-
function (x1, x2)
ops$kron(x1, x2)


#' Compute the least common multiple element-wise.
#'
#' # Examples
#' ```{r}
#' op_lcm(c(2L, 3L, 4L), c(5L, 6L, 7L))
#' ```
#'
#' @returns An integer tensor containing element-wise least common multiples.
#'
#' @param x1,x2 Integer input tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.lcm
op_lcm <-
function (x1, x2)
{
    args <- capture_args(list(x1 = as_integer, x2 = as_integer))
    do.call(ops$lcm, args)
}


#' Multiply by an integer power of two element-wise.
#'
#' @description
#' Computes `x1 * 2^x2`.
#'
#' # Examples
#' ```{r}
#' op_ldexp(c(0.75, 1.5), c(1L, 2L))
#' ```
#'
#' @returns A tensor containing the element-wise result.
#'
#' @param x1 Floating-point input tensor.
#' @param x2 Integer exponent tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.ldexp
op_ldexp <-
function (x1, x2)
{
    args <- capture_args(list(x2 = as_integer))
    do.call(ops$ldexp, args)
}


#' Compute the base-two logarithm of summed exponentials.
#'
#' @description
#' Computes `log2(2^x1 + 2^x2)` element-wise.
#'
#' # Examples
#' ```{r}
#' op_logaddexp2(c(1, 2), c(1, 2))
#' ```
#'
#' @returns A tensor containing the element-wise result.
#'
#' @param x1,x2 Input tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.logaddexp2
op_logaddexp2 <-
function (x1, x2)
ops$logaddexp2(x1, x2)


#' Compute matrix rank using singular value decomposition.
#'
#' The rank is the number of singular values greater than `tol`. When `tol` is
#' `NULL`, each backend derives its default threshold from the largest singular
#' value and the matrix dimensions.
#'
#' # Examples
#' ```{r}
#' x <- op_array(rbind(c(1, 2), c(2, 4)), dtype = "float32")
#' op_matrix_rank(x)
#' ```
#'
#' @returns An integer tensor of shape `(...)` containing each matrix rank.
#'
#' @param x Input tensor of shape `(..., M, N)`.
#' @param tol Optional absolute threshold below which singular values are
#'   treated as zero. If `NULL`, the backend default described above is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.matrix_rank
op_matrix_rank <-
function (x, tol = NULL)
ops$matrix_rank(x, tol)
