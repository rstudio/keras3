# Subset elements from a tensor

Extract elements from a tensor using common R-style `[` indexing idioms.
This function can also be conveniently accessed via the syntax
`tensor@r[...]`.

## Usage

``` r
op_subset(x, ...)

op_subset(x, ...) <- value

op_subset_set(x, ..., value)
```

## Arguments

- x:

  Input tensor.

- ...:

  Indices specifying elements to extract. Each argument in `...` can be:

  - An integer scalar

  - A 1-d integer or logical vector

  - `NULL` or `newaxis`

  - The `..` symbol

  - A slice expression using `:`

  If only a single argument is supplied to `...`, then `..1` can also
  be:

  - A logical array with the same shape as `x`

  - An integer matrix where `ncol(..1) == op_rank(x)`

- value:

  new value to replace the selected subset with.

## Value

A tensor containing the subset of elements.

## Details

While the semantics are similar to R's `[`, there are some differences:

## Differences from R's `[`:

- Negative indices follow Python-style indexing, counting from the end
  of the array.

- `NULL` or `newaxis` adds a new dimension (equivalent to
  [`op_expand_dims()`](https://keras3.posit.co/dev/reference/op_expand_dims.md)).

- If fewer indices than dimensions (`op_rank(x)`) are provided, missing
  dimensions are implicitly filled. For example, if `x` is a matrix,
  `x[1]` returns the first row.

- `..` or
  [`all_dims()`](https://rdrr.io/pkg/tensorflow/man/all_dims.html)
  expands to include all unspecified dimensions (see examples).

- Extended slicing syntax (`:`) is supported, including:

  - Strided steps: `x@r[start:end:step]`

  - `NA` values for `start` and `end`. `NA` for `start` defaults to `1`,
    and `NA` for `end` defaults to the axis size.

- A logical array matching the shape of `x` selects elements in row-wise
  order.

## Similarities with R's `[`:

Similarities to R's `[` (differences from Python's `[`):

- Positive indices are 1-based.

- Slices (`x[start:end]`) are inclusive of `end`.

- 1-d logical/integer arrays subset along their respective axis.
  Multiple vectors provided for different axes return intersected
  subsets.

- A single integer matrix with `ncol(i) == op_rank(x)` selects elements
  by coordinates. Each row in the matrix specifies the location of one
  value, where each column corresponds to an axis in the tensor being
  subsetted. This means you use a 2-column matrix to subset a matrix, a
  3-column matrix to subset a 3d array, and so on.

## Examples

    (x <- op_arange(5L) + 10L)

    ## tf.Tensor([11 12 13 14 15], shape=(5), dtype=int32)

    # Basic example, get first element
    op_subset(x, 1)

    ## tf.Tensor(11, shape=(), dtype=int32)

    # Use `@r[` syntax
    x@r[1]           # same as `op_subset(x, 1)`

    ## tf.Tensor(11, shape=(), dtype=int32)

    x@r[1:2]         # get the first 2 elements

    ## tf.Tensor([11 12], shape=(2), dtype=int32)

    x@r[c(1, 3)]     # first and third element

    ## tf.Tensor([11 13], shape=(2), dtype=int32)

    # Negative indices
    x@r[-1]          # last element

    ## tf.Tensor(15, shape=(), dtype=int32)

    x@r[-2]          # second to last element

    ## tf.Tensor(14, shape=(), dtype=int32)

    x@r[c(-1, -2)]   # last and second to last elements

    ## tf.Tensor([15 14], shape=(2), dtype=int32)

    x@r[c(-2, -1)]   # second to last and last elements

    ## tf.Tensor([14 15], shape=(2), dtype=int32)

    x@r[c(1, -1)]    # first and last elements

    ## tf.Tensor([11 15], shape=(2), dtype=int32)

    # Slices
    x@r[1:3]          # first 3 elements

    ## tf.Tensor([11 12 13], shape=(3), dtype=int32)

    x@r[NA:3]         # first 3 elements

    ## tf.Tensor([11 12 13], shape=(3), dtype=int32)

    x@r[1:5]          # all elements

    ## tf.Tensor([11 12 13 14 15], shape=(5), dtype=int32)

    x@r[1:-1]         # all elements

    ## tf.Tensor([11 12 13 14 15], shape=(5), dtype=int32)

    x@r[NA:NA]        # all elements

    ## tf.Tensor([11 12 13 14 15], shape=(5), dtype=int32)

    x@r[]             # all elements

    ## tf.Tensor([11 12 13 14 15], shape=(5), dtype=int32)

    x@r[1:-2]         # drop last element

    ## tf.Tensor([11 12 13 14], shape=(4), dtype=int32)

    x@r[NA:-2]        # drop last element

    ## tf.Tensor([11 12 13 14], shape=(4), dtype=int32)

    x@r[2:NA]         # drop first element

    ## tf.Tensor([12 13 14 15], shape=(4), dtype=int32)

    # 2D array examples
    xr <- array(1:12, c(3, 4))
    x <- op_convert_to_tensor(xr)

    # Basic subsetting
    x@r[1, ]      # first row

    ## tf.Tensor([ 1  4  7 10], shape=(4), dtype=int32)

    x@r[1]        # also first row! Missing axes are implicitly inserted

    ## tf.Tensor([ 1  4  7 10], shape=(4), dtype=int32)

    x@r[-1]       # last row

    ## tf.Tensor([ 3  6  9 12], shape=(4), dtype=int32)

    x@r[, 2]      # second column

    ## tf.Tensor([4 5 6], shape=(3), dtype=int32)

    x@r[, 2:2]    # second column, but shape preserved (like [, drop=FALSE])

    ## tf.Tensor(
    ## [[4]
    ##  [5]
    ##  [6]], shape=(3, 1), dtype=int32)

    # Subsetting with a boolean array
    # Note: extracted elements are selected row-wise, not column-wise
    mask <- x >= 6
    x@r[mask]             # returns a 1D tensor

    ## tf.Tensor([ 7 10  8 11  6  9 12], shape=(7), dtype=int32)

    x.r <- as.array(x)
    mask.r <- as.array(mask)
    # as.array(x)[mask] selects column-wise. Use `aperm()` to reverse search order.
    all(aperm(x.r)[aperm(mask.r)] == as.array(x@r[mask]))

    ## [1] TRUE

    # Subsetting with a matrix of index positions
    indices <- rbind(c(1, 1), c(2, 2), c(3, 3))
    x@r[indices] # get diagonal elements

    ## tf.Tensor([1 5 9], shape=(3), dtype=int32)

    x.r[indices] # same as subsetting an R array

    ## [1] 1 5 9

    # 3D array examples
    # Image: 4x4 pixels, 3 colors (RGB)
    # Tensor shape: (img_height, img_width, img_color_channels)
    shp <- shape(4, 4, 3)
    x <- op_arange(prod(shp)) |> op_reshape(shp)

    # Convert to a batch of images by inserting a new axis
    # New shape: (batch_size, img_height, img_width, img_color_channels)
    x@r[newaxis, , , ] |> op_shape()

    ## shape(1, 4, 4, 3)

    x@r[newaxis] |> op_shape()  # same as above

    ## shape(1, 4, 4, 3)

    x@r[NULL] |> op_shape()     # same as above

    ## shape(1, 4, 4, 3)

    x <- x@r[newaxis]
    # Extract color channels
    x@r[, , , 1]          # red channel

    ## tf.Tensor(
    ## [[[ 1.  4.  7. 10.]
    ##   [13. 16. 19. 22.]
    ##   [25. 28. 31. 34.]
    ##   [37. 40. 43. 46.]]], shape=(1, 4, 4), dtype=float32)

    x@r[.., 1]            # red channel, same as above using .. shorthand

    ## tf.Tensor(
    ## [[[ 1.  4.  7. 10.]
    ##   [13. 16. 19. 22.]
    ##   [25. 28. 31. 34.]
    ##   [37. 40. 43. 46.]]], shape=(1, 4, 4), dtype=float32)

    x@r[.., 2]            # green channel

    ## tf.Tensor(
    ## [[[ 2.  5.  8. 11.]
    ##   [14. 17. 20. 23.]
    ##   [26. 29. 32. 35.]
    ##   [38. 41. 44. 47.]]], shape=(1, 4, 4), dtype=float32)

    x@r[.., 3]            # blue channel

    ## tf.Tensor(
    ## [[[ 3.  6.  9. 12.]
    ##   [15. 18. 21. 24.]
    ##   [27. 30. 33. 36.]
    ##   [39. 42. 45. 48.]]], shape=(1, 4, 4), dtype=float32)

    # .. expands to all unspecified axes.
    op_shape(x@r[])

    ## shape(1, 4, 4, 3)

    op_shape(x@r[..])

    ## shape(1, 4, 4, 3)

    op_shape(x@r[1, ..])

    ## shape(4, 4, 3)

    op_shape(x@r[1, .., 1, 1])

    ## shape(4)

    op_shape(x@r[1, 1, 1, .., 1])

    ## shape()

    # op_subset<- uses the same semantics, but note that not all tensors
    # support modification. E.g., TensorFlow constant tensors cannot be modified,
    # while TensorFlow Variables can be.

    (x <- tensorflow::tf$Variable(matrix(1, nrow = 2, ncol = 3)))

    ## <tf.Variable 'Variable:0' shape=(2, 3) dtype=float64, numpy=
    ## array([[1., 1., 1.],
    ##        [1., 1., 1.]])>

    op_subset(x, 1) <- 9
    x

    ## <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float64, numpy=
    ## array([[9., 9., 9.],
    ##        [1., 1., 1.]])>

    x@r[1,1] <- 33
    x

    ## <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float64, numpy=
    ## array([[33.,  9.,  9.],
    ##        [ 1.,  1.,  1.]])>

## See also

Other core ops:  
[`op_associative_scan()`](https://keras3.posit.co/dev/reference/op_associative_scan.md)  
[`op_cast()`](https://keras3.posit.co/dev/reference/op_cast.md)  
[`op_cond()`](https://keras3.posit.co/dev/reference/op_cond.md)  
[`op_convert_to_numpy()`](https://keras3.posit.co/dev/reference/op_convert_to_numpy.md)  
[`op_convert_to_tensor()`](https://keras3.posit.co/dev/reference/op_convert_to_tensor.md)  
[`op_custom_gradient()`](https://keras3.posit.co/dev/reference/op_custom_gradient.md)  
[`op_dtype()`](https://keras3.posit.co/dev/reference/op_dtype.md)  
[`op_fori_loop()`](https://keras3.posit.co/dev/reference/op_fori_loop.md)  
[`op_is_tensor()`](https://keras3.posit.co/dev/reference/op_is_tensor.md)  
[`op_map()`](https://keras3.posit.co/dev/reference/op_map.md)  
[`op_rearrange()`](https://keras3.posit.co/dev/reference/op_rearrange.md)  
[`op_scan()`](https://keras3.posit.co/dev/reference/op_scan.md)  
[`op_scatter()`](https://keras3.posit.co/dev/reference/op_scatter.md)  
[`op_scatter_update()`](https://keras3.posit.co/dev/reference/op_scatter_update.md)  
[`op_searchsorted()`](https://keras3.posit.co/dev/reference/op_searchsorted.md)  
[`op_shape()`](https://keras3.posit.co/dev/reference/op_shape.md)  
[`op_slice()`](https://keras3.posit.co/dev/reference/op_slice.md)  
[`op_slice_update()`](https://keras3.posit.co/dev/reference/op_slice_update.md)  
[`op_stop_gradient()`](https://keras3.posit.co/dev/reference/op_stop_gradient.md)  
[`op_switch()`](https://keras3.posit.co/dev/reference/op_switch.md)  
[`op_unstack()`](https://keras3.posit.co/dev/reference/op_unstack.md)  
[`op_vectorized_map()`](https://keras3.posit.co/dev/reference/op_vectorized_map.md)  
[`op_while_loop()`](https://keras3.posit.co/dev/reference/op_while_loop.md)  

Other ops:  
[`op_abs()`](https://keras3.posit.co/dev/reference/op_abs.md)  
[`op_add()`](https://keras3.posit.co/dev/reference/op_add.md)  
[`op_all()`](https://keras3.posit.co/dev/reference/op_all.md)  
[`op_angle()`](https://keras3.posit.co/dev/reference/op_angle.md)  
[`op_any()`](https://keras3.posit.co/dev/reference/op_any.md)  
[`op_append()`](https://keras3.posit.co/dev/reference/op_append.md)  
[`op_arange()`](https://keras3.posit.co/dev/reference/op_arange.md)  
[`op_arccos()`](https://keras3.posit.co/dev/reference/op_arccos.md)  
[`op_arccosh()`](https://keras3.posit.co/dev/reference/op_arccosh.md)  
[`op_arcsin()`](https://keras3.posit.co/dev/reference/op_arcsin.md)  
[`op_arcsinh()`](https://keras3.posit.co/dev/reference/op_arcsinh.md)  
[`op_arctan()`](https://keras3.posit.co/dev/reference/op_arctan.md)  
[`op_arctan2()`](https://keras3.posit.co/dev/reference/op_arctan2.md)  
[`op_arctanh()`](https://keras3.posit.co/dev/reference/op_arctanh.md)  
[`op_argmax()`](https://keras3.posit.co/dev/reference/op_argmax.md)  
[`op_argmin()`](https://keras3.posit.co/dev/reference/op_argmin.md)  
[`op_argpartition()`](https://keras3.posit.co/dev/reference/op_argpartition.md)  
[`op_argsort()`](https://keras3.posit.co/dev/reference/op_argsort.md)  
[`op_array()`](https://keras3.posit.co/dev/reference/op_array.md)  
[`op_associative_scan()`](https://keras3.posit.co/dev/reference/op_associative_scan.md)  
[`op_average()`](https://keras3.posit.co/dev/reference/op_average.md)  
[`op_average_pool()`](https://keras3.posit.co/dev/reference/op_average_pool.md)  
[`op_bartlett()`](https://keras3.posit.co/dev/reference/op_bartlett.md)  
[`op_batch_normalization()`](https://keras3.posit.co/dev/reference/op_batch_normalization.md)  
[`op_binary_crossentropy()`](https://keras3.posit.co/dev/reference/op_binary_crossentropy.md)  
[`op_bincount()`](https://keras3.posit.co/dev/reference/op_bincount.md)  
[`op_bitwise_and()`](https://keras3.posit.co/dev/reference/op_bitwise_and.md)  
[`op_bitwise_invert()`](https://keras3.posit.co/dev/reference/op_bitwise_invert.md)  
[`op_bitwise_left_shift()`](https://keras3.posit.co/dev/reference/op_bitwise_left_shift.md)  
[`op_bitwise_not()`](https://keras3.posit.co/dev/reference/op_bitwise_not.md)  
[`op_bitwise_or()`](https://keras3.posit.co/dev/reference/op_bitwise_or.md)  
[`op_bitwise_right_shift()`](https://keras3.posit.co/dev/reference/op_bitwise_right_shift.md)  
[`op_bitwise_xor()`](https://keras3.posit.co/dev/reference/op_bitwise_xor.md)  
[`op_blackman()`](https://keras3.posit.co/dev/reference/op_blackman.md)  
[`op_broadcast_to()`](https://keras3.posit.co/dev/reference/op_broadcast_to.md)  
[`op_cast()`](https://keras3.posit.co/dev/reference/op_cast.md)  
[`op_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/op_categorical_crossentropy.md)  
[`op_cbrt()`](https://keras3.posit.co/dev/reference/op_cbrt.md)  
[`op_ceil()`](https://keras3.posit.co/dev/reference/op_ceil.md)  
[`op_celu()`](https://keras3.posit.co/dev/reference/op_celu.md)  
[`op_cholesky()`](https://keras3.posit.co/dev/reference/op_cholesky.md)  
[`op_clip()`](https://keras3.posit.co/dev/reference/op_clip.md)  
[`op_concatenate()`](https://keras3.posit.co/dev/reference/op_concatenate.md)  
[`op_cond()`](https://keras3.posit.co/dev/reference/op_cond.md)  
[`op_conj()`](https://keras3.posit.co/dev/reference/op_conj.md)  
[`op_conv()`](https://keras3.posit.co/dev/reference/op_conv.md)  
[`op_conv_transpose()`](https://keras3.posit.co/dev/reference/op_conv_transpose.md)  
[`op_convert_to_numpy()`](https://keras3.posit.co/dev/reference/op_convert_to_numpy.md)  
[`op_convert_to_tensor()`](https://keras3.posit.co/dev/reference/op_convert_to_tensor.md)  
[`op_copy()`](https://keras3.posit.co/dev/reference/op_copy.md)  
[`op_corrcoef()`](https://keras3.posit.co/dev/reference/op_corrcoef.md)  
[`op_correlate()`](https://keras3.posit.co/dev/reference/op_correlate.md)  
[`op_cos()`](https://keras3.posit.co/dev/reference/op_cos.md)  
[`op_cosh()`](https://keras3.posit.co/dev/reference/op_cosh.md)  
[`op_count_nonzero()`](https://keras3.posit.co/dev/reference/op_count_nonzero.md)  
[`op_cross()`](https://keras3.posit.co/dev/reference/op_cross.md)  
[`op_ctc_decode()`](https://keras3.posit.co/dev/reference/op_ctc_decode.md)  
[`op_ctc_loss()`](https://keras3.posit.co/dev/reference/op_ctc_loss.md)  
[`op_cumprod()`](https://keras3.posit.co/dev/reference/op_cumprod.md)  
[`op_cumsum()`](https://keras3.posit.co/dev/reference/op_cumsum.md)  
[`op_custom_gradient()`](https://keras3.posit.co/dev/reference/op_custom_gradient.md)  
[`op_deg2rad()`](https://keras3.posit.co/dev/reference/op_deg2rad.md)  
[`op_depthwise_conv()`](https://keras3.posit.co/dev/reference/op_depthwise_conv.md)  
[`op_det()`](https://keras3.posit.co/dev/reference/op_det.md)  
[`op_diag()`](https://keras3.posit.co/dev/reference/op_diag.md)  
[`op_diagflat()`](https://keras3.posit.co/dev/reference/op_diagflat.md)  
[`op_diagonal()`](https://keras3.posit.co/dev/reference/op_diagonal.md)  
[`op_diff()`](https://keras3.posit.co/dev/reference/op_diff.md)  
[`op_digitize()`](https://keras3.posit.co/dev/reference/op_digitize.md)  
[`op_divide()`](https://keras3.posit.co/dev/reference/op_divide.md)  
[`op_divide_no_nan()`](https://keras3.posit.co/dev/reference/op_divide_no_nan.md)  
[`op_dot()`](https://keras3.posit.co/dev/reference/op_dot.md)  
[`op_dot_product_attention()`](https://keras3.posit.co/dev/reference/op_dot_product_attention.md)  
[`op_dtype()`](https://keras3.posit.co/dev/reference/op_dtype.md)  
[`op_eig()`](https://keras3.posit.co/dev/reference/op_eig.md)  
[`op_eigh()`](https://keras3.posit.co/dev/reference/op_eigh.md)  
[`op_einsum()`](https://keras3.posit.co/dev/reference/op_einsum.md)  
[`op_elu()`](https://keras3.posit.co/dev/reference/op_elu.md)  
[`op_empty()`](https://keras3.posit.co/dev/reference/op_empty.md)  
[`op_equal()`](https://keras3.posit.co/dev/reference/op_equal.md)  
[`op_erf()`](https://keras3.posit.co/dev/reference/op_erf.md)  
[`op_erfinv()`](https://keras3.posit.co/dev/reference/op_erfinv.md)  
[`op_exp()`](https://keras3.posit.co/dev/reference/op_exp.md)  
[`op_exp2()`](https://keras3.posit.co/dev/reference/op_exp2.md)  
[`op_expand_dims()`](https://keras3.posit.co/dev/reference/op_expand_dims.md)  
[`op_expm1()`](https://keras3.posit.co/dev/reference/op_expm1.md)  
[`op_extract_sequences()`](https://keras3.posit.co/dev/reference/op_extract_sequences.md)  
[`op_eye()`](https://keras3.posit.co/dev/reference/op_eye.md)  
[`op_fft()`](https://keras3.posit.co/dev/reference/op_fft.md)  
[`op_fft2()`](https://keras3.posit.co/dev/reference/op_fft2.md)  
[`op_flip()`](https://keras3.posit.co/dev/reference/op_flip.md)  
[`op_floor()`](https://keras3.posit.co/dev/reference/op_floor.md)  
[`op_floor_divide()`](https://keras3.posit.co/dev/reference/op_floor_divide.md)  
[`op_fori_loop()`](https://keras3.posit.co/dev/reference/op_fori_loop.md)  
[`op_full()`](https://keras3.posit.co/dev/reference/op_full.md)  
[`op_full_like()`](https://keras3.posit.co/dev/reference/op_full_like.md)  
[`op_gelu()`](https://keras3.posit.co/dev/reference/op_gelu.md)  
[`op_get_item()`](https://keras3.posit.co/dev/reference/op_get_item.md)  
[`op_glu()`](https://keras3.posit.co/dev/reference/op_glu.md)  
[`op_greater()`](https://keras3.posit.co/dev/reference/op_greater.md)  
[`op_greater_equal()`](https://keras3.posit.co/dev/reference/op_greater_equal.md)  
[`op_hamming()`](https://keras3.posit.co/dev/reference/op_hamming.md)  
[`op_hanning()`](https://keras3.posit.co/dev/reference/op_hanning.md)  
[`op_hard_shrink()`](https://keras3.posit.co/dev/reference/op_hard_shrink.md)  
[`op_hard_sigmoid()`](https://keras3.posit.co/dev/reference/op_hard_sigmoid.md)  
[`op_hard_silu()`](https://keras3.posit.co/dev/reference/op_hard_silu.md)  
[`op_hard_tanh()`](https://keras3.posit.co/dev/reference/op_hard_tanh.md)  
[`op_heaviside()`](https://keras3.posit.co/dev/reference/op_heaviside.md)  
[`op_histogram()`](https://keras3.posit.co/dev/reference/op_histogram.md)  
[`op_hstack()`](https://keras3.posit.co/dev/reference/op_hstack.md)  
[`op_identity()`](https://keras3.posit.co/dev/reference/op_identity.md)  
[`op_ifft2()`](https://keras3.posit.co/dev/reference/op_ifft2.md)  
[`op_imag()`](https://keras3.posit.co/dev/reference/op_imag.md)  
[`op_image_affine_transform()`](https://keras3.posit.co/dev/reference/op_image_affine_transform.md)  
[`op_image_crop()`](https://keras3.posit.co/dev/reference/op_image_crop.md)  
[`op_image_extract_patches()`](https://keras3.posit.co/dev/reference/op_image_extract_patches.md)  
[`op_image_gaussian_blur()`](https://keras3.posit.co/dev/reference/op_image_gaussian_blur.md)  
[`op_image_hsv_to_rgb()`](https://keras3.posit.co/dev/reference/op_image_hsv_to_rgb.md)  
[`op_image_map_coordinates()`](https://keras3.posit.co/dev/reference/op_image_map_coordinates.md)  
[`op_image_pad()`](https://keras3.posit.co/dev/reference/op_image_pad.md)  
[`op_image_perspective_transform()`](https://keras3.posit.co/dev/reference/op_image_perspective_transform.md)  
[`op_image_resize()`](https://keras3.posit.co/dev/reference/op_image_resize.md)  
[`op_image_rgb_to_grayscale()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_grayscale.md)  
[`op_image_rgb_to_hsv()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_hsv.md)  
[`op_in_top_k()`](https://keras3.posit.co/dev/reference/op_in_top_k.md)  
[`op_inner()`](https://keras3.posit.co/dev/reference/op_inner.md)  
[`op_inv()`](https://keras3.posit.co/dev/reference/op_inv.md)  
[`op_irfft()`](https://keras3.posit.co/dev/reference/op_irfft.md)  
[`op_is_tensor()`](https://keras3.posit.co/dev/reference/op_is_tensor.md)  
[`op_isclose()`](https://keras3.posit.co/dev/reference/op_isclose.md)  
[`op_isfinite()`](https://keras3.posit.co/dev/reference/op_isfinite.md)  
[`op_isinf()`](https://keras3.posit.co/dev/reference/op_isinf.md)  
[`op_isnan()`](https://keras3.posit.co/dev/reference/op_isnan.md)  
[`op_istft()`](https://keras3.posit.co/dev/reference/op_istft.md)  
[`op_kaiser()`](https://keras3.posit.co/dev/reference/op_kaiser.md)  
[`op_layer_normalization()`](https://keras3.posit.co/dev/reference/op_layer_normalization.md)  
[`op_leaky_relu()`](https://keras3.posit.co/dev/reference/op_leaky_relu.md)  
[`op_left_shift()`](https://keras3.posit.co/dev/reference/op_left_shift.md)  
[`op_less()`](https://keras3.posit.co/dev/reference/op_less.md)  
[`op_less_equal()`](https://keras3.posit.co/dev/reference/op_less_equal.md)  
[`op_linspace()`](https://keras3.posit.co/dev/reference/op_linspace.md)  
[`op_log()`](https://keras3.posit.co/dev/reference/op_log.md)  
[`op_log10()`](https://keras3.posit.co/dev/reference/op_log10.md)  
[`op_log1p()`](https://keras3.posit.co/dev/reference/op_log1p.md)  
[`op_log2()`](https://keras3.posit.co/dev/reference/op_log2.md)  
[`op_log_sigmoid()`](https://keras3.posit.co/dev/reference/op_log_sigmoid.md)  
[`op_log_softmax()`](https://keras3.posit.co/dev/reference/op_log_softmax.md)  
[`op_logaddexp()`](https://keras3.posit.co/dev/reference/op_logaddexp.md)  
[`op_logdet()`](https://keras3.posit.co/dev/reference/op_logdet.md)  
[`op_logical_and()`](https://keras3.posit.co/dev/reference/op_logical_and.md)  
[`op_logical_not()`](https://keras3.posit.co/dev/reference/op_logical_not.md)  
[`op_logical_or()`](https://keras3.posit.co/dev/reference/op_logical_or.md)  
[`op_logical_xor()`](https://keras3.posit.co/dev/reference/op_logical_xor.md)  
[`op_logspace()`](https://keras3.posit.co/dev/reference/op_logspace.md)  
[`op_logsumexp()`](https://keras3.posit.co/dev/reference/op_logsumexp.md)  
[`op_lstsq()`](https://keras3.posit.co/dev/reference/op_lstsq.md)  
[`op_lu_factor()`](https://keras3.posit.co/dev/reference/op_lu_factor.md)  
[`op_map()`](https://keras3.posit.co/dev/reference/op_map.md)  
[`op_matmul()`](https://keras3.posit.co/dev/reference/op_matmul.md)  
[`op_max()`](https://keras3.posit.co/dev/reference/op_max.md)  
[`op_max_pool()`](https://keras3.posit.co/dev/reference/op_max_pool.md)  
[`op_maximum()`](https://keras3.posit.co/dev/reference/op_maximum.md)  
[`op_mean()`](https://keras3.posit.co/dev/reference/op_mean.md)  
[`op_median()`](https://keras3.posit.co/dev/reference/op_median.md)  
[`op_meshgrid()`](https://keras3.posit.co/dev/reference/op_meshgrid.md)  
[`op_min()`](https://keras3.posit.co/dev/reference/op_min.md)  
[`op_minimum()`](https://keras3.posit.co/dev/reference/op_minimum.md)  
[`op_mod()`](https://keras3.posit.co/dev/reference/op_mod.md)  
[`op_moments()`](https://keras3.posit.co/dev/reference/op_moments.md)  
[`op_moveaxis()`](https://keras3.posit.co/dev/reference/op_moveaxis.md)  
[`op_multi_hot()`](https://keras3.posit.co/dev/reference/op_multi_hot.md)  
[`op_multiply()`](https://keras3.posit.co/dev/reference/op_multiply.md)  
[`op_nan_to_num()`](https://keras3.posit.co/dev/reference/op_nan_to_num.md)  
[`op_ndim()`](https://keras3.posit.co/dev/reference/op_ndim.md)  
[`op_negative()`](https://keras3.posit.co/dev/reference/op_negative.md)  
[`op_nonzero()`](https://keras3.posit.co/dev/reference/op_nonzero.md)  
[`op_norm()`](https://keras3.posit.co/dev/reference/op_norm.md)  
[`op_normalize()`](https://keras3.posit.co/dev/reference/op_normalize.md)  
[`op_not_equal()`](https://keras3.posit.co/dev/reference/op_not_equal.md)  
[`op_one_hot()`](https://keras3.posit.co/dev/reference/op_one_hot.md)  
[`op_ones()`](https://keras3.posit.co/dev/reference/op_ones.md)  
[`op_ones_like()`](https://keras3.posit.co/dev/reference/op_ones_like.md)  
[`op_outer()`](https://keras3.posit.co/dev/reference/op_outer.md)  
[`op_pad()`](https://keras3.posit.co/dev/reference/op_pad.md)  
[`op_polar()`](https://keras3.posit.co/dev/reference/op_polar.md)  
[`op_power()`](https://keras3.posit.co/dev/reference/op_power.md)  
[`op_prod()`](https://keras3.posit.co/dev/reference/op_prod.md)  
[`op_psnr()`](https://keras3.posit.co/dev/reference/op_psnr.md)  
[`op_qr()`](https://keras3.posit.co/dev/reference/op_qr.md)  
[`op_quantile()`](https://keras3.posit.co/dev/reference/op_quantile.md)  
[`op_ravel()`](https://keras3.posit.co/dev/reference/op_ravel.md)  
[`op_real()`](https://keras3.posit.co/dev/reference/op_real.md)  
[`op_rearrange()`](https://keras3.posit.co/dev/reference/op_rearrange.md)  
[`op_reciprocal()`](https://keras3.posit.co/dev/reference/op_reciprocal.md)  
[`op_relu()`](https://keras3.posit.co/dev/reference/op_relu.md)  
[`op_relu6()`](https://keras3.posit.co/dev/reference/op_relu6.md)  
[`op_repeat()`](https://keras3.posit.co/dev/reference/op_repeat.md)  
[`op_reshape()`](https://keras3.posit.co/dev/reference/op_reshape.md)  
[`op_rfft()`](https://keras3.posit.co/dev/reference/op_rfft.md)  
[`op_right_shift()`](https://keras3.posit.co/dev/reference/op_right_shift.md)  
[`op_rms_normalization()`](https://keras3.posit.co/dev/reference/op_rms_normalization.md)  
[`op_roll()`](https://keras3.posit.co/dev/reference/op_roll.md)  
[`op_rot90()`](https://keras3.posit.co/dev/reference/op_rot90.md)  
[`op_round()`](https://keras3.posit.co/dev/reference/op_round.md)  
[`op_rsqrt()`](https://keras3.posit.co/dev/reference/op_rsqrt.md)  
[`op_saturate_cast()`](https://keras3.posit.co/dev/reference/op_saturate_cast.md)  
[`op_scan()`](https://keras3.posit.co/dev/reference/op_scan.md)  
[`op_scatter()`](https://keras3.posit.co/dev/reference/op_scatter.md)  
[`op_scatter_update()`](https://keras3.posit.co/dev/reference/op_scatter_update.md)  
[`op_searchsorted()`](https://keras3.posit.co/dev/reference/op_searchsorted.md)  
[`op_segment_max()`](https://keras3.posit.co/dev/reference/op_segment_max.md)  
[`op_segment_sum()`](https://keras3.posit.co/dev/reference/op_segment_sum.md)  
[`op_select()`](https://keras3.posit.co/dev/reference/op_select.md)  
[`op_selu()`](https://keras3.posit.co/dev/reference/op_selu.md)  
[`op_separable_conv()`](https://keras3.posit.co/dev/reference/op_separable_conv.md)  
[`op_shape()`](https://keras3.posit.co/dev/reference/op_shape.md)  
[`op_sigmoid()`](https://keras3.posit.co/dev/reference/op_sigmoid.md)  
[`op_sign()`](https://keras3.posit.co/dev/reference/op_sign.md)  
[`op_signbit()`](https://keras3.posit.co/dev/reference/op_signbit.md)  
[`op_silu()`](https://keras3.posit.co/dev/reference/op_silu.md)  
[`op_sin()`](https://keras3.posit.co/dev/reference/op_sin.md)  
[`op_sinh()`](https://keras3.posit.co/dev/reference/op_sinh.md)  
[`op_size()`](https://keras3.posit.co/dev/reference/op_size.md)  
[`op_slice()`](https://keras3.posit.co/dev/reference/op_slice.md)  
[`op_slice_update()`](https://keras3.posit.co/dev/reference/op_slice_update.md)  
[`op_slogdet()`](https://keras3.posit.co/dev/reference/op_slogdet.md)  
[`op_soft_shrink()`](https://keras3.posit.co/dev/reference/op_soft_shrink.md)  
[`op_softmax()`](https://keras3.posit.co/dev/reference/op_softmax.md)  
[`op_softplus()`](https://keras3.posit.co/dev/reference/op_softplus.md)  
[`op_softsign()`](https://keras3.posit.co/dev/reference/op_softsign.md)  
[`op_solve()`](https://keras3.posit.co/dev/reference/op_solve.md)  
[`op_solve_triangular()`](https://keras3.posit.co/dev/reference/op_solve_triangular.md)  
[`op_sort()`](https://keras3.posit.co/dev/reference/op_sort.md)  
[`op_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/op_sparse_categorical_crossentropy.md)  
[`op_sparse_plus()`](https://keras3.posit.co/dev/reference/op_sparse_plus.md)  
[`op_sparse_sigmoid()`](https://keras3.posit.co/dev/reference/op_sparse_sigmoid.md)  
[`op_sparsemax()`](https://keras3.posit.co/dev/reference/op_sparsemax.md)  
[`op_split()`](https://keras3.posit.co/dev/reference/op_split.md)  
[`op_sqrt()`](https://keras3.posit.co/dev/reference/op_sqrt.md)  
[`op_square()`](https://keras3.posit.co/dev/reference/op_square.md)  
[`op_squareplus()`](https://keras3.posit.co/dev/reference/op_squareplus.md)  
[`op_squeeze()`](https://keras3.posit.co/dev/reference/op_squeeze.md)  
[`op_stack()`](https://keras3.posit.co/dev/reference/op_stack.md)  
[`op_std()`](https://keras3.posit.co/dev/reference/op_std.md)  
[`op_stft()`](https://keras3.posit.co/dev/reference/op_stft.md)  
[`op_stop_gradient()`](https://keras3.posit.co/dev/reference/op_stop_gradient.md)  
[`op_subtract()`](https://keras3.posit.co/dev/reference/op_subtract.md)  
[`op_sum()`](https://keras3.posit.co/dev/reference/op_sum.md)  
[`op_svd()`](https://keras3.posit.co/dev/reference/op_svd.md)  
[`op_swapaxes()`](https://keras3.posit.co/dev/reference/op_swapaxes.md)  
[`op_switch()`](https://keras3.posit.co/dev/reference/op_switch.md)  
[`op_take()`](https://keras3.posit.co/dev/reference/op_take.md)  
[`op_take_along_axis()`](https://keras3.posit.co/dev/reference/op_take_along_axis.md)  
[`op_tan()`](https://keras3.posit.co/dev/reference/op_tan.md)  
[`op_tanh()`](https://keras3.posit.co/dev/reference/op_tanh.md)  
[`op_tanh_shrink()`](https://keras3.posit.co/dev/reference/op_tanh_shrink.md)  
[`op_tensordot()`](https://keras3.posit.co/dev/reference/op_tensordot.md)  
[`op_threshold()`](https://keras3.posit.co/dev/reference/op_threshold.md)  
[`op_tile()`](https://keras3.posit.co/dev/reference/op_tile.md)  
[`op_top_k()`](https://keras3.posit.co/dev/reference/op_top_k.md)  
[`op_trace()`](https://keras3.posit.co/dev/reference/op_trace.md)  
[`op_transpose()`](https://keras3.posit.co/dev/reference/op_transpose.md)  
[`op_tri()`](https://keras3.posit.co/dev/reference/op_tri.md)  
[`op_tril()`](https://keras3.posit.co/dev/reference/op_tril.md)  
[`op_triu()`](https://keras3.posit.co/dev/reference/op_triu.md)  
[`op_trunc()`](https://keras3.posit.co/dev/reference/op_trunc.md)  
[`op_unravel_index()`](https://keras3.posit.co/dev/reference/op_unravel_index.md)  
[`op_unstack()`](https://keras3.posit.co/dev/reference/op_unstack.md)  
[`op_var()`](https://keras3.posit.co/dev/reference/op_var.md)  
[`op_vdot()`](https://keras3.posit.co/dev/reference/op_vdot.md)  
[`op_vectorize()`](https://keras3.posit.co/dev/reference/op_vectorize.md)  
[`op_vectorized_map()`](https://keras3.posit.co/dev/reference/op_vectorized_map.md)  
[`op_view_as_complex()`](https://keras3.posit.co/dev/reference/op_view_as_complex.md)  
[`op_view_as_real()`](https://keras3.posit.co/dev/reference/op_view_as_real.md)  
[`op_vstack()`](https://keras3.posit.co/dev/reference/op_vstack.md)  
[`op_where()`](https://keras3.posit.co/dev/reference/op_where.md)  
[`op_while_loop()`](https://keras3.posit.co/dev/reference/op_while_loop.md)  
[`op_zeros()`](https://keras3.posit.co/dev/reference/op_zeros.md)  
[`op_zeros_like()`](https://keras3.posit.co/dev/reference/op_zeros_like.md)  

Other core ops:  
[`op_associative_scan()`](https://keras3.posit.co/dev/reference/op_associative_scan.md)  
[`op_cast()`](https://keras3.posit.co/dev/reference/op_cast.md)  
[`op_cond()`](https://keras3.posit.co/dev/reference/op_cond.md)  
[`op_convert_to_numpy()`](https://keras3.posit.co/dev/reference/op_convert_to_numpy.md)  
[`op_convert_to_tensor()`](https://keras3.posit.co/dev/reference/op_convert_to_tensor.md)  
[`op_custom_gradient()`](https://keras3.posit.co/dev/reference/op_custom_gradient.md)  
[`op_dtype()`](https://keras3.posit.co/dev/reference/op_dtype.md)  
[`op_fori_loop()`](https://keras3.posit.co/dev/reference/op_fori_loop.md)  
[`op_is_tensor()`](https://keras3.posit.co/dev/reference/op_is_tensor.md)  
[`op_map()`](https://keras3.posit.co/dev/reference/op_map.md)  
[`op_rearrange()`](https://keras3.posit.co/dev/reference/op_rearrange.md)  
[`op_scan()`](https://keras3.posit.co/dev/reference/op_scan.md)  
[`op_scatter()`](https://keras3.posit.co/dev/reference/op_scatter.md)  
[`op_scatter_update()`](https://keras3.posit.co/dev/reference/op_scatter_update.md)  
[`op_searchsorted()`](https://keras3.posit.co/dev/reference/op_searchsorted.md)  
[`op_shape()`](https://keras3.posit.co/dev/reference/op_shape.md)  
[`op_slice()`](https://keras3.posit.co/dev/reference/op_slice.md)  
[`op_slice_update()`](https://keras3.posit.co/dev/reference/op_slice_update.md)  
[`op_stop_gradient()`](https://keras3.posit.co/dev/reference/op_stop_gradient.md)  
[`op_switch()`](https://keras3.posit.co/dev/reference/op_switch.md)  
[`op_unstack()`](https://keras3.posit.co/dev/reference/op_unstack.md)  
[`op_vectorized_map()`](https://keras3.posit.co/dev/reference/op_vectorized_map.md)  
[`op_while_loop()`](https://keras3.posit.co/dev/reference/op_while_loop.md)  

Other ops:  
[`op_abs()`](https://keras3.posit.co/dev/reference/op_abs.md)  
[`op_add()`](https://keras3.posit.co/dev/reference/op_add.md)  
[`op_all()`](https://keras3.posit.co/dev/reference/op_all.md)  
[`op_angle()`](https://keras3.posit.co/dev/reference/op_angle.md)  
[`op_any()`](https://keras3.posit.co/dev/reference/op_any.md)  
[`op_append()`](https://keras3.posit.co/dev/reference/op_append.md)  
[`op_arange()`](https://keras3.posit.co/dev/reference/op_arange.md)  
[`op_arccos()`](https://keras3.posit.co/dev/reference/op_arccos.md)  
[`op_arccosh()`](https://keras3.posit.co/dev/reference/op_arccosh.md)  
[`op_arcsin()`](https://keras3.posit.co/dev/reference/op_arcsin.md)  
[`op_arcsinh()`](https://keras3.posit.co/dev/reference/op_arcsinh.md)  
[`op_arctan()`](https://keras3.posit.co/dev/reference/op_arctan.md)  
[`op_arctan2()`](https://keras3.posit.co/dev/reference/op_arctan2.md)  
[`op_arctanh()`](https://keras3.posit.co/dev/reference/op_arctanh.md)  
[`op_argmax()`](https://keras3.posit.co/dev/reference/op_argmax.md)  
[`op_argmin()`](https://keras3.posit.co/dev/reference/op_argmin.md)  
[`op_argpartition()`](https://keras3.posit.co/dev/reference/op_argpartition.md)  
[`op_argsort()`](https://keras3.posit.co/dev/reference/op_argsort.md)  
[`op_array()`](https://keras3.posit.co/dev/reference/op_array.md)  
[`op_associative_scan()`](https://keras3.posit.co/dev/reference/op_associative_scan.md)  
[`op_average()`](https://keras3.posit.co/dev/reference/op_average.md)  
[`op_average_pool()`](https://keras3.posit.co/dev/reference/op_average_pool.md)  
[`op_bartlett()`](https://keras3.posit.co/dev/reference/op_bartlett.md)  
[`op_batch_normalization()`](https://keras3.posit.co/dev/reference/op_batch_normalization.md)  
[`op_binary_crossentropy()`](https://keras3.posit.co/dev/reference/op_binary_crossentropy.md)  
[`op_bincount()`](https://keras3.posit.co/dev/reference/op_bincount.md)  
[`op_bitwise_and()`](https://keras3.posit.co/dev/reference/op_bitwise_and.md)  
[`op_bitwise_invert()`](https://keras3.posit.co/dev/reference/op_bitwise_invert.md)  
[`op_bitwise_left_shift()`](https://keras3.posit.co/dev/reference/op_bitwise_left_shift.md)  
[`op_bitwise_not()`](https://keras3.posit.co/dev/reference/op_bitwise_not.md)  
[`op_bitwise_or()`](https://keras3.posit.co/dev/reference/op_bitwise_or.md)  
[`op_bitwise_right_shift()`](https://keras3.posit.co/dev/reference/op_bitwise_right_shift.md)  
[`op_bitwise_xor()`](https://keras3.posit.co/dev/reference/op_bitwise_xor.md)  
[`op_blackman()`](https://keras3.posit.co/dev/reference/op_blackman.md)  
[`op_broadcast_to()`](https://keras3.posit.co/dev/reference/op_broadcast_to.md)  
[`op_cast()`](https://keras3.posit.co/dev/reference/op_cast.md)  
[`op_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/op_categorical_crossentropy.md)  
[`op_cbrt()`](https://keras3.posit.co/dev/reference/op_cbrt.md)  
[`op_ceil()`](https://keras3.posit.co/dev/reference/op_ceil.md)  
[`op_celu()`](https://keras3.posit.co/dev/reference/op_celu.md)  
[`op_cholesky()`](https://keras3.posit.co/dev/reference/op_cholesky.md)  
[`op_clip()`](https://keras3.posit.co/dev/reference/op_clip.md)  
[`op_concatenate()`](https://keras3.posit.co/dev/reference/op_concatenate.md)  
[`op_cond()`](https://keras3.posit.co/dev/reference/op_cond.md)  
[`op_conj()`](https://keras3.posit.co/dev/reference/op_conj.md)  
[`op_conv()`](https://keras3.posit.co/dev/reference/op_conv.md)  
[`op_conv_transpose()`](https://keras3.posit.co/dev/reference/op_conv_transpose.md)  
[`op_convert_to_numpy()`](https://keras3.posit.co/dev/reference/op_convert_to_numpy.md)  
[`op_convert_to_tensor()`](https://keras3.posit.co/dev/reference/op_convert_to_tensor.md)  
[`op_copy()`](https://keras3.posit.co/dev/reference/op_copy.md)  
[`op_corrcoef()`](https://keras3.posit.co/dev/reference/op_corrcoef.md)  
[`op_correlate()`](https://keras3.posit.co/dev/reference/op_correlate.md)  
[`op_cos()`](https://keras3.posit.co/dev/reference/op_cos.md)  
[`op_cosh()`](https://keras3.posit.co/dev/reference/op_cosh.md)  
[`op_count_nonzero()`](https://keras3.posit.co/dev/reference/op_count_nonzero.md)  
[`op_cross()`](https://keras3.posit.co/dev/reference/op_cross.md)  
[`op_ctc_decode()`](https://keras3.posit.co/dev/reference/op_ctc_decode.md)  
[`op_ctc_loss()`](https://keras3.posit.co/dev/reference/op_ctc_loss.md)  
[`op_cumprod()`](https://keras3.posit.co/dev/reference/op_cumprod.md)  
[`op_cumsum()`](https://keras3.posit.co/dev/reference/op_cumsum.md)  
[`op_custom_gradient()`](https://keras3.posit.co/dev/reference/op_custom_gradient.md)  
[`op_deg2rad()`](https://keras3.posit.co/dev/reference/op_deg2rad.md)  
[`op_depthwise_conv()`](https://keras3.posit.co/dev/reference/op_depthwise_conv.md)  
[`op_det()`](https://keras3.posit.co/dev/reference/op_det.md)  
[`op_diag()`](https://keras3.posit.co/dev/reference/op_diag.md)  
[`op_diagflat()`](https://keras3.posit.co/dev/reference/op_diagflat.md)  
[`op_diagonal()`](https://keras3.posit.co/dev/reference/op_diagonal.md)  
[`op_diff()`](https://keras3.posit.co/dev/reference/op_diff.md)  
[`op_digitize()`](https://keras3.posit.co/dev/reference/op_digitize.md)  
[`op_divide()`](https://keras3.posit.co/dev/reference/op_divide.md)  
[`op_divide_no_nan()`](https://keras3.posit.co/dev/reference/op_divide_no_nan.md)  
[`op_dot()`](https://keras3.posit.co/dev/reference/op_dot.md)  
[`op_dot_product_attention()`](https://keras3.posit.co/dev/reference/op_dot_product_attention.md)  
[`op_dtype()`](https://keras3.posit.co/dev/reference/op_dtype.md)  
[`op_eig()`](https://keras3.posit.co/dev/reference/op_eig.md)  
[`op_eigh()`](https://keras3.posit.co/dev/reference/op_eigh.md)  
[`op_einsum()`](https://keras3.posit.co/dev/reference/op_einsum.md)  
[`op_elu()`](https://keras3.posit.co/dev/reference/op_elu.md)  
[`op_empty()`](https://keras3.posit.co/dev/reference/op_empty.md)  
[`op_equal()`](https://keras3.posit.co/dev/reference/op_equal.md)  
[`op_erf()`](https://keras3.posit.co/dev/reference/op_erf.md)  
[`op_erfinv()`](https://keras3.posit.co/dev/reference/op_erfinv.md)  
[`op_exp()`](https://keras3.posit.co/dev/reference/op_exp.md)  
[`op_exp2()`](https://keras3.posit.co/dev/reference/op_exp2.md)  
[`op_expand_dims()`](https://keras3.posit.co/dev/reference/op_expand_dims.md)  
[`op_expm1()`](https://keras3.posit.co/dev/reference/op_expm1.md)  
[`op_extract_sequences()`](https://keras3.posit.co/dev/reference/op_extract_sequences.md)  
[`op_eye()`](https://keras3.posit.co/dev/reference/op_eye.md)  
[`op_fft()`](https://keras3.posit.co/dev/reference/op_fft.md)  
[`op_fft2()`](https://keras3.posit.co/dev/reference/op_fft2.md)  
[`op_flip()`](https://keras3.posit.co/dev/reference/op_flip.md)  
[`op_floor()`](https://keras3.posit.co/dev/reference/op_floor.md)  
[`op_floor_divide()`](https://keras3.posit.co/dev/reference/op_floor_divide.md)  
[`op_fori_loop()`](https://keras3.posit.co/dev/reference/op_fori_loop.md)  
[`op_full()`](https://keras3.posit.co/dev/reference/op_full.md)  
[`op_full_like()`](https://keras3.posit.co/dev/reference/op_full_like.md)  
[`op_gelu()`](https://keras3.posit.co/dev/reference/op_gelu.md)  
[`op_get_item()`](https://keras3.posit.co/dev/reference/op_get_item.md)  
[`op_glu()`](https://keras3.posit.co/dev/reference/op_glu.md)  
[`op_greater()`](https://keras3.posit.co/dev/reference/op_greater.md)  
[`op_greater_equal()`](https://keras3.posit.co/dev/reference/op_greater_equal.md)  
[`op_hamming()`](https://keras3.posit.co/dev/reference/op_hamming.md)  
[`op_hanning()`](https://keras3.posit.co/dev/reference/op_hanning.md)  
[`op_hard_shrink()`](https://keras3.posit.co/dev/reference/op_hard_shrink.md)  
[`op_hard_sigmoid()`](https://keras3.posit.co/dev/reference/op_hard_sigmoid.md)  
[`op_hard_silu()`](https://keras3.posit.co/dev/reference/op_hard_silu.md)  
[`op_hard_tanh()`](https://keras3.posit.co/dev/reference/op_hard_tanh.md)  
[`op_heaviside()`](https://keras3.posit.co/dev/reference/op_heaviside.md)  
[`op_histogram()`](https://keras3.posit.co/dev/reference/op_histogram.md)  
[`op_hstack()`](https://keras3.posit.co/dev/reference/op_hstack.md)  
[`op_identity()`](https://keras3.posit.co/dev/reference/op_identity.md)  
[`op_ifft2()`](https://keras3.posit.co/dev/reference/op_ifft2.md)  
[`op_imag()`](https://keras3.posit.co/dev/reference/op_imag.md)  
[`op_image_affine_transform()`](https://keras3.posit.co/dev/reference/op_image_affine_transform.md)  
[`op_image_crop()`](https://keras3.posit.co/dev/reference/op_image_crop.md)  
[`op_image_extract_patches()`](https://keras3.posit.co/dev/reference/op_image_extract_patches.md)  
[`op_image_gaussian_blur()`](https://keras3.posit.co/dev/reference/op_image_gaussian_blur.md)  
[`op_image_hsv_to_rgb()`](https://keras3.posit.co/dev/reference/op_image_hsv_to_rgb.md)  
[`op_image_map_coordinates()`](https://keras3.posit.co/dev/reference/op_image_map_coordinates.md)  
[`op_image_pad()`](https://keras3.posit.co/dev/reference/op_image_pad.md)  
[`op_image_perspective_transform()`](https://keras3.posit.co/dev/reference/op_image_perspective_transform.md)  
[`op_image_resize()`](https://keras3.posit.co/dev/reference/op_image_resize.md)  
[`op_image_rgb_to_grayscale()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_grayscale.md)  
[`op_image_rgb_to_hsv()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_hsv.md)  
[`op_in_top_k()`](https://keras3.posit.co/dev/reference/op_in_top_k.md)  
[`op_inner()`](https://keras3.posit.co/dev/reference/op_inner.md)  
[`op_inv()`](https://keras3.posit.co/dev/reference/op_inv.md)  
[`op_irfft()`](https://keras3.posit.co/dev/reference/op_irfft.md)  
[`op_is_tensor()`](https://keras3.posit.co/dev/reference/op_is_tensor.md)  
[`op_isclose()`](https://keras3.posit.co/dev/reference/op_isclose.md)  
[`op_isfinite()`](https://keras3.posit.co/dev/reference/op_isfinite.md)  
[`op_isinf()`](https://keras3.posit.co/dev/reference/op_isinf.md)  
[`op_isnan()`](https://keras3.posit.co/dev/reference/op_isnan.md)  
[`op_istft()`](https://keras3.posit.co/dev/reference/op_istft.md)  
[`op_kaiser()`](https://keras3.posit.co/dev/reference/op_kaiser.md)  
[`op_layer_normalization()`](https://keras3.posit.co/dev/reference/op_layer_normalization.md)  
[`op_leaky_relu()`](https://keras3.posit.co/dev/reference/op_leaky_relu.md)  
[`op_left_shift()`](https://keras3.posit.co/dev/reference/op_left_shift.md)  
[`op_less()`](https://keras3.posit.co/dev/reference/op_less.md)  
[`op_less_equal()`](https://keras3.posit.co/dev/reference/op_less_equal.md)  
[`op_linspace()`](https://keras3.posit.co/dev/reference/op_linspace.md)  
[`op_log()`](https://keras3.posit.co/dev/reference/op_log.md)  
[`op_log10()`](https://keras3.posit.co/dev/reference/op_log10.md)  
[`op_log1p()`](https://keras3.posit.co/dev/reference/op_log1p.md)  
[`op_log2()`](https://keras3.posit.co/dev/reference/op_log2.md)  
[`op_log_sigmoid()`](https://keras3.posit.co/dev/reference/op_log_sigmoid.md)  
[`op_log_softmax()`](https://keras3.posit.co/dev/reference/op_log_softmax.md)  
[`op_logaddexp()`](https://keras3.posit.co/dev/reference/op_logaddexp.md)  
[`op_logdet()`](https://keras3.posit.co/dev/reference/op_logdet.md)  
[`op_logical_and()`](https://keras3.posit.co/dev/reference/op_logical_and.md)  
[`op_logical_not()`](https://keras3.posit.co/dev/reference/op_logical_not.md)  
[`op_logical_or()`](https://keras3.posit.co/dev/reference/op_logical_or.md)  
[`op_logical_xor()`](https://keras3.posit.co/dev/reference/op_logical_xor.md)  
[`op_logspace()`](https://keras3.posit.co/dev/reference/op_logspace.md)  
[`op_logsumexp()`](https://keras3.posit.co/dev/reference/op_logsumexp.md)  
[`op_lstsq()`](https://keras3.posit.co/dev/reference/op_lstsq.md)  
[`op_lu_factor()`](https://keras3.posit.co/dev/reference/op_lu_factor.md)  
[`op_map()`](https://keras3.posit.co/dev/reference/op_map.md)  
[`op_matmul()`](https://keras3.posit.co/dev/reference/op_matmul.md)  
[`op_max()`](https://keras3.posit.co/dev/reference/op_max.md)  
[`op_max_pool()`](https://keras3.posit.co/dev/reference/op_max_pool.md)  
[`op_maximum()`](https://keras3.posit.co/dev/reference/op_maximum.md)  
[`op_mean()`](https://keras3.posit.co/dev/reference/op_mean.md)  
[`op_median()`](https://keras3.posit.co/dev/reference/op_median.md)  
[`op_meshgrid()`](https://keras3.posit.co/dev/reference/op_meshgrid.md)  
[`op_min()`](https://keras3.posit.co/dev/reference/op_min.md)  
[`op_minimum()`](https://keras3.posit.co/dev/reference/op_minimum.md)  
[`op_mod()`](https://keras3.posit.co/dev/reference/op_mod.md)  
[`op_moments()`](https://keras3.posit.co/dev/reference/op_moments.md)  
[`op_moveaxis()`](https://keras3.posit.co/dev/reference/op_moveaxis.md)  
[`op_multi_hot()`](https://keras3.posit.co/dev/reference/op_multi_hot.md)  
[`op_multiply()`](https://keras3.posit.co/dev/reference/op_multiply.md)  
[`op_nan_to_num()`](https://keras3.posit.co/dev/reference/op_nan_to_num.md)  
[`op_ndim()`](https://keras3.posit.co/dev/reference/op_ndim.md)  
[`op_negative()`](https://keras3.posit.co/dev/reference/op_negative.md)  
[`op_nonzero()`](https://keras3.posit.co/dev/reference/op_nonzero.md)  
[`op_norm()`](https://keras3.posit.co/dev/reference/op_norm.md)  
[`op_normalize()`](https://keras3.posit.co/dev/reference/op_normalize.md)  
[`op_not_equal()`](https://keras3.posit.co/dev/reference/op_not_equal.md)  
[`op_one_hot()`](https://keras3.posit.co/dev/reference/op_one_hot.md)  
[`op_ones()`](https://keras3.posit.co/dev/reference/op_ones.md)  
[`op_ones_like()`](https://keras3.posit.co/dev/reference/op_ones_like.md)  
[`op_outer()`](https://keras3.posit.co/dev/reference/op_outer.md)  
[`op_pad()`](https://keras3.posit.co/dev/reference/op_pad.md)  
[`op_polar()`](https://keras3.posit.co/dev/reference/op_polar.md)  
[`op_power()`](https://keras3.posit.co/dev/reference/op_power.md)  
[`op_prod()`](https://keras3.posit.co/dev/reference/op_prod.md)  
[`op_psnr()`](https://keras3.posit.co/dev/reference/op_psnr.md)  
[`op_qr()`](https://keras3.posit.co/dev/reference/op_qr.md)  
[`op_quantile()`](https://keras3.posit.co/dev/reference/op_quantile.md)  
[`op_ravel()`](https://keras3.posit.co/dev/reference/op_ravel.md)  
[`op_real()`](https://keras3.posit.co/dev/reference/op_real.md)  
[`op_rearrange()`](https://keras3.posit.co/dev/reference/op_rearrange.md)  
[`op_reciprocal()`](https://keras3.posit.co/dev/reference/op_reciprocal.md)  
[`op_relu()`](https://keras3.posit.co/dev/reference/op_relu.md)  
[`op_relu6()`](https://keras3.posit.co/dev/reference/op_relu6.md)  
[`op_repeat()`](https://keras3.posit.co/dev/reference/op_repeat.md)  
[`op_reshape()`](https://keras3.posit.co/dev/reference/op_reshape.md)  
[`op_rfft()`](https://keras3.posit.co/dev/reference/op_rfft.md)  
[`op_right_shift()`](https://keras3.posit.co/dev/reference/op_right_shift.md)  
[`op_rms_normalization()`](https://keras3.posit.co/dev/reference/op_rms_normalization.md)  
[`op_roll()`](https://keras3.posit.co/dev/reference/op_roll.md)  
[`op_rot90()`](https://keras3.posit.co/dev/reference/op_rot90.md)  
[`op_round()`](https://keras3.posit.co/dev/reference/op_round.md)  
[`op_rsqrt()`](https://keras3.posit.co/dev/reference/op_rsqrt.md)  
[`op_saturate_cast()`](https://keras3.posit.co/dev/reference/op_saturate_cast.md)  
[`op_scan()`](https://keras3.posit.co/dev/reference/op_scan.md)  
[`op_scatter()`](https://keras3.posit.co/dev/reference/op_scatter.md)  
[`op_scatter_update()`](https://keras3.posit.co/dev/reference/op_scatter_update.md)  
[`op_searchsorted()`](https://keras3.posit.co/dev/reference/op_searchsorted.md)  
[`op_segment_max()`](https://keras3.posit.co/dev/reference/op_segment_max.md)  
[`op_segment_sum()`](https://keras3.posit.co/dev/reference/op_segment_sum.md)  
[`op_select()`](https://keras3.posit.co/dev/reference/op_select.md)  
[`op_selu()`](https://keras3.posit.co/dev/reference/op_selu.md)  
[`op_separable_conv()`](https://keras3.posit.co/dev/reference/op_separable_conv.md)  
[`op_shape()`](https://keras3.posit.co/dev/reference/op_shape.md)  
[`op_sigmoid()`](https://keras3.posit.co/dev/reference/op_sigmoid.md)  
[`op_sign()`](https://keras3.posit.co/dev/reference/op_sign.md)  
[`op_signbit()`](https://keras3.posit.co/dev/reference/op_signbit.md)  
[`op_silu()`](https://keras3.posit.co/dev/reference/op_silu.md)  
[`op_sin()`](https://keras3.posit.co/dev/reference/op_sin.md)  
[`op_sinh()`](https://keras3.posit.co/dev/reference/op_sinh.md)  
[`op_size()`](https://keras3.posit.co/dev/reference/op_size.md)  
[`op_slice()`](https://keras3.posit.co/dev/reference/op_slice.md)  
[`op_slice_update()`](https://keras3.posit.co/dev/reference/op_slice_update.md)  
[`op_slogdet()`](https://keras3.posit.co/dev/reference/op_slogdet.md)  
[`op_soft_shrink()`](https://keras3.posit.co/dev/reference/op_soft_shrink.md)  
[`op_softmax()`](https://keras3.posit.co/dev/reference/op_softmax.md)  
[`op_softplus()`](https://keras3.posit.co/dev/reference/op_softplus.md)  
[`op_softsign()`](https://keras3.posit.co/dev/reference/op_softsign.md)  
[`op_solve()`](https://keras3.posit.co/dev/reference/op_solve.md)  
[`op_solve_triangular()`](https://keras3.posit.co/dev/reference/op_solve_triangular.md)  
[`op_sort()`](https://keras3.posit.co/dev/reference/op_sort.md)  
[`op_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/op_sparse_categorical_crossentropy.md)  
[`op_sparse_plus()`](https://keras3.posit.co/dev/reference/op_sparse_plus.md)  
[`op_sparse_sigmoid()`](https://keras3.posit.co/dev/reference/op_sparse_sigmoid.md)  
[`op_sparsemax()`](https://keras3.posit.co/dev/reference/op_sparsemax.md)  
[`op_split()`](https://keras3.posit.co/dev/reference/op_split.md)  
[`op_sqrt()`](https://keras3.posit.co/dev/reference/op_sqrt.md)  
[`op_square()`](https://keras3.posit.co/dev/reference/op_square.md)  
[`op_squareplus()`](https://keras3.posit.co/dev/reference/op_squareplus.md)  
[`op_squeeze()`](https://keras3.posit.co/dev/reference/op_squeeze.md)  
[`op_stack()`](https://keras3.posit.co/dev/reference/op_stack.md)  
[`op_std()`](https://keras3.posit.co/dev/reference/op_std.md)  
[`op_stft()`](https://keras3.posit.co/dev/reference/op_stft.md)  
[`op_stop_gradient()`](https://keras3.posit.co/dev/reference/op_stop_gradient.md)  
[`op_subtract()`](https://keras3.posit.co/dev/reference/op_subtract.md)  
[`op_sum()`](https://keras3.posit.co/dev/reference/op_sum.md)  
[`op_svd()`](https://keras3.posit.co/dev/reference/op_svd.md)  
[`op_swapaxes()`](https://keras3.posit.co/dev/reference/op_swapaxes.md)  
[`op_switch()`](https://keras3.posit.co/dev/reference/op_switch.md)  
[`op_take()`](https://keras3.posit.co/dev/reference/op_take.md)  
[`op_take_along_axis()`](https://keras3.posit.co/dev/reference/op_take_along_axis.md)  
[`op_tan()`](https://keras3.posit.co/dev/reference/op_tan.md)  
[`op_tanh()`](https://keras3.posit.co/dev/reference/op_tanh.md)  
[`op_tanh_shrink()`](https://keras3.posit.co/dev/reference/op_tanh_shrink.md)  
[`op_tensordot()`](https://keras3.posit.co/dev/reference/op_tensordot.md)  
[`op_threshold()`](https://keras3.posit.co/dev/reference/op_threshold.md)  
[`op_tile()`](https://keras3.posit.co/dev/reference/op_tile.md)  
[`op_top_k()`](https://keras3.posit.co/dev/reference/op_top_k.md)  
[`op_trace()`](https://keras3.posit.co/dev/reference/op_trace.md)  
[`op_transpose()`](https://keras3.posit.co/dev/reference/op_transpose.md)  
[`op_tri()`](https://keras3.posit.co/dev/reference/op_tri.md)  
[`op_tril()`](https://keras3.posit.co/dev/reference/op_tril.md)  
[`op_triu()`](https://keras3.posit.co/dev/reference/op_triu.md)  
[`op_trunc()`](https://keras3.posit.co/dev/reference/op_trunc.md)  
[`op_unravel_index()`](https://keras3.posit.co/dev/reference/op_unravel_index.md)  
[`op_unstack()`](https://keras3.posit.co/dev/reference/op_unstack.md)  
[`op_var()`](https://keras3.posit.co/dev/reference/op_var.md)  
[`op_vdot()`](https://keras3.posit.co/dev/reference/op_vdot.md)  
[`op_vectorize()`](https://keras3.posit.co/dev/reference/op_vectorize.md)  
[`op_vectorized_map()`](https://keras3.posit.co/dev/reference/op_vectorized_map.md)  
[`op_view_as_complex()`](https://keras3.posit.co/dev/reference/op_view_as_complex.md)  
[`op_view_as_real()`](https://keras3.posit.co/dev/reference/op_view_as_real.md)  
[`op_vstack()`](https://keras3.posit.co/dev/reference/op_vstack.md)  
[`op_where()`](https://keras3.posit.co/dev/reference/op_where.md)  
[`op_while_loop()`](https://keras3.posit.co/dev/reference/op_while_loop.md)  
[`op_zeros()`](https://keras3.posit.co/dev/reference/op_zeros.md)  
[`op_zeros_like()`](https://keras3.posit.co/dev/reference/op_zeros_like.md)  

Other core ops:  
[`op_associative_scan()`](https://keras3.posit.co/dev/reference/op_associative_scan.md)  
[`op_cast()`](https://keras3.posit.co/dev/reference/op_cast.md)  
[`op_cond()`](https://keras3.posit.co/dev/reference/op_cond.md)  
[`op_convert_to_numpy()`](https://keras3.posit.co/dev/reference/op_convert_to_numpy.md)  
[`op_convert_to_tensor()`](https://keras3.posit.co/dev/reference/op_convert_to_tensor.md)  
[`op_custom_gradient()`](https://keras3.posit.co/dev/reference/op_custom_gradient.md)  
[`op_dtype()`](https://keras3.posit.co/dev/reference/op_dtype.md)  
[`op_fori_loop()`](https://keras3.posit.co/dev/reference/op_fori_loop.md)  
[`op_is_tensor()`](https://keras3.posit.co/dev/reference/op_is_tensor.md)  
[`op_map()`](https://keras3.posit.co/dev/reference/op_map.md)  
[`op_rearrange()`](https://keras3.posit.co/dev/reference/op_rearrange.md)  
[`op_scan()`](https://keras3.posit.co/dev/reference/op_scan.md)  
[`op_scatter()`](https://keras3.posit.co/dev/reference/op_scatter.md)  
[`op_scatter_update()`](https://keras3.posit.co/dev/reference/op_scatter_update.md)  
[`op_searchsorted()`](https://keras3.posit.co/dev/reference/op_searchsorted.md)  
[`op_shape()`](https://keras3.posit.co/dev/reference/op_shape.md)  
[`op_slice()`](https://keras3.posit.co/dev/reference/op_slice.md)  
[`op_slice_update()`](https://keras3.posit.co/dev/reference/op_slice_update.md)  
[`op_stop_gradient()`](https://keras3.posit.co/dev/reference/op_stop_gradient.md)  
[`op_switch()`](https://keras3.posit.co/dev/reference/op_switch.md)  
[`op_unstack()`](https://keras3.posit.co/dev/reference/op_unstack.md)  
[`op_vectorized_map()`](https://keras3.posit.co/dev/reference/op_vectorized_map.md)  
[`op_while_loop()`](https://keras3.posit.co/dev/reference/op_while_loop.md)  

Other ops:  
[`op_abs()`](https://keras3.posit.co/dev/reference/op_abs.md)  
[`op_add()`](https://keras3.posit.co/dev/reference/op_add.md)  
[`op_all()`](https://keras3.posit.co/dev/reference/op_all.md)  
[`op_angle()`](https://keras3.posit.co/dev/reference/op_angle.md)  
[`op_any()`](https://keras3.posit.co/dev/reference/op_any.md)  
[`op_append()`](https://keras3.posit.co/dev/reference/op_append.md)  
[`op_arange()`](https://keras3.posit.co/dev/reference/op_arange.md)  
[`op_arccos()`](https://keras3.posit.co/dev/reference/op_arccos.md)  
[`op_arccosh()`](https://keras3.posit.co/dev/reference/op_arccosh.md)  
[`op_arcsin()`](https://keras3.posit.co/dev/reference/op_arcsin.md)  
[`op_arcsinh()`](https://keras3.posit.co/dev/reference/op_arcsinh.md)  
[`op_arctan()`](https://keras3.posit.co/dev/reference/op_arctan.md)  
[`op_arctan2()`](https://keras3.posit.co/dev/reference/op_arctan2.md)  
[`op_arctanh()`](https://keras3.posit.co/dev/reference/op_arctanh.md)  
[`op_argmax()`](https://keras3.posit.co/dev/reference/op_argmax.md)  
[`op_argmin()`](https://keras3.posit.co/dev/reference/op_argmin.md)  
[`op_argpartition()`](https://keras3.posit.co/dev/reference/op_argpartition.md)  
[`op_argsort()`](https://keras3.posit.co/dev/reference/op_argsort.md)  
[`op_array()`](https://keras3.posit.co/dev/reference/op_array.md)  
[`op_associative_scan()`](https://keras3.posit.co/dev/reference/op_associative_scan.md)  
[`op_average()`](https://keras3.posit.co/dev/reference/op_average.md)  
[`op_average_pool()`](https://keras3.posit.co/dev/reference/op_average_pool.md)  
[`op_bartlett()`](https://keras3.posit.co/dev/reference/op_bartlett.md)  
[`op_batch_normalization()`](https://keras3.posit.co/dev/reference/op_batch_normalization.md)  
[`op_binary_crossentropy()`](https://keras3.posit.co/dev/reference/op_binary_crossentropy.md)  
[`op_bincount()`](https://keras3.posit.co/dev/reference/op_bincount.md)  
[`op_bitwise_and()`](https://keras3.posit.co/dev/reference/op_bitwise_and.md)  
[`op_bitwise_invert()`](https://keras3.posit.co/dev/reference/op_bitwise_invert.md)  
[`op_bitwise_left_shift()`](https://keras3.posit.co/dev/reference/op_bitwise_left_shift.md)  
[`op_bitwise_not()`](https://keras3.posit.co/dev/reference/op_bitwise_not.md)  
[`op_bitwise_or()`](https://keras3.posit.co/dev/reference/op_bitwise_or.md)  
[`op_bitwise_right_shift()`](https://keras3.posit.co/dev/reference/op_bitwise_right_shift.md)  
[`op_bitwise_xor()`](https://keras3.posit.co/dev/reference/op_bitwise_xor.md)  
[`op_blackman()`](https://keras3.posit.co/dev/reference/op_blackman.md)  
[`op_broadcast_to()`](https://keras3.posit.co/dev/reference/op_broadcast_to.md)  
[`op_cast()`](https://keras3.posit.co/dev/reference/op_cast.md)  
[`op_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/op_categorical_crossentropy.md)  
[`op_cbrt()`](https://keras3.posit.co/dev/reference/op_cbrt.md)  
[`op_ceil()`](https://keras3.posit.co/dev/reference/op_ceil.md)  
[`op_celu()`](https://keras3.posit.co/dev/reference/op_celu.md)  
[`op_cholesky()`](https://keras3.posit.co/dev/reference/op_cholesky.md)  
[`op_clip()`](https://keras3.posit.co/dev/reference/op_clip.md)  
[`op_concatenate()`](https://keras3.posit.co/dev/reference/op_concatenate.md)  
[`op_cond()`](https://keras3.posit.co/dev/reference/op_cond.md)  
[`op_conj()`](https://keras3.posit.co/dev/reference/op_conj.md)  
[`op_conv()`](https://keras3.posit.co/dev/reference/op_conv.md)  
[`op_conv_transpose()`](https://keras3.posit.co/dev/reference/op_conv_transpose.md)  
[`op_convert_to_numpy()`](https://keras3.posit.co/dev/reference/op_convert_to_numpy.md)  
[`op_convert_to_tensor()`](https://keras3.posit.co/dev/reference/op_convert_to_tensor.md)  
[`op_copy()`](https://keras3.posit.co/dev/reference/op_copy.md)  
[`op_corrcoef()`](https://keras3.posit.co/dev/reference/op_corrcoef.md)  
[`op_correlate()`](https://keras3.posit.co/dev/reference/op_correlate.md)  
[`op_cos()`](https://keras3.posit.co/dev/reference/op_cos.md)  
[`op_cosh()`](https://keras3.posit.co/dev/reference/op_cosh.md)  
[`op_count_nonzero()`](https://keras3.posit.co/dev/reference/op_count_nonzero.md)  
[`op_cross()`](https://keras3.posit.co/dev/reference/op_cross.md)  
[`op_ctc_decode()`](https://keras3.posit.co/dev/reference/op_ctc_decode.md)  
[`op_ctc_loss()`](https://keras3.posit.co/dev/reference/op_ctc_loss.md)  
[`op_cumprod()`](https://keras3.posit.co/dev/reference/op_cumprod.md)  
[`op_cumsum()`](https://keras3.posit.co/dev/reference/op_cumsum.md)  
[`op_custom_gradient()`](https://keras3.posit.co/dev/reference/op_custom_gradient.md)  
[`op_deg2rad()`](https://keras3.posit.co/dev/reference/op_deg2rad.md)  
[`op_depthwise_conv()`](https://keras3.posit.co/dev/reference/op_depthwise_conv.md)  
[`op_det()`](https://keras3.posit.co/dev/reference/op_det.md)  
[`op_diag()`](https://keras3.posit.co/dev/reference/op_diag.md)  
[`op_diagflat()`](https://keras3.posit.co/dev/reference/op_diagflat.md)  
[`op_diagonal()`](https://keras3.posit.co/dev/reference/op_diagonal.md)  
[`op_diff()`](https://keras3.posit.co/dev/reference/op_diff.md)  
[`op_digitize()`](https://keras3.posit.co/dev/reference/op_digitize.md)  
[`op_divide()`](https://keras3.posit.co/dev/reference/op_divide.md)  
[`op_divide_no_nan()`](https://keras3.posit.co/dev/reference/op_divide_no_nan.md)  
[`op_dot()`](https://keras3.posit.co/dev/reference/op_dot.md)  
[`op_dot_product_attention()`](https://keras3.posit.co/dev/reference/op_dot_product_attention.md)  
[`op_dtype()`](https://keras3.posit.co/dev/reference/op_dtype.md)  
[`op_eig()`](https://keras3.posit.co/dev/reference/op_eig.md)  
[`op_eigh()`](https://keras3.posit.co/dev/reference/op_eigh.md)  
[`op_einsum()`](https://keras3.posit.co/dev/reference/op_einsum.md)  
[`op_elu()`](https://keras3.posit.co/dev/reference/op_elu.md)  
[`op_empty()`](https://keras3.posit.co/dev/reference/op_empty.md)  
[`op_equal()`](https://keras3.posit.co/dev/reference/op_equal.md)  
[`op_erf()`](https://keras3.posit.co/dev/reference/op_erf.md)  
[`op_erfinv()`](https://keras3.posit.co/dev/reference/op_erfinv.md)  
[`op_exp()`](https://keras3.posit.co/dev/reference/op_exp.md)  
[`op_exp2()`](https://keras3.posit.co/dev/reference/op_exp2.md)  
[`op_expand_dims()`](https://keras3.posit.co/dev/reference/op_expand_dims.md)  
[`op_expm1()`](https://keras3.posit.co/dev/reference/op_expm1.md)  
[`op_extract_sequences()`](https://keras3.posit.co/dev/reference/op_extract_sequences.md)  
[`op_eye()`](https://keras3.posit.co/dev/reference/op_eye.md)  
[`op_fft()`](https://keras3.posit.co/dev/reference/op_fft.md)  
[`op_fft2()`](https://keras3.posit.co/dev/reference/op_fft2.md)  
[`op_flip()`](https://keras3.posit.co/dev/reference/op_flip.md)  
[`op_floor()`](https://keras3.posit.co/dev/reference/op_floor.md)  
[`op_floor_divide()`](https://keras3.posit.co/dev/reference/op_floor_divide.md)  
[`op_fori_loop()`](https://keras3.posit.co/dev/reference/op_fori_loop.md)  
[`op_full()`](https://keras3.posit.co/dev/reference/op_full.md)  
[`op_full_like()`](https://keras3.posit.co/dev/reference/op_full_like.md)  
[`op_gelu()`](https://keras3.posit.co/dev/reference/op_gelu.md)  
[`op_get_item()`](https://keras3.posit.co/dev/reference/op_get_item.md)  
[`op_glu()`](https://keras3.posit.co/dev/reference/op_glu.md)  
[`op_greater()`](https://keras3.posit.co/dev/reference/op_greater.md)  
[`op_greater_equal()`](https://keras3.posit.co/dev/reference/op_greater_equal.md)  
[`op_hamming()`](https://keras3.posit.co/dev/reference/op_hamming.md)  
[`op_hanning()`](https://keras3.posit.co/dev/reference/op_hanning.md)  
[`op_hard_shrink()`](https://keras3.posit.co/dev/reference/op_hard_shrink.md)  
[`op_hard_sigmoid()`](https://keras3.posit.co/dev/reference/op_hard_sigmoid.md)  
[`op_hard_silu()`](https://keras3.posit.co/dev/reference/op_hard_silu.md)  
[`op_hard_tanh()`](https://keras3.posit.co/dev/reference/op_hard_tanh.md)  
[`op_heaviside()`](https://keras3.posit.co/dev/reference/op_heaviside.md)  
[`op_histogram()`](https://keras3.posit.co/dev/reference/op_histogram.md)  
[`op_hstack()`](https://keras3.posit.co/dev/reference/op_hstack.md)  
[`op_identity()`](https://keras3.posit.co/dev/reference/op_identity.md)  
[`op_ifft2()`](https://keras3.posit.co/dev/reference/op_ifft2.md)  
[`op_imag()`](https://keras3.posit.co/dev/reference/op_imag.md)  
[`op_image_affine_transform()`](https://keras3.posit.co/dev/reference/op_image_affine_transform.md)  
[`op_image_crop()`](https://keras3.posit.co/dev/reference/op_image_crop.md)  
[`op_image_extract_patches()`](https://keras3.posit.co/dev/reference/op_image_extract_patches.md)  
[`op_image_gaussian_blur()`](https://keras3.posit.co/dev/reference/op_image_gaussian_blur.md)  
[`op_image_hsv_to_rgb()`](https://keras3.posit.co/dev/reference/op_image_hsv_to_rgb.md)  
[`op_image_map_coordinates()`](https://keras3.posit.co/dev/reference/op_image_map_coordinates.md)  
[`op_image_pad()`](https://keras3.posit.co/dev/reference/op_image_pad.md)  
[`op_image_perspective_transform()`](https://keras3.posit.co/dev/reference/op_image_perspective_transform.md)  
[`op_image_resize()`](https://keras3.posit.co/dev/reference/op_image_resize.md)  
[`op_image_rgb_to_grayscale()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_grayscale.md)  
[`op_image_rgb_to_hsv()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_hsv.md)  
[`op_in_top_k()`](https://keras3.posit.co/dev/reference/op_in_top_k.md)  
[`op_inner()`](https://keras3.posit.co/dev/reference/op_inner.md)  
[`op_inv()`](https://keras3.posit.co/dev/reference/op_inv.md)  
[`op_irfft()`](https://keras3.posit.co/dev/reference/op_irfft.md)  
[`op_is_tensor()`](https://keras3.posit.co/dev/reference/op_is_tensor.md)  
[`op_isclose()`](https://keras3.posit.co/dev/reference/op_isclose.md)  
[`op_isfinite()`](https://keras3.posit.co/dev/reference/op_isfinite.md)  
[`op_isinf()`](https://keras3.posit.co/dev/reference/op_isinf.md)  
[`op_isnan()`](https://keras3.posit.co/dev/reference/op_isnan.md)  
[`op_istft()`](https://keras3.posit.co/dev/reference/op_istft.md)  
[`op_kaiser()`](https://keras3.posit.co/dev/reference/op_kaiser.md)  
[`op_layer_normalization()`](https://keras3.posit.co/dev/reference/op_layer_normalization.md)  
[`op_leaky_relu()`](https://keras3.posit.co/dev/reference/op_leaky_relu.md)  
[`op_left_shift()`](https://keras3.posit.co/dev/reference/op_left_shift.md)  
[`op_less()`](https://keras3.posit.co/dev/reference/op_less.md)  
[`op_less_equal()`](https://keras3.posit.co/dev/reference/op_less_equal.md)  
[`op_linspace()`](https://keras3.posit.co/dev/reference/op_linspace.md)  
[`op_log()`](https://keras3.posit.co/dev/reference/op_log.md)  
[`op_log10()`](https://keras3.posit.co/dev/reference/op_log10.md)  
[`op_log1p()`](https://keras3.posit.co/dev/reference/op_log1p.md)  
[`op_log2()`](https://keras3.posit.co/dev/reference/op_log2.md)  
[`op_log_sigmoid()`](https://keras3.posit.co/dev/reference/op_log_sigmoid.md)  
[`op_log_softmax()`](https://keras3.posit.co/dev/reference/op_log_softmax.md)  
[`op_logaddexp()`](https://keras3.posit.co/dev/reference/op_logaddexp.md)  
[`op_logdet()`](https://keras3.posit.co/dev/reference/op_logdet.md)  
[`op_logical_and()`](https://keras3.posit.co/dev/reference/op_logical_and.md)  
[`op_logical_not()`](https://keras3.posit.co/dev/reference/op_logical_not.md)  
[`op_logical_or()`](https://keras3.posit.co/dev/reference/op_logical_or.md)  
[`op_logical_xor()`](https://keras3.posit.co/dev/reference/op_logical_xor.md)  
[`op_logspace()`](https://keras3.posit.co/dev/reference/op_logspace.md)  
[`op_logsumexp()`](https://keras3.posit.co/dev/reference/op_logsumexp.md)  
[`op_lstsq()`](https://keras3.posit.co/dev/reference/op_lstsq.md)  
[`op_lu_factor()`](https://keras3.posit.co/dev/reference/op_lu_factor.md)  
[`op_map()`](https://keras3.posit.co/dev/reference/op_map.md)  
[`op_matmul()`](https://keras3.posit.co/dev/reference/op_matmul.md)  
[`op_max()`](https://keras3.posit.co/dev/reference/op_max.md)  
[`op_max_pool()`](https://keras3.posit.co/dev/reference/op_max_pool.md)  
[`op_maximum()`](https://keras3.posit.co/dev/reference/op_maximum.md)  
[`op_mean()`](https://keras3.posit.co/dev/reference/op_mean.md)  
[`op_median()`](https://keras3.posit.co/dev/reference/op_median.md)  
[`op_meshgrid()`](https://keras3.posit.co/dev/reference/op_meshgrid.md)  
[`op_min()`](https://keras3.posit.co/dev/reference/op_min.md)  
[`op_minimum()`](https://keras3.posit.co/dev/reference/op_minimum.md)  
[`op_mod()`](https://keras3.posit.co/dev/reference/op_mod.md)  
[`op_moments()`](https://keras3.posit.co/dev/reference/op_moments.md)  
[`op_moveaxis()`](https://keras3.posit.co/dev/reference/op_moveaxis.md)  
[`op_multi_hot()`](https://keras3.posit.co/dev/reference/op_multi_hot.md)  
[`op_multiply()`](https://keras3.posit.co/dev/reference/op_multiply.md)  
[`op_nan_to_num()`](https://keras3.posit.co/dev/reference/op_nan_to_num.md)  
[`op_ndim()`](https://keras3.posit.co/dev/reference/op_ndim.md)  
[`op_negative()`](https://keras3.posit.co/dev/reference/op_negative.md)  
[`op_nonzero()`](https://keras3.posit.co/dev/reference/op_nonzero.md)  
[`op_norm()`](https://keras3.posit.co/dev/reference/op_norm.md)  
[`op_normalize()`](https://keras3.posit.co/dev/reference/op_normalize.md)  
[`op_not_equal()`](https://keras3.posit.co/dev/reference/op_not_equal.md)  
[`op_one_hot()`](https://keras3.posit.co/dev/reference/op_one_hot.md)  
[`op_ones()`](https://keras3.posit.co/dev/reference/op_ones.md)  
[`op_ones_like()`](https://keras3.posit.co/dev/reference/op_ones_like.md)  
[`op_outer()`](https://keras3.posit.co/dev/reference/op_outer.md)  
[`op_pad()`](https://keras3.posit.co/dev/reference/op_pad.md)  
[`op_polar()`](https://keras3.posit.co/dev/reference/op_polar.md)  
[`op_power()`](https://keras3.posit.co/dev/reference/op_power.md)  
[`op_prod()`](https://keras3.posit.co/dev/reference/op_prod.md)  
[`op_psnr()`](https://keras3.posit.co/dev/reference/op_psnr.md)  
[`op_qr()`](https://keras3.posit.co/dev/reference/op_qr.md)  
[`op_quantile()`](https://keras3.posit.co/dev/reference/op_quantile.md)  
[`op_ravel()`](https://keras3.posit.co/dev/reference/op_ravel.md)  
[`op_real()`](https://keras3.posit.co/dev/reference/op_real.md)  
[`op_rearrange()`](https://keras3.posit.co/dev/reference/op_rearrange.md)  
[`op_reciprocal()`](https://keras3.posit.co/dev/reference/op_reciprocal.md)  
[`op_relu()`](https://keras3.posit.co/dev/reference/op_relu.md)  
[`op_relu6()`](https://keras3.posit.co/dev/reference/op_relu6.md)  
[`op_repeat()`](https://keras3.posit.co/dev/reference/op_repeat.md)  
[`op_reshape()`](https://keras3.posit.co/dev/reference/op_reshape.md)  
[`op_rfft()`](https://keras3.posit.co/dev/reference/op_rfft.md)  
[`op_right_shift()`](https://keras3.posit.co/dev/reference/op_right_shift.md)  
[`op_rms_normalization()`](https://keras3.posit.co/dev/reference/op_rms_normalization.md)  
[`op_roll()`](https://keras3.posit.co/dev/reference/op_roll.md)  
[`op_rot90()`](https://keras3.posit.co/dev/reference/op_rot90.md)  
[`op_round()`](https://keras3.posit.co/dev/reference/op_round.md)  
[`op_rsqrt()`](https://keras3.posit.co/dev/reference/op_rsqrt.md)  
[`op_saturate_cast()`](https://keras3.posit.co/dev/reference/op_saturate_cast.md)  
[`op_scan()`](https://keras3.posit.co/dev/reference/op_scan.md)  
[`op_scatter()`](https://keras3.posit.co/dev/reference/op_scatter.md)  
[`op_scatter_update()`](https://keras3.posit.co/dev/reference/op_scatter_update.md)  
[`op_searchsorted()`](https://keras3.posit.co/dev/reference/op_searchsorted.md)  
[`op_segment_max()`](https://keras3.posit.co/dev/reference/op_segment_max.md)  
[`op_segment_sum()`](https://keras3.posit.co/dev/reference/op_segment_sum.md)  
[`op_select()`](https://keras3.posit.co/dev/reference/op_select.md)  
[`op_selu()`](https://keras3.posit.co/dev/reference/op_selu.md)  
[`op_separable_conv()`](https://keras3.posit.co/dev/reference/op_separable_conv.md)  
[`op_shape()`](https://keras3.posit.co/dev/reference/op_shape.md)  
[`op_sigmoid()`](https://keras3.posit.co/dev/reference/op_sigmoid.md)  
[`op_sign()`](https://keras3.posit.co/dev/reference/op_sign.md)  
[`op_signbit()`](https://keras3.posit.co/dev/reference/op_signbit.md)  
[`op_silu()`](https://keras3.posit.co/dev/reference/op_silu.md)  
[`op_sin()`](https://keras3.posit.co/dev/reference/op_sin.md)  
[`op_sinh()`](https://keras3.posit.co/dev/reference/op_sinh.md)  
[`op_size()`](https://keras3.posit.co/dev/reference/op_size.md)  
[`op_slice()`](https://keras3.posit.co/dev/reference/op_slice.md)  
[`op_slice_update()`](https://keras3.posit.co/dev/reference/op_slice_update.md)  
[`op_slogdet()`](https://keras3.posit.co/dev/reference/op_slogdet.md)  
[`op_soft_shrink()`](https://keras3.posit.co/dev/reference/op_soft_shrink.md)  
[`op_softmax()`](https://keras3.posit.co/dev/reference/op_softmax.md)  
[`op_softplus()`](https://keras3.posit.co/dev/reference/op_softplus.md)  
[`op_softsign()`](https://keras3.posit.co/dev/reference/op_softsign.md)  
[`op_solve()`](https://keras3.posit.co/dev/reference/op_solve.md)  
[`op_solve_triangular()`](https://keras3.posit.co/dev/reference/op_solve_triangular.md)  
[`op_sort()`](https://keras3.posit.co/dev/reference/op_sort.md)  
[`op_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/op_sparse_categorical_crossentropy.md)  
[`op_sparse_plus()`](https://keras3.posit.co/dev/reference/op_sparse_plus.md)  
[`op_sparse_sigmoid()`](https://keras3.posit.co/dev/reference/op_sparse_sigmoid.md)  
[`op_sparsemax()`](https://keras3.posit.co/dev/reference/op_sparsemax.md)  
[`op_split()`](https://keras3.posit.co/dev/reference/op_split.md)  
[`op_sqrt()`](https://keras3.posit.co/dev/reference/op_sqrt.md)  
[`op_square()`](https://keras3.posit.co/dev/reference/op_square.md)  
[`op_squareplus()`](https://keras3.posit.co/dev/reference/op_squareplus.md)  
[`op_squeeze()`](https://keras3.posit.co/dev/reference/op_squeeze.md)  
[`op_stack()`](https://keras3.posit.co/dev/reference/op_stack.md)  
[`op_std()`](https://keras3.posit.co/dev/reference/op_std.md)  
[`op_stft()`](https://keras3.posit.co/dev/reference/op_stft.md)  
[`op_stop_gradient()`](https://keras3.posit.co/dev/reference/op_stop_gradient.md)  
[`op_subtract()`](https://keras3.posit.co/dev/reference/op_subtract.md)  
[`op_sum()`](https://keras3.posit.co/dev/reference/op_sum.md)  
[`op_svd()`](https://keras3.posit.co/dev/reference/op_svd.md)  
[`op_swapaxes()`](https://keras3.posit.co/dev/reference/op_swapaxes.md)  
[`op_switch()`](https://keras3.posit.co/dev/reference/op_switch.md)  
[`op_take()`](https://keras3.posit.co/dev/reference/op_take.md)  
[`op_take_along_axis()`](https://keras3.posit.co/dev/reference/op_take_along_axis.md)  
[`op_tan()`](https://keras3.posit.co/dev/reference/op_tan.md)  
[`op_tanh()`](https://keras3.posit.co/dev/reference/op_tanh.md)  
[`op_tanh_shrink()`](https://keras3.posit.co/dev/reference/op_tanh_shrink.md)  
[`op_tensordot()`](https://keras3.posit.co/dev/reference/op_tensordot.md)  
[`op_threshold()`](https://keras3.posit.co/dev/reference/op_threshold.md)  
[`op_tile()`](https://keras3.posit.co/dev/reference/op_tile.md)  
[`op_top_k()`](https://keras3.posit.co/dev/reference/op_top_k.md)  
[`op_trace()`](https://keras3.posit.co/dev/reference/op_trace.md)  
[`op_transpose()`](https://keras3.posit.co/dev/reference/op_transpose.md)  
[`op_tri()`](https://keras3.posit.co/dev/reference/op_tri.md)  
[`op_tril()`](https://keras3.posit.co/dev/reference/op_tril.md)  
[`op_triu()`](https://keras3.posit.co/dev/reference/op_triu.md)  
[`op_trunc()`](https://keras3.posit.co/dev/reference/op_trunc.md)  
[`op_unravel_index()`](https://keras3.posit.co/dev/reference/op_unravel_index.md)  
[`op_unstack()`](https://keras3.posit.co/dev/reference/op_unstack.md)  
[`op_var()`](https://keras3.posit.co/dev/reference/op_var.md)  
[`op_vdot()`](https://keras3.posit.co/dev/reference/op_vdot.md)  
[`op_vectorize()`](https://keras3.posit.co/dev/reference/op_vectorize.md)  
[`op_vectorized_map()`](https://keras3.posit.co/dev/reference/op_vectorized_map.md)  
[`op_view_as_complex()`](https://keras3.posit.co/dev/reference/op_view_as_complex.md)  
[`op_view_as_real()`](https://keras3.posit.co/dev/reference/op_view_as_real.md)  
[`op_vstack()`](https://keras3.posit.co/dev/reference/op_vstack.md)  
[`op_where()`](https://keras3.posit.co/dev/reference/op_where.md)  
[`op_while_loop()`](https://keras3.posit.co/dev/reference/op_while_loop.md)  
[`op_zeros()`](https://keras3.posit.co/dev/reference/op_zeros.md)  
[`op_zeros_like()`](https://keras3.posit.co/dev/reference/op_zeros_like.md)  
