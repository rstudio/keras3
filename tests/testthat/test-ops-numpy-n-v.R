test_that("NaN-aware NumPy ops use R axes and indices", {
  skip_if_no_keras("3.15.1")

  x <- op_array(
    matrix(c(1, NaN, 3, NaN, 2, 1), nrow = 2, byrow = TRUE),
    dtype = "float32"
  )

  expect_equal_array(op_nanargmax(x, axis = 2), c(3L, 2L))
  expect_equal_array(op_nanargmax(x, axis = 2, zero_indexed = TRUE),
                     c(2L, 1L))
  expect_equal_array(op_nanargmin(x, axis = 2), c(1L, 3L))
  expect_equal_array(op_nanargmin(x, axis = 2, zero_indexed = TRUE),
                     c(0L, 2L))

  expect_equal_array(
    op_nancumprod(x, axis = 2),
    matrix(c(1, 1, 3, 1, 2, 2), nrow = 2, byrow = TRUE)
  )
  expect_equal_array(
    op_nancumsum(x, axis = 2),
    matrix(c(1, 1, 4, 0, 2, 3), nrow = 2, byrow = TRUE)
  )

  expect_equal_array(op_nanmax(x, axis = 2), c(3, 2))
  expect_equal_array(op_nanmean(x, axis = 2), c(2, 1.5))
  expect_equal_array(op_nanmedian(x, axis = 2), c(2, 1.5))
  expect_equal_array(op_nanmin(x, axis = 2), c(1, 1))
  expect_equal_array(op_nanpercentile(x, 50, axis = 2), c(2, 1.5))
  expect_equal_array(op_nanprod(x, axis = 2), c(3, 2))
  expect_equal_array(op_nanquantile(x, 0.5, axis = 2), c(2, 1.5))
  expect_equal_array(op_nanstd(x, axis = 2), c(1, 0.5))
  expect_equal_array(op_nansum(x, axis = 2), c(4, 3))
  expect_equal_array(op_nanvar(x, axis = 2), c(1, 0.25))
})

test_that("Keras 3.15.1 numeric NumPy ops are available", {
  skip_if_no_keras("3.15.1")

  adjacent <- as.numeric(as.array(op_nextafter(c(1, 1), c(2, 0))))
  expect_gt(adjacent[[1]], 1)
  expect_lt(adjacent[[2]], 1)

  x <- op_array(matrix(c(1, 3, 2, 4, 0, 5), nrow = 2, byrow = TRUE))
  expect_equal_array(op_percentile(x, 50, axis = 2), c(2, 4))
  expect_equal_array(op_ptp(x, axis = 2), c(2, 5))

  a <- op_array(matrix(c(1, 2, 3, 4, 5, 6), ncol = 2),
                dtype = "float32")
  expect_equal(as.array(op_matmul(op_pinv(a), a)), diag(2),
               tolerance = 1e-5)

  expect_equal_array(op_rad2deg(c(0, pi)), c(0, 180))
  expect_equal(as.numeric(as.array(op_sinc(c(0, 1)))), c(1, 0),
               tolerance = 1e-6)
  expect_equal_array(
    op_trapezoid(matrix(1:6, nrow = 2, byrow = TRUE), axis = 2),
    c(4, 10)
  )

  expect_equal_array(
    op_vander(c(1, 2, 3), N = 3L, increasing = TRUE),
    matrix(c(1, 1, 1, 1, 2, 4, 1, 3, 9), nrow = 3, byrow = TRUE)
  )

  viewed <- op_view(op_array(c(1L, 2L), dtype = "int32"), "float32")
  expect_identical(op_dtype(viewed), "float32")
  expect_equal(op_shape(viewed), shape(2))
  expect_true(all(abs(as.array(viewed)) < 1e-40))
})

test_that("op_unique returns R-indexed metadata", {
  skip_if_no_keras("3.15.1")

  x <- c(3L, 1L, 2L, 1L, 3L, 2L)
  result <- op_unique(
    x,
    return_index = TRUE,
    return_inverse = TRUE,
    return_counts = TRUE
  )

  expect_equal_array(result[[1]], c(1L, 2L, 3L))
  expect_equal_array(result[[2]], c(2L, 3L, 1L))
  expect_equal_array(result[[3]], c(3L, 1L, 2L, 1L, 3L, 2L))
  expect_equal_array(result[[4]], c(2L, 2L, 2L))

  zero_indexed <- op_unique(
    x,
    return_index = TRUE,
    return_inverse = TRUE,
    zero_indexed = TRUE
  )
  expect_equal_array(zero_indexed[[2]], c(1L, 2L, 0L))
  expect_equal_array(zero_indexed[[3]], c(2L, 0L, 1L, 0L, 2L, 1L))
})

test_that("op_vsplit distinguishes section counts from R split positions", {
  skip_if_no_keras("3.15.1")

  x <- op_reshape(op_arange(12), c(4, 3))
  sections <- op_vsplit(x, 2L)
  expect_equal(lapply(sections, op_shape), rep(list(shape(2, 3)), 2))

  x <- op_reshape(op_arange(10), c(5, 2))
  positions <- op_vsplit(x, array(c(2L, 4L)))
  expect_equal(
    lapply(positions, op_shape),
    list(shape(1, 2), shape(2, 2), shape(2, 2))
  )
})
