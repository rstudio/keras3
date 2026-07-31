test_that("Keras 3.15 comparison and classification ops work", {
  skip_if_no_keras("3.15.1")

  expect_true(as.logical(as.array(op_allclose(c(1, 2), c(1, 2)))))
  expect_equal_array(op_isin(c(0L, 1L, 2L), c(0L, 2L)),
                     c(TRUE, FALSE, TRUE))
  expect_equal_array(op_isneginf(c(-Inf, Inf, 0)), c(TRUE, FALSE, FALSE))
  expect_equal_array(op_isposinf(c(-Inf, Inf, 0)), c(FALSE, TRUE, FALSE))

  x <- op_array(c(1 + 0i, 1 + 1i), dtype = "complex64")
  expect_equal_array(op_isreal(x), c(TRUE, FALSE))
})

test_that("Keras 3.15 element-wise numeric ops work", {
  skip_if_no_keras("3.15.1")

  expect_equal_array(op_erfc(0), 1)
  expect_equal_array(op_fabs(c(-2L, 3L)), c(2, 3))
  expect_equal_array(op_fmax(c(2, NaN), c(1, 4)), c(2, 4))
  expect_equal_array(op_fmin(c(2, NaN), c(1, 4)), c(1, 4))
  expect_equal_array(op_fmod(c(-5, 5), 3), c(-2, 2))
  expect_equal_array(op_gcd(c(12L, 18L), c(8L, 12L)), c(4L, 6L))
  expect_equal_array(op_hypot(c(3, 5), c(4, 12)), c(5, 13))
  expect_equal_array(op_i0(0), 1)
  expect_equal_array(op_kron(c(1, 2), c(3, 4)), c(3, 4, 6, 8))
  expect_equal_array(op_lcm(c(2L, 3L, 4L), c(5L, 6L, 7L)),
                     c(10L, 6L, 28L))
  expect_equal_array(op_ldexp(c(0.75, 1.5), c(1L, 2L)), c(1.5, 6))
  expect_equal_array(op_logaddexp2(c(1, 2), c(1, 2)), c(2, 3))
})

test_that("Keras 3.15 array-shaping ops follow R axes", {
  skip_if_no_keras("3.15.1")

  x <- op_reshape(op_arange(12), c(2, 6))
  pieces <- op_array_split(x, 3L, axis = 2L)
  expect_equal(lapply(pieces, op_shape), rep(list(shape(2, 2)), 3))

  x3 <- op_reshape(op_arange(24), c(2, 3, 4))
  depth <- op_dsplit(x3, 2L)
  expect_equal(lapply(depth, op_shape), rep(list(shape(2, 3, 2)), 2))

  horizontal <- op_hsplit(x, 3L)
  expect_equal(lapply(horizontal, op_shape), rep(list(shape(2, 2)), 3))

  indexed <- op_hsplit(x, c(3L, 5L))
  expect_equal(lapply(indexed, op_shape), rep(list(shape(2, 2)), 3))

  stacked <- op_dstack(list(op_array(1:3), op_array(4:6)))
  expect_equal(op_shape(stacked), shape(1, 3, 2))

  empty <- op_empty_like(op_ones(c(2, 3), dtype = "float32"))
  expect_equal(op_shape(empty), shape(2, 3))
  expect_identical(op_dtype(empty), "float32")

  matrix <- op_array(matrix(1:6, nrow = 2, byrow = TRUE))
  expect_equal_array(op_fliplr(matrix),
                     matrix(c(3, 2, 1, 6, 5, 4), nrow = 2, byrow = TRUE))
  expect_equal_array(op_flipud(matrix),
                     matrix(c(4, 5, 6, 1, 2, 3), nrow = 2, byrow = TRUE))
})

test_that("Keras 3.15 distance, matrix, and range ops work", {
  skip_if_no_keras("3.15.1")

  x <- op_array(rbind(c(0, 0), c(3, 4)), dtype = "float32")
  y <- op_array(rbind(c(0, 0)), dtype = "float32")
  expect_equal_array(op_cdist(x, y), matrix(c(0, 5), ncol = 1))

  factor <- op_array(diag(c(2, 3)), dtype = "float32")
  expect_equal_array(op_cholesky_inverse(factor), diag(c(1 / 4, 1 / 9)))
  expect_equal_array(op_matrix_rank(factor), 2L)

  values <- op_geomspace(1, 1000, num = 4L)
  expect_equal(as.numeric(as.array(values)), c(1, 10, 100, 1000),
               tolerance = 1e-5)

  ranged <- op_geomspace(c(1, 100), c(100, 10000), num = 3L, axis = 1L)
  expect_equal(op_shape(ranged), shape(3, 2))
})
