test_that("linear algebra and scatter ops forward new options", {
  skip_if_no_keras("3.14.0")

  input <- matrix(c(4, 2, 2, 3), nrow = 2)
  lower <- as.array(op_cholesky(input))
  upper <- as.array(op_cholesky(input, upper = TRUE))
  scattered <- op_scatter_update(
    op_zeros(3),
    matrix(c(1L, 1L, 2L), ncol = 1),
    c(1, 2, 4),
    reduction = "add"
  )

  expect_equal(upper, t(lower))
  expect_equal(as.numeric(scattered), c(3, 4, 0))
})
