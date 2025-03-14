


test_that("op_subset() works", {

  xr <- array(1:20, c(4, 5))
  xt <- op_convert_to_tensor(xr)

  r <- xr[1:2, c(TRUE, TRUE, TRUE, TRUE, FALSE)]
  expect_equal_array(r, xt@r[1:2, c(TRUE, TRUE, TRUE, TRUE, FALSE)])
  expect_equal_array(r, xt@r[(1:2), c(TRUE, TRUE, TRUE, TRUE, FALSE)])
  expect_equal_array(r, xt@r[ct(1:2), c(TRUE, TRUE, TRUE, TRUE, FALSE)])
  expect_equal_array(r, xt@r[c(1:2), ct(TRUE, TRUE, TRUE, TRUE, FALSE)])
  expect_equal_array(r, xt@r[ct(1:2), ct(TRUE, TRUE, TRUE, TRUE, FALSE)])

  r <- xr[c(1:2, 4), c(TRUE, FALSE, TRUE, TRUE, FALSE)]
  expect_equal_array(r, xt@r[c(1:2, 4), c(TRUE, FALSE, TRUE, TRUE, FALSE)])


  i <- cbind(3:1, 1)
  r <- xr[i]
  expect_equal_array(r, xt@r[i])
  expect_equal_array(r, xt@r[ct(i)])

})



test_that("op_subset() works", {

  xr <- array(1:20, c(4, 5))
  xt <- op_convert_to_tensor(xr)

  expect_same_semantics <- function(expr) {
    expr <- substitute(expr)
    array_result <- as.array(eval(expr, env(parent.frame(), x = xr)))
    tensor_result <- eval(expr, env(parent.frame(), `[` = op_subset, x = xt))
    expect_identical(array_result, op_convert_to_array(tensor_result))

    vars <- setdiff(all.vars(expr), "x")
    if(length(vars)) {
      vars <- mget(vars, parent.frame(), inherits = TRUE)
      # Convert variables to tensors where appropriate
      vars <- lapply(vars, function(x) {
        if(is.double(x))
          storage.mode(x) <- "integer"
        if(length(x) > 1)
          x <- as.array(x)
        op_convert_to_tensor(x)
      })
      tensor_result2 <- eval(expr, env(parent.frame(), `[` = op_subset, x = xt, !!!vars))
      expect_identical(array_result, op_convert_to_array(tensor_result2))
    }
  }


  # Basic subsetting
  expect_same_semantics(x[1:2, c(TRUE, TRUE, TRUE, TRUE, FALSE)])
  expect_same_semantics(x[(1:2), c(TRUE, TRUE, TRUE, TRUE, FALSE)])

  # Mixed tensor and regular indexing
  r <- xr[1:2, c(TRUE, TRUE, TRUE, TRUE, FALSE)]
  expect_equal_array(r, xt@r[ct(1:2), c(TRUE, TRUE, TRUE, TRUE, FALSE)])
  expect_equal_array(r, xt@r[c(1:2), ct(TRUE, TRUE, TRUE, TRUE, FALSE)])
  expect_equal_array(r, xt@r[ct(1:2), ct(TRUE, TRUE, TRUE, TRUE, FALSE)])

  # More complex indexing
  r <- xr[c(1:2, 4), c(TRUE, FALSE, TRUE, TRUE, FALSE)]
  expect_equal_array(r, xt@r[c(1:2, 4), c(TRUE, FALSE, TRUE, TRUE, FALSE)])

  # Matrix-based indexing
  i <- cbind(3:1, 1)
  r <- xr[i]
  expect_equal_array(r, xt@r[i])
  expect_equal_array(r, xt@r[ct(i)])

  # Edge cases from the additional list
  # Mixed numeric and logical indexing
  expect_same_semantics(x[c(1, 3, 4), c(TRUE, FALSE, TRUE, TRUE, FALSE)])
  expect_same_semantics(x[c(1, 3, 4), c(2, 4, 5)])
  expect_same_semantics(x[c(1, 3, 4), 1:2])
  expect_same_semantics(x[c(TRUE, FALSE, TRUE, TRUE), c(1, 3, 4)])

  # Empty dimension specifications
  expect_same_semantics(x[c(1, 3, 4),])
  expect_same_semantics(x[1:3, ])
  expect_same_semantics(x[, 1:3])

  # Nested parentheses
  expect_same_semantics(x[((1:3)), (1:2)])

  # Mixed vector types
  expect_same_semantics(x[c(1, 3), c(1:2)])

  # Matrix indexing
  i <- cbind(1:3, c(3, 3, 4))
  # storage.mode(i) <- "integer"
  expect_same_semantics(x[i])

  # Single dimension subsetting
  indices_x <- array(1:2)
  expect_same_semantics(x[indices_x, indices_x])

  # More complex combinations
  expect_same_semantics(x[c(TRUE, FALSE, TRUE, TRUE), c(1:2)])
  expect_same_semantics(x[c(TRUE, FALSE, TRUE, TRUE), c(1, 3)])
  expect_same_semantics(x[1:3, 1:3])

  # Tensor-specific functions (assuming tuple is defined)
  indices_x <- array(1:2)
  r0 <- xr[indices_x, indices_x]
  r1 <- xt@r[indices_x, indices_x]
  r2 <- xt@r[indices_x]@r[,indices_x]
  r3 <- xt@r[tuple(indices_x ,indices_x)]
  expect_equal_array(r0, r1)
  expect_equal_array(r0, r2)
  expect_equal_array(r0, r3)

  expect_equal_array(xt>10, xr>10)
  expect_equal_array(sort(xr[xr>10]), op_sort(xt[xt>10]))
  expect_equal_array(sort(xr[xr>10]), op_sort(xt@r[xt>10]))
  expect_equal_array(sort(xr[xr>10]), op_sort(xt@py[xt>10]))
  # search order is different
  expect_equal_array(t(xr)[t(xr)>10], xt[xt>10])

})



test_that("op_subset() works", {
  # test pythonic features

  xr <- array(1:20, c(4, 5))
  xt <- op_convert_to_tensor(xr)

  # .. and newaxis
  expect_equal(op_shape(xt[newaxis]), shape(1, 4, 5))
  expect_equal(op_shape(xt[newaxis, ..]), shape(1, 4, 5))
  expect_equal(op_shape(xt[newaxis, .., newaxis]), shape(1, 4, 5, 1))

  # negative numbers
  expect_equal_array(xt@r[-1], xr[4,])
  expect_equal_array(xt@r[-2], xr[3,])
  expect_equal_array(xt@r[1:-1], xr[1:4,])
  expect_equal_array(xt@r[1:-1], xr[, ])
  expect_equal_array(xt@r[1:-2], xr[1:3,])
  expect_equal_array(xt@r[2:NA], xr[2:4,])


})

