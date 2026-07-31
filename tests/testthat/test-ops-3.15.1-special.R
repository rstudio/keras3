test_that("adaptive pooling and spatial rearrangement ops are available", {
  skip_if_no_keras("3.15.1")

  image <- op_ones(c(1, 4, 4, 1))
  expect_tensor(
    op_adaptive_average_pool(image, c(2, 2)),
    shape = c(1L, 2L, 2L, 1L)
  )
  expect_tensor(
    op_adaptive_max_pool(image, c(2, 2)),
    shape = c(1L, 2L, 2L, 1L)
  )

  pixels <- op_reshape(op_arange(16), c(1, 2, 2, 4))
  shuffled <- op_depth_to_space(pixels, block_size = 2)
  expect_tensor(shuffled, shape = c(1L, 4L, 4L, 1L))
  expect_equal(
    as.array(op_space_to_depth(shuffled, block_size = 2)),
    as.array(pixels)
  )
})

test_that("fold and unfold round-trip non-overlapping patches", {
  skip_if_no_keras("3.15.1")

  image <- op_ones(c(1, 2, 4, 4))
  patches <- op_unfold(image, kernel_size = 2, stride = 2)
  expect_tensor(patches, shape = c(1L, 8L, 4L))

  restored <- op_fold(
    patches,
    output_size = c(4, 4),
    kernel_size = 2,
    stride = 2
  )
  expect_equal(as.array(restored), as.array(image))
})

test_that("forward autodiff and segment reductions use R conventions", {
  skip_if_no_keras("3.15.1")

  x <- op_convert_to_tensor(3)
  result <- op_jvp(\(x) op_square(x), list(x), list(op_convert_to_tensor(1)))
  expect_equal(as.numeric(result[[1]]), 9)
  expect_equal(as.numeric(result[[2]]), 6)

  data <- op_array(c(2, 3, 5, 7))
  segment_ids <- op_array(c(1, 1, 2, 2), dtype = "int32")
  expect_equal(as.numeric(op_segment_min(data, segment_ids)), c(2, 5))
  expect_equal(as.numeric(op_segment_prod(data, segment_ids)), c(6, 35))
})
