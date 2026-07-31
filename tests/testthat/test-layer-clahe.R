test_that("CLAHE preserves image shape", {
  image <- op_reshape(op_arange(16, dtype = "float32"), c(1, 4, 4, 1))

  equalized <- layer_contrast_limited_adaptive_histogram_equalization(
    image,
    value_range = c(0, 15),
    tile_grid_size = c(2, 2)
  )

  expect_equal(unlist(shape(equalized)), c(1L, 4L, 4L, 1L))
})
