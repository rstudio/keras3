test_that("adaptive pooling layers produce requested spatial shapes", {
  skip_if_no_keras("3.15.1")

  cases <- list(
    list(layer_adaptive_average_pooling_1d, c(2, 8, 3), 4, c(2L, 4L, 3L)),
    list(layer_adaptive_average_pooling_2d, c(2, 8, 6, 3), c(4, 2), c(2L, 4L, 2L, 3L)),
    list(layer_adaptive_average_pooling_3d, c(2, 8, 6, 4, 3), c(4, 3, 2), c(2L, 4L, 3L, 2L, 3L)),
    list(layer_adaptive_max_pooling_1d, c(2, 8, 3), 4, c(2L, 4L, 3L)),
    list(layer_adaptive_max_pooling_2d, c(2, 8, 6, 3), c(4, 2), c(2L, 4L, 2L, 3L)),
    list(layer_adaptive_max_pooling_3d, c(2, 8, 6, 4, 3), c(4, 3, 2), c(2L, 4L, 3L, 2L, 3L))
  )

  for (case in cases) {
    layer_constructor <- case[[1]]
    input_shape <- case[[2]]
    output_size <- case[[3]]
    expected_shape <- case[[4]]
    x <- op_reshape(op_arange(prod(input_shape)), input_shape)

    output <- layer_constructor(x, output_size = output_size)

    expect_equal(unlist(shape(output)), expected_shape)
  }
})
