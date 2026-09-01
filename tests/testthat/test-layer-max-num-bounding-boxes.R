test_that("maximum bounding boxes forwards padding_value", {
  skip_if_no_keras("3.14.0")

  layer <- layer_max_num_bounding_boxes(
    max_number = 3L,
    padding_value = -7L
  )

  expect_identical(layer$fill_value, -7L)
})
