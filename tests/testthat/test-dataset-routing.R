test_that("dataset helpers forward backend and format choices", {
  skip_if_no_keras("3.14.0")

  splits <- split_dataset(
    matrix(1:10, nrow = 5),
    left_size = 2L,
    preferred_backend = "tensorflow"
  )
  series <- timeseries_dataset_from_array(
    data = matrix(1:8, ncol = 1),
    targets = NULL,
    sequence_length = 3L,
    batch_size = 2L,
    format = "tf"
  )

  expect_true(inherits(splits[[1]], "tensorflow.python.types.data.DatasetV2"))
  expect_true(inherits(splits[[2]], "tensorflow.python.types.data.DatasetV2"))
  expect_identical(as.integer(splits[[1]]$cardinality()), 2L)
  expect_identical(as.integer(splits[[2]]$cardinality()), 3L)
  expect_true(inherits(series, "tensorflow.python.types.data.DatasetV2"))
  expect_error(timeseries_dataset_from_array(
    data = matrix(1:8, ncol = 1),
    targets = NULL,
    sequence_length = 3L,
    format = "invalid"
  ))
})
