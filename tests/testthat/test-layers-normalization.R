context("normalization")
test_succeeds("layer normalization", {

  d <- matrix(1:10, ncol = 2, nrow = 5, byrow = TRUE)*10
  data <- tensorflow::tf$constant(d, dtype=tensorflow::tf$float32)

  layer <- layer_layer_normalization(axis=1)
  output <- as.matrix(layer(data))

  expect_equal(output[,1], rep(-1+2e-5+2e-8, 5), tolerance = 1e-6)
  expect_equal(output[,2], rep(1-2e-5-2e-8, 5), tolerance = 1e-6)
})

if(tf_version() >= "2.9")
test_succeeds("layer unit normalization", {
  data <- as_tensor(1:6, shape = c(2, 3), dtype = "float64")
  normalized_data <- data %>%
    layer_unit_normalization(dtype = "float64") %>%
    as.array()

  for (row in 1:2)
    expect_equal(sum(normalized_data[row,] ^ 2), 1)
})
