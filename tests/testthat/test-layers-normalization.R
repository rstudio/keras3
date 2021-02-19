test_succeeds("layer normalization", {
  
  d <- matrix(1:10, ncol = 2, nrow = 5, byrow = TRUE)*10
  data <- tensorflow::tf$constant(d, dtype=tensorflow::tf$float32)
  
  layer <- layer_layer_normalization(axis=1)
  output <- as.matrix(layer(data))
  
  expect_equal(output[,1], rep(-1+2e-5+2e-8, 5), tolerance = 1e-6)
  expect_equal(output[,2], rep(1-2e-5-2e-8, 5), tolerance = 1e-6)
})
