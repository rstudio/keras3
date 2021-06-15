test_succeeds("multi_head_attention", {

  if (tensorflow::tf_version() < "2.4")
    skip("requires tf_version() >= 2.4")

  layer <- layer_multi_head_attention(num_heads=2, key_dim=2, name = "hello")
  target <- layer_input(shape=c(8, 16))
  source <- layer_input(shape=c(4, 16))

  expect_equal(layer$name, "hello")

  c(output_tensor, weights) %<-% layer(target, source,return_attention_scores=TRUE)

  expect_equal(output_tensor$shape$as_list(), list(NULL, 8, 16))
  expect_equal(weights$shape$as_list(), list(NULL, 2, 8, 4))
})
