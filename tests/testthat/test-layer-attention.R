test_succeeds("multi_head_attention", {
#
  if (tensorflow::tf_version() < "2.4")
    skip("requires tf_version() >= 2.4")
# devtools::load_all()
  layer <- layer_multi_head_attention(num_heads=2, key_dim=2, name = "hello")
  target <- layer_input(shape=c(8, 16))
  source <- layer_input(shape=c(4, 16))

  expect_equal(layer$name, "hello")

  skip("MultiHeadAttention upstream bug")

  c(output_tensor, weights) %<-%
    layer(target, source, return_attention_scores=TRUE)

    ## FOUND IT:
    ## MultiHeadAttention.compute_output_shape() method doesn't
    ## take the 'return_attention_scores' kwarg, doesn't do anything,
    ## returns the wrong result.
    ##
    # tracked it to Operation.symbolic_call(), which seems to never
    # actually invoke the MultiHeadAttention.call() function, only
    # lazily get the output shape somehow...
    # actually, it calls compute_output_spec() for the output shape

    # tracked this to the Operation.__call__ method.
    # layer.__call__ method eventually
    # calls Operation.__call__ w/ the right kwarg `return_attention_score: True`,
    # calling layer.call() directly gives the correct result
    # so the bug is somewhere between Operation.__call__ entry and layer.call() invocation.
#
  expect_equal(output_tensor$shape$as_list(), list(NULL, 8, 16))
  expect_equal(weights$shape$as_list(), list(NULL, 2, 8, 4))
})
test_that("attention layers preserve gated configurations", {
  skip_if_no_keras("3.14.0")

  grouped <- layer_group_query_attention(
    head_dim = 4L,
    num_query_heads = 4L,
    num_key_value_heads = 2L,
    use_gate = TRUE
  )
  multi <- layer_multi_head_attention(
    num_heads = 4L,
    key_dim = 4L,
    use_gate = TRUE
  )

  expect_true(grouped$get_config()$use_gate)
  expect_true(multi$get_config()$use_gate)
})
