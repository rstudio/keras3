test_that("reversible embedding projects in both directions", {
  embedding <- layer_reversible_embedding(input_dim = 8, output_dim = 4)
  token_ids <- op_array(matrix(c(0L, 1L, 2L, 3L), nrow = 1), dtype = "int32")

  hidden_states <- embedding(token_ids)
  logits <- embedding(hidden_states, reverse = TRUE)

  expect_equal(unlist(shape(hidden_states)), c(1L, 4L, 4L))
  expect_equal(unlist(shape(logits)), c(1L, 4L, 8L))
})
