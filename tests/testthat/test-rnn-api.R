
context("rnn api")


test_succeeds("layer_lstm_cell", {
  # LSTMCell
  inputs <- k_random_normal(c(32, 10, 8))
  rnn <- layer_rnn(cell = layer_lstm_cell(units = 4))
  output <- rnn(inputs)
  expect_equal(dim(output),  c(32, 4))

  rnn <- layer_rnn(
    cell = layer_lstm_cell(units = 4),
    return_sequences = TRUE,
    return_state = TRUE
  )
  c(whole_seq_output, final_memory_state, final_carry_state) %<-% rnn(inputs)

  expect_equal(dim(whole_seq_output)  , c(32, 10, 4))
  expect_equal(dim(final_memory_state), c(32, 4))
  expect_equal(dim(final_carry_state) , c(32, 4))
})

test_succeeds("layer_gru_cell", {
  # GRUCell
 inputs <- k_random_uniform(c(32, 10, 8))
 output <- inputs %>% layer_rnn(layer_gru_cell(4))
 expect_true(output$shape == shape(32, 4))

 rnn <- layer_rnn(cell = layer_gru_cell(4),
                  return_sequences = TRUE,
                  return_state = TRUE)
 c(whole_sequence_output, final_state) %<-% rnn(inputs)

  expect_true(whole_sequence_output$shape == shape(32, 10, 4))
  expect_true(final_state$shape           == shape(32, 4))
})

test_succeeds("layer_rnn", {
  batch_size <- 64
  # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
  # Each input sequence will be of size (28, 28) (height is treated like time).
  input_dim <- 28

  units  <- 64
  output_size <- 10  # labels are from 0 to 9

  # Build the RNN model
  build_model <- function(allow_cudnn_kernel = TRUE) {
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `layer_lstm(units=units)` will use the CuDNN kernel,
    # while layer_rnn(layer_lstm_cell(units)) will run on non-CuDNN kernel.
    if (allow_cudnn_kernel)
      # The LSTM layer with default options uses CuDNN.
      lstm_layer <- layer_lstm(units = units)
    else
      # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
      lstm_layer <- layer_rnn(cell = layer_lstm_cell(units = units))

    model <-
      keras_model_sequential(input_shape = shape(NULL, input_dim)) %>%
      lstm_layer() %>%
      layer_batch_normalization() %>%
      layer_dense(output_size)

    model
  }

  expect_error(build_model(TRUE),  NA)
  expect_error(build_model(FALSE), NA)

})
