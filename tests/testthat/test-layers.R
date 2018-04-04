context("layers")

source("utils.R")

test_call_succeeds("layer_input", {
  layer_input(shape = c(32))
})

test_call_succeeds("layer_dense", {
  layer_dense(keras_model_sequential(), 32, input_shape = c(784))
})

test_call_succeeds("layer_activation", {
  keras_model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activation('relu')
})

test_call_succeeds("layer_activation_leaky_relu", {
  keras_model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activation_leaky_relu()
})

test_call_succeeds("layer_activation_parametric_relu", {
  keras_model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activation_parametric_relu()
})

test_call_succeeds("layer_activation_thresholded_relu", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_activation_thresholded_relu()
})

test_call_succeeds("layer_activation_elu", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_activation_elu()
})

test_call_succeeds("layer_activity_regularization", {
  skip_if_tensorflow_implementation()
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_activity_regularization()
})

test_call_succeeds("layer_dropout", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_dropout(rate = 0.5, noise_shape = c(1))
})

test_call_succeeds("layer_spatial_dropout_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_spatial_dropout_1d(rate = 0.5)
})

test_call_succeeds("layer_spatial_dropout_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_spatial_dropout_2d(rate = 0.5)
})

test_call_succeeds("layer_spatial_dropout_3d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,2,2,4)) %>%
    layer_spatial_dropout_3d(rate = 0.5)
})


test_call_succeeds("layer_lambda", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_lambda(function(t) t, output_shape = c(784))
})

test_call_succeeds("layer_masking", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_masking(mask_value = 0.5)
})

test_call_succeeds("layer_repeat_vector", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_repeat_vector(3)
})


test_call_succeeds("layer_reshape", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16))
})

test_call_succeeds("layer_permute", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_permute(dims = c(1))
})

test_call_succeeds("layer_flatten", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_flatten()
})

test_call_succeeds("layer_conv_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_conv_1d(filters = 3, kernel_size = 2)
})

test_call_succeeds("layer_conv_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_conv_2d(filters = 3, kernel_size = c(2, 2))
})

test_call_succeeds("layer_conv_3d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,2,2,4)) %>%
    layer_conv_3d(filters = 3, kernel_size = c(2, 2, 2))
})

test_call_succeeds("layer_conv_2d_transpose", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_conv_2d_transpose(filters = 3, kernel_size = c(2, 2))
})

test_call_succeeds("layer_conv_3d_transpose", required_version = "2.0.6", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,2,2,4)) %>%
    layer_conv_3d_transpose(filters = 3, kernel_size = c(2, 2, 2))
})

test_call_succeeds("layer_separable_conv_2d", {
  if (is_tensorflow_implementation()) {
    keras_model_sequential() %>%
      layer_dense(32, input_shape = c(784)) %>%
      layer_reshape(target_shape = c(2,4,4)) %>%
      layer_separable_conv_2d(filters = 4, kernel_size = c(2,2))
  }
})

test_call_succeeds("layer_depthwise_conv_2d", required_version = "2.1.5", {
  if (is_tensorflow_implementation()) {
    keras_model_sequential() %>%
      layer_dense(32, input_shape = c(784)) %>%
      layer_reshape(target_shape = c(2,4,4)) %>%
      layer_depthwise_conv_2d(kernel_size = c(2,2))
  }
})


test_call_succeeds("layer_conv_lstm_2d", {
  keras_model_sequential() %>%
    layer_dense(16, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,2,2,2)) %>%
    layer_conv_lstm_2d(filters = 3, kernel_size = c(1, 1))
})

test_call_succeeds("layer_upsampling_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_upsampling_1d()
})

test_call_succeeds("layer_upsampling_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_upsampling_2d()
})

test_call_succeeds("layer_upsampling_3d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,2,2)) %>%
    layer_upsampling_3d()
    
})


test_call_succeeds("layer_zero_padding_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_zero_padding_1d()
})

test_call_succeeds("layer_zero_padding_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_zero_padding_2d()
})


test_call_succeeds("layer_zero_padding_3d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,2,2)) %>%
    layer_zero_padding_3d()
})


test_call_succeeds("layer_cropping_1d", {
  skip_if_cntk() # crashes CNTK
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_cropping_1d()
})

test_call_succeeds("layer_cropping_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_cropping_2d()
})


test_call_succeeds("layer_cropping_3d", {
  skip_if_cntk() # crashes CNTK
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,2,2)) %>%
    layer_cropping_3d()
})

test_call_succeeds("layer_max_pooling_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_max_pooling_1d()
})

test_call_succeeds("layer_max_pooling_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_max_pooling_2d()
})


test_call_succeeds("layer_max_pooling_3d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,2,2,4)) %>%
    layer_max_pooling_3d()
})

test_call_succeeds("layer_average_pooling_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_average_pooling_1d()
})

test_call_succeeds("layer_average_pooling_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_average_pooling_2d()
})


test_call_succeeds("layer_average_pooling_3d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,2,2,4)) %>%
    layer_average_pooling_3d()
})

test_call_succeeds("layer_global_average_pooling_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_global_average_pooling_1d()
})

test_call_succeeds("layer_global_average_pooling_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_global_average_pooling_2d()
})


test_call_succeeds("layer_global_average_pooling_3d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,2,2,4)) %>%
    layer_global_average_pooling_3d()
})

test_call_succeeds("layer_global_max_pooling_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_global_max_pooling_1d()
})

test_call_succeeds("layer_global_max_pooling_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_global_max_pooling_2d()
})


test_call_succeeds("layer_global_max_pooling_3d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,2,2,4)) %>%
    layer_global_max_pooling_3d()
})

test_call_succeeds("layer_locally_connected_1d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_locally_connected_1d(filters = 3, kernel_size = 2)
})

test_call_succeeds("layer_locally_connected_2d", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4)) %>%
    layer_locally_connected_2d(filters = 3, kernel_size = c(2, 2))
})

test_call_succeeds("layer_simple_rnn", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_simple_rnn(units = 2)
})

test_call_succeeds("layer_gru", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_gru(units = 2)
})

test_call_succeeds("layer_lstm", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_lstm(units = 2)
})

test_call_succeeds("layer_embedding", {
  keras_model_sequential() %>%
    layer_embedding(1000, 64, input_length = 10)
})

get_merge_inputs <- function() {
  c(layer_input(shape = c(4, 5)),
    layer_input(shape = c(4, 5)),
    layer_input(shape = c(4, 5)))
}


test_call_succeeds("layer_add", {
  merge_inputs <- get_merge_inputs()
  output <- layer_add(merge_inputs)
  keras_model(merge_inputs, output)
})

test_call_succeeds(required_version = "2.0.7", "layer_subtract", {
  merge_inputs <- c(layer_input(shape = c(4, 5)),
                    layer_input(shape = c(4, 5)))
  output <- layer_subtract(merge_inputs)
  keras_model(merge_inputs, output)
})

test_call_succeeds("layer_multiply", {
  merge_inputs <- get_merge_inputs()
  output <- layer_multiply(merge_inputs)
  keras_model(merge_inputs, output)
})

test_call_succeeds("layer_maximum", {
  merge_inputs <- get_merge_inputs()
  output <- layer_maximum(merge_inputs)
  keras_model(merge_inputs, output)
})

test_call_succeeds("layer_minumum", required_version = "2.0.9", {
  merge_inputs <- get_merge_inputs()
  output <- layer_minimum(merge_inputs)
  keras_model(merge_inputs, output)
})

test_call_succeeds("layer_average", {
  merge_inputs <- get_merge_inputs()
  output <- layer_average(merge_inputs)
  keras_model(merge_inputs, output)
})

test_call_succeeds("layer_concatenate", {
  merge_inputs <- get_merge_inputs()
  output <- layer_concatenate(merge_inputs)
  keras_model(merge_inputs, output)
})

test_call_succeeds("layer_batch_normalization", {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_batch_normalization()
})

test_call_succeeds("layer_gaussian_noise", {
  skip_if_cntk()
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_gaussian_noise(stddev = 0.5)
})

test_call_succeeds("layer_gaussian_dropout", {
  skip_if_cntk()
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_gaussian_dropout(rate = 0.5)
})

test_call_succeeds("layer_alpha_dropout", required_version = "2.0.6", {
  skip_if_cntk()
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,16)) %>%
    layer_alpha_dropout(rate = 0.5)
})



test_call_succeeds("time_distributed", {
  keras_model_sequential() %>%
    time_distributed(layer_dense(units = 8), input_shape = c(10, 16))
})

test_call_succeeds("bidirectional", {
  keras_model_sequential() %>%
    bidirectional(layer_lstm(units = 10, return_sequences = TRUE), input_shape = c(5, 10)) %>%
    bidirectional(layer_lstm(units = 10)) %>%
    layer_dense(units = 5) %>%
    layer_activation(activation = "softmax")
})

test_call_succeeds("layer_activation_softmax", required_version = "2.1.3", {
  if (is_tensorflow_implementation()) {
    keras_model_sequential() %>% 
      layer_dense(32, input_shape = c(784)) %>% 
      layer_activation_softmax()
  }
})

test_call_succeeds("layer_separable_conv_1d", required_version = "2.1.3", {
  if (is_tensorflow_implementation()) {
    keras_model_sequential() %>% 
      layer_dense(32, input_shape = c(784)) %>%
      layer_reshape(target_shape = c(4, 8)) %>%
      layer_separable_conv_1d(filters = 4, kernel_size = c(4))
  }
})



