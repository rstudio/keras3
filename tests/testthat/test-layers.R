context("layers")

source("utils.R")

test_call_succeeds("layer_input", {
  layer_input(shape = c(32))
})

test_call_succeeds("layer_dense", {
  layer_dense(model_sequential(), 32, input_shape = c(784))
})

test_call_succeeds("layer_activation", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activation('relu')
})

test_call_succeeds("layer_activation_leaky_relu", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activation_leaky_relu()
})

test_call_succeeds("layer_activation_parametric_relu", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activation_parametric_relu()
})

test_call_succeeds("layer_activation_thresholded_relu", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activation_thresholded_relu()
})

test_call_succeeds("layer_activation_elu", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activation_elu()
})

test_call_succeeds("layer_activity_regularization", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_activity_regularization()
})

test_call_succeeds("layer_dropout", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_dropout(rate = 0.5, noise_shape = c(1))
})

test_call_succeeds("layer_spatial_dropout_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_spatial_dropout_1d(rate = 0.5)
})

test_call_succeeds("layer_spatial_dropout_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_spatial_dropout_2d(rate = 0.5)
})

test_call_succeeds("layer_spatial_dropout_3d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,2,2,4)) %>% 
    layer_spatial_dropout_3d(rate = 0.5)
})


test_call_succeeds("layer_lambda", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_lambda(function(t) t)
})

test_call_succeeds("layer_masking", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_masking(mask_value = 0.5)
})

test_call_succeeds("layer_repeat_vector", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_repeat_vector(3)
})


test_call_succeeds("layer_reshape", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16))
})
 
test_call_succeeds("layer_permute", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_permute(dims = 1)
})

test_call_succeeds("layer_flatten", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_flatten()
})

test_call_succeeds("layer_conv_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_conv_1d(filters = 3, kernel_size = 2)
})

test_call_succeeds("layer_conv_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_conv_2d(filters = 3, kernel_size = c(2, 2))
})

test_call_succeeds("layer_conv_3d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,2,2,4)) %>% 
    layer_conv_3d(filters = 3, kernel_size = c(2, 2, 2))
})

test_call_succeeds("layer_conv_2d_transpose", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_conv_2d_transpose(filters = 3, kernel_size = c(2, 2))
})

test_call_succeeds("layer_separable_conv_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_separable_conv_2d(filters = 4, kernel_size = c(2,2))
})


test_call_succeeds("layer_conv_lstm_2d", {
  model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4,4)) %>%
    layer_conv_lstm_2d(filters = 3, kernel_size = c(2, 2))
})

test_call_succeeds("layer_upsampling_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_upsampling_1d()
})

test_call_succeeds("layer_upsampling_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,16)) %>% 
    layer_upsampling_2d()
})

test_call_succeeds("layer_upsampling_3d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4,4)) %>% 
    layer_upsampling_3d()
})


test_call_succeeds("layer_zero_padding_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_zero_padding_1d()
})

test_call_succeeds("layer_zero_padding_2d", {
  model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,16)) %>%
    layer_zero_padding_2d()
})


test_call_succeeds("layer_zero_padding_3d", {
  model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4,4)) %>%
    layer_zero_padding_3d()
})


test_call_succeeds("layer_cropping_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_cropping_1d()
})

test_call_succeeds("layer_cropping_2d", {
  model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,16)) %>%
    layer_cropping_2d()
})


test_call_succeeds("layer_cropping_3d", {
  model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    layer_reshape(target_shape = c(2,4,4,4)) %>%
    layer_cropping_3d()
})

test_call_succeeds("layer_max_pooling_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_max_pooling_1d()
})

test_call_succeeds("layer_max_pooling_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_max_pooling_2d()
})


test_call_succeeds("layer_max_pooling_3d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,2,2,4)) %>% 
    layer_max_pooling_3d()
})

test_call_succeeds("layer_average_pooling_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_average_pooling_1d()
})

test_call_succeeds("layer_average_pooling_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_average_pooling_2d()
})


test_call_succeeds("layer_average_pooling_3d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,2,2,4)) %>% 
    layer_average_pooling_3d()
})

test_call_succeeds("layer_global_average_pooling_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_global_average_pooling_1d()
})

test_call_succeeds("layer_global_average_pooling_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_global_average_pooling_2d()
})


test_call_succeeds("layer_global_average_pooling_3d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,2,2,4)) %>% 
    layer_global_average_pooling_3d()
})

test_call_succeeds("layer_global_max_pooling_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_global_max_pooling_1d()
})

test_call_succeeds("layer_global_max_pooling_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_global_max_pooling_2d()
})


test_call_succeeds("layer_global_max_pooling_3d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,2,2,4)) %>% 
    layer_global_max_pooling_3d()
})

test_call_succeeds("layer_locally_connected_1d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_locally_connected_1d(filters = 3, kernel_size = 2)
})

test_call_succeeds("layer_locally_connected_2d", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,4,4)) %>% 
    layer_locally_connected_2d(filters = 3, kernel_size = c(2, 2))
})

test_call_succeeds("layer_simple_rnn", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_simple_rnn(units = 2)
})

test_call_succeeds("layer_gru", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_gru(units = 2)
})

test_call_succeeds("layer_lstm", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_lstm(units = 2)
})

test_call_succeeds("layer_embedding", {
  model_sequential() %>% 
    layer_embedding(1000, 64, input_length = 10)
})

merge_inputs <- c(
  layer_input(shape = c(4, 5)),
  layer_input(shape = c(4, 5)),
  layer_input(shape = c(4, 5))
)


test_call_succeeds("layer_add", {
  output <- layer_add(merge_inputs)
  model(merge_inputs, output)
})

test_call_succeeds("layer_multiply", {
  output <- layer_multiply(merge_inputs)
  model(merge_inputs, output)
})

test_call_succeeds("layer_maximum", {
  output <- layer_maximum(merge_inputs)
  model(merge_inputs, output)
})

test_call_succeeds("layer_average", {
  output <- layer_average(merge_inputs)
  model(merge_inputs, output)
})

test_call_succeeds("layer_concatenate", {
  output <- layer_concatenate(merge_inputs)
  model(merge_inputs, output)
})

test_call_succeeds("layer_batch_normalization", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_batch_normalization()
})

test_call_succeeds("layer_gaussian_noise", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_gaussian_noise(stddev = 0.5)
})


test_call_succeeds("layer_gaussian_dropout", {
  model_sequential() %>% 
    layer_dense(32, input_shape = c(784)) %>% 
    layer_reshape(target_shape = c(2,16)) %>% 
    layer_gaussian_dropout(rate = 0.5)
})


test_call_succeeds("time_distributed", {
  model_sequential() %>%
    time_distributed(layer_dense(units = 8), input_shape = c(10, 16))
})

test_call_succeeds("bidirectional", {
  model_sequential() %>%
    bidirectional(layer_lstm(units = 10, return_sequences = TRUE), input_shape = c(5, 10)) %>%
    bidirectional(layer_lstm(units = 10)) %>%
    layer_dense(units = 5) %>%
    layer_activation(activation = "softmax")
})





