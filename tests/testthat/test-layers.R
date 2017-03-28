context("layers")

source("utils.R")

test_call_succeeds("layer_input", {
  layer_input(shape = shape(32))
})

test_call_succeeds("layer_dense", {
  layer_dense(model_sequential(), 32, input_shape = shape(784))
})

test_call_succeeds("layer_activation", {
  model_sequential() %>% 
    layer_dense(32, input_shape = shape(784)) %>% 
    layer_activation('relu')
})

test_call_succeeds("layer_activation_leaky_relu", {
  model_sequential() %>% 
    layer_dense(32, input_shape = shape(784)) %>% 
    layer_activation_leaky_relu()
})

test_call_succeeds("layer_activation_parametric_relu", {
  model_sequential() %>% 
    layer_dense(32, input_shape = shape(784)) %>% 
    layer_activation_parametric_relu()
})

test_call_succeeds("layer_activation_thresholded_relu", {
  model_sequential() %>% 
    layer_dense(32, input_shape = shape(784)) %>% 
    layer_activation_thresholded_relu()
})

test_call_succeeds("layer_activation_elu", {
  model_sequential() %>% 
    layer_dense(32, input_shape = shape(784)) %>% 
    layer_activation_elu()
})

test_call_succeeds("layer_reshape", {
  model_sequential() %>% 
    layer_dense(32, input_shape = shape(784)) %>% 
    layer_reshape(target_shape = shape(2,16))
})
 
test_call_succeeds("layer_permute", {
  model_sequential() %>% 
    layer_dense(32, input_shape = shape(784)) %>% 
    layer_permute(dims = 1)
})

test_call_succeeds("layer_flatten", {
  model_sequential() %>% 
    layer_dense(32, input_shape = shape(784)) %>% 
    layer_reshape(target_shape = shape(2,16)) %>% 
    layer_flatten()
})





