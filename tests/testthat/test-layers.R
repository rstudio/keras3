context("layers")

source("utils.R")


test_call <- function(layer_name, expr) {
  test_that(paste(layer_name, "call succeeds"), {
    skip_if_no_keras()
    force(expr)
    expect_equal(TRUE, TRUE)
  })
}

test_call("layer_input", {
  layer_input(shape = shape(32))
})

test_call("layer_dense", {
  layer_dense(model_sequential(), 32, input_dim = 784)
})

test_call("layer_activation", {
  model_sequential() %>% 
    layer_dense(32, input_dim = 784) %>% 
    layer_activation('relu')
})

test_call("layer_reshape", {
  model_sequential() %>% 
    layer_dense(32, input_dim = 784) %>% 
    layer_reshape(target_shape = c(2,16))
})
 
test_call("layer_permute", {
  model_sequential() %>% 
    layer_dense(32, input_dim = 784) %>% 
    layer_permute(dims = 1)
})





