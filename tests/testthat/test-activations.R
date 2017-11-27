context("activations")

source("utils.R")

test_activation <- function(name, required_version = NULL) {
  test_succeeds(paste("use activation", name), {
    skip_if_no_keras(required_version)
    activation_fn <- eval(parse(text = name))
    test_call_succeeds(name, {
      keras_model_sequential() %>% 
        layer_dense(32, input_shape = 784) %>% 
        layer_activation(activation = activation_fn)
    }) 
    tensor <- k_constant(matrix(runif(100), nrow = 10, ncol = 10), shape = c(10, 10))
    activation_fn(tensor)
  })
}


test_activation("activation_elu")
test_activation("activation_selu", required_version = "2.0.6")
test_activation("activation_hard_sigmoid")
test_activation("activation_linear")
test_activation("activation_relu")
test_activation("activation_sigmoid")
test_activation("activation_softmax")
test_activation("activation_softplus")
test_activation("activation_softsign")
test_activation("activation_tanh")



