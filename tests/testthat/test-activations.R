context("activations")



test_activation <- function(name, required_version = NULL, required_tf_version=tf_version()) {
  test_succeeds(paste("use activation", name), {
    # browser()
    skip_if_no_keras(required_version)
    skip_if_not_tensorflow_version(required_tf_version)
    activation_fn <- eval(parse(text = name))

    # test in a model
    keras_model_sequential(input_shape = 784) %>%
      layer_dense(32) %>%
      layer_activation(activation = activation_fn)

    # test stand alone
    tensor <- op_array(matrix(runif(100), nrow = 10, ncol = 10))
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
test_activation("activation_exponential", required_version = "2.2.3")
test_activation("activation_gelu", required_tf_version = "2.4.1")

skip("activation_swish")
test_activation("activation_swish", required_tf_version = "2.2.0")

# tf$`__version__` tf$keras$`__version__`
# tf-ver keras-ver
# 2.1.4 2.3.0-tf
# 2.2.0 2.3.0-tf
# 2.3.0 2.4.0
# 2.4.1 2.4.0
# 2.5.0 2.5.0
