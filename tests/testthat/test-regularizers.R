context("regularizers")

source("utils.R")

test_regularizer <- function(name) {
  regularizer_fn <- eval(parse(text = name))
  test_call_succeeds(name, {
    keras_model_sequential() %>% 
      layer_dense(32, input_shape = c(784), 
                  kernel_regularizer = regularizer_fn(),
                  activity_regularizer = regularizer_fn())
  }) 
}

test_regularizer("regularizer_l1")
test_regularizer("regularizer_l1_l2")
test_regularizer("regularizer_l2")



