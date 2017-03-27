context("regularizers")

source("utils.R")

test_regularizer <- function(name, regularizer) {
  test_call_succeeds(name, {
    model_sequential() %>% 
      layer_dense(32, input_shape = shape(784), 
                  kernel_regularizer = regularizer,
                  activity_regularizer = regularizer)
  }) 
 
}

test_regularizer("regularizer_l1", regularizer_l1())
test_regularizer("regularizer_l1_l2", regularizer_l1_l2())
test_regularizer("regularizer_l2", regularizer_l2())



