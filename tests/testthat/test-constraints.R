context("constraints")

source("utils.R")

test_constraint <- function(name) {
  constraint_fn <- eval(parse(text = name))
  test_call_succeeds(name, {
    keras_model_sequential() %>% 
      layer_dense(32, input_shape = c(784), 
                  kernel_constraint = constraint_fn(),
                  bias_constraint = constraint_fn())
  }) 
}


test_constraint("constraint_maxnorm")
test_constraint("constraint_minmaxnorm")
test_constraint("constraint_nonneg")
test_constraint("constraint_unitnorm")



