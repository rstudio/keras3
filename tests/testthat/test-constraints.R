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

test_succeeds("R custom constraints", {
  
  nonneg_constraint <- function(w) {
    w * k_cast(k_greater_equal(w, 0), k_floatx())
  }
  
  CustomNonNegConstraint <- R6::R6Class(
    "CustomNonNegConstraint",
    inherit = KerasConstraint,
    public = list(
      call = nonneg_constraint
    )
  )
  
  model <- keras_model_sequential() %>% 
    layer_dense(32, input_shape = c(784), 
                kernel_constraint = CustomNonNegConstraint$new(),
                bias_constraint = nonneg_constraint) %>% 
    layer_dense(10, activation = 'softmax') %>% 
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics='accuracy'
    )
    
    if (!is_backend("theano")) {
      data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
      labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)
      
      model %>% fit(
        data, labels
      )
    }
})





