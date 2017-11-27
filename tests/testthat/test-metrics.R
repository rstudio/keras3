context("metrics")

source("utils.R")

test_succeeds("metrics can be used when compiling models", {
  define_model() %>% 
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics=list(
        metric_binary_accuracy,
        metric_binary_crossentropy,
        metric_hinge
      )
    )
})

test_succeeds("metrics be can called directly", {
  y_true <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
  y_pred <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
  metric_binary_accuracy(y_true, y_pred)
  metric_binary_crossentropy(y_true, y_pred)
  metric_hinge(y_true, y_pred)
  
  skip_if_cntk() # top_k doesn't work on CNTK, see 
                 # https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras#known-issues)
  
  y_pred <- k_variable(matrix(c(0.3, 0.2, 0.1, 0.1, 0.2, 0.7), nrow=2, ncol = 3))
  y_true <- k_variable(matrix(c(0L, 1L), nrow = 2, ncol = 1))
  metric_top_k_categorical_accuracy(y_true, y_pred, k = 3)
  if (is_keras_available("2.0.5"))
    metric_sparse_top_k_categorical_accuracy(y_true, y_pred, k = 3)
    
})