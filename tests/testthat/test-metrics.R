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

test_succeeds("metrics can be referred to by executing the method function", {
  define_model() %>% 
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics=list(
        metric_binary_accuracy(),
        metric_binary_crossentropy(),
        metric_hinge()
      )
    )
})

test_succeeds("metrics be can called directly", {
  tf <- tensorflow::tf
  y_true <- tf$constant(matrix(runif(100), nrow = 10, ncol = 10))
  y_pred <- tf$constant(matrix(runif(100), nrow = 10, ncol = 10))
  metric_binary_accuracy(y_true, y_pred)
  metric_binary_crossentropy(y_true, y_pred)
  metric_hinge(y_true, y_pred)
})