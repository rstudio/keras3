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
    ) %>% 
    fit(x = matrix(0, ncol = 784, nrow = 100), y = matrix(0, ncol = 10, nrow = 100), 
        epochs = 1, verbose = 0)
})

test_succeeds("custom metrics can be used when compiling models", {
  
  metric_mean_pred <- custom_metric("mean_pred", function(y_true, y_pred) {
    k_mean(y_pred) 
  })
  
  define_model() %>% 
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics=list(
        metric_binary_accuracy,
        metric_binary_crossentropy,
        metric_hinge,
        metric_mean_pred
      )
    ) %>% 
    fit(x = matrix(0, ncol = 784, nrow = 100), y = matrix(0, ncol = 10, nrow = 100),
        epochs = 1, verbose = 0)
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

test_succeeds("metrics for multiple output models", {
  
  input <- layer_input(shape = 1)
  
  output1 <- layer_dense(input, units = 1, name = "out1")
  output2 <- layer_dense(input, units = 1, name = "out2")
  
  model <- keras_model(input, list(output1, output2))
  
  model %>% compile(
    loss = "mse",
    optimizer = "adam",
    metrics = list(out1 = "mse", out2 = "mae")
  )
  
  history <- model %>% fit(
    x = matrix(0, ncol = 1, nrow = 100),
    y = list(rep(0, 100), rep(0, 100)),
    epochs = 1
  )
  
  if (tensorflow::tf_version() < "2.0") {
    expect_true(all(c("out2_mean_absolute_error", "out1_mean_squared_error") %in% names(history$metrics)))
    expect_true(all(!c("out1_mean_absolute_error", "out2_mean_squared_error") %in% names(history$metrics)))  
  } else {
    expect_true(all(c("out2_mae", "out1_mse") %in% names(history$metrics)))
    expect_true(all(!c("out1_mae", "out2_mse") %in% names(history$metrics)))  
  }
  
})


test_succeeds("get warning when passing using named list of metrics", {
  
  input <- layer_input(shape = 1)
  
  output1 <- layer_dense(input, units = 1, name = "out1")
  output2 <- layer_dense(input, units = 1, name = "out2")
  
  model <- keras_model(input, list(output1, output2))
  
  expect_warning({
    model %>% compile(
      loss = "mse",
      optimizer = "adam",
      metrics = list("metric1" = function(y_true, y_pred) k_mean(y_pred))
    )  
  })
  
})