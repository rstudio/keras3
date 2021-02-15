context("losses")



test_loss <- function(name, test_direct_call = TRUE) {
  
  loss_fn_name <- paste0("loss_", name)
  loss_fn <- eval(parse(text = loss_fn_name))
  test_call_succeeds(name, {

    keras_model_sequential() %>% 
      layer_dense(32, input_shape = c(784)) %>% 
      layer_dropout(rate = 0.5) %>% 
      compile( 
        optimizer = optimizer_sgd(),
        loss = loss_fn, 
        metrics='accuracy'
      )
    if (test_direct_call) {
      y_true <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
      y_pred <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
      loss_fn(y_true, y_pred)
    }
  }) 
}


test_loss("mean_squared_error")
test_loss("mean_absolute_error")
test_loss("mean_absolute_percentage_error")
test_loss("mean_squared_logarithmic_error")
test_loss("squared_hinge")
test_loss("hinge")
test_loss("categorical_crossentropy")
test_loss("sparse_categorical_crossentropy", test_direct_call = FALSE)
test_loss("binary_crossentropy")
test_loss("kullback_leibler_divergence")
test_loss("poisson")
test_loss("cosine_similarity", test_direct_call = FALSE)

test_succeeds("binary_crossentropy new args", {
  
  y_true <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
  y_pred <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
  
  if (tensorflow::tf_version() >= "2.2") {
    out <- loss_binary_crossentropy(y_true, y_pred, from_logits = TRUE, label_smoothing = 0.5)  
  } else {
    expect_warning(
      out <- loss_binary_crossentropy(y_true, y_pred, from_logits = TRUE, label_smoothing = 0.5)    
    )
  }
  
  expect_equal(out$shape$as_list(),10)
})

