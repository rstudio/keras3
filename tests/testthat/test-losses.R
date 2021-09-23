context("losses")



test_loss <- function(name, test_direct_call = TRUE, test_callable_call = test_direct_call) {

  loss_fn_name <- paste0("loss_", name)
  loss_fn <- eval(parse(text = loss_fn_name))
  test_call_succeeds(name, {

    # pass loss_fn
    keras_model_sequential() %>%
      layer_dense(32, input_shape = c(784)) %>%
      layer_dropout(rate = 0.5) %>%
      compile(
        optimizer = optimizer_sgd(),
        loss = loss_fn,
        metrics='accuracy'
      )

    # pass loss_fn()
    keras_model_sequential() %>%
      layer_dense(32, input_shape = c(784)) %>%
      layer_dropout(rate = 0.5) %>%
      compile(
        optimizer = optimizer_sgd(),
        loss = loss_fn(),
        metrics='accuracy'
      )

    y_true <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
    y_pred <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
    if (test_direct_call)
      loss_fn(y_true, y_pred)
    if (test_callable_call) {
      callable <- loss_fn()
      callable(y_true, y_pred)
    }
  })
}




test_loss("binary_crossentropy")
test_loss("categorical_crossentropy")
test_loss("categorical_hinge")
test_loss("cosine_similarity", test_direct_call = FALSE)
test_loss("hinge")
test_loss("kl_divergence")
test_loss("kullback_leibler_divergence")
test_loss("logcosh")
test_loss("mean_absolute_error")
test_loss("mean_absolute_percentage_error")
test_loss("mean_squared_error")
test_loss("mean_squared_logarithmic_error")
test_loss("poisson")
test_loss("sparse_categorical_crossentropy", test_direct_call = FALSE)
test_loss("squared_hinge")
if(tf_version() >= "2.3")
  test_loss("huber")

## deprecated
expect_warning(loss_cosine_proximity(), "cosine_similarity")
expect_warning(loss_cosine_proximity(random_array(c(3, 4)), random_array(c(3, 4))),
               "cosine_similarity")

# names(asNamespace("keras")) %>%
#   grep("^loss_", ., value = TRUE) %>%
#   sub("^loss_", "", .) %>%
#   sort() %>%
#   sprintf('test_loss(name = "%s")', .) %>%
#   writeLines()


test_succeeds("binary_crossentropy new args", {

  y_true <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))
  y_pred <- k_constant(matrix(runif(100), nrow = 10, ncol = 10))

    out <- loss_binary_crossentropy(y_true, y_pred, from_logits = TRUE, label_smoothing = 0.5)

  expect_equal(out$shape$as_list(),10)
})



test_succeeds("passing R fn to compile(loss=)", {
  # passing an R function to compile(loss = r_fn) can sometimes
  # result in the py_func having a malformed name and keras throwing an error
  model <- define_model()

  # generate dummy training data
  N <- 10
  data <- matrix(rexp(N * 784), nrow = N, ncol = 784)
  labels <- matrix(round(runif(N * 10, min = 0, max = 9)),
                   nrow = N, ncol = 10)


  new_loss_fn <- function() {
    function(y_true, y_pred, ...)
      loss_binary_crossentropy(y_true, y_pred, ...)
  }

  compile(model,
          loss = new_loss_fn(),
          optimizer = optimizer_sgd(),
          metrics = 'accuracy')

  fit(model, data, labels, epochs = 2)

})
