context("utils")



test_call_succeeds("to_categorical", {
  runif(1000, min = 0, max = 9) %>%
    round() %>%
    matrix(nrow = 1000, ncol = 1) %>%
    to_categorical(num_classes = 10)
})


# test_call_succeeds("get_file", {
#   # file moved.
#   get_file("im.jpg",
#            origin = "https://camo.githubusercontent.com/0d08dc4f9466d347e8d28a951ea51e3430c6f92c/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6b657261732e696f2f696d672f6b657261732d6c6f676f2d323031382d6c617267652d313230302e706e67",
#            cache_subdir = "tests")
# })


test_call_succeeds("hdf5_matrix", {

  if (tensorflow::tf_version() >= "2.4")
    skip("hdf5 matrix have been removed in tf >= 2.4")

  if (!keras:::have_h5py())
    skip("h5py not available for testing")

  X_train = hdf5_matrix('test.h5', 'my_data', start=0, end=150)
  y_train = hdf5_matrix('test.h5', 'my_labels', start=0, end=150)
})


test_call_succeeds("normalize", {
  data <- runif(1000, min = 0, max = 9) %>%  round() %>% matrix(nrow = 1000, ncol = 1)
  normalize(data)
})


test_call_succeeds("with_custom_object_scope", {

  if (!keras:::have_h5py())
    skip("h5py not available for testing")


  metric_mean_pred <- custom_metric("mean_pred", function(y_true, y_pred) {
    k_mean(y_pred)
  })

  with_custom_object_scope(c(mean_pred = metric_mean_pred), {

    model <- define_model()

    model %>% compile(
      loss = "binary_crossentropy",
      optimizer = optimizer_nadam(),
      metrics = metric_mean_pred
    )

    tmp <- tempfile("model", fileext = ".hdf5")
    save_model_hdf5(model, tmp)
    model <- load_model_hdf5(tmp)

    # https://github.com/tensorflow/tensorflow/issues/45903#issuecomment-804973541
    # broken in tf 2.4 and 2.5, fixed in nightly already
    if (tf_version() == "2.5")
      model$compile(optimizer=model$optimizer,
                    loss = "binary_crossentropy",
                    metrics = metric_mean_pred)

    # generate dummy training data
    data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
    labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

    model %>% fit(data, labels, epochs = 2, verbose = 0)

  })


})


test_call_succeeds("with_custom_object_scope", {

  gradients <- list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3")
  weights <- list("weight_1", "weight_2", "weight_3")
  expect_identical(zip_lists(gradients, weights),
                   list(
                     list("grad_for_wt_1", "weight_1"),
                     list("grad_for_wt_2", "weight_2"),
                     list("grad_for_wt_3", "weight_3")
                   ))

  expect_identical(zip_lists(gradient = gradients, weight = weights),
                   list(
                     list(gradient = "grad_for_wt_1", weight = "weight_1"),
                     list(gradient = "grad_for_wt_2", weight = "weight_2"),
                     list(gradient = "grad_for_wt_3", weight = "weight_3")
                   ))

  names(gradients) <- names(weights) <- paste0("layer_", 1:3)
  expected <-
    list(
      layer_1 = list("grad_for_wt_1", "weight_1"),
      layer_2 = list("grad_for_wt_2", "weight_2"),
      layer_3 = list("grad_for_wt_3", "weight_3")
    )

  expect_identical(zip_lists(gradients, weights),
                   expected)
  expect_identical(zip_lists(gradients, weights[c(3, 1, 2)]),
                   expected)


  expect_identical(
    zip_lists(gradient = gradients, weight = weights),
    list(
      layer_1 = list(gradient = "grad_for_wt_1", weight = "weight_1"),
      layer_2 = list(gradient = "grad_for_wt_2", weight = "weight_2"),
      layer_3 = list(gradient = "grad_for_wt_3", weight = "weight_3")
    )
  )

  names(gradients) <- paste0("gradient_", 1:3)
  expect_error(zip_lists(gradients, weights)) # error, names don't match
  # call unname directly for positional matching
  expect_identical(zip_lists(unname(gradients), unname(weights)),
                   list(
                     list("grad_for_wt_1", "weight_1"),
                     list("grad_for_wt_2", "weight_2"),
                     list("grad_for_wt_3", "weight_3")
                   ))
})
