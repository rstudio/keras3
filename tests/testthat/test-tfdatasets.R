context("tfdatasets")


test_succeeds("Use tfdatasets to train a keras model", {

  model <- keras_model_sequential() %>%
    layer_dense(units = 1, input_shape = 1)
  model %>% compile(loss='mse', optimizer='sgd')

  dataset <- tfdatasets::tensors_dataset(reticulate::tuple(list(1), list(1))) %>%
    tfdatasets::dataset_repeat(100) %>%
    tfdatasets::dataset_shuffle(buffer_size = 100) %>%
    tfdatasets::dataset_batch(10)

  if (tensorflow::tf_version() >= "2.0") {
    model %>% fit(dataset, epochs = 2)
    evaluate(model, dataset)
    preds <- predict(model, dataset)
  } else {
    model %>% fit(dataset, epochs = 2, steps_per_epoch = 5)
    evaluate(model, dataset, steps = 10)
    preds <- predict(model, dataset, steps = 10)
  }

})

test_that("Error when specifying batch_size with tfdatasets", {
  skip_if_no_keras()
  if (!is_tensorflow_implementation())
    skip("Datasets need TensorFlow implementation.")

  model <- keras_model_sequential() %>%
    layer_dense(units = 1, input_shape = 1)
  model %>% compile(loss='mse', optimizer='sgd')

  dataset <- tfdatasets::tensors_dataset(reticulate::tuple(list(1), list(1))) %>%
    tfdatasets::dataset_repeat(100) %>%
    tfdatasets::dataset_shuffle(buffer_size = 100) %>%
    tfdatasets::dataset_batch(10)

  expect_error(
    model %>% fit(dataset, epochs = 2, batch_size = 5)
  )

})


test_succeeds("Works with tf$distribute", {

  if (tensorflow::tf_version() < "1.14.0")
    skip("tf$distribute is not available in TF prior to v1.14")

  strategy <- tensorflow::tf$distribute$MirroredStrategy()

  with (strategy$scope(), {

    model <- keras_model_sequential() %>%
      layer_dense(units = 1, input_shape = 1)
    model %>% compile(loss='mse', optimizer='sgd')

  })

  dataset <- tfdatasets::tensors_dataset(reticulate::tuple(list(1), list(1))) %>%
    tfdatasets::dataset_repeat(100) %>%
    tfdatasets::dataset_shuffle(buffer_size = 100) %>%
    tfdatasets::dataset_batch(10)

  # no clean way to silence the massively verbose output this test produces
  # TF_CPP_MIN_LOG_LEVEL is only effective if set before tf is initialized.
  # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
  # https://github.com/tensorflow/tensorflow/issues/45157

  # (function() {
  #   ol <- Sys.getenv("TF_CPP_MIN_LOG_LEVEL")
  #   Sys.setenv("TF_CPP_MIN_LOG_LEVEL" = "3")
  #   on.exit(Sys.setenv("TF_CPP_MIN_LOG_LEVEL" = ol))
  model %>%
    fit(dataset, epochs = 10, verbose = 0)
  # })()

})
