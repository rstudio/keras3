
context("model-persistence")



test_succeeds("model can be saved and loaded", {

  if (!keras:::have_h5py())
    skip("h5py not available for testing")

  model <- define_and_compile_model()
  tmp <- tempfile("model", fileext = ".hdf5")
  save_model_hdf5(model, tmp)
  model <- load_model_hdf5(tmp)
})

test_succeeds("model with custom loss and metrics can be saved and loaded", {
  
  if (!keras:::have_h5py())
    skip("h5py not available for testing")
  
  model <- define_model()
  
  metric_mean_pred <- custom_metric("mean_pred", function(y_true, y_pred) {
    k_mean(y_pred) 
  })
  
  custom_loss <- function(y_pred, y_true) {
    loss_categorical_crossentropy(y_pred, y_true)
  }
  
  model %>% compile(
    loss = custom_loss,
    optimizer = optimizer_nadam(),
    metrics = metric_mean_pred
  )
  
  tmp <- tempfile("model", fileext = ".hdf5")
  save_model_hdf5(model, tmp)
  model <- load_model_hdf5(tmp, custom_objects = c(mean_pred = metric_mean_pred,
                                                   custom_loss = custom_loss))
  
  # generate dummy training data
  data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
  labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)
  
  
  model %>% fit(data, labels, epochs = 2, verbose = 0)
  
})

test_succeeds("model weights can be saved and loaded", {

  if (!keras:::have_h5py())
    skip("h5py not available for testing")

  model <- define_and_compile_model()
  tmp <- tempfile("model", fileext = ".hdf5")
  save_model_weights_hdf5(model, tmp)
  load_model_weights_hdf5(model, tmp)
})

test_succeeds("model can be saved and loaded from json", {
  model <- define_model()
  json <- model_to_json(model)
  model_from <- model_from_json(json)
  expect_equal(json, model_to_json(model_from))
})

test_succeeds("model can be saved and loaded from yaml", {

  if (!keras:::have_pyyaml())
    skip("yaml not available for testing")

  model <- define_model()
  yaml <- model_to_yaml(model)
  model_from <- model_from_yaml(yaml)
  expect_equal(yaml, model_to_yaml(model_from))
})

test_succeeds("model can be saved and loaded from R 'raw' object", {

  if (!keras:::have_h5py())
    skip("h5py not available for testing")

  model <- define_and_compile_model()

  mdl_raw <- serialize_model(model)
  model <- unserialize_model(mdl_raw)

})

test_succeeds("saved models/weights are mirrored in the run_dir", {
  run <- tfruns::training_run("train.R", echo = FALSE)
  run_dir <- run$run_dir
  expect_true(file.exists(file.path(run_dir, "model.h5")))
  expect_true(file.exists(file.path(run_dir, "weights", "weights.h5")))
})

test_succeeds("callback output is redirected to run_dir", {
  run <- tfruns::training_run("train.R", echo = FALSE)
  run_dir <- run$run_dir
  if (is_backend("tensorflow"))
    expect_true(file_test("-d", file.path(run_dir, "tflogs")))
  expect_true(file.exists(file.path(run_dir, "cbk_checkpoint.h5")))
  expect_true(file.exists(file.path(run_dir, "cbk_history.csv")))
})

test_succeeds("model can be exported to TensorFlow", {
  if (!is_backend("tensorflow")) skip("not a tensorflow backend")
  
  model <- define_and_compile_model()
  model_dir <- tempfile()
  
  export <- function() tensorflow::export_savedmodel(model, model_dir)
  
  export()
  model_files <- dir(model_dir, recursive = TRUE)
  expect_true(any(grepl("saved_model\\.pb", model_files)))

})

test_succeeds("model can be exported to saved model format", {
  if (!is_backend("tensorflow")) skip("not a tensorflow backend")
  if (!tensorflow::tf_version() >= "1.14") skip("Needs TF >= 1.14")
  if (tensorflow::tf_version() > "2.0") skip("Is deprecated in TF 2.1")
  
  model <- define_and_compile_model()
  data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
  labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)
  
  model %>% fit(data, labels, epochs = 2, verbose = 0)
  
  model_dir <- tempfile()
  dir.create(model_dir)
  
  if (tensorflow::tf_version() == "2.0") {
    expect_warning({
      model_to_saved_model(model, model_dir)
      loaded <- model_from_saved_model(model_dir)    
    })
  } else {
    model_to_saved_model(model, model_dir)
    loaded <- model_from_saved_model(model_dir)  
  }
  
  
  expect_equal(
    predict(model, matrix(rep(1, 784), nrow = 1)),
    predict(loaded, matrix(rep(1, 784), nrow = 1))
  )
})

test_succeeds("model can be exported to saved model format using save_model_tf", {
  
  if (!is_backend("tensorflow")) skip("not a tensorflow backend")
  if (!tensorflow::tf_version() >= "2.0.0") skip("Needs TF >= 2.0")
  
  model <- define_and_compile_model()
  model_dir <- tempfile()
  
  s <- save_model_tf(model, model_dir)
  loaded <- load_model_tf(model_dir)
  
  expect_equal(
    predict(model, matrix(rep(1, 784), nrow = 1)),
    predict(loaded, matrix(rep(1, 784), nrow = 1))
  )
})



