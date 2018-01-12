
context("model persistence")

source("utils.R")

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
  
  sparse_top_k_cat_acc <- function(y_pred, y_true){
    metric_sparse_top_k_categorical_accuracy(y_pred, y_true, k = 5)
  }
  
  custom_loss <- function(y_pred, y_true) {
    loss_categorical_crossentropy(y_pred, y_true)
  }
  
  model %>% compile(
    loss = custom_loss,
    optimizer = optimizer_nadam(),
    metrics = c(top_k_acc = sparse_top_k_cat_acc)
  )
  
  tmp <- tempfile("model", fileext = ".hdf5")
  save_model_hdf5(model, tmp)
  model <- load_model_hdf5(tmp, custom_objects = c(top_k_acc = sparse_top_k_cat_acc,
                                                   custom_loss = custom_loss))
  
  # generate dummy training data
  data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
  labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)
  
  
  model %>% fit(data, labels, epochs = 2)
  
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
  run <- tfruns::training_run("train.R")
  run_dir <- run$run_dir
  expect_true(file.exists(file.path(run_dir, "model.h5")))
  expect_true(file.exists(file.path(run_dir, "weights", "weights.h5")))
})

test_succeeds("callback output is redirected to run_dir", {
  run <- tfruns::training_run("train.R")
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
  
  if (grepl("^tensorflow", Sys.getenv("KERAS_IMPLEMENTATION"))) {
    expect_error(export())
  }
  else {
    export()
    model_files <- dir(model_dir, recursive = TRUE)
    expect_true(any(grepl("saved_model\\.pb", model_files)))
  }
})
