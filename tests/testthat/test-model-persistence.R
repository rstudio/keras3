
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


