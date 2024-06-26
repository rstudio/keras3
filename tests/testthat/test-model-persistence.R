
context("model-persistence")



test_succeeds("model can be saved and loaded", {

  if (!keras3:::have_h5py())
    skip("h5py not available for testing")

  model <- define_and_compile_model()
  tmp <- tempfile("model", fileext = ".hdf5")
  skip("save_model_hdf5")
  save_model_hdf5(model, tmp)
  model <- load_model_hdf5(tmp)
})

test_succeeds("model with custom loss and metrics can be saved and loaded", {

  if (!keras3:::have_h5py())
    skip("h5py not available for testing")

  model <- define_model()

  metric_mean_pred <- custom_metric("mean_pred", function(y_true, y_pred) {
    op_mean(y_pred)
  })

  custom_loss <- function(y_pred, y_true) {
    loss_categorical_crossentropy(y_pred, y_true)
  }

  model %>% compile(
    loss = custom_loss,
    optimizer = optimizer_nadam(),
    metrics = metric_mean_pred
  )

  tmp <- tempfile("model", fileext = ".keras")
  save_model(model, tmp)
  restored_model <- load_model(tmp, custom_objects = c(mean_pred = metric_mean_pred,
                                              custom_loss = custom_loss))

  # generate dummy training data
  data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
  labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

  expect_equal(
    model %>% predict(data, verbose = 0),
    restored_model %>% predict(data, verbose = 0)
  )


  model %>% fit(data, labels, epochs = 2, verbose = 0)
  expect_no_error({
    restored_model %>% fit(data, labels, epochs = 2, verbose = 0)
  })

})

test_succeeds("model load with unnamed custom_objects", {

  layer_my_dense <-  new_layer_class(
    "MyDense",
    initialize = function(units, ...) {
      super$initialize(...)
      private$units <- units
      self$dense <- layer_dense(units = units)
    },
    # TODO: warning emitted from upstream if missing build method...
    # but this simple case should not need a build method
    build = function(input_shape) {
      self$dense$build(input_shape)
    },

    call = function(x, ...) {
      # TODO: a call() method without any named args breaks shape inference
      # for tracing, symbolic builds, and auto-calling build() w/ the correct
      # input shape. Emit a warning from `new_layer_class()` if that happens?
      self$dense(x, ...)
    },
    get_config = function() {
      config <- super$get_config()
      config$units <- private$units
      config
    }
  )

  # l <- layer_my_dense(,10)
  # x <- random_array(3, 4)
  # l(random_array(3, 4))

  model <- keras_model_sequential(input_shape = 32) %>%
    layer_dense(10) %>%
    layer_my_dense(10) %>%
    layer_dense(10)


  metric_mean_pred <- custom_metric("mean_pred", function(y_true, y_pred) {
    op_mean(y_pred)
  })

  custom_loss <- function(y_pred, y_true) {
    loss_categorical_crossentropy(y_pred, y_true)
  }
  # TODO:
  # attr(custom_loss, "name") <- "custom_loss"
  # custom_loss <- py_func2(function(y_pred, y_true) {
  #     loss_categorical_crossentropy(y_pred, y_true)
  #   }, TRUE, name = "custom_loss")

  model %>% compile(
    loss = custom_loss,
    optimizer = optimizer_nadam(),
    metrics = metric_mean_pred
  )

  # generate dummy training data
  data <- x <- random_normal(c(10, 32))
  # y <- to_categorical(sample(0:9, 10, replace = TRUE))
  y <- to_categorical(random_integer(10, 0, 10), 10)

  model %>% fit(x, y, verbose = FALSE)

  res1 <- as.array(model(data))

  tmp <- tempfile("model", fileext = ".keras")
  if (is_windows())
    skip("save_model() errors on Windows")
    # need to investigate next time on Windows
    "  ── Failure ('test-model-persistence.R:128:3'): model load with unnamed custom_objects ──
  Expected `{ ... }` to run without any errors.
  ℹ Actually got a <python.builtin.SystemError> with text:
    SystemError: <built-in function call_r_function> returned a result with an exception set
    Run `reticulate::py_last_error()` for details.

  [ FAIL 1 | WARN 0 | SKIP 40 | PASS 513 ]"

  save_model(model, tmp)
  model2 <- load_model(tmp, custom_objects = list(
    metric_mean_pred,
    layer_my_dense,
    custom_loss = custom_loss)
  )
  res2 <- as.array(model2(data))

  expect_identical(res1, res2)
  expect_no_error({
    model2 %>% fit(x, y, verbose = 0)
  })
})


test_succeeds("model weights can be saved and loaded", {

  if (!keras3:::have_h5py())
    skip("h5py not available for testing")

  model <- define_and_compile_model()
  tmp <- tempfile("model", fileext = ".hdf5")
  skip("save_model_weights_hdf5")
  save_model_weights_hdf5(model, tmp)
  load_model_weights_hdf5(model, tmp)
})

test_succeeds("model can be saved and loaded from json", {
  model <- define_model()

  json_file <- tempfile("config-", fileext = ".json")
  save_model_config(model, json_file)

  model2 <- load_model_config(json_file)

  json_file2 <- tempfile("config-2-", fileext = ".json")
  save_model_config(model2, json_file2)

  expect_identical(jsonlite::read_json(json_file),
                   jsonlite::read_json(json_file2))

  config <- get_config(model)
  attributes(config) <- attributes(config)['names']
  expect_identical(jsonlite::read_json(json_file)$config,
                   config)
})

## patch releases removed ability to serialize to/from yaml in all the version
## going back to 2.2

# test_succeeds("model can be saved and loaded from yaml", {
#
#   if (!keras3:::have_pyyaml())
#     skip("yaml not available for testing")
#
#   if(tf_version() >= "2.5.1")
#     skip("model$to_yaml() removed in 2.6")
#
#   model <- define_model()
#   yaml <- model_to_yaml(model)
#   model_from <- model_from_yaml(yaml)
#   expect_equal(yaml, model_to_yaml(model_from))
# })

test_succeeds("model can be saved and loaded from R 'raw' object", {

  if (!keras3:::have_h5py())
    skip("h5py not available for testing")

  model <- define_and_compile_model()

  skip("serialize_model")
  mdl_raw <- serialize_model(model)
  model <- unserialize_model(mdl_raw)

})

test_succeeds("saved models/weights are mirrored in the run_dir", {
  skip("tfruns")
  run <- tfruns::training_run("train.R", echo = FALSE)
  run_dir <- run$run_dir
  expect_true(file.exists(file.path(run_dir, "model.h5")))
  expect_true(file.exists(file.path(run_dir, "weights", "weights.h5")))
})

test_succeeds("callback output is redirected to run_dir", {
  skip("tfruns")
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

  skip("tensorflow::export_savedmodel")
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

  skip("save_model_tf")
  s <- save_model_tf(model, model_dir)
  loaded <- load_model_tf(model_dir)

  expect_equal(
    predict(model, matrix(rep(1, 784), nrow = 1)),
    predict(loaded, matrix(rep(1, 784), nrow = 1))
  )
})
