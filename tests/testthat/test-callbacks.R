
context("callbacks")




# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)

define_compile_and_fit <- function(callbacks) {
  model <- define_and_compile_model()
  fit(model, data, labels, callbacks = callbacks, epochs = 1)
}

test_callback <- function(name, callback, h5py = FALSE, required_version = NULL) {

  test_succeeds(required_version = required_version,
                paste0("callback_", name, " is called back"),  {
    if (h5py && !have_h5py())
      skip(paste(name, "test requires h5py package"))
    define_compile_and_fit(callbacks = list(callback))
  })
}

# disable progbar test as per: https://github.com/tensorflow/tensorflow/issues/38618#issuecomment-617907735
if (tensorflow::tf_version() <= "2.1")
  test_callback("progbar_logger", callback_progbar_logger())


test_callback("model_checkpoint", callback_model_checkpoint(tempfile(fileext = ".h5")), h5py = TRUE)

if(tf_version() >= "2.8")
  test_callback("backup_and_restore", callback_backup_and_restore(tempfile()))

test_callback("learning_rate_scheduler", callback_learning_rate_scheduler(schedule = function (index, ...) {
  0.1
}))
if (is_keras_available() && is_backend("tensorflow"))
  test_callback("tensorboard", callback_tensorboard(log_dir = "./tb_logs"))

test_callback("terminate_on_naan", callback_terminate_on_naan(), required_version = "2.0.5")

test_callback("reduce_lr_on_plateau", callback_reduce_lr_on_plateau(monitor = "loss"))

test_callback("csv_logger", callback_csv_logger(tempfile(fileext = ".csv")))
test_callback("lambd", callback_lambda(
  on_epoch_begin = function(epoch, logs) {
    cat("Epoch Begin\n")
  },
  on_epoch_end = function(epoch, logs) {
    cat("Epoch End\n")
  }
))

test_succeeds("lambda callbacks other args", {

  x <- layer_input(shape = 1)
  y <- layer_dense(x, units = 1)
  model <- keras_model(x, y)
  model %>% compile(optimizer = "adam", loss = "mae")

  warns <- capture_warnings(
    clb <- callback_lambda(
      on_epoch_begin = function(epoch, logs) {
        cat("Epoch Begin")
      },
      on_epoch_end = function(epoch, logs) {
        cat("Epoch End")
      },
      on_predict_begin = function(epoch, logs) {
        cat("Prediction Begin")
      },
      on_test_begin = function(epoch, logs) {
        cat("Test Begin")
      }
    )
  )

  if (get_keras_implementation() == "tensorflow" &&
      tensorflow::tf_version() >= "2.0") {
    expect_equal(length(warns), 0)
  } else {
    expect_equal(length(warns), 2)
  }

  warns <- capture_warnings(
    out <- capture_output(
      pred <- predict(model, matrix(1:10, ncol = 1), callbacks = list(clb))
    )
  )

  if (get_keras_implementation() == "tensorflow" &&
      tensorflow::tf_version() >= "2.0") {
    expect_equal(length(warns), 0)
    expect_equal(out, "Prediction Begin")
  } else {
    expect_equal(length(warns), 1)
    expect_equal(out, "")
  }

  warns <- capture_warnings(
    out <- capture_output(
      pred <- evaluate(model, matrix(1:10, ncol = 1), y = 1:10,
                       callbacks = list(clb))
    )
  )

  if (get_keras_implementation() == "tensorflow" &&
      tensorflow::tf_version() >= "2.0") {
    expect_equal(length(warns), 0)
    expect_equal(out, "Test Begin")
  } else {
    expect_equal(length(warns), 1)
    expect_equal(out, "")
  }

})


test_succeeds("custom callbacks", {

  CustomCallback <- R6::R6Class("CustomCallback",
    inherit = KerasCallback,
    public = list(
      on_train_begin = function(logs) {
        print("TRAIN BEGIN\n")
      },
      on_train_end = function(logs) {
        print("TRAIN END\n")
      }
    )
  )

  LossHistory <- R6::R6Class("LossHistory",
    inherit = KerasCallback,
    public = list(
      losses = NULL,

      on_batch_end = function(batch, logs = list()) {
        self$losses <- c(self$losses, logs[["loss"]])
      }

    ))

  cc <- CustomCallback$new()
  lh <- LossHistory$new()

  define_compile_and_fit(callbacks = list(cc, lh))

  expect_is(lh$losses, "numeric")

})


test_succeeds("custom callbacks, new-style", {

  CustomMetric <- R6::R6Class(
    "CustomMetric",
    inherit = keras$callbacks$Callback,
    public = list(
      on_epoch_end = function(epoch, logs = NULL) {
        logs[["my_epoch"]] <- epoch
        logs
      }
    )
  )
  CustomMetric <- r_to_py(CustomMetric, convert = FALSE)


  CustomMetric2 <- R6::R6Class(
    "CustomMetric2",
    inherit = keras$callbacks$Callback,
    public = list(
      on_epoch_end = function(epoch, logs = NULL) {
        expect_true("my_epoch" %in% names(logs))
        logs[['my_epoch2']] <- epoch
        logs
      }
    )
  )
  CustomMetric2 <- r_to_py(CustomMetric2, convert=TRUE)

  cm <- CustomMetric()
  cm2 <- CustomMetric2()

  hist <- define_compile_and_fit(callbacks = list(cm, cm2))

  expect_is(hist$metrics$my_epoch, "numeric")
  expect_equal(hist$metrics$my_epoch, 0L)
  expect_false("my_epoch2" %in% names(hist$metrics))

})


expect_warns_and_out <- function(warns, out) {
  if (get_keras_implementation() == "tensorflow" &&
      tensorflow::tf_version() >= "2.0") {
    expect_equal(out, c("PREDICT BEGINPREDICT END"))
    expect_equal(warns, character())
  } else {
    expect_equal(out, "")
    expect_true(warns != "")
  }
}

test_succeeds("on predict/evaluation callbacks", {

  if (tensorflow::tf_version() <= "2.1")
    skip("don't work in tf2.1")

  CustomCallback <- R6::R6Class(
    "CustomCallback",
    inherit = KerasCallback,
    public = list(
      on_predict_begin = function(logs) {
        cat("PREDICT BEGIN")
      },
      on_predict_end = function(logs) {
        cat("PREDICT END")
      },
      on_test_begin = function(logs) {
        cat("PREDICT BEGIN")
      },
      on_test_end = function(logs) {
        cat("PREDICT END")
      }
    )
  )

  input <- layer_input(shape = 1)
  output <- layer_dense(input, 1)
  model <- keras_model(input, output)
  model %>% compile(optimizer = "adam", loss = "mae")

  cc <- CustomCallback$new()

  # test for prediction
  warns <- capture_warnings(
    out <- capture_output(
      pred <- predict(model, x = matrix(1:10, ncol = 1), callbacks = cc)
    )
  )
  expect_warns_and_out(warns, out)

  gen <- function() {
    list(matrix(1:10, ncol = 1))
  }

  warns <- capture_warnings(
    out <- capture_output(
      pred <- predict(model, gen, callbacks = cc, steps = 1)
    )
  )
  expect_warns_and_out(warns, out)

  # tests for evaluation
  warns <- capture_warnings(
    out <- capture_output(
      ev <- evaluate(model, x = matrix(1:10, ncol = 1), y = 1:10, callbacks = cc)
    )
  )
  expect_warns_and_out(warns, out)

  gen <- function() {
    list(matrix(1:10, ncol = 1), 1:10)
  }

  warns <- capture_warnings(
    out <- capture_output(
      ev <- evaluate(model, gen, callbacks = cc, steps = 1)
    )
  )
  expect_warns_and_out(warns, out)

})

test_succeeds("warnings for new callback moment", {

  CustomCallback <- R6::R6Class(
    "CustomCallback",
    inherit = KerasCallback,
    public = list(
      on_predict_begin = function(logs) {
        cat("PREDICT BEGIN")
      },
      on_predict_end = function(logs) {
        cat("PREDICT END")
      },
      on_test_begin = function(logs) {
        cat("PREDICT BEGIN")
      },
      on_test_end = function(logs) {
        cat("PREDICT END")
      }
    )
  )

  cc <- CustomCallback$new()

  input <- layer_input(shape = 1)
  output <- layer_dense(input, 1)
  model <- keras_model(input, output)
  model %>% compile(optimizer = "adam", loss = "mae")

  warns <- capture_warnings(
    model %>%
      fit(x = matrix(1:10, ncol = 1), y = 1:10, callbacks = list(cc),
          verbose = 0, epochs = 2)
  )

  if (get_keras_implementation() == "tensorflow" && tensorflow::tf_version() < "2.0")
    expect_equal(length(warns), 4)
  else
    expect_equal(length(warns), 0)

})
