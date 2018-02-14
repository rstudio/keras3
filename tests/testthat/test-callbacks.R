
context("callbacks")

source("utils.R")


# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)

define_compile_and_fit <- function(callbacks) {
  model <- define_and_compile_model()
  fit(model, data, labels, callbacks = callbacks)
}

test_callback <- function(name, callback, h5py = FALSE, required_version = NULL) {

  test_succeeds(required_version = required_version,
                paste0("callback_", name, " is called back"),  {
    if (h5py && !have_h5py())
      skip(paste(name, "test requires h5py package"))
    define_compile_and_fit(callbacks = list(callback))   
  })
}

test_callback("progbar_logger", callback_progbar_logger())
test_callback("model_checkpoint", callback_model_checkpoint("checkpoint.h5"), h5py = TRUE)
test_callback("learning_rate_scheduler", callback_learning_rate_scheduler(schedule = function (index, ...) {
  0.1
}))
if (is_keras_available() && is_backend("tensorflow"))
  test_callback("tensorboard", callback_tensorboard(log_dir = "./tb_logs"))

test_callback("terminate_on_naan", callback_terminate_on_naan(), required_version = "2.0.5")

test_callback("reduce_lr_on_plateau", callback_reduce_lr_on_plateau(monitor = "loss"))
test_callback("csv_logger", callback_csv_logger("training.csv"))
test_callback("lambd", callback_lambda(
  on_epoch_begin = function(epoch, logs) {
    cat("Epoch Begin\n")
  },
  on_epoch_end = function(epoch, logs) {
    cat("Epoch End\n")
  }
))


test_succeeds("custom callbacks", {
  
  CustomCallback <- R6::R6Class("CustomCallback",
    inherit = KerasCallback,
    public = list(
      on_train_begin = function(logs) {
        cat("TRAIN BEGIN\n")
      },
      on_train_end = function(logs) {
        cat("TRAIN END\n")
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





