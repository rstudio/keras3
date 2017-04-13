
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

test_callback <- function(name, callback, h5py = FALSE) {

  if (h5py && !have_h5py())
    skip(paste(name, "test requires h5py package"))
  
  test_succeeds(paste0("callback_", name, " is called back"),  {
    define_compile_and_fit(callbacks = list(callback))   
  })
}

test_callback("progbar_logger", callback_progbar_logger())
test_callback("model_checkpoint", callback_model_checkpoint("checkpoint.h5"), h5py = TRUE)
test_callback("learning_rate_scheduler", callback_learning_rate_scheduler(schedule = function (index) {
  0.1
}))
test_callback("tensorboard", callback_tensorboard(log_dir = "./tb_logs"))
test_callback("reduce_lr_on_plateau", callback_reduce_lr_on_plateau())
test_callback("csv_logger", callback_csv_logger("training.csv"))
test_callback("lambd", callback_lambda(
  on_epoch_begin = function(epoch, logs) {
    cat("Epoch Begin\n")
  },
  on_epoch_end = function(epoch, logs) {
    cat("Epoch End\n")
  }
))




