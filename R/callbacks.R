

#' Callback that prints metrics to stdout.
#' 
#' @param count_mode One of "steps" or "samples". Whether the progress bar
#'   should count samples seens or steps (batches) seen.
#'   
#' @family callbacks   
#'   
#' @export
callback_progbar_logger <- function(count_mode = "samples") {
  keras$callbacks$ProgbarLogger(
    count_mode = count_mode
  )
}



#' Save the model after every epoch.
#' 
#' `filepath` can contain named formatting options, which will be filled the 
#' value of `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: 
#' if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model 
#' checkpoints will be saved with the epoch number and the validation loss in 
#' the filename.
#' 
#' @param filepath string, path to save the model file.
#' @param monitor quantity to monitor.
#' @param verbose verbosity mode, 0 or 1.
#' @param save_best_only if `save_best_only=TRUE`, the latest best model 
#'   according to the quantity monitored will not be overwritten.
#' @param save_weights_only  if `TRUE`, then only the model's weights will be 
#'   saved (`save_model_weights_hdf5(filepath)`), else the full model is saved 
#'   (`save_model_hdf5(filepath)`).
#' @param mode one of "auto", "min", "max". If `save_best_only=TRUE`, the decision to
#'   overwrite the current save file is made based on either the maximization or
#'   the minimization of the monitored quantity. For val_acc, this should be
#'   max, for val_loss this should be min, etc. In auto mode, the direction is
#'   automatically inferred from the name of the monitored quantity.
#' @param period Interval (number of epochs) between checkpoints.
#'   
#' @section For example: if `filepath` is 
#'   `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,: then the model checkpoints will
#'   be saved with the epoch number and the validation loss in the filename.
#'  
#' @family callbacks  
#'   
#' @export
callback_model_checkpoint <- function(filepath, monitor = "val_loss", verbose = 0, 
                                      save_best_only = FALSE, save_weights_only = FALSE, 
                                      mode = c("auto", "min", "max"), period = 1) {
  
  if (!save_weights_only && !have_h5py())
    stop("The h5py Python package is required to save model checkpoints")
  
  keras$callbacks$ModelCheckpoint(
    filepath = normalize_path(filepath),
    monitor = monitor,
    verbose = as.integer(verbose),
    save_best_only = save_best_only,
    save_weights_only = save_weights_only,
    mode = match.arg(mode),
    period = as.integer(period)
  )
}


#' Stop training when a monitored quantity has stopped improving.
#' 
#' @inheritParams callback_model_checkpoint
#'   
#' @param monitor quantity to be monitored.
#' @param min_delta minimum change in the monitored quantity to qualify as an 
#'   improvement, i.e. an absolute change of less than min_delta, will count as 
#'   no improvement.
#' @param patience number of epochs with no improvement after which training 
#'   will be stopped.
#' @param mode  one of "auto", "min", "max". In `min` mode, training will stop when 
#'   the quantity monitored has stopped decreasing; in `max` mode it will stop 
#'   when the quantity monitored has stopped increasing; in `auto` mode, the
#'   direction is automatically inferred from the name of the monitored
#'   quantity.
#' 
#' @family callbacks 
#'       
#' @export
callback_early_stopping <- function(monitor = "val_loss", min_delta = 0, patience = 0, 
                                    verbose = 0, mode = c("auto", "min", "max")) {
  keras$callbacks$EarlyStopping(
    monitor = monitor,
    min_delta = min_delta,
    patience = as.integer(patience),
    verbose = as.integer(verbose),
    mode = match.arg(mode)
  )
}


#' Callback used to stream events to a server.
#' 
#' @param root root url of the target server.
#' @param path path relative to root to which the events will be sent.
#' @param field JSON field under which the data will be stored.
#' @param headers Optional named list of custom HTTP headers. Defaults to:
#'   `list(Accept = "application/json", `Content-Type` = "application/json")`
#' 
#' @family callbacks 
#' 
#' @export
callback_remote_monitor <- function(root = "http://localhost:9000", path = "/publish/epoch/end/", 
                                    field = "data", headers = NULL) {
  
  if (!have_requests())
    stop("The requests Python package is required for remote monitoring")
  
  keras$callbacks$RemoteMonitor(
    root = root,
    path = path,
    field = field,
    headers = headers
  )
}



#' Learning rate scheduler.
#' 
#' @param schedule a function that takes an epoch index as input (integer,
#'   indexed from 0) and returns a new learning rate as output (float).
#'
#' @family callbacks 
#'            
#' @export
callback_learning_rate_scheduler <- function(schedule) {
  keras$callbacks$LearningRateScheduler(
    schedule = schedule
  )
}


#' Callback that terminates training when a NaN loss is encountered.
#' 
#' @family callbacks
#' 
#' @export
callback_terminate_on_naan <- function() {
  keras$callbacks$TerminateOnNaN()
}


#' TensorBoard basic visualizations
#' 
#' This callback writes a log for TensorBoard, which allows you to visualize 
#' dynamic graphs of your training and test metrics, as well as activation 
#' histograms for the different layers in your model.
#' 
#' @param log_dir The path of the directory where to save the log files to be
#'   parsed by Tensorboard. The default is `NULL`, which will use the active
#'   run directory (if available) and otherwise will use "logs".
#' @param histogram_freq frequency (in epochs) at which to compute activation 
#'   histograms for the layers of the model. If set to 0, histograms won't be
#'   computed.
#' @param batch_size size of batch of inputs to feed to the network
#'   for histograms computation.
#' @param write_graph whether to visualize the graph in Tensorboard. The log
#'   file can become quite large when write_graph is set to `TRUE`
#' @param write_grads whether to visualize gradient histograms in TensorBoard.
#'   `histogram_freq` must be greater than 0.
#' @param write_images whether to write model weights to visualize as image in
#'   Tensorboard.
#' @param embeddings_freq frequency (in epochs) at which selected embedding 
#'   layers will be saved.
#' @param embeddings_layer_names a list of names of layers to keep eye on. If 
#'   `NULL` or empty list all the embedding layers will be watched.
#' @param  embeddings_metadata a named list which maps layer name to a file name in
#'   which metadata for this embedding layer is saved. See the 
#'   [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
#'    about the metadata file format. In case if the same metadata file is used
#'   for all embedding layers, string can be passed.

#' @details TensorBoard is a visualization tool provided with TensorFlow.
#'   
#' You can find more information about TensorBoard
#' [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
#' 
#' @family callbacks 
#'    
#' @export
callback_tensorboard <- function(log_dir = NULL, histogram_freq = 0,
                                 batch_size = 32,
                                 write_graph = TRUE, 
                                 write_grads = FALSE,
                                 write_images = FALSE,
                                 embeddings_freq = 0, 
                                 embeddings_layer_names = NULL,
                                 embeddings_metadata = NULL) {
  
  # establish the log_dir
  if (is.null(log_dir)) {
    if (tfruns::is_run_active())
      log_dir <- tfruns::run_dir()
    else
      log_dir <- "logs"
  }
   
  args <- list(
    log_dir = normalize_path(log_dir),
    histogram_freq = as.integer(histogram_freq),
    write_graph = write_graph,
    write_images = write_images,
    embeddings_freq = as.integer(embeddings_freq),
    embeddings_layer_names = embeddings_layer_names,
    embeddings_metadata = embeddings_metadata
  )
  
  if (keras_version() >= "2.0.5") {
    args$batch_size <- as.integer(batch_size)
    args$write_grads <- write_grads
  }
  
  do.call(keras$callbacks$TensorBoard, args)
}


#' Reduce learning rate when a metric has stopped improving.
#' 
#' Models often benefit from reducing the learning rate by a factor of 2-10 once
#' learning stagnates. This callback monitors a quantity and if no improvement 
#' is seen for a 'patience' number of epochs, the learning rate is reduced.
#' 
#' @param monitor quantity to be monitored.
#' @param factor factor by which the learning rate will be reduced. new_lr = lr 
#'   * factor
#' @param patience number of epochs with no improvement after which learning 
#'   rate will be reduced.
#' @param verbose int. 0: quiet, 1: update messages.
#' @param mode one of "auto", "min", "max". In min mode, lr will be reduced when
#'   the quantity monitored has stopped decreasing; in max mode it will be 
#'   reduced when the quantity monitored has stopped increasing; in auto mode, 
#'   the direction is automatically inferred from the name of the monitored 
#'   quantity.
#' @param epsilon threshold for measuring the new optimum, to only focus on 
#'   significant changes.
#' @param cooldown number of epochs to wait before resuming normal operation 
#'   after lr has been reduced.
#' @param min_lr lower bound on the learning rate.
#' 
#' @family callbacks
#'   
#' @export
callback_reduce_lr_on_plateau <- function(monitor = "val_loss", factor = 0.1, patience = 10, 
                                          verbose = 0, mode = c("auto", "min", "max"), 
                                          epsilon = 0.0001, cooldown = 0, min_lr = 0.0) {
  keras$callbacks$ReduceLROnPlateau(
    monitor = monitor,
    factor = factor,
    patience = as.integer(patience),
    verbose = as.integer(verbose),
    mode = match.arg(mode),
    epsilon = epsilon,
    cooldown = as.integer(cooldown),
    min_lr = min_lr
  )
}

#' Callback that streams epoch results to a csv file
#' 
#' Supports all values that can be represented as a string
#' 
#' @param filename filename of the csv file, e.g. 'run/log.csv'.
#' @param separator string used to separate elements in the csv file.
#' @param append `TRUE`: append if file exists (useful for continuing training).
#'   `FALSE`: overwrite existing file,
#'   
#' @family callbacks
#'   
#' @export
callback_csv_logger <- function(filename, separator = ",", append = FALSE) {
  keras$callbacks$CSVLogger(
    filename = normalize_path(filename),
    separator = separator,
    append = append
  )
}



#' Create a custom callback
#' 
#' This callback is constructed with anonymous functions that will be called at
#' the appropriate time. Note that the callbacks expects positional arguments,
#' as:
#'  
#' - `on_epoch_begin` and `on_epoch_end` expect two positional arguments: `epoch`, `logs` 
#' - `on_batch_begin` and `on_batch_end` expect two positional arguments: `batch`, `logs` 
#' - `on_train_begin` and `on_train_end` expect one positional argument: `logs`
#' 
#' @param on_epoch_begin called at the beginning of every epoch.
#' @param on_epoch_end called at the end of every epoch.
#' @param on_batch_begin called at the beginning of every batch.
#' @param on_batch_end called at the end of every batch.
#' @param on_train_begin called at the beginning of model training.
#' @param on_train_end called at the end of model training.
#' 
#' @family callbacks
#'   
#' @export
callback_lambda <- function(on_epoch_begin = NULL, on_epoch_end = NULL, 
                            on_batch_begin = NULL, on_batch_end = NULL, 
                            on_train_begin = NULL, on_train_end = NULL) {
  keras$callbacks$LambdaCallback(
    on_epoch_begin = on_epoch_begin,
    on_epoch_end = on_epoch_end,
    on_batch_begin = on_batch_begin,
    on_batch_end = on_batch_end,
    on_train_begin = on_train_begin,
    on_train_end = on_train_end
  )
}

#' Base R6 class for Keras callbacks
#' 
#' @docType class
#' 
#' @format An [R6Class] generator object
#' 
#' @field params Named list with training parameters (eg. verbosity, batch size, number of epochs...).
#' @field model Reference to the Keras model being trained.
#' 
#' @section Methods:
#' \describe{
#'  \item{\code{on_epoch_begin(epoch, logs)}}{Called at the beginning of each epoch.}
#'  \item{\code{on_epoch_end(epoch, logs)}}{Called at the end of each epoch.}
#'  \item{\code{on_batch_begin(batch, logs)}}{Called at the beginning of each batch.}
#'  \item{\code{on_batch_end(batch, logs)}}{Called at the end of each batch.}
#'  \item{\code{on_train_begin(logs)}}{Called at the beginning of training.}
#'  \item{\code{on_train_end(logs)}}{Called at the end of training.}
#' }
#' 
#' @details  The `logs` named list that callback methods take as argument will 
#' contain keys for quantities relevant to the current batch or epoch.
#' 
#' Currently, the `fit()` method for sequential models will include the following quantities in the `logs` that
#' it passes to its callbacks:
#'
#' - `on_epoch_end`: logs include `acc` and `loss`, and optionally include `val_loss` (if validation is enabled in `fit`), and `val_acc` (if validation and accuracy monitoring are enabled).
#' - `on_batch_begin`: logs include `size`, the number of samples in the current batch.
#' - `on_batch_end`: logs include `loss`, and optionally `acc` (if accuracy monitoring is enabled).
#' 
#' @return [KerasCallback].
#' 
#' @examples 
#' \dontrun{
#' library(keras)
#' 
#' LossHistory <- R6::R6Class("LossHistory",
#'   inherit = KerasCallback,
#'   
#'   public = list(
#'   
#'     losses = NULL,
#' 
#'     on_batch_end = function(batch, logs = list()) {
#'       self$losses <- c(self$losses, logs[["loss"]])
#'     }
#'   )
#' )
#' }
#' @export
KerasCallback <- R6Class("KerasCallback",
                         
  public = list(
    
    params = NULL,
    model = NULL,
    
    set_context = function(params = NULL, model = NULL) {
      self$params <- params
      self$model <- model
    },
    
    on_epoch_begin = function(epoch, logs = NULL) {
      
    },
    
    on_epoch_end = function(epoch, logs = NULL) {
      
    },
    
    on_batch_begin = function(batch, logs = NULL) {
      
    },
    
    on_batch_end = function(batch, logs = NULL) {
      
    },
    
    on_train_begin = function(logs = NULL) {
      
    },
    
    on_train_end = function(logs = NULL) {
      
    }
  )
)

normalize_callbacks <- function(view_metrics, callbacks) {
  
  # if callbacks isn't a list then make it one
  if (!is.null(callbacks) && !is.list(callbacks))
    callbacks <- list(callbacks)
  
  # always include the metrics callback
  callbacks <- append(callbacks, KerasMetricsCallback$new(view_metrics))  
 
  # import callback utility module
  python_path <- system.file("python", package = "keras")
  tools <- import_from_path("kerastools", path = python_path)
  
  # convert R callbacks to Python and check whether the user
  # has already included the tensorboard callback
  have_tensorboard_callback <- FALSE
  callbacks <- lapply(callbacks, function(callback) {
    
    # track whether we have a tensorboard callback
    if (inherits(callback, "keras.callbacks.TensorBoard"))
      have_tensorboard_callback <<- TRUE
    
    if (inherits(callback, "KerasCallback")) {
      # create a python callback to map to our R callback
      tools$callback$RCallback(
        r_set_context = callback$set_context,
        r_on_epoch_begin = callback$on_epoch_begin,
        r_on_epoch_end = callback$on_epoch_end,
        r_on_batch_begin = callback$on_batch_begin,
        r_on_batch_end = callback$on_batch_end,
        r_on_train_begin = callback$on_train_begin,
        r_on_train_end = callback$on_train_end
      )
    } else {
      callback
    }
  })
  
  # add the tensorboard callback if necessary
  if (is_backend("tensorflow") && tfruns::is_run_active() && !have_tensorboard_callback)
    callbacks <- append(callbacks, callback_tensorboard())
  
  # return the callbacks
  callbacks
}






