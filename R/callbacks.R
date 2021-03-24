

#' Callback that prints metrics to stdout.
#' 
#' @param count_mode One of "steps" or "samples". Whether the progress bar
#'   should count samples seens or steps (batches) seen.
#' @param stateful_metrics List of metric names that should *not*
#'   be averaged onver an epoch. Metrics in this list will be logged
#'   as-is in `on_epoch_end`. All others will be averaged in 
#'   `on_epoch_end`.
#'   
#' @family callbacks   
#'   
#' @export
callback_progbar_logger <- function(count_mode = "samples", stateful_metrics = NULL) {
  args <- list(
    count_mode = count_mode
  )
  if (keras_version() >= "2.1.4")
    args$stateful_metrics <- stateful_metrics
  
  do.call(keras$callbacks$ProgbarLogger, args)
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
#' @param save_freq `'epoch'` or integer. When using 'epoch', the callback saves 
#'   the model after each epoch. When using integer, the callback saves the model 
#'   at end of a batch at which this many samples have been seen since last saving. 
#'   Note that if the saving isn't aligned to epochs, the monitored metric may 
#'   potentially be less reliable (it could reflect as little as 1 batch, since 
#'   the metrics get reset every epoch). Defaults to `'epoch'`
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
                                      mode = c("auto", "min", "max"), period = NULL,
                                      save_freq = "epoch") {
  
  if (!save_weights_only && !have_h5py())
    stop("The h5py Python package is required to save model checkpoints")
  
  args <- list(
    filepath = normalize_path(filepath),
    monitor = monitor,
    verbose = as.integer(verbose),
    save_best_only = save_best_only,
    save_weights_only = save_weights_only,
    mode = match.arg(mode)
  )
  
  if (is_tensorflow_implementation()) {
    if (tensorflow::tf_version() < "1.14") {
      
      if (!is.null(save_freq))
        warning(
          "The save_freq argument is only used by TensorFlow >= 1.14. ",
          "Update TensorFlow or use save_freq = NULL"
        )
      
      if (is.null(period))
        period <- 1L
      
      args$period <- as.integer(period)
    } else {
      
      if (!is.null(period))
        warning(
          "The period argument is deprecated since TF v1.14 and will be ignored. ",
          "Use save_freq instead."
        )
      
      # save_freq can be a string or an integer
      if (is.character(save_freq))
        args$save_freq <- save_freq
      else 
        args$save_freq <- as_nullable_integer(save_freq)
    }
  } else if (is_backend("plaidml")) {
    
    if (!is.null(save_freq))
      warning("`save_freq` is ignored in plaidml. Use the `period` argument.")
    
    if (is.null(save_freq) && is.null(period))
      period <- 1L
    
    args$period <- as.integer(period)
  }
  
  do.call(keras$callbacks$ModelCheckpoint, args)
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
#' @param baseline Baseline value for the monitored quantity to reach.
#'   Training will stop if the model doesn't show improvement
#'   over the baseline.
#' @param restore_best_weights Whether to restore model weights from
#'   the epoch with the best value of the monitored quantity.
#'   If `FALSE`, the model weights obtained at the last step of
#'   training are used.  
#'
#' 
#' @family callbacks 
#'       
#' @export
callback_early_stopping <- function(monitor = "val_loss", min_delta = 0, patience = 0, 
                                    verbose = 0, mode = c("auto", "min", "max"), 
                                    baseline = NULL, restore_best_weights = FALSE) {
  
  args <- list(
    monitor = monitor,
    min_delta = min_delta,
    patience = as.integer(patience),
    verbose = as.integer(verbose),
    mode = match.arg(mode)
  )
  
  if (keras_version() >= "2.2")
    args$baseline <- baseline
  
  if (keras_version() >= "2.2.3")
    args$restore_best_weights <- restore_best_weights
  
  do.call(keras$callbacks$EarlyStopping, args)
}


#' Callback used to stream events to a server.
#'
#' @param root root url of the target server.
#' @param path path relative to root to which the events will be sent.
#' @param field JSON field under which the data will be stored.
#' @param headers Optional named list of custom HTTP headers. Defaults to:
#'   `list(Accept = "application/json", `Content-Type` = "application/json")`
#' @param send_as_json Whether the request should be sent as application/json.
#'
#' @details Events are sent to `root + '/publish/epoch/end/'` by default. Calls
#'   are HTTP POST, with a `data` argument which is a JSON-encoded dictionary
#'   of event data. If send_as_json is set to True, the content type of the
#'   request will be application/json. Otherwise the serialized JSON will be
#'   send within a form
#'
#' @family callbacks
#'
#' @export
callback_remote_monitor <- function(root = "https://localhost:9000", path = "/publish/epoch/end/", 
                                    field = "data", headers = NULL, send_as_json = FALSE) {
  
  if (!have_requests())
    stop("The requests Python package is required for remote monitoring")
  
  args <- list(
    root = root,
    path = path,
    field = field,
    headers = headers
  )
  
  if (keras_version() >= "2.1.6")
    args$send_as_json <- send_as_json
  
  do.call(keras$callbacks$RemoteMonitor, args)
}



#' Learning rate scheduler.
#' 
#' @param schedule a function that takes an epoch index as input (integer,
#'   indexed from 0) and current learning rate and returns a new learning rate
#'   as output (float).
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
#'   for histograms computation. No longer needed, ignored since TF 1.14.
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
#'   [details](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin#saving_data_for_tensorboard)
#'    about the metadata file format. In case if the same metadata file is used
#'   for all embedding layers, string can be passed.
#' @param embeddings_data Data to be embedded at layers specified in
#'   `embeddings_layer_names`. Array (if the model has a single input) or list 
#'   of arrays (if the model has multiple inputs). Learn [more about embeddings](https://www.tensorflow.org/tutorials/text/word_embeddings)
#' @param update_freq `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
#'   the losses and metrics to TensorBoard after each batch. The same
#'   applies for `'epoch'`. If using an integer, let's say `10000`,
#'   the callback will write the metrics and losses to TensorBoard every
#'   10000 samples. Note that writing too frequently to TensorBoard
#'   can slow down your training.
#' @param profile_batch Profile the batch to sample compute characteristics. By 
#'   default, it will disbale profiling. Set profile_batch=2 profile the second
#'   batch. Must run in TensorFlow eager mode. (TF >= 1.14)
#'  
#' @details TensorBoard is a visualization tool provided with TensorFlow.
#'   
#' You can find more information about TensorBoard
#' [here](https://www.tensorflow.org/tensorboard/get_started).
#' 
#' When using a backend other than TensorFlow, TensorBoard will still work
#' (if you have TensorFlow installed), but the only feature available will
#' be the display of the losses and metrics plots.
#' 
#' @family callbacks 
#'    
#' @export
callback_tensorboard <- function(log_dir = NULL, histogram_freq = 0,
                                 batch_size = NULL,
                                 write_graph = TRUE, 
                                 write_grads = FALSE,
                                 write_images = FALSE,
                                 embeddings_freq = 0, 
                                 embeddings_layer_names = NULL,
                                 embeddings_metadata = NULL,
                                 embeddings_data = NULL,
                                 update_freq = "epoch",
                                 profile_batch = 0) {
  
  # establish the log_dir
  if (is.null(log_dir)) {
    if (tfruns::is_run_active())
      log_dir <- file.path(tfruns::run_dir(), "logs")
    else
      log_dir <- "logs"
  }
   
  args <- list(
    log_dir = normalize_path(log_dir),
    histogram_freq = as.integer(histogram_freq),
    write_graph = write_graph,
    write_images = write_images
  )
  
  if (tensorflow::tf_version() >= 1.14) {
    args[["profile_batch"]] = as.integer(profile_batch)
  } else if (profile_batch > 0) {
    warning("profile_batch can only be used with TensorFlow >= 1.14", call. = FALSE)
  }
  
  if (!missing(embeddings_data) && keras_version() < "2.2.0")
    stop("embeddings_data requires keras >= 2.2. Please update with install_keras()")
  
  # embeddings arguments seem to have been excluded in the TF implementation
  # (even though they are stil part of the docs there)
  if (!is_tensorflow_implementation()) {
    args$embeddings_freq <- as.integer(embeddings_freq)
    args$embeddings_layer_names <- embeddings_layer_names
    args$embeddings_metadata <- embeddings_metadata
    args$embeddings_data <- embeddings_data
  }
  
  if (keras_version() >= "2.0.5" & tensorflow::tf_version() < "1.14") {
    
    if (is.null(batch_size))
      batch_size <- 32L
    
    args$batch_size <- as.integer(batch_size)
    args$write_grads <- write_grads
  } else if (!is.null(batch_size)) {
    warning("Batch size is ignored since TensorFlow 1.14.0")
  }
  
  if (keras_version() >= "2.2.3")
    args$update_freq <- update_freq
  
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
#' @param min_delta threshold for measuring the new optimum, to only focus on 
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
                                          min_delta = 0.0001, cooldown = 0, min_lr = 0.0) {
  
  args <- list(
    monitor = monitor,
    factor = factor,
    patience = as.integer(patience),
    verbose = as.integer(verbose),
    mode = match.arg(mode),
    cooldown = as.integer(cooldown),
    min_lr = min_lr
  )
  
  if (keras_version() >= "2.1.6")
    args$min_delta <- min_delta
  else
    args$epsilon <- min_delta
  
  do.call(keras$callbacks$ReduceLROnPlateau, args)
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
#' - `on_batch_*`, `on_train_batch_*`, `on_predict_batch_*` and `on_test_batch_*`, expect 
#'    two positional arguments: `batch`, `logs` 
#' - `on_train_*`, `on_test_*` and `on_predict_*` expect one positional argument: `logs`
#' 
#' @param on_epoch_begin called at the beginning of every epoch.
#' @param on_epoch_end called at the end of every epoch.
#' @param on_batch_begin called at the beginning of every training batch.
#' @param on_batch_end called at the end of every training batch.
#' @param on_train_batch_begin called at the beginning of every batch.
#' @param on_train_batch_end called at the end of every batch.
#' @param on_train_begin called at the beginning of model training.
#' @param on_train_end called at the end of model training.
#' @param on_predict_batch_begin called at the beginning of a batch in predict methods.
#' @param on_predict_batch_end called at the end of a batch in predict methods.
#' @param on_predict_begin called at the beginning of prediction.
#' @param on_predict_end called at the end of prediction.
#' @param on_test_batch_begin called at the beginning of a batch in evaluate methods.
#'   Also called at the beginning of a validation batch in the fit methods, 
#'   if validation data is provided.
#' @param on_test_batch_end called at the end of a batch in evaluate methods.
#'   Also called at the end of a validation batch in the fit methods, 
#'   if validation data is provided.
#' @param on_test_begin called at the beginning of evaluation or validation.
#' @param on_test_end called at the end of evaluation or validation.
#' 
#' @family callbacks
#'   
#' @export
callback_lambda <- function(on_epoch_begin = NULL, on_epoch_end = NULL, 
                            on_batch_begin = NULL, on_batch_end = NULL,
                            on_train_batch_begin = NULL, on_train_batch_end = NULL,
                            on_train_begin = NULL, on_train_end = NULL,
                            on_predict_batch_begin = NULL, on_predict_batch_end = NULL,
                            on_predict_begin = NULL, on_predict_end = NULL,
                            on_test_batch_begin = NULL, on_test_batch_end = NULL,
                            on_test_begin = NULL, on_test_end = NULL
                            ) {
  
  
  args <- list(
    on_epoch_begin = on_epoch_begin,
    on_epoch_end = on_epoch_end,
    on_batch_begin = on_batch_begin,
    on_batch_end = on_batch_end,
    on_train_begin = on_train_begin,
    on_train_end = on_train_end,
    on_train_batch_begin = on_train_batch_begin,
    on_train_batch_end = on_train_batch_end,
    on_predict_batch_begin = on_predict_batch_begin,
    on_predict_batch_end = on_predict_batch_end,
    on_predict_begin = on_predict_begin,
    on_test_batch_begin = on_test_batch_begin,
    on_test_batch_end = on_test_batch_end,
    on_test_begin = on_test_begin,
    on_test_end = on_test_end
  )
  
  # remove NULL arguments from args.
  args <- Filter(function(x) !is.null(x), args)
  warn_callback(args)
  
  do.call(keras$callbacks$LambdaCallback, args)
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
#' Currently, the `fit.keras.engine.training.Model()` method for sequential 
#' models will include the following quantities in the `logs` that
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
      
    },
    
    on_predict_batch_begin = function(batch, logs = NULL) {
      
    },
    
    on_predict_batch_end = function(batch, logs = NULL) {
      
    },
    
    on_predict_begin = function(logs = NULL) {
      
    },
    
    on_predict_end = function(logs = NULL) {
      
    },
    
    on_test_batch_begin = function(batch, logs = NULL) {
      
    },
    
    on_test_batch_end = function(batch, logs = NULL) {
      
    },
    
    on_test_begin = function(logs = NULL) {
      
    },
    
    on_test_end = function(logs = NULL) {
      
    },
    
    on_train_batch_begin = function(batch, logs = NULL) {
      
    },
    
    on_train_batch_end = function(batch, logs = NULL) {
      
    }
    
  )
)

normalize_callbacks_with_metrics <- function(view_metrics, initial_epoch, callbacks) {
  
  # if callbacks isn't a list then make it one
  if (!is.null(callbacks) && !is.list(callbacks))
    callbacks <- list(callbacks)
  
  # always include the metrics callback
  if (tensorflow::tf_version() >= "2.2.0")
    metrics_callback <- KerasMetricsCallbackV2$new(view_metrics, initial_epoch)
  else
    metrics_callback <- KerasMetricsCallback$new(view_metrics)
  
  callbacks <- append(callbacks, metrics_callback)  
 
  normalize_callbacks(callbacks) 
}

warn_callback <- function(callback) {
  
  new_callbacks <- c("on_predict_batch_begin", "on_predict_batch_end", 
    "on_predict_begin", "on_predict_end",
    "on_test_batch_begin", "on_test_batch_end",
    "on_test_begin", "on_test_end",
    "on_train_batch_begin", "on_train_batch_end"
    )
  
  lapply(new_callbacks, function(x) {
    
    
    if (!(get_keras_implementation() == "tensorflow" && 
          tensorflow::tf_version() >= "2.0")) {
      
      if (inherits(callback, "KerasCallback")) {
        
        # workaround to find out if the body is empty as expected.
        bdy <- paste(as.character(body(callback[[x]])), collapse = "")
        
        if (is.null(body) || bdy != "{") {
          warning("Callback '", x, "' only works with Keras TensorFlow",
                  " implementation and Tensorflow >= 2.0")
        }
        
      } else if (inherits(callback, "list")) {
        
        if (!is.null(callback[[x]])) {
          warning("Callback '", x, "' only works with Keras TensorFlow",
                  " implementation and Tensorflow >= 2.0")
        }
        
      }
      
    }
    
  })
  
  invisible(NULL)
}

normalize_callbacks <- function(callbacks) {
  
  # if callbacks isn't a list then make it one
  if (!is.null(callbacks) && !is.list(callbacks))
    callbacks <- list(callbacks)
  
  # import callback utility module
  python_path <- system.file("python", package = "keras")
  tools <- import_from_path("kerastools", path = python_path)
  
  # convert R callbacks to Python and check whether the user
  # has already included the tensorboard callback
  have_tensorboard_callback <- FALSE
  callbacks <- lapply(callbacks, function(callback) {
    
    warn_callback(callback)
    
    # track whether we have a TensorBoard callback
    if (inherits(callback, "keras.callbacks.TensorBoard"))
      have_tensorboard_callback <<- TRUE
    
    if (inherits(callback, "KerasCallback")) {
      
      args <- list(
        r_set_context = callback$set_context,
        r_on_epoch_begin = callback$on_epoch_begin,
        r_on_epoch_end = callback$on_epoch_end,
        r_on_train_begin = callback$on_train_begin,
        r_on_train_end = callback$on_train_end,
        r_on_batch_begin = callback$on_batch_begin,
        r_on_batch_end = callback$on_batch_end,
        r_on_predict_batch_begin = callback$on_predict_batch_begin,
        r_on_predict_batch_end = callback$on_predict_batch_end,
        r_on_predict_begin = callback$on_predict_begin,
        r_on_predict_end = callback$on_predict_end,
        r_on_test_batch_begin = callback$on_test_batch_begin,
        r_on_test_batch_end = callback$on_test_batch_end,
        r_on_test_begin = callback$on_test_begin,
        r_on_test_end = callback$on_test_end,
        r_on_train_batch_begin = callback$on_train_batch_begin,
        r_on_train_batch_end = callback$on_train_batch_end
      )
      
      # on_batch_* -> on_train_batch_*
      if (!isTRUE(all.equal(callback$on_batch_begin, empty_fun))) {
        args$r_on_train_batch_begin <- callback$on_batch_begin
      }
      
      if (!isTRUE(all.equal(callback$on_batch_end, empty_fun))) {
        args$r_on_train_batch_end <- callback$on_batch_end
      }
      
      # create a python callback to map to our R callback
      do.call(tools$callback$RCallback, args)
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

empty_fun <- function(batch, logs = NULL) {}
