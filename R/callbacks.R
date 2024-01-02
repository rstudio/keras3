


#' Callback to back up and restore the training state.
#'
#' @description
#' `callback_backup_and_restore` callback is intended to recover training from an
#' interruption that has happened in the middle of a `fit` execution, by
#' backing up the training states in a temporary checkpoint file, at the end of
#' each epoch. Each backup overwrites the previously written checkpoint file,
#' so at any given time there is at most one such checkpoint file for
#' backup/restoring purpose.
#'
#' If training restarts before completion, the training state (which includes
#' the model weights and epoch number) is restored to the most recently saved
#' state at the beginning of a new `fit` run. At the completion of a
#' `fit` run, the temporary checkpoint file is deleted.
#'
#' Note that the user is responsible to bring jobs back after the interruption.
#' This callback is important for the backup and restore mechanism for fault
#' tolerance purpose, and the model to be restored from a previous checkpoint
#' is expected to be the same as the one used to back up. If user changes
#' arguments passed to `compile` or `fit`, the checkpoint saved for fault tolerance
#' can become invalid.
#'
#' # Examples
#'
#' ```{r}
#' callback_interrupting <- new_callback_class(
#'   "InterruptingCallback",
#'   on_epoch_begin = function(epoch, logs = NULL) {
#'     if (epoch == 4) {
#'       stop('Interrupting!')
#'     }
#'   }
#' )
#'
#' backup_dir <- tempfile()
#' callback <- callback_backup_and_restore(backup_dir = backup_dir)
#' model <- keras_model_sequential() %>%
#'   layer_dense(10)
#' model %>% compile(optimizer = optimizer_sgd(), loss = 'mse')
#'
#' tryCatch({
#'   model %>% fit(x = op_ones(c(5, 20)),
#'                 y = op_zeros(5),
#'                 epochs = 10, batch_size = 1,
#'                 callbacks = list(callback, callback_interrupting()),
#'                 verbose = 0)
#' }, python.builtin.RuntimeError = function(e) message("Interrupted!"))
#'
#' model$history$epoch
#' # model$history %>% keras3:::to_keras_training_history() %>% as.data.frame() %>% print()
#'
#' history <- model %>% fit(x = op_ones(c(5, 20)),
#'                          y = op_zeros(5),
#'                          epochs = 10, batch_size = 1,
#'                          callbacks = list(callback),
#'                          verbose = 0)
#'
#' # Only 6 more epochs are run, since first training got interrupted at
#' # zero-indexed epoch 4, second training will continue from 4 to 9.
#' nrow(as.data.frame(history))
#' ```
#'
#' @param backup_dir
#' String, path of directory where to store the data
#' needed to restore the model. The directory
#' cannot be reused elsewhere to store other files, e.g. by the
#' `backup_and_restore` callback of another training run,
#' or by another callback (e.g. `callback_model_checkpoint`)
#' of the same training run.
#'
#' @param save_freq
#' `"epoch"`, integer, or `FALSE`. When set to `"epoch"`,
#' the callback saves the checkpoint at the end of each epoch.
#' When set to an integer, the callback saves the checkpoint every
#' `save_freq` batches. Set `save_freq = FALSE` only if using
#' preemption checkpointing (i.e. with `save_before_preemption = TRUE`).
#'
#' @param delete_checkpoint
#' Boolean, defaults to `TRUE`. This `backup_and_restore`
#' callback works by saving a checkpoint to back up the training state.
#' If `delete_checkpoint = TRUE`, the checkpoint will be deleted after
#' training is finished. Use `FALSE` if you'd like to keep the checkpoint
#' for future usage.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/backup_and_restore#backupandrestore-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/BackupAndRestore>
#' @tether keras.callbacks.BackupAndRestore
callback_backup_and_restore <-
function (backup_dir, save_freq = "epoch", delete_checkpoint = TRUE)
{
    args <- capture_args2(list(save_freq = as_integer))
    do.call(keras$callbacks$BackupAndRestore, args)
}


#' Callback that streams epoch results to a CSV file.
#'
#' @description
#' Supports all values that can be represented as a string,
#' including 1D iterables such as atomic vectors.
#'
#' # Examples
#' ```r
#' csv_logger <- callback_csv_logger('training.log')
#' model %>% fit(X_train, Y_train, callbacks = list(csv_logger))
#' ```
#'
#' @param filename
#' Filename of the CSV file, e.g. `'run/log.csv'`.
#'
#' @param separator
#' String used to separate elements in the CSV file.
#'
#' @param append
#' Boolean. `TRUE`: append if file exists (useful for continuing
#' training). `FALSE`: overwrite existing file.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/csv_logger#csvlogger-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger>
#' @tether keras.callbacks.CSVLogger
callback_csv_logger <-
function (filename, separator = ",", append = FALSE)
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$CSVLogger, args)
}


#' Stop training when a monitored metric has stopped improving.
#'
#' @description
#' Assuming the goal of a training is to minimize the loss. With this, the
#' metric to be monitored would be `'loss'`, and mode would be `'min'`. A
#' `model$fit()` training loop will check at end of every epoch whether
#' the loss is no longer decreasing, considering the `min_delta` and
#' `patience` if applicable. Once it's found no longer decreasing,
#' `model$stop_training` is marked `TRUE` and the training terminates.
#'
#' The quantity to be monitored needs to be available in `logs` list.
#' To make it so, pass the loss or metrics at `model$compile()`.
#'
#' # Examples
#' ```{r}
#' callback <- callback_early_stopping(monitor = 'loss',
#'                                    patience = 3)
#' # This callback will stop the training when there is no improvement in
#' # the loss for three consecutive epochs.
#' model <- keras_model_sequential() %>%
#'   layer_dense(10)
#' model %>% compile(optimizer = optimizer_sgd(), loss = 'mse')
#' history <- model %>% fit(x = op_ones(c(5, 20)),
#'                          y = op_zeros(5),
#'                          epochs = 10, batch_size = 1,
#'                          callbacks = list(callback),
#'                          verbose = 0)
#' nrow(as.data.frame(history))  # Only 4 epochs are run.
#' ```
#'
#' @param monitor
#' Quantity to be monitored. Defaults to `"val_loss"`.
#'
#' @param min_delta
#' Minimum change in the monitored quantity to qualify as an
#' improvement, i.e. an absolute change of less than min_delta, will
#' count as no improvement. Defaults to `0`.
#'
#' @param patience
#' Number of epochs with no improvement after which training will
#' be stopped. Defaults to `0`.
#'
#' @param verbose
#' Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays
#' messages when the callback takes an action. Defaults to `0`.
#'
#' @param mode
#' One of `{"auto", "min", "max"}`. In `min` mode, training will stop
#' when the quantity monitored has stopped decreasing; in `"max"` mode
#' it will stop when the quantity monitored has stopped increasing; in
#' `"auto"` mode, the direction is automatically inferred from the name
#' of the monitored quantity. Defaults to `"auto"`.
#'
#' @param baseline
#' Baseline value for the monitored quantity. If not `NULL`,
#' training will stop if the model doesn't show improvement over the
#' baseline. Defaults to `NULL`.
#'
#' @param restore_best_weights
#' Whether to restore model weights from the epoch
#' with the best value of the monitored quantity. If `FALSE`, the model
#' weights obtained at the last step of training are used. An epoch
#' will be restored regardless of the performance relative to the
#' `baseline`. If no epoch improves on `baseline`, training will run
#' for `patience` epochs and restore weights from the best epoch in
#' that set. Defaults to `FALSE`.
#'
#' @param start_from_epoch
#' Number of epochs to wait before starting to monitor
#' improvement. This allows for a warm-up period in which no
#' improvement is expected and thus training will not be stopped.
#' Defaults to `0`.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/early_stopping#earlystopping-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping>
#' @tether keras.callbacks.EarlyStopping
callback_early_stopping <-
function (monitor = "val_loss", min_delta = 0L, patience = 0L,
    verbose = 0L, mode = "auto", baseline = NULL, restore_best_weights = FALSE,
    start_from_epoch = 0L)
{
    args <- capture_args2(list(min_delta = as_integer, patience = as_integer,
        verbose = as_integer, start_from_epoch = as_integer))
    do.call(keras$callbacks$EarlyStopping, args)
}


#' Callback for creating simple, custom callbacks on-the-fly.
#'
#' @description
#' This callback is constructed with anonymous functions that will be called
#' at the appropriate time (during `Model.{fit | evaluate | predict}`).
#' Note that the callbacks expects positional arguments, as:
#'
#' - `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
#'   `epoch`, `logs`
#' - `on_train_begin` and `on_train_end` expect one positional argument:
#'   `logs`
#' - `on_train_batch_begin` and `on_train_batch_end` expect two positional
#'   arguments: `batch`, `logs`
#' - See `Callback` class definition for the full list of functions and their
#'   expected arguments.
#'
#' # Examples
#'
#' ```r
#' # Print the batch number at the beginning of every batch.
#' batch_print_callback <- callback_lambda(
#'   on_train_batch_begin = function(batch, logs) {
#'     print(batch)
#'   }
#' )
#'
#' # Stream the epoch loss to a file in new-line delimited JSON format
#' # (one valid JSON object per line)
#' json_log <- file('loss_log.json', open = 'wt')
#' json_logging_callback <- callback_lambda(
#'   on_epoch_end = function(epoch, logs) {
#'     jsonlite::write_json(
#'       list(epoch = epoch, loss = logs$loss),
#'       json_log,
#'       append = TRUE
#'     )
#'   },
#'   on_train_end = function(logs) {
#'     close(json_log)
#'   }
#' )
#'
#' # Terminate some processes after having finished model training.
#' processes <- ...
#' cleanup_callback <- callback_lambda(
#'   on_train_end = function(logs) {
#'     for (p in processes) {
#'       if (is_alive(p)) {
#'         terminate(p)
#'       }
#'     }
#'   }
#' )
#'
#' model %>% fit(
#'   ...,
#'   callbacks = list(
#'     batch_print_callback,
#'     json_logging_callback,
#'     cleanup_callback
#'   )
#' )
#' ```
#'
#' @param on_epoch_begin
#' called at the beginning of every epoch.
#'
#' @param on_epoch_end
#' called at the end of every epoch.
#'
#' @param on_train_begin
#' called at the beginning of model training.
#'
#' @param on_train_end
#' called at the end of model training.
#'
#' @param on_train_batch_begin
#' called at the beginning of every train batch.
#'
#' @param on_train_batch_end
#' called at the end of every train batch.
#'
#' @param ...
#' Any function in `Callback` that you want to override by
#' passing `function_name = function`. For example,
#' `callback_lambda(.., on_train_end = train_end_fn)`. The custom function
#' needs to have same arguments as the ones defined in `Callback`.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/lambda_callback#lambdacallback-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback>
#' @tether keras.callbacks.LambdaCallback
callback_lambda <-
function (on_epoch_begin = NULL, on_epoch_end = NULL, on_train_begin = NULL,
    on_train_end = NULL, on_train_batch_begin = NULL, on_train_batch_end = NULL,
    ...)
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$LambdaCallback, args)
}


#' Learning rate scheduler.
#'
#' @description
#' At the beginning of every epoch, this callback gets the updated learning
#' rate value from `schedule` function provided at `__init__`, with the current
#' epoch and current learning rate, and applies the updated learning rate on
#' the optimizer.
#'
#' # Examples
#' ```{r}
#' # This function keeps the initial learning rate for the first ten epochs
#' # and decreases it exponentially after that.
#' scheduler <- function(epoch, lr) {
#'   if (epoch < 10)
#'     return(lr)
#'   else
#'     return(lr * exp(-0.1))
#' }
#'
#' model <- keras_model_sequential() |> layer_dense(units = 10)
#' model |> compile(optimizer = optimizer_sgd(), loss = 'mse')
#' model$optimizer$learning_rate |> as.array() |> round(5)
#' ```
#'
#' ```{r}
#' callback <- callback_learning_rate_scheduler(schedule = scheduler)
#' history <- model |> fit(x = array(runif(100), c(5, 20)),
#'                         y = array(0, c(5, 1)),
#'                         epochs = 15, callbacks = list(callback), verbose = 0)
#' model$optimizer$learning_rate |> as.array() |> round(5)
#' ```
#'
#' @param schedule
#' A function that takes an epoch index (integer, indexed from 0)
#' and current learning rate (float) as inputs and returns a new
#' learning rate as output (float).
#'
#' @param verbose
#' Integer. 0: quiet, 1: log update messages.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/learning_rate_scheduler#learningratescheduler-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler>
#' @tether keras.callbacks.LearningRateScheduler
callback_learning_rate_scheduler <-
function (schedule, verbose = 0L)
{
    args <- capture_args2(list(schedule = as_integer, verbose = as_integer))
    do.call(keras$callbacks$LearningRateScheduler, args)
}


#' Callback to save the Keras model or model weights at some frequency.
#'  @description
#' `callback_model_checkpoint()` is used in conjunction with training using
#' `model |> fit()` to save a model or weights (in a checkpoint file) at some
#' interval, so the model or weights can be loaded later to continue the
#' training from the state saved.
#'
#' A few options this callback provides include:
#'
#' - Whether to only keep the model that has achieved the "best performance" so
#'   far, or whether to save the model at the end of every epoch regardless of
#'   performance.
#' - Definition of "best"; which quantity to monitor and whether it should be
#'   maximized or minimized.
#' - The frequency it should save at. Currently, the callback supports saving
#'   at the end of every epoch, or after a fixed number of training batches.
#' - Whether only weights are saved, or the whole model is saved.
#'
#' # Examples
#' ```{r}
#' model <- keras_model_sequential(input_shape = c(10)) |>
#'   layer_dense(1, activation = "sigmoid") |>
#'   compile(loss = "binary_crossentropy", optimizer = "adam",
#'           metrics = c('accuracy'))
#'
#' EPOCHS <- 10
#' checkpoint_filepath <- tempfile('checkpoint-model-', fileext = ".keras")
#' model_checkpoint_callback <- callback_model_checkpoint(
#'   filepath = checkpoint_filepath,
#'   monitor = 'val_accuracy',
#'   mode = 'max',
#'   save_best_only = TRUE
#' )
#'
#' # Model is saved at the end of every epoch, if it's the best seen so far.
#' model |> fit(x = random_uniform(c(2, 10)), y = op_ones(2, 1),
#'              epochs = EPOCHS, validation_split = .5, verbose = 0,
#'              callbacks = list(model_checkpoint_callback))
#'
#' # The model (that are considered the best) can be loaded as -
#' load_model(checkpoint_filepath)
#'
#' # Alternatively, one could checkpoint just the model weights as -
#' checkpoint_filepath <- tempfile('checkpoint-', fileext = ".weights.h5")
#' model_checkpoint_callback <- callback_model_checkpoint(
#'   filepath = checkpoint_filepath,
#'   save_weights_only = TRUE,
#'   monitor = 'val_accuracy',
#'   mode = 'max',
#'   save_best_only = TRUE
#' )
#'
#' # Model weights are saved at the end of every epoch, if it's the best seen
#' # so far.
#' # same as above
#' model |> fit(x = random_uniform(c(2, 10)), y = op_ones(2, 1),
#'              epochs = EPOCHS, validation_split = .5, verbose = 0,
#'              callbacks = list(model_checkpoint_callback))
#'
#' # The model weights (that are considered the best) can be loaded
#' model |> load_model_weights(checkpoint_filepath)
#' ```
#'
#' @param filepath
#' string, path to save the model file.
#' `filepath` can contain named formatting options,
#' which will be filled the value of `epoch` and keys in `logs`
#' (passed in `on_epoch_end`).
#' The `filepath` name needs to end with `".weights.h5"` when
#' `save_weights_only = TRUE` or should end with `".keras"` when
#' checkpoint saving the whole model (default).
#' For example:
#' if `filepath` is `"{epoch:02d}-{val_loss:.2f}.keras"`, then the
#' model checkpoints will be saved with the epoch number and the
#' validation loss in the filename. The directory of the filepath
#' should not be reused by any other callbacks to avoid conflicts.
#'
#' @param monitor
#' The metric name to monitor. Typically the metrics are set by
#' the `model |> compile()` method. Note:
#' * Prefix the name with `"val_"` to monitor validation metrics.
#' * Use `"loss"` or `"val_loss"` to monitor the model's total loss.
#' * If you specify metrics as strings, like `"accuracy"`, pass the
#'     same string (with or without the `"val_"` prefix).
#' * If you pass `Metric` objects (created by one of `metric_*()`), `monitor` should be set to
#'     `metric$name`.
#' * If you're not sure about the metric names you can check the
#'     contents of the `history$metrics` list returned by
#'     `history <- model |> fit()`
#' * Multi-output models set additional prefixes on the metric names.
#'
#' @param verbose
#' Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
#' displays messages when the callback takes an action.
#'
#' @param save_best_only
#' if `save_best_only = TRUE`, it only saves when the model
#' is considered the "best" and the latest best model according to the
#' quantity monitored will not be overwritten. If `filepath` doesn't
#' contain formatting options like `{epoch}` then `filepath` will be
#' overwritten by each new better model.
#'
#' @param mode
#' one of {`"auto"`, `"min"`, `"max"`}. If `save_best_only = TRUE`, the
#' decision to overwrite the current save file is made based on either
#' the maximization or the minimization of the monitored quantity.
#' For `val_acc`, this should be `"max"`, for `val_loss` this should be
#' `"min"`, etc. In `"auto"` mode, the mode is set to `"max"` if the
#' quantities monitored are `"acc"` or start with `"fmeasure"` and are
#' set to `"min"` for the rest of the quantities.
#'
#' @param save_weights_only
#' if TRUE, then only the model's weights will be saved
#' (`model |> save_model_weights(filepath)`), else the full model is saved
#' (`model |> save_model(filepath)`).
#'
#' @param save_freq
#' `"epoch"` or integer. When using `"epoch"`, the callback
#' saves the model after each epoch. When using integer, the callback
#' saves the model at end of this many batches. If the `Model` is
#' compiled with `steps_per_execution = N`, then the saving criteria will
#' be checked every Nth batch. Note that if the saving isn't aligned to
#' epochs, the monitored metric may potentially be less reliable (it
#' could reflect as little as 1 batch, since the metrics get reset
#' every epoch). Defaults to `"epoch"`.
#'
#' @param initial_value_threshold
#' Floating point initial "best" value of the
#' metric to be monitored. Only applies if `save_best_value = TRUE`. Only
#' overwrites the model weights already saved if the performance of
#' current model is better than this value.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/model_checkpoint#modelcheckpoint-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint>
#' @tether keras.callbacks.ModelCheckpoint
callback_model_checkpoint <-
function (filepath, monitor = "val_loss", verbose = 0L, save_best_only = FALSE,
    save_weights_only = FALSE, mode = "auto", save_freq = "epoch",
    initial_value_threshold = NULL)
{
    args <- capture_args2(list(verbose = as_integer, save_freq = as_integer))
    do.call(keras$callbacks$ModelCheckpoint, args)
}


#' Callback that prints metrics to stdout.
#'
#' @description
#'
#' # Raises
#' ValueError: In case of invalid `count_mode`.
#'
#' @param count_mode
#' One of `"steps"` or `"samples"`.
#' Whether the progress bar should
#' count samples seen or steps (batches) seen.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/progbar_logger#progbarlogger-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ProgbarLogger>
#' @tether keras.callbacks.ProgbarLogger
callback_progbar_logger <-
function (count_mode = NULL)
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$ProgbarLogger, args)
}


#' Reduce learning rate when a metric has stopped improving.
#'
#' @description
#' Models often benefit from reducing the learning rate by a factor
#' of 2-10 once learning stagnates. This callback monitors a
#' quantity and if no improvement is seen for a 'patience' number
#' of epochs, the learning rate is reduced.
#'
#' # Examples
#' ```{r, eval = FALSE}
#' reduce_lr <- callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.2,
#'                                            patience = 5, min_lr = 0.001)
#' model %>% fit(x_train, y_train, callbacks = list(reduce_lr))
#' ```
#'
#' @param monitor
#' String. Quantity to be monitored.
#'
#' @param factor
#' Numeric. Factor by which the learning rate will be reduced.
#' `new_lr = lr * factor`.
#'
#' @param patience
#' Integer. Number of epochs with no improvement after which
#' learning rate will be reduced.
#'
#' @param verbose
#' Integer. 0: quiet, 1: update messages.
#'
#' @param mode
#' String. One of `{'auto', 'min', 'max'}`. In `'min'` mode,
#' the learning rate will be reduced when the
#' quantity monitored has stopped decreasing; in `'max'` mode it will
#' be reduced when the quantity monitored has stopped increasing; in
#' `'auto'` mode, the direction is automatically inferred from the name
#' of the monitored quantity.
#'
#' @param min_delta
#' Numeric. Threshold for measuring the new optimum, to only focus
#' on significant changes.
#'
#' @param cooldown
#' Integer. Number of epochs to wait before resuming normal
#' operation after the learning rate has been reduced.
#'
#' @param min_lr
#' Numeric. Lower bound on the learning rate.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/reduce_lr_on_plateau#reducelronplateau-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau>
#' @tether keras.callbacks.ReduceLROnPlateau
callback_reduce_lr_on_plateau <-
function (monitor = "val_loss", factor = 0.1, patience = 10L,
    verbose = 0L, mode = "auto", min_delta = 1e-04, cooldown = 0L,
    min_lr = 0L, ...)
{
    args <- capture_args2(list(patience = as_integer, verbose = as_integer,
        cooldown = as_integer, min_lr = as_integer))
    do.call(keras$callbacks$ReduceLROnPlateau, args)
}


#' Callback used to stream events to a server.
#'
#' @description
#' Requires the `requests` library.
#' Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
#' HTTP POST, with a `data` argument which is a
#' JSON-encoded named list of event data.
#' If `send_as_json = TRUE`, the content type of the request will be
#' `"application/json"`.
#' Otherwise the serialized JSON will be sent within a form.
#'
#' @param root
#' String; root url of the target server.
#'
#' @param path
#' String; path relative to `root` to which the events will be sent.
#'
#' @param field
#' String; JSON field under which the data will be stored.
#' The field is used only if the payload is sent within a form
#' (i.e. send_as_json is set to `FALSE`).
#'
#' @param headers
#' Named list; optional custom HTTP headers.
#'
#' @param send_as_json
#' Boolean; whether the request should be
#' sent as `"application/json"`.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/remote_monitor#remotemonitor-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/RemoteMonitor>
#' @tether keras.callbacks.RemoteMonitor
callback_remote_monitor <-
function (root = "http://localhost:9000", path = "/publish/epoch/end/",
    field = "data", headers = NULL, send_as_json = FALSE)
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$RemoteMonitor, args)
}


#' Enable visualizations for TensorBoard.
#'
#' @description
#' TensorBoard is a visualization tool provided with TensorFlow. A TensorFlow
#' installation is required to use this callback.
#'
#' This callback logs events for TensorBoard, including:
#'
#' * Metrics summary plots
#' * Training graph visualization
#' * Weight histograms
#' * Sampled profiling
#'
#' When used in `model |> evaluate()` or regular validation
#' in addition to epoch summaries, there will be a summary that records
#' evaluation metrics vs `model$optimizer$iterations` written. The metric names
#' will be prepended with `evaluation`, with `model$optimizer$iterations` being
#' the step in the visualized TensorBoard.
#'
#' If you have installed TensorFlow with `pip` or `reticulate::py_install()`, you should be able
#' to launch TensorBoard from the command line:
#'
#' ```
#' tensorboard --logdir=path_to_your_logs
#' ```
#' or from R with `tensorflow::tensorboard()`.
#'
#' You can find more information about TensorBoard
#' [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
#'
#' # Examples
#' Basic usage:
#'
#' ```{r, eval = FALSE}
#' tensorboard_callback <- callback_tensorboard(log_dir = "./logs")
#' model %>% fit(x_train, y_train, epochs = 2, callbacks = list(tensorboard_callback))
#' # Then run the tensorboard command to view the visualizations.
#' ```
#'
#' Custom batch-level summaries in a subclassed Model:
#'
#' ```{r, eval = FALSE}
#' MyModel <- new_model_class("MyModel",
#'   initialize = function() {
#'     self$dense <- layer_dense(units = 10)
#'   },
#'   call = function(x) {
#'     outputs <- x |> self$dense()
#'     tf$summary$histogram('outputs', outputs)
#'     outputs
#'   }
#' )
#'
#' model <- MyModel()
#' model |> compile(optimizer = 'sgd', loss = 'mse')
#'
#' # Make sure to set `update_freq = N` to log a batch-level summary every N
#' # batches. In addition to any `tf$summary` contained in `model$call()`,
#' # metrics added in `model |>compile` will be logged every N batches.
#' tb_callback <- callback_tensorboard(log_dir = './logs', update_freq = 1)
#' model |> fit(x_train, y_train, callbacks = list(tb_callback))
#' ```
#'
#' Custom batch-level summaries in a Functional API Model:
#'
#' ```{r, eval = FALSE}
#' my_summary <- function(x) {
#'   tf$summary$histogram('x', x)
#'   x
#' }
#'
#' inputs <- layer_input(10)
#' outputs <- inputs |>
#'   layer_dense(10) |>
#'   layer_lambda(my_summary)
#'
#' model <- keras_model(inputs, outputs)
#' model |> compile(optimizer = 'sgd', loss = 'mse')
#'
#' # Make sure to set `update_freq = N` to log a batch-level summary every N
#' # batches. In addition to any `tf.summary` contained in `Model.call`,
#' # metrics added in `Model.compile` will be logged every N batches.
#' tb_callback <- callback_tensorboard(log_dir = './logs', update_freq = 1)
#' model |> fit(x_train, y_train, callbacks = list(tb_callback))
#' ```
#'
#' Profiling:
#'
#' ```{r, eval = FALSE}
#' # Profile a single batch, e.g. the 5th batch.
#' tensorboard_callback <- callback_tensorboard(
#'   log_dir = './logs', profile_batch = 5)
#' model |> fit(x_train, y_train, epochs = 2,
#'              callbacks = list(tensorboard_callback))
#'
#' # Profile a range of batches, e.g. from 10 to 20.
#' tensorboard_callback <- callback_tensorboard(
#'   log_dir = './logs', profile_batch = c(10, 20))
#' model |> fit(x_train, y_train, epochs = 2,
#'              callbacks = list(tensorboard_callback))
#' ```
#'
#' @param log_dir
#' the path of the directory where to save the log files to be
#' parsed by TensorBoard. e.g.,
#' `log_dir = file.path(working_dir, 'logs')`.
#' This directory should not be reused by any other callbacks.
#'
#' @param histogram_freq
#' frequency (in epochs) at which to compute
#' weight histograms for the layers of the model. If set to 0,
#' histograms won't be computed. Validation data (or split) must be
#' specified for histogram visualizations.
#'
#' @param write_graph
#' (Not supported at this time)
#' Whether to visualize the graph in TensorBoard.
#' Note that the log file can become quite large
#' when `write_graph` is set to `TRUE`.
#'
#' @param write_images
#' whether to write model weights to visualize as image in
#' TensorBoard.
#'
#' @param write_steps_per_second
#' whether to log the training steps per second
#' into TensorBoard. This supports both epoch and batch frequency
#' logging.
#'
#' @param update_freq
#' `"batch"` or `"epoch"` or integer. When using `"epoch"`,
#' writes the losses and metrics to TensorBoard after every epoch.
#' If using an integer, let's say `1000`, all metrics and losses
#' (including custom ones added by `Model.compile`) will be logged to
#' TensorBoard every 1000 batches. `"batch"` is a synonym for 1,
#' meaning that they will be written every batch.
#' Note however that writing too frequently to TensorBoard can slow
#' down your training, especially when used with distribution
#' strategies as it will incur additional synchronization overhead.
#' Batch-level summary writing is also available via `train_step`
#' override. Please see
#' [TensorBoard Scalars tutorial](
#'     https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)  # noqa: E501
#' for more details.
#'
#' @param profile_batch
#' (Not supported at this time)
#' Profile the batch(es) to sample compute characteristics.
#' profile_batch must be a non-negative integer or a tuple of integers.
#' A pair of positive integers signify a range of batches to profile.
#' By default, profiling is disabled.
#'
#' @param embeddings_freq
#' frequency (in epochs) at which embedding layers will be
#' visualized. If set to 0, embeddings won't be visualized.
#'
#' @param embeddings_metadata
#' Named list which maps embedding layer names to the
#' filename of a file in which to save metadata for the embedding layer.
#' In case the same metadata file is to be
#' used for all embedding layers, a single filename can be passed.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/tensorboard#tensorboard-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard>
#' @tether keras.callbacks.TensorBoard
callback_tensorboard <-
function (log_dir = "logs", histogram_freq = 0L, write_graph = TRUE,
    write_images = FALSE, write_steps_per_second = FALSE, update_freq = "epoch",
    profile_batch = 0L, embeddings_freq = 0L, embeddings_metadata = NULL)
{
    args <- capture_args2(list(histogram_freq = as_integer, update_freq = as_integer,
        profile_batch = as_integer, embeddings_freq = as_integer))
    do.call(keras$callbacks$TensorBoard, args)
}


#' Callback that terminates training when a NaN loss is encountered.
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/terminate_on_nan#terminateonnan-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TerminateOnNaN>
#' @tether keras.callbacks.TerminateOnNaN
callback_terminate_on_nan <-
function ()
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$TerminateOnNaN, args)
}


# --------------------------------------------------------------------------------




#' (Deprecated) Base R6 class for Keras callbacks
#'
#' New custom callbacks implemented as R6 classes are encouraged to inherit from
#' `keras$callbacks$Callback` directly.
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
#' Currently, the `fit.keras.src.models.model.Model()` method for sequential
#' models will include the following quantities in the `logs` that
#' it passes to its callbacks:
#'
#' - `on_epoch_end`: logs include `acc` and `loss`, and optionally include `val_loss` (if validation is enabled in `fit`), and `val_acc` (if validation and accuracy monitoring are enabled).
#' - `on_batch_begin`: logs include `size`, the number of samples in the current batch.
#' - `on_batch_end`: logs include `loss`, and optionally `acc` (if accuracy monitoring is enabled).
#'
#' @return [KerasCallback].
#' @keywords internal
#' @examples
#' \dontrun{
#' library(keras3)
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
  python_path <- system.file("python", package = "keras3")
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
  if ((nzchar(Sys.getenv("RUN_DIR")) || tfruns::is_run_active()) &&
      !have_tensorboard_callback)
    callbacks <- append(callbacks, callback_tensorboard())

  # return the callbacks
  callbacks
}

empty_fun <- function(batch, logs = NULL) {}
