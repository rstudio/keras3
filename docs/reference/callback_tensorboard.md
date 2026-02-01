# Enable visualizations for TensorBoard.

TensorBoard is a visualization tool provided with TensorFlow. A
TensorFlow installation is required to use this callback.

This callback logs events for TensorBoard, including:

- Metrics summary plots

- Training graph visualization

- Weight histograms

- Sampled profiling

When used in `model |> evaluate()` or regular validation in addition to
epoch summaries, there will be a summary that records evaluation metrics
vs `model$optimizer$iterations` written. The metric names will be
prepended with `evaluation`, with `model$optimizer$iterations` being the
step in the visualized TensorBoard.

If you have installed TensorFlow with `pip` or
[`reticulate::py_install()`](https://rstudio.github.io/reticulate/reference/py_install.html),
you should be able to launch TensorBoard from the command line:

    tensorboard --logdir=path_to_your_logs

or from R with
[`tensorflow::tensorboard()`](https://rdrr.io/pkg/tensorflow/man/tensorboard.html).

You can find more information about TensorBoard
[here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

## Usage

``` r
callback_tensorboard(
  log_dir = "logs",
  histogram_freq = 0L,
  write_graph = TRUE,
  write_images = FALSE,
  write_steps_per_second = FALSE,
  update_freq = "epoch",
  profile_batch = 0L,
  embeddings_freq = 0L,
  embeddings_metadata = NULL
)
```

## Arguments

- log_dir:

  the path of the directory where to save the log files to be parsed by
  TensorBoard. e.g., `log_dir = file.path(working_dir, 'logs')`. This
  directory should not be reused by any other callbacks.

- histogram_freq:

  frequency (in epochs) at which to compute weight histograms for the
  layers of the model. If set to 0, histograms won't be computed.
  Validation data (or split) must be specified for histogram
  visualizations.

- write_graph:

  (Not supported at this time) Whether to visualize the graph in
  TensorBoard. Note that the log file can become quite large when
  `write_graph` is set to `TRUE`.

- write_images:

  whether to write model weights to visualize as image in TensorBoard.

- write_steps_per_second:

  whether to log the training steps per second into TensorBoard. This
  supports both epoch and batch frequency logging.

- update_freq:

  `"batch"` or `"epoch"` or integer. When using `"epoch"`, writes the
  losses and metrics to TensorBoard after every epoch. If using an
  integer, let's say `1000`, all metrics and losses (including custom
  ones added by `Model.compile`) will be logged to TensorBoard every
  1000 batches. `"batch"` is a synonym for 1, meaning that they will be
  written every batch. Note however that writing too frequently to
  TensorBoard can slow down your training, especially when used with
  distribution strategies as it will incur additional synchronization
  overhead. Batch-level summary writing is also available via
  `train_step` override. Please see [TensorBoard Scalars
  tutorial](https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)
  for more details.

- profile_batch:

  Profile the batch(es) to sample compute characteristics. profile_batch
  must be a non-negative integer or a tuple of integers. A pair of
  positive integers signify a range of batches to profile. By default,
  profiling is disabled.

- embeddings_freq:

  frequency (in epochs) at which embedding layers will be visualized. If
  set to `0`, embeddings won't be visualized.

- embeddings_metadata:

  Named list which maps embedding layer names to the filename of a file
  in which to save metadata for the embedding layer. In case the same
  metadata file is to be used for all embedding layers, a single
  filename can be passed.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/reference/fit.keras.src.models.model.Model.md).

## Examples

    tensorboard_callback <- callback_tensorboard(log_dir = "./logs")
    model %>% fit(x_train, y_train, epochs = 2, callbacks = list(tensorboard_callback))
    # Then run the tensorboard command to view the visualizations.

Custom batch-level summaries in a subclassed Model:

    MyModel <- new_model_class("MyModel",
      initialize = function() {
        self$dense <- layer_dense(units = 10)
      },
      call = function(x) {
        outputs <- x |> self$dense()
        tf$summary$histogram('outputs', outputs)
        outputs
      }
    )

    model <- MyModel()
    model |> compile(optimizer = 'sgd', loss = 'mse')

    # Make sure to set `update_freq = N` to log a batch-level summary every N
    # batches. In addition to any `tf$summary` contained in `model$call()`,
    # metrics added in `model |>compile` will be logged every N batches.
    tb_callback <- callback_tensorboard(log_dir = './logs', update_freq = 1)
    model |> fit(x_train, y_train, callbacks = list(tb_callback))

Custom batch-level summaries in a Functional API Model:

    my_summary <- function(x) {
      tf$summary$histogram('x', x)
      x
    }

    inputs <- layer_input(10)
    outputs <- inputs |>
      layer_dense(10) |>
      layer_lambda(my_summary)

    model <- keras_model(inputs, outputs)
    model |> compile(optimizer = 'sgd', loss = 'mse')

    # Make sure to set `update_freq = N` to log a batch-level summary every N
    # batches. In addition to any `tf.summary` contained in `Model.call`,
    # metrics added in `Model.compile` will be logged every N batches.
    tb_callback <- callback_tensorboard(log_dir = './logs', update_freq = 1)
    model |> fit(x_train, y_train, callbacks = list(tb_callback))

Profiling:

    # Profile a single batch, e.g. the 5th batch.
    tensorboard_callback <- callback_tensorboard(
      log_dir = './logs', profile_batch = 5)
    model |> fit(x_train, y_train, epochs = 2,
                 callbacks = list(tensorboard_callback))

    # Profile a range of batches, e.g. from 10 to 20.
    tensorboard_callback <- callback_tensorboard(
      log_dir = './logs', profile_batch = c(10, 20))
    model |> fit(x_train, y_train, epochs = 2,
                 callbacks = list(tensorboard_callback))

## See also

- <https://keras.io/api/callbacks/tensorboard#tensorboard-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/reference/callback_swap_ema_weights.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/reference/callback_terminate_on_nan.md)  
