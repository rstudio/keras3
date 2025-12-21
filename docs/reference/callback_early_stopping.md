# Stop training when a monitored metric has stopped improving.

Assuming the goal of a training is to minimize the loss. With this, the
metric to be monitored would be `'loss'`, and mode would be `'min'`. A
`model$fit()` training loop will check at end of every epoch whether the
loss is no longer decreasing, considering the `min_delta` and `patience`
if applicable. Once it's found no longer decreasing,
`model$stop_training` is marked `TRUE` and the training terminates.

The quantity to be monitored needs to be available in `logs` list. To
make it so, pass the loss or metrics at `model$compile()`.

## Usage

``` r
callback_early_stopping(
  monitor = "val_loss",
  min_delta = 0L,
  patience = 0L,
  verbose = 0L,
  mode = "auto",
  baseline = NULL,
  restore_best_weights = FALSE,
  start_from_epoch = 0L
)
```

## Arguments

- monitor:

  Quantity to be monitored. Defaults to `"val_loss"`.

- min_delta:

  Minimum change in the monitored quantity to qualify as an improvement,
  i.e. an absolute change of less than min_delta, will count as no
  improvement. Defaults to `0`.

- patience:

  Number of epochs with no improvement after which training will be
  stopped. Defaults to `0`.

- verbose:

  Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages
  when the callback takes an action. Defaults to `0`.

- mode:

  One of `{"auto", "min", "max"}`. In `min` mode, training will stop
  when the quantity monitored has stopped decreasing; in `"max"` mode it
  will stop when the quantity monitored has stopped increasing; in
  `"auto"` mode, the direction is automatically inferred from the name
  of the monitored quantity. Defaults to `"auto"`.

- baseline:

  Baseline value for the monitored quantity. If not `NULL`, training
  will stop if the model doesn't show improvement over the baseline.
  Defaults to `NULL`.

- restore_best_weights:

  Whether to restore model weights from the epoch with the best value of
  the monitored quantity. If `FALSE`, the model weights obtained at the
  last step of training are used. An epoch will be restored regardless
  of the performance relative to the `baseline`. If no epoch improves on
  `baseline`, training will run for `patience` epochs and restore
  weights from the best epoch in that set. Defaults to `FALSE`.

- start_from_epoch:

  Number of epochs to wait before starting to monitor improvement. This
  allows for a warm-up period in which no improvement is expected and
  thus training will not be stopped. Defaults to `0`.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/reference/fit.keras.src.models.model.Model.md).

## Examples

    callback <- callback_early_stopping(monitor = 'loss',
                                       patience = 3)
    # This callback will stop the training when there is no improvement in
    # the loss for three consecutive epochs.
    model <- keras_model_sequential() %>%
      layer_dense(10)
    model %>% compile(optimizer = optimizer_sgd(), loss = 'mse')
    history <- model %>% fit(x = op_ones(c(5, 20)),
                             y = op_zeros(5),
                             epochs = 10, batch_size = 1,
                             callbacks = list(callback),
                             verbose = 0)
    nrow(as.data.frame(history))  # Only 4 epochs are run.

    ## [1] 10

## See also

- <https://keras.io/api/callbacks/early_stopping#earlystopping-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/reference/callback_csv_logger.md)  
[`callback_lambda()`](https://keras3.posit.co/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/reference/callback_terminate_on_nan.md)  
