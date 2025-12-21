# Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor of 2-10
once learning stagnates. This callback monitors a quantity and if no
improvement is seen for a 'patience' number of epochs, the learning rate
is reduced.

## Usage

``` r
callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.1,
  patience = 10L,
  verbose = 0L,
  mode = "auto",
  min_delta = 0.0001,
  cooldown = 0L,
  min_lr = 0,
  ...
)
```

## Arguments

- monitor:

  String. Quantity to be monitored.

- factor:

  Float. Factor by which the learning rate will be reduced.
  `new_lr = lr * factor`.

- patience:

  Integer. Number of epochs with no improvement after which learning
  rate will be reduced.

- verbose:

  Integer. 0: quiet, 1: update messages.

- mode:

  String. One of `{'auto', 'min', 'max'}`. In `'min'` mode, the learning
  rate will be reduced when the quantity monitored has stopped
  decreasing; in `'max'` mode it will be reduced when the quantity
  monitored has stopped increasing; in `'auto'` mode, the direction is
  automatically inferred from the name of the monitored quantity.

- min_delta:

  Float. Threshold for measuring the new optimum, to only focus on
  significant changes.

- cooldown:

  Integer. Number of epochs to wait before resuming normal operation
  after the learning rate has been reduced.

- min_lr:

  Float. Lower bound on the learning rate.

- ...:

  For forward/backward compatability.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/reference/fit.keras.src.models.model.Model.md).

## Examples

    reduce_lr <- callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.2,
                                               patience = 5, min_lr = 0.001)
    model %>% fit(x_train, y_train, callbacks = list(reduce_lr))

## See also

- <https://keras.io/api/callbacks/reduce_lr_on_plateau#reducelronplateau-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/reference/callback_model_checkpoint.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/reference/callback_terminate_on_nan.md)  
