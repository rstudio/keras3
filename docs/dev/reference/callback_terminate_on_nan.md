# Callback that terminates training when a NaN loss is encountered.

This callback monitors the loss during training and terminates training
when a NaN or Inf loss is detected. By default, training stops
gracefully by setting the model's `stop_training` flag, which allows
callback cleanup methods such as `on_train_end()` to run.

Set `raise_error = TRUE` to raise an error immediately when a NaN or Inf
is detected. In this mode, `on_train_end()` is not called on other
callbacks. This can preserve backup states or prevent unintended cleanup
after a training failure.

## Usage

``` r
callback_terminate_on_nan(raise_error = FALSE)
```

## Arguments

- raise_error:

  If `FALSE`, stop training gracefully. If `TRUE`, raise an error
  immediately when a NaN or Inf loss is detected, bypassing callback
  cleanup methods.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/fit.keras.src.models.model.Model.md).

## Examples

    # Graceful termination (default)
    callback <- callback_terminate_on_nan()
    model |> fit(x, y, callbacks = list(callback))

    # Immediate error
    callback <- callback_terminate_on_nan(raise_error = TRUE)
    model |> fit(x, y, callbacks = list(callback))

## See also

- <https://keras.io/api/callbacks/terminate_on_nan#terminateonnan-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/dev/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/dev/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/dev/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/dev/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/dev/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/dev/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/dev/reference/callback_model_checkpoint.md)  
[`callback_orbax_checkpoint()`](https://keras3.posit.co/dev/reference/callback_orbax_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/dev/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/dev/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/dev/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/dev/reference/callback_tensorboard.md)  
