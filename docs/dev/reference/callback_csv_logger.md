# Callback that streams epoch results to a CSV file.

Supports all values that can be represented as a string, including 1D
iterables such as atomic vectors.

## Usage

``` r
callback_csv_logger(filename, separator = ",", append = FALSE)
```

## Arguments

- filename:

  Filename of the CSV file, e.g. `'run/log.csv'`.

- separator:

  String used to separate elements in the CSV file.

- append:

  Boolean. `TRUE`: append if file exists (useful for continuing
  training). `FALSE`: overwrite existing file.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/fit.keras.src.models.model.Model.md).

## Examples

    csv_logger <- callback_csv_logger('training.log')
    model %>% fit(X_train, Y_train, callbacks = list(csv_logger))

## See also

- <https://keras.io/api/callbacks/csv_logger#csvlogger-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/dev/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/dev/reference/callback_backup_and_restore.md)  
[`callback_early_stopping()`](https://keras3.posit.co/dev/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/dev/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/dev/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/dev/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/dev/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/dev/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/dev/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/dev/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/dev/reference/callback_terminate_on_nan.md)  
