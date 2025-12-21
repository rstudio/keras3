# Callback for creating simple, custom callbacks on-the-fly.

This callback is constructed with anonymous functions that will be
called at the appropriate time (during
`Model.{fit | evaluate | predict}`). Note that the callbacks expects
positional arguments, as:

- `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
  `epoch`, `logs`

- `on_train_begin` and `on_train_end` expect one positional argument:
  `logs`

- `on_train_batch_begin` and `on_train_batch_end` expect a positional
  argument `batch` and a named argument `logs`

- See `Callback` class definition for the full list of functions and
  their expected arguments.

## Usage

``` r
callback_lambda(
  on_epoch_begin = NULL,
  on_epoch_end = NULL,
  on_train_begin = NULL,
  on_train_end = NULL,
  on_train_batch_begin = NULL,
  on_train_batch_end = NULL,
  ...
)
```

## Arguments

- on_epoch_begin:

  called at the beginning of every epoch.

- on_epoch_end:

  called at the end of every epoch.

- on_train_begin:

  called at the beginning of model training.

- on_train_end:

  called at the end of model training.

- on_train_batch_begin:

  called at the beginning of every train batch.

- on_train_batch_end:

  called at the end of every train batch.

- ...:

  Any function in
  [`Callback()`](https://keras3.posit.co/reference/Callback.md) that you
  want to override by passing `function_name = function`. For example,
  `callback_lambda(.., on_train_end = train_end_fn)`. The custom
  function needs to have the same arguments as the ones defined in
  [`Callback()`](https://keras3.posit.co/reference/Callback.md).

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/reference/fit.keras.src.models.model.Model.md).

## Examples

    # Print the batch number at the beginning of every batch.
    batch_print_callback <- callback_lambda(
      on_train_batch_begin = function(batch, logs) {
        print(batch)
      }
    )

    # Stream the epoch loss to a file in new-line delimited JSON format
    # (one valid JSON object per line)
    json_log <- file('loss_log.json', open = 'wt')
    json_logging_callback <- callback_lambda(
      on_epoch_end = function(epoch, logs) {
        jsonlite::write_json(
          list(epoch = epoch, loss = logs$loss),
          json_log,
          append = TRUE
        )
      },
      on_train_end = function(logs) {
        close(json_log)
      }
    )

    # Terminate some processes after having finished model training.
    processes <- ...
    cleanup_callback <- callback_lambda(
      on_train_end = function(logs) {
        for (p in processes) {
          if (is_alive(p)) {
            terminate(p)
          }
        }
      }
    )

    model %>% fit(
      ...,
      callbacks = list(
        batch_print_callback,
        json_logging_callback,
        cleanup_callback
      )
    )

## See also

- <https://keras.io/api/callbacks/lambda_callback#lambdacallback-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/reference/callback_early_stopping.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/reference/callback_terminate_on_nan.md)  
