# Save and restore model state with Orbax

Saves model weights and optimizer state asynchronously with Orbax. On
the JAX backend, checkpointing is coordinated across hosts. The Python
`orbax-checkpoint` package is required.

## Usage

``` r
callback_orbax_checkpoint(
  directory,
  monitor = "val_loss",
  verbose = 0L,
  save_best_only = FALSE,
  mode = "auto",
  save_freq = "epoch",
  initial_value_threshold = NULL,
  max_to_keep = 1L,
  save_on_background = TRUE,
  save_weights_only = FALSE
)
```

## Arguments

- directory:

  Directory in which to save checkpoints.

- monitor:

  Metric name to monitor.

- verbose:

  Verbosity mode, 0 or 1.

- save_best_only:

  Whether to save only when `monitor` improves.

- mode:

  One of `"auto"`, `"min"`, or `"max"`.

- save_freq:

  `"epoch"` or an integer number of batches between saves.

- initial_value_threshold:

  Initial best value for `monitor`.

- max_to_keep:

  Maximum number of recent checkpoints to retain. Use `NULL` to retain
  all checkpoints.

- save_on_background:

  Whether to save asynchronously in the background.

- save_weights_only:

  Whether to save only trainable and non-trainable variables, excluding
  model configuration, optimizer state, and assets.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/fit.keras.src.models.model.Model.md).

## Examples

    checkpoint <- callback_orbax_checkpoint(
      directory = tempfile("orbax-checkpoints-"),
      monitor = "val_accuracy",
      mode = "max",
      save_best_only = TRUE
    )
    model |> fit(x, y, validation_split = 0.2, callbacks = list(checkpoint))

## See also

Other callbacks:  
[`Callback()`](https://keras3.posit.co/dev/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/dev/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/dev/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/dev/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/dev/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/dev/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/dev/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/dev/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/dev/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/dev/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/dev/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/dev/reference/callback_terminate_on_nan.md)  
