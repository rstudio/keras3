# Swaps model weights and EMA weights before and after evaluation.

This callbacks replaces the model's weight values with the values of the
optimizer's EMA weights (the exponential moving average of the past
model weights values, implementing "Polyak averaging") before model
evaluation, and restores the previous weights after evaluation.

The `SwapEMAWeights` callback is to be used in conjunction with an
optimizer that sets `use_ema = TRUE`.

Note that the weights are swapped in-place in order to save memory. The
behavior is undefined if you modify the EMA weights or model weights in
other callbacks.

## Usage

``` r
callback_swap_ema_weights(swap_on_epoch = FALSE)
```

## Arguments

- swap_on_epoch:

  Whether to perform swapping at `on_epoch_begin()` and
  `on_epoch_end()`. This is useful if you want to use EMA weights for
  other callbacks such as
  [`callback_model_checkpoint()`](https://keras3.posit.co/reference/callback_model_checkpoint.md).
  Defaults to `FALSE`.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/reference/fit.keras.src.models.model.Model.md).

## Examples

    # Remember to set `use_ema=TRUE` in the optimizer
    optimizer <- optimizer_sgd(use_ema = TRUE)
    model |> compile(optimizer = optimizer, loss = ..., metrics = ...)

    # Metrics will be computed with EMA weights
    model |> fit(X_train, Y_train,
                 callbacks = c(callback_swap_ema_weights()))

    # If you want to save model checkpoint with EMA weights, you can set
    # `swap_on_epoch=TRUE` and place ModelCheckpoint after SwapEMAWeights.
    model |> fit(
      X_train, Y_train,
      callbacks = c(
        callback_swap_ema_weights(swap_on_epoch = TRUE),
        callback_model_checkpoint(...)
      )
    )

## See also

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
[`callback_tensorboard()`](https://keras3.posit.co/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/reference/callback_terminate_on_nan.md)  
