# Learning rate scheduler.

At the beginning of every epoch, this callback gets the updated learning
rate value from `schedule` function provided, with the current epoch and
current learning rate, and applies the updated learning rate on the
optimizer.

## Usage

``` r
callback_learning_rate_scheduler(schedule, verbose = 0L)
```

## Arguments

- schedule:

  A function that takes an epoch index (integer, indexed from 0) and
  current learning rate (float) as inputs and returns a new learning
  rate as output (float).

- verbose:

  Integer. 0: quiet, 1: log update messages.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/fit.keras.src.models.model.Model.md).

## Examples

    # This function keeps the initial learning rate steady for the first ten epochs
    # and decreases it exponentially after that.
    scheduler <- function(epoch, lr) {
      if (epoch < 10)
        return(lr)
      else
        return(lr * exp(-0.1))
    }

    model <- keras_model_sequential() |> layer_dense(units = 10)
    model |> compile(optimizer = optimizer_sgd(), loss = 'mse')
    model$optimizer$learning_rate |> as.array() |> round(5)

    ## [1] 0.01

    callback <- callback_learning_rate_scheduler(schedule = scheduler)
    history <- model |> fit(x = array(runif(100), c(5, 20)),
                            y = array(0, c(5, 1)),
                            epochs = 15, callbacks = list(callback), verbose = 0)
    model$optimizer$learning_rate |> as.array() |> round(5)

    ## [1] 0.00607

## See also

- <https://keras.io/api/callbacks/learning_rate_scheduler#learningratescheduler-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/dev/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/dev/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/dev/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/dev/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/dev/reference/callback_lambda.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/dev/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/dev/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/dev/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/dev/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/dev/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/dev/reference/callback_terminate_on_nan.md)  
