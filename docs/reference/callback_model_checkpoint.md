# Callback to save the Keras model or model weights at some frequency.

`callback_model_checkpoint()` is used in conjunction with training using
`model |> fit()` to save a model or weights (in a checkpoint file) at
some interval, so the model or weights can be loaded later to continue
the training from the state saved.

A few options this callback provides include:

- Whether to only keep the model that has achieved the "best
  performance" so far, or whether to save the model at the end of every
  epoch regardless of performance.

- Definition of "best"; which quantity to monitor and whether it should
  be maximized or minimized.

- The frequency it should save at. Currently, the callback supports
  saving at the end of every epoch, or after a fixed number of training
  batches.

- Whether only weights are saved, or the whole model is saved.

## Usage

``` r
callback_model_checkpoint(
  filepath,
  monitor = "val_loss",
  verbose = 0L,
  save_best_only = FALSE,
  save_weights_only = FALSE,
  mode = "auto",
  save_freq = "epoch",
  initial_value_threshold = NULL
)
```

## Arguments

- filepath:

  string, path to save the model file. `filepath` can contain named
  formatting options, which will be filled the value of `epoch` and keys
  in `logs` (passed in `on_epoch_end`). The `filepath` name needs to end
  with `".weights.h5"` when `save_weights_only = TRUE` or should end
  with `".keras"` or `".h5"` when checkpoint saving the whole model
  (default). For example: if `filepath` is
  `"{epoch:02d}-{val_loss:.2f}.keras"` or
  `"{epoch:02d}-{val_loss:.2f}.weights.h5"`, then the model checkpoints
  will be saved with the epoch number and the validation loss in the
  filename. The directory of the filepath should not be reused by any
  other callbacks to avoid conflicts.

- monitor:

  The metric name to monitor. Typically the metrics are set by the
  `model |> compile()` method. Note:

  - Prefix the name with `"val_"` to monitor validation metrics.

  - Use `"loss"` or `"val_loss"` to monitor the model's total loss.

  - If you specify metrics as strings, like `"accuracy"`, pass the same
    string (with or without the `"val_"` prefix).

  - If you pass `Metric` objects (created by one of `metric_*()`),
    `monitor` should be set to `metric$name`.

  - If you're not sure about the metric names you can check the contents
    of the `history$metrics` list returned by
    `history <- model |> fit()`

  - Multi-output models set additional prefixes on the metric names.

- verbose:

  Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages
  when the callback takes an action.

- save_best_only:

  if `save_best_only = TRUE`, it only saves when the model is considered
  the "best" and the latest best model according to the quantity
  monitored will not be overwritten. If `filepath` doesn't contain
  formatting options like `{epoch}` then `filepath` will be overwritten
  by each new better model.

- save_weights_only:

  if TRUE, then only the model's weights will be saved
  (`model |> save_model_weights(filepath)`), else the full model is
  saved (`model |> save_model(filepath)`).

- mode:

  one of {`"auto"`, `"min"`, `"max"`}. If `save_best_only = TRUE`, the
  decision to overwrite the current save file is made based on either
  the maximization or the minimization of the monitored quantity. For
  `val_acc`, this should be `"max"`, for `val_loss` this should be
  `"min"`, etc. In `"auto"` mode, the direction is automatically
  inferred from the name of the monitored quantity.

- save_freq:

  `"epoch"` or integer. When using `"epoch"`, the callback saves the
  model after each epoch. When using integer, the callback saves the
  model at end of this many batches. If the `Model` is compiled with
  `steps_per_execution = N`, then the saving criteria will be checked
  every Nth batch. Note that if the saving isn't aligned to epochs, the
  monitored metric may potentially be less reliable (it could reflect as
  little as 1 batch, since the metrics get reset every epoch). Defaults
  to `"epoch"`.

- initial_value_threshold:

  Floating point initial "best" value of the metric to be monitored.
  Only applies if `save_best_value = TRUE`. Only overwrites the model
  weights already saved if the performance of current model is better
  than this value.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/reference/fit.keras.src.models.model.Model.md).

## Examples

    model <- keras_model_sequential(input_shape = c(10)) |>
      layer_dense(1, activation = "sigmoid") |>
      compile(loss = "binary_crossentropy", optimizer = "adam",
              metrics = c('accuracy'))

    EPOCHS <- 10
    checkpoint_filepath <- tempfile('checkpoint-model-', fileext = ".keras")
    model_checkpoint_callback <- callback_model_checkpoint(
      filepath = checkpoint_filepath,
      monitor = 'val_accuracy',
      mode = 'max',
      save_best_only = TRUE
    )

    # Model is saved at the end of every epoch, if it's the best seen so far.
    model |> fit(x = random_uniform(c(2, 10)), y = op_ones(2, 1),
                 epochs = EPOCHS, validation_split = .5, verbose = 0,
                 callbacks = list(model_checkpoint_callback))

    # The model (that are considered the best) can be loaded as -
    load_model(checkpoint_filepath)

    ## Model: "sequential"
    ## +---------------------------------+------------------------+---------------+
    ## | Layer (type)                    | Output Shape           |       Param # |
    ## +=================================+========================+===============+
    ## | dense (Dense)                   | (None, 1)              |            11 |
    ## +---------------------------------+------------------------+---------------+
    ##  Total params: 35 (144.00 B)
    ##  Trainable params: 11 (44.00 B)
    ##  Non-trainable params: 0 (0.00 B)
    ##  Optimizer params: 24 (100.00 B)

    # Alternatively, one could checkpoint just the model weights as -
    checkpoint_filepath <- tempfile('checkpoint-', fileext = ".weights.h5")
    model_checkpoint_callback <- callback_model_checkpoint(
      filepath = checkpoint_filepath,
      save_weights_only = TRUE,
      monitor = 'val_accuracy',
      mode = 'max',
      save_best_only = TRUE
    )

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    # same as above
    model |> fit(x = random_uniform(c(2, 10)), y = op_ones(2, 1),
                 epochs = EPOCHS, validation_split = .5, verbose = 0,
                 callbacks = list(model_checkpoint_callback))

    # The model weights (that are considered the best) can be loaded
    model |> load_model_weights(checkpoint_filepath)

## See also

- <https://keras.io/api/callbacks/model_checkpoint#modelcheckpoint-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/reference/Callback.md)  
[`callback_backup_and_restore()`](https://keras3.posit.co/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/reference/callback_learning_rate_scheduler.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/reference/callback_terminate_on_nan.md)  
