# Callback to back up and restore the training state.

`callback_backup_and_restore()` callback is intended to recover training
from an interruption that has happened in the middle of a `fit`
execution, by backing up the training states in a temporary checkpoint
file, at the end of each epoch. Each backup overwrites the previously
written checkpoint file, so at any given time there is at most one such
checkpoint file for backup/restoring purpose.

If training restarts before completion, the training state (which
includes the model weights and epoch number) is restored to the most
recently saved state at the beginning of a new `fit` run. At the
completion of a `fit` run, the temporary checkpoint file is deleted.

Note that the user is responsible to bring jobs back after the
interruption. This callback is important for the backup and restore
mechanism for fault tolerance purpose, and the model to be restored from
a previous checkpoint is expected to be the same as the one used to back
up. If user changes arguments passed to `compile` or `fit`, the
checkpoint saved for fault tolerance can become invalid.

## Usage

``` r
callback_backup_and_restore(
  backup_dir,
  save_freq = "epoch",
  double_checkpoint = FALSE,
  delete_checkpoint = TRUE
)
```

## Arguments

- backup_dir:

  String, path of directory where to store the data needed to restore
  the model. The directory cannot be reused elsewhere to store other
  files, e.g. by the `backup_and_restore` callback of another training
  run, or by another callback (e.g. `callback_model_checkpoint`) of the
  same training run.

- save_freq:

  `"epoch"`, integer, or `FALSE`. When set to `"epoch"`, the callback
  saves the checkpoint at the end of each epoch. When set to an integer,
  the callback saves the checkpoint every `save_freq` batches. Set
  `save_freq = FALSE` only if using preemption checkpointing (i.e. with
  `save_before_preemption = TRUE`).

- double_checkpoint:

  Boolean. If enabled, `BackupAndRestore` callback will save 2 last
  training states (current and previous). After interruption if current
  state can't be loaded due to IO error (e.g. file corrupted) it will
  try to restore previous one. Such behaviour will consume twice more
  space on disk, but increase fault tolerance. Defaults to `FALSE`.

- delete_checkpoint:

  Boolean. This `backup_and_restore` callback works by saving a
  checkpoint to back up the training state. If
  `delete_checkpoint = TRUE`, the checkpoint will be deleted after
  training is finished. Use `FALSE` if you'd like to keep the checkpoint
  for future usage. Defaults to `TRUE`.

## Value

A `Callback` instance that can be passed to
[`fit.keras.src.models.model.Model()`](https://keras3.posit.co/reference/fit.keras.src.models.model.Model.md).

## Examples

    callback_interrupting <- new_callback_class(
      "InterruptingCallback",
      on_epoch_begin = function(epoch, logs = NULL) {
        if (epoch == 4) {
          stop('Interrupting!')
        }
      }
    )

    backup_dir <- tempfile()
    callback <- callback_backup_and_restore(backup_dir = backup_dir)
    model <- keras_model_sequential() %>%
      layer_dense(10)
    model %>% compile(optimizer = optimizer_sgd(), loss = 'mse')

    # ensure model is built (i.e., weights are initialized) for
    # callback_backup_and_restore()
    model(op_ones(c(5, 20))) |> invisible()

    tryCatch({
      model %>% fit(x = op_ones(c(5, 20)),
                    y = op_zeros(5),
                    epochs = 10, batch_size = 1,
                    callbacks = list(callback, callback_interrupting()),
                    verbose = 0)
    }, python.builtin.RuntimeError = function(e) message("Interrupted!"))

    ## Interrupted!

    model$history$epoch

    ## [1] 0 1 2

    # model$history %>% keras3:::to_keras_training_history() %>% as.data.frame() %>% print()

    history <- model %>% fit(x = op_ones(c(5, 20)),
                             y = op_zeros(5),
                             epochs = 10, batch_size = 1,
                             callbacks = list(callback),
                             verbose = 0)

    # Only 6 more epochs are run, since first training got interrupted at
    # zero-indexed epoch 4, second training will continue from 4 to 9.
    nrow(as.data.frame(history))

    ## [1] 10

## See also

- <https://keras.io/api/callbacks/backup_and_restore#backupandrestore-class>

Other callbacks:  
[`Callback()`](https://keras3.posit.co/reference/Callback.md)  
[`callback_csv_logger()`](https://keras3.posit.co/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/reference/callback_terminate_on_nan.md)  
