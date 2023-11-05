Callback to back up and restore the training state.

`BackupAndRestore` callback is intended to recover training from an
interruption that has happened in the middle of a `Model.fit` execution, by
backing up the training states in a temporary checkpoint file, at the end of
each epoch. Each backup overwrites the previously written checkpoint file,
so at any given time there is at most one such checkpoint file for
backup/restoring purpose.

If training restarts before completion, the training state (which includes
the `Model` weights and epoch number) is restored to the most recently saved
state at the beginning of a new `Model.fit` run. At the completion of a
`Model.fit` run, the temporary checkpoint file is deleted.

Note that the user is responsible to bring jobs back after the interruption.
This callback is important for the backup and restore mechanism for fault
tolerance purpose, and the model to be restored from a previous checkpoint
is expected to be the same as the one used to back up. If user changes
arguments passed to compile or fit, the checkpoint saved for fault tolerance
can become invalid.

Example:

>>> class InterruptingCallback(keras.callbacks.Callback):
...   def on_epoch_begin(self, epoch, logs=None):
...     if epoch == 4:
...       raise RuntimeError('Interrupting!')
>>> callback = keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup")
>>> model = keras.models.Sequential([keras.layers.Dense(10)])
>>> model.compile(keras.optimizers.SGD(), loss='mse')
>>> try:
...   model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
...             batch_size=1, callbacks=[callback, InterruptingCallback()],
...             verbose=0)
... except:
...   pass
>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
...                     epochs=10, batch_size=1, callbacks=[callback],
...                     verbose=0)
>>> # Only 6 more epochs are run, since first training got interrupted at
>>> # zero-indexed epoch 4, second training will continue from 4 to 9.
>>> len(history.history['loss'])
>>> 6

Args:
    backup_dir: String, path of directory where to store the data
        needed to restore the model. The directory
        cannot be reused elsewhere to store other files, e.g. by the
        `BackupAndRestore` callback of another training run,
        or by another callback (e.g. `ModelCheckpoint`)
        of the same training run.
    save_freq: `"epoch"`, integer, or `False`. When set to `"epoch"`
      the callback saves the checkpoint at the end of each epoch.
      When set to an integer, the callback saves the checkpoint every
      `save_freq` batches. Set `save_freq=False` only if using
      preemption checkpointing (i.e. with `save_before_preemption=True`).
    delete_checkpoint: Boolean, defaults to `True`. This `BackupAndRestore`
      callback works by saving a checkpoint to back up the training state.
      If `delete_checkpoint=True`, the checkpoint will be deleted after
      training is finished. Use `False` if you'd like to keep the checkpoint
      for future usage.
