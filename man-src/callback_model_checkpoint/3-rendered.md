Callback to save the Keras model or model weights at some frequency.

@description
`ModelCheckpoint` callback is used in conjunction with training using
`model %>% fit()` to save a model or weights (in a checkpoint file) at some
interval, so the model or weights can be loaded later to continue the
training from the state saved.

A few options this callback provides include:

- Whether to only keep the model that has achieved the "best performance" so
  far, or whether to save the model at the end of every epoch regardless of
  performance.
- Definition of "best"; which quantity to monitor and whether it should be
  maximized or minimized.
- The frequency it should save at. Currently, the callback supports saving
  at the end of every epoch, or after a fixed number of training batches.
- Whether only weights are saved, or the whole model is saved.

# Examples


