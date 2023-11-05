Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This callback monitors a
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.

Example:

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(x_train, y_train, callbacks=[reduce_lr])
```

Args:
    monitor: String. Quantity to be monitored.
    factor: Float. Factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
    patience: Integer. Number of epochs with no improvement after which
        learning rate will be reduced.
    verbose: Integer. 0: quiet, 1: update messages.
    mode: String. One of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will
        be reduced when the quantity monitored has stopped increasing; in
        `'auto'` mode, the direction is automatically inferred from the name
        of the monitored quantity.
    min_delta: Float. Threshold for measuring the new optimum, to only focus
        on significant changes.
    cooldown: Integer. Number of epochs to wait before resuming normal
        operation after the learning rate has been reduced.
    min_lr: Float. Lower bound on the learning rate.
