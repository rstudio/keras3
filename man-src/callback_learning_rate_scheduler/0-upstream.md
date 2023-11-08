keras.callbacks.LearningRateScheduler
__signature__
(schedule, verbose=0)
__doc__
Learning rate scheduler.

At the beginning of every epoch, this callback gets the updated learning
rate value from `schedule` function provided at `__init__`, with the current
epoch and current learning rate, and applies the updated learning rate on
the optimizer.

Args:
    schedule: A function that takes an epoch index (integer, indexed from 0)
        and current learning rate (float) as inputs and returns a new
        learning rate as output (float).
    verbose: Integer. 0: quiet, 1: log update messages.

Example:

>>> # This function keeps the initial learning rate for the first ten epochs
>>> # and decreases it exponentially after that.
>>> def scheduler(epoch, lr):
...     if epoch < 10:
...         return lr
...     else:
...         return lr * ops.exp(-0.1)
>>>
>>> model = keras.models.Sequential([keras.layers.Dense(10)])
>>> model.compile(keras.optimizers.SGD(), loss='mse')
>>> round(model.optimizer.learning_rate, 5)
0.01

>>> callback = keras.callbacks.LearningRateScheduler(scheduler)
>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
...                     epochs=15, callbacks=[callback], verbose=0)
>>> round(model.optimizer.learning_rate, 5)
0.00607
