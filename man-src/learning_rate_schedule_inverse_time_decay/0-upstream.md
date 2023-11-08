keras.optimizers.schedules.InverseTimeDecay
__signature__
(
  initial_learning_rate,
  decay_steps,
  decay_rate,
  staircase=False,
  name='InverseTimeDecay'
)
__doc__
A `LearningRateSchedule` that uses an inverse time decay schedule.

When training a model, it is often useful to lower the learning rate as
the training progresses. This schedule applies the inverse decay function
to an optimizer step, given a provided initial learning rate.
It requires a `step` value to compute the decayed learning rate. You can
just pass a backend variable that you increment at each training step.

The schedule is a 1-arg callable that produces a decayed learning
rate when passed the current optimizer step. This can be useful for changing
the learning rate value across different invocations of optimizer functions.
It is computed as:

```python
def decayed_learning_rate(step):
    return initial_learning_rate / (1 + decay_rate * step / decay_step)
```

or, if `staircase` is `True`, as:

```python
def decayed_learning_rate(step):
    return initial_learning_rate /
           (1 + decay_rate * floor(step / decay_step))
```

You can pass this schedule directly into a `keras.optimizers.Optimizer`
as the learning rate.
Example: Fit a Keras model when decaying 1/t with a rate of 0.5:

```python
...
initial_learning_rate = 0.1
decay_steps = 1.0
decay_rate = 0.5
learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate, decay_steps, decay_rate)

model.compile(optimizer=keras.optimizers.SGD(
                  learning_rate=learning_rate_fn),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=5)
```

Args:
    initial_learning_rate: A Python float. The initial learning rate.
    decay_steps: How often to apply decay.
    decay_rate: A Python number.  The decay rate.
    staircase: Whether to apply decay in a discrete staircase, as o
    pposed to continuous, fashion.
    name: String.  Optional name of the operation.  Defaults to
        `"InverseTimeDecay"`.

Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar tensor of the
    same type as `initial_learning_rate`.
