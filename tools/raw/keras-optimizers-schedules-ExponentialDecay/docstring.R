A `LearningRateSchedule` that uses an exponential decay schedule.

When training a model, it is often useful to lower the learning rate as
the training progresses. This schedule applies an exponential decay function
to an optimizer step, given a provided initial learning rate.

The schedule is a 1-arg callable that produces a decayed learning
rate when passed the current optimizer step. This can be useful for changing
the learning rate value across different invocations of optimizer functions.
It is computed as:

```python
def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ^ (step / decay_steps)
```

If the argument `staircase` is `True`, then `step / decay_steps` is
an integer division and the decayed learning rate follows a
staircase function.

You can pass this schedule directly into a `keras.optimizers.Optimizer`
as the learning rate.
Example: When fitting a Keras model, decay every 100000 steps with a base
of 0.96:

```python
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=5)
```

The learning rate schedule is also serializable and deserializable using
`keras.optimizers.schedules.serialize` and
`keras.optimizers.schedules.deserialize`.

Args:
    initial_learning_rate: A Python float. The initial learning rate.
    decay_steps: A Python integer. Must be positive. See the decay
        computation above.
    decay_rate: A Python float. The decay rate.
    staircase: Boolean.  If `True` decay the learning rate at discrete
        intervals.
    name: String.  Optional name of the operation.  Defaults to
        `"ExponentialDecay`".

Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar tensor of the
    same type as `initial_learning_rate`.
