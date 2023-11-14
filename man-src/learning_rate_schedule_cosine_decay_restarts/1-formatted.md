A `LearningRateSchedule` that uses a cosine decay schedule with restarts.

@description
See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
SGDR: Stochastic Gradient Descent with Warm Restarts.

When training a model, it is often useful to lower the learning rate as
the training progresses. This schedule applies a cosine decay function with
restarts to an optimizer step, given a provided initial learning rate.
It requires a `step` value to compute the decayed learning rate. You can
just pass a backend variable that you increment at each training step.

The schedule is a 1-arg callable that produces a decayed learning
rate when passed the current optimizer step. This can be useful for changing
the learning rate value across different invocations of optimizer functions.

The learning rate multiplier first decays
from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
restart is performed. Each new warm restart runs for `t_mul` times more
steps and with `m_mul` times initial learning rate as the new learning rate.

Example usage:
```python
first_decay_steps = 1000
lr_decayed_fn = (
    keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps))
```

You can pass this schedule directly into a `keras.optimizers.Optimizer`
as the learning rate. The learning rate schedule is also serializable and
deserializable using `keras.optimizers.schedules.serialize` and
`keras.optimizers.schedules.deserialize`.

@returns
A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the
same type as `initial_learning_rate`.

@param initial_learning_rate
A Python float. The initial learning rate.

@param first_decay_steps
A Python integer. Number of steps to decay over.

@param t_mul
A Python float. Used to derive the number of iterations in
the i-th period.

@param m_mul
A Python float. Used to derive the initial learning rate of
the i-th period.

@param alpha
A Python float. Minimum learning rate value as a fraction of
the `initial_learning_rate`.

@param name
String. Optional name of the operation. Defaults to
`"SGDRDecay"`.

@export
@family optimizer learing rate schedules
@seealso
+ <https:/keras.io/api/optimizers/learning_rate_schedules/cosine_decay_restarts#cosinedecayrestarts-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts>
