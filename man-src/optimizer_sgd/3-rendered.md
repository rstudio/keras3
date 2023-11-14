Gradient descent (with momentum) optimizer.

@description
Update rule for parameter `w` with gradient `g` when `momentum` is 0:

```python
w = w - learning_rate * g
```

Update rule when `momentum` is larger than 0:

```python
velocity = momentum * velocity - learning_rate * g
w = w + velocity
```

When `nesterov=True`, this rule becomes:

```python
velocity = momentum * velocity - learning_rate * g
w = w + momentum * velocity - learning_rate * g
```

@param learning_rate
A float, a
`keras.optimizers.schedules.LearningRateSchedule` instance, or
a callable that takes no arguments and returns the actual value to
use. The learning rate. Defaults to `0.01`.

@param momentum
float hyperparameter >= 0 that accelerates gradient descent in
the relevant direction and dampens oscillations. 0 is vanilla
gradient descent. Defaults to `0.0`.

@param nesterov
boolean. Whether to apply Nesterov momentum.
Defaults to `False`.

@param name
String. The name to use
for momentum accumulator weights created by
the optimizer.

@param weight_decay
Float. If set, weight decay is applied.

@param clipnorm
Float. If set, the gradient of each weight is individually
clipped so that its norm is no higher than this value.

@param clipvalue
Float. If set, the gradient of each weight is clipped to be
no higher than this value.

@param global_clipnorm
Float. If set, the gradient of all weights is clipped
so that their global norm is no higher than this value.

@param use_ema
Boolean, defaults to False. If True, exponential moving average
(EMA) is applied. EMA consists of computing an exponential moving
average of the weights of the model (as the weight values change after
each training batch), and periodically overwriting the weights with
their moving average.

@param ema_momentum
Float, defaults to 0.99. Only used if `use_ema=True`.
This is the momentum to use when computing
the EMA of the model's weights:
`new_average = ema_momentum * old_average + (1 - ema_momentum) *
current_variable_value`.

@param ema_overwrite_frequency
Int or None, defaults to None. Only used if
`use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
we overwrite the model variable by its moving average.
If None, the optimizer
does not overwrite model variables in the middle of training, and you
need to explicitly overwrite the variables at the end of training
by calling `optimizer.finalize_variable_values()`
(which updates the model
variables in-place). When using the built-in `fit()` training loop,
this happens automatically after the last epoch,
and you don't need to do anything.

@param loss_scale_factor
Float or `None`. If a float, the scale factor will
be multiplied the loss before computing gradients, and the inverse of
the scale factor will be multiplied by the gradients before updating
variables. Useful for preventing underflow during mixed precision
training. Alternately, `keras.optimizers.LossScaleOptimizer` will
automatically set a loss scale factor.

@param ...
Passed on to the Python callable

@export
@family optimizers
@seealso
+ <https:/keras.io/api/optimizers/sgd#sgd-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>
