# Gradient descent (with momentum) optimizer.

Update rule for parameter `w` with gradient `g` when `momentum` is 0:

    w <- w - learning_rate * g

Update rule when `momentum` is larger than 0:

    velocity <- momentum * velocity - learning_rate * g
    w <- w + velocity

When `nesterov=TRUE`, this rule becomes:

    velocity <- momentum * velocity - learning_rate * g
    w <- w + momentum * velocity - learning_rate * g

## Usage

``` r
optimizer_sgd(
  learning_rate = 0.01,
  momentum = 0,
  nesterov = FALSE,
  weight_decay = NULL,
  clipnorm = NULL,
  clipvalue = NULL,
  global_clipnorm = NULL,
  use_ema = FALSE,
  ema_momentum = 0.99,
  ema_overwrite_frequency = NULL,
  name = "SGD",
  ...,
  loss_scale_factor = NULL,
  gradient_accumulation_steps = NULL
)
```

## Arguments

- learning_rate:

  A float, a `learning_rate_schedule_*` instance, or a callable that
  takes no arguments and returns the actual value to use. The learning
  rate. Defaults to `0.01`.

- momentum:

  float hyperparameter \>= 0 that accelerates gradient descent in the
  relevant direction and dampens oscillations. 0 is vanilla gradient
  descent. Defaults to `0.0`.

- nesterov:

  boolean. Whether to apply Nesterov momentum. Defaults to `FALSE`.

- weight_decay:

  Float. If set, weight decay is applied.

- clipnorm:

  Float. If set, the gradient of each weight is individually clipped so
  that its norm is no higher than this value.

- clipvalue:

  Float. If set, the gradient of each weight is clipped to be no higher
  than this value.

- global_clipnorm:

  Float. If set, the gradient of all weights is clipped so that their
  global norm is no higher than this value.

- use_ema:

  Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
  (EMA) is applied. EMA consists of computing an exponential moving
  average of the weights of the model (as the weight values change after
  each training batch), and periodically overwriting the weights with
  their moving average.

- ema_momentum:

  Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is the
  momentum to use when computing the EMA of the model's weights:
  `new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value`.

- ema_overwrite_frequency:

  Int or NULL, defaults to NULL. Only used if `use_ema=TRUE`. Every
  `ema_overwrite_frequency` steps of iterations, we overwrite the model
  variable by its moving average. If NULL, the optimizer does not
  overwrite model variables in the middle of training, and you need to
  explicitly overwrite the variables at the end of training by calling
  `optimizer$finalize_variable_values()` (which updates the model
  variables in-place). When using the built-in
  [`fit()`](https://generics.r-lib.org/reference/fit.html) training
  loop, this happens automatically after the last epoch, and you don't
  need to do anything.

- name:

  String. The name to use for momentum accumulator weights created by
  the optimizer.

- ...:

  For forward/backward compatability.

- loss_scale_factor:

  Float or `NULL`. If a float, the scale factor will be multiplied the
  loss before computing gradients, and the inverse of the scale factor
  will be multiplied by the gradients before updating variables. Useful
  for preventing underflow during mixed precision training. Alternately,
  [`optimizer_loss_scale()`](https://keras3.posit.co/dev/reference/optimizer_loss_scale.md)
  will automatically set a loss scale factor.

- gradient_accumulation_steps:

  Int or `NULL`. If an int, model and optimizer variables will not be
  updated at every step; instead they will be updated every
  `gradient_accumulation_steps` steps, using the average value of the
  gradients since the last update. This is known as "gradient
  accumulation". This can be useful when your batch size is very small,
  in order to reduce gradient noise at each update step. EMA frequency
  will look at "accumulated" iterations value (optimizer steps //
  gradient_accumulation_steps). Learning rate schedules will look at
  "real" iterations value (optimizer steps).

## Value

an `Optimizer` instance

## See also

- <https://keras.io/api/optimizers/sgd#sgd-class>

Other optimizers:  
[`optimizer_adadelta()`](https://keras3.posit.co/dev/reference/optimizer_adadelta.md)  
[`optimizer_adafactor()`](https://keras3.posit.co/dev/reference/optimizer_adafactor.md)  
[`optimizer_adagrad()`](https://keras3.posit.co/dev/reference/optimizer_adagrad.md)  
[`optimizer_adam()`](https://keras3.posit.co/dev/reference/optimizer_adam.md)  
[`optimizer_adam_w()`](https://keras3.posit.co/dev/reference/optimizer_adam_w.md)  
[`optimizer_adamax()`](https://keras3.posit.co/dev/reference/optimizer_adamax.md)  
[`optimizer_ftrl()`](https://keras3.posit.co/dev/reference/optimizer_ftrl.md)  
[`optimizer_lamb()`](https://keras3.posit.co/dev/reference/optimizer_lamb.md)  
[`optimizer_lion()`](https://keras3.posit.co/dev/reference/optimizer_lion.md)  
[`optimizer_loss_scale()`](https://keras3.posit.co/dev/reference/optimizer_loss_scale.md)  
[`optimizer_muon()`](https://keras3.posit.co/dev/reference/optimizer_muon.md)  
[`optimizer_nadam()`](https://keras3.posit.co/dev/reference/optimizer_nadam.md)  
[`optimizer_rmsprop()`](https://keras3.posit.co/dev/reference/optimizer_rmsprop.md)  
