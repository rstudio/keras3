# Optimizer that implements the Adafactor algorithm.

Adafactor is commonly used in NLP tasks, and has the advantage of taking
less memory because it only saves partial information of previous
gradients.

The default argument setup is based on the original paper (see
reference). When gradients are of dimension \> 2, Adafactor optimizer
will delete the last 2 dimensions separately in its accumulator
variables.

## Usage

``` r
optimizer_adafactor(
  learning_rate = 0.001,
  beta_2_decay = -0.8,
  epsilon_1 = 1e-30,
  epsilon_2 = 0.001,
  clip_threshold = 1,
  relative_step = TRUE,
  weight_decay = NULL,
  clipnorm = NULL,
  clipvalue = NULL,
  global_clipnorm = NULL,
  use_ema = FALSE,
  ema_momentum = 0.99,
  ema_overwrite_frequency = NULL,
  name = "adafactor",
  ...,
  loss_scale_factor = NULL,
  gradient_accumulation_steps = NULL
)
```

## Arguments

- learning_rate:

  A float, a
  [`LearningRateSchedule()`](https://keras3.posit.co/reference/LearningRateSchedule.md)
  instance, or a callable that takes no arguments and returns the actual
  value to use. The learning rate. Defaults to `0.001`.

- beta_2_decay:

  float, defaults to -0.8. The decay rate of `beta_2`.

- epsilon_1:

  float, defaults to 1e-30. A small offset to keep denominator away from
  0.

- epsilon_2:

  float, defaults to 1e-3. A small offset to avoid learning rate
  becoming too small by time.

- clip_threshold:

  float, defaults to 1.0. Clipping threshold. This is a part of
  Adafactor algorithm, independent from `clipnorm`, `clipvalue`, and
  `global_clipnorm`.

- relative_step:

  bool, defaults to `TRUE`. If `learning_rate` is a constant and
  `relative_step=TRUE`, learning rate will be adjusted based on current
  iterations. This is a default learning rate decay in Adafactor.

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

  Float, defaults to 0.99. Only used if `use_ema = TRUE`. This is the
  momentum to use when computing the EMA of the model's weights:
  `new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value`.

- ema_overwrite_frequency:

  Int or `NULL`, defaults to `NULL`. Only used if `use_ema=TRUE`. Every
  `ema_overwrite_frequency` steps of iterations, we overwrite the model
  variable by its moving average. If `NULL`, the optimizer does not
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
  [`optimizer_loss_scale()`](https://keras3.posit.co/reference/optimizer_loss_scale.md)
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

## Reference

- [Shazeer, Noam et al., 2018](https://arxiv.org/abs/1804.04235).

## See also

- <https://keras.io/api/optimizers/adafactor#adafactor-class>

Other optimizers:  
[`optimizer_adadelta()`](https://keras3.posit.co/reference/optimizer_adadelta.md)  
[`optimizer_adagrad()`](https://keras3.posit.co/reference/optimizer_adagrad.md)  
[`optimizer_adam()`](https://keras3.posit.co/reference/optimizer_adam.md)  
[`optimizer_adam_w()`](https://keras3.posit.co/reference/optimizer_adam_w.md)  
[`optimizer_adamax()`](https://keras3.posit.co/reference/optimizer_adamax.md)  
[`optimizer_ftrl()`](https://keras3.posit.co/reference/optimizer_ftrl.md)  
[`optimizer_lamb()`](https://keras3.posit.co/reference/optimizer_lamb.md)  
[`optimizer_lion()`](https://keras3.posit.co/reference/optimizer_lion.md)  
[`optimizer_loss_scale()`](https://keras3.posit.co/reference/optimizer_loss_scale.md)  
[`optimizer_muon()`](https://keras3.posit.co/reference/optimizer_muon.md)  
[`optimizer_nadam()`](https://keras3.posit.co/reference/optimizer_nadam.md)  
[`optimizer_rmsprop()`](https://keras3.posit.co/reference/optimizer_rmsprop.md)  
[`optimizer_sgd()`](https://keras3.posit.co/reference/optimizer_sgd.md)  
