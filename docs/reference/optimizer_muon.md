# Optimizer that implements the Muon algorithm.

Note that this optimizer should not be used in the following layers:

1.  Embedding layer

2.  Final output fully connected layer

3.  Any 0- or 1-D variables

These should all be optimized using AdamW.

The Muon optimizer can use both the Muon update step or the AdamW update
step based on the following:

- For any variable that isn't 2D, 3D or 4D, the AdamW step will be used.
  This is not configurable.

- If the argument `exclude_embeddings` (defaults to `TRUE`) is set to
  `TRUE`, the AdamW step will be used.

- For any variable with a name that matches an expression listed in the
  argument `exclude_layers` (a list), the AdamW step will be used.

- Any other variable uses the Muon step.

Typically, you only need to pass the name of your densely-connected
output layer to `exclude_layers`, e.g.
`exclude_layers = "output_dense"`.

## Usage

``` r
optimizer_muon(
  learning_rate = 0.001,
  adam_beta_1 = 0.9,
  adam_beta_2 = 0.999,
  epsilon = 1e-07,
  weight_decay = 0.1,
  clipnorm = NULL,
  clipvalue = NULL,
  global_clipnorm = NULL,
  use_ema = FALSE,
  ema_momentum = 0.99,
  ema_overwrite_frequency = NULL,
  loss_scale_factor = NULL,
  gradient_accumulation_steps = NULL,
  name = "muon",
  exclude_layers = NULL,
  exclude_embeddings = TRUE,
  muon_a = 3.4445,
  muon_b = -4.775,
  muon_c = 2.0315,
  adam_lr_ratio = 0.1,
  momentum = 0.95,
  ns_steps = 6L,
  nesterov = TRUE,
  ...
)
```

## Arguments

- learning_rate:

  A float,
  [`LearningRateSchedule()`](https://keras3.posit.co/reference/LearningRateSchedule.md)
  instance, or a callable that takes no arguments and returns the actual
  value to use. The learning rate. Defaults to `0.001`.

- adam_beta_1:

  A float value or a constant float tensor, or a callable that takes no
  arguments and returns the actual value to use. The exponential decay
  rate for the 1st moment estimates. Defaults to `0.9`.

- adam_beta_2:

  A float value or a constant float tensor, ora callable that takes no
  arguments and returns the actual value to use. The exponential decay
  rate for the 2nd moment estimates. Defaults to `0.999`.

- epsilon:

  A small constant for numerical stability. This is "epsilon hat" in the
  Kingma and Ba paper (in the formula just before Section 2.1), not the
  epsilon in Algorithm 1 of the paper. It is used as in AdamW. Defaults
  to `1e-7`.

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

  Float, defaults to `0.99`. Only used if `use_ema = TRUE`. This is the
  momentum to use when computing the EMA of the model's weights:
  `new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value`.

- ema_overwrite_frequency:

  Int or `NULL`, defaults to `NULL`. Only used if `use_ema = TRUE`.
  Every `ema_overwrite_frequency` steps of iterations, we overwrite the
  model variable by its moving average. If `NULL`, the optimizer does
  not overwrite model variables in the middle of training, and you need
  to explicitly overwrite the variables at the end of training by
  calling `optimizer$finalize_variable_values()` (which updates the
  model variables in-place). When using the built-in
  [`fit()`](https://generics.r-lib.org/reference/fit.html) training
  loop, this happens automatically after the last epoch, and you don't
  need to do anything.

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

- name:

  String, name for the object

- exclude_layers:

  List of strings, keywords of layer names to exclude. All layers with
  keywords in their path will use AdamW.

- exclude_embeddings:

  Boolean value. If `TRUE`, embedding layers will use AdamW.

- muon_a:

  Float, parameter a of the muon algorithm. It is recommended to use the
  default value.

- muon_b:

  Float, parameter b of the muon algorithm. It is recommended to use the
  default value.

- muon_c:

  Float, parameter c of the muon algorithm. It is recommended to use the
  default value.

- adam_lr_ratio:

  Float, the ratio of the learning rate when using Adam to the main
  learning rate. it is recommended to set it to `0.1`.

- momentum:

  Float, momentum used by internal SGD.

- ns_steps:

  Integer, number of Newton-Schulz iterations to run.

- nesterov:

  Boolean, whether to use Nesterov-style momentum.

- ...:

  For forward/backward compatibility.

## Value

an `Optimizer` instance

## References

- [Original implementation](https://github.com/KellerJordan/Muon)

- [Liu et al, 2025](https://arxiv.org/abs/2502.16982)

## See also

Other optimizers:  
[`optimizer_adadelta()`](https://keras3.posit.co/reference/optimizer_adadelta.md)  
[`optimizer_adafactor()`](https://keras3.posit.co/reference/optimizer_adafactor.md)  
[`optimizer_adagrad()`](https://keras3.posit.co/reference/optimizer_adagrad.md)  
[`optimizer_adam()`](https://keras3.posit.co/reference/optimizer_adam.md)  
[`optimizer_adam_w()`](https://keras3.posit.co/reference/optimizer_adam_w.md)  
[`optimizer_adamax()`](https://keras3.posit.co/reference/optimizer_adamax.md)  
[`optimizer_ftrl()`](https://keras3.posit.co/reference/optimizer_ftrl.md)  
[`optimizer_lamb()`](https://keras3.posit.co/reference/optimizer_lamb.md)  
[`optimizer_lion()`](https://keras3.posit.co/reference/optimizer_lion.md)  
[`optimizer_loss_scale()`](https://keras3.posit.co/reference/optimizer_loss_scale.md)  
[`optimizer_nadam()`](https://keras3.posit.co/reference/optimizer_nadam.md)  
[`optimizer_rmsprop()`](https://keras3.posit.co/reference/optimizer_rmsprop.md)  
[`optimizer_sgd()`](https://keras3.posit.co/reference/optimizer_sgd.md)  
