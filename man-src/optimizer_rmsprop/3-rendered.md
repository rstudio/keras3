Optimizer that implements the RMSprop algorithm.

@description
The gist of RMSprop is to:

- Maintain a moving (discounted) average of the square of gradients
- Divide the gradient by the root of this average

This implementation of RMSprop uses plain momentum, not Nesterov momentum.

The centered version additionally maintains a moving average of the
gradients, and uses that average to estimate the variance.

# Usage

```r
opt <- optimizer_rmsprop(learning_rate=0.1)
```

# Reference
- [Hinton, 2012](
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

@param learning_rate
A float, a
`learning_rate_schedule_*` instance, or
a callable that takes no arguments and returns the actual value to
use. The learning rate. Defaults to `0.001`.

@param rho
float, defaults to 0.9. Discounting factor for the old gradients.

@param momentum
float, defaults to 0.0. If not 0.0., the optimizer tracks the
momentum value, with a decay rate equals to `1 - momentum`.

@param epsilon
A small constant for numerical stability. This epsilon is
"epsilon hat" in the Kingma and Ba paper (in the formula just before
Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
to 1e-7.

@param centered
Boolean. If `TRUE`, gradients are normalized by the estimated
variance of the gradient; if FALSE, by the uncentered second moment.
Setting this to `TRUE` may help with training, but is slightly more
expensive in terms of computation and memory. Defaults to `FALSE`.

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
Boolean, defaults to FALSE. If TRUE, exponential moving average
(EMA) is applied. EMA consists of computing an exponential moving
average of the weights of the model (as the weight values change after
each training batch), and periodically overwriting the weights with
their moving average.

@param ema_momentum
Float, defaults to 0.99. Only used if `use_ema=TRUE`.
This is the momentum to use when computing
the EMA of the model's weights:
`new_average = ema_momentum * old_average + (1 - ema_momentum) *
current_variable_value`.

@param ema_overwrite_frequency
Int or NULL, defaults to NULL. Only used if
`use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
we overwrite the model variable by its moving average.
If NULL, the optimizer
does not overwrite model variables in the middle of training, and you
need to explicitly overwrite the variables at the end of training
by calling `optimizer.finalize_variable_values()`
(which updates the model
variables in-place). When using the built-in `fit()` training loop,
this happens automatically after the last epoch,
and you don't need to do anything.

@param loss_scale_factor
Float or `NULL`. If a float, the scale factor will
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
+ <https:/keras.io/api/optimizers/rmsprop#rmsprop-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop>

