Optimizer that implements the RMSprop algorithm.

The gist of RMSprop is to:

- Maintain a moving (discounted) average of the square of gradients
- Divide the gradient by the root of this average

This implementation of RMSprop uses plain momentum, not Nesterov momentum.

The centered version additionally maintains a moving average of the
gradients, and uses that average to estimate the variance.

Args:
    learning_rate: A float, a
        `keras.optimizers.schedules.LearningRateSchedule` instance, or
        a callable that takes no arguments and returns the actual value to
        use. The learning rate. Defaults to `0.001`.
    rho: float, defaults to 0.9. Discounting factor for the old gradients.
    momentum: float, defaults to 0.0. If not 0.0., the optimizer tracks the
        momentum value, with a decay rate equals to `1 - momentum`.
    epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
        to 1e-7.
    centered: Boolean. If `True`, gradients are normalized by the estimated
        variance of the gradient; if False, by the uncentered second moment.
        Setting this to `True` may help with training, but is slightly more
        expensive in terms of computation and memory. Defaults to `False`.
    name: String. The name to use
      for momentum accumulator weights created by
      the optimizer.
  weight_decay: Float, defaults to None. If set, weight decay is applied.
  clipnorm: Float. If set, the gradient of each weight is individually
      clipped so that its norm is no higher than this value.
  clipvalue: Float. If set, the gradient of each weight is clipped to be no
      higher than this value.
  global_clipnorm: Float. If set, the gradient of all weights is clipped so
      that their global norm is no higher than this value.
  use_ema: Boolean, defaults to False. If True, exponential moving average
      (EMA) is applied. EMA consists of computing an exponential moving
      average of the weights of the model (as the weight values change after
      each training batch), and periodically overwriting the weights with
      their moving average.
  ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
      This is the momentum to use when computing
      the EMA of the model's weights:
      `new_average = ema_momentum * old_average + (1 - ema_momentum) *
      current_variable_value`.
  ema_overwrite_frequency: Int or None, defaults to None. Only used if
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
  loss_scale_factor: Float or `None`. If a float, the scale factor will
      be multiplied the loss before computing gradients, and the inverse of
      the scale factor will be multiplied by the gradients before updating
      variables. Useful for preventing underflow during mixed precision
      training. Alternately, `keras.optimizers.LossScaleOptimizer` will
      automatically set a loss scale factor.


Usage:

>>> opt = keras.optimizers.RMSprop(learning_rate=0.1)
>>> var1 = keras.backend.Variable(10.0)
>>> loss = lambda: (var1 ** 2) / 2.0  # d(loss) / d(var1) = var1
>>> opt.minimize(loss, [var1])
>>> var1
9.683772

Reference:

- [Hinton, 2012](
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
