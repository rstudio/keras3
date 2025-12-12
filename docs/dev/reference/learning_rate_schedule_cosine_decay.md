# A `LearningRateSchedule` that uses a cosine decay with optional warmup.

See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
SGDR: Stochastic Gradient Descent with Warm Restarts.

For the idea of a linear warmup of our learning rate, see [Goyal et
al.](https://arxiv.org/pdf/1706.02677).

When we begin training a model, we often want an initial increase in our
learning rate followed by a decay. If `warmup_target` is an int, this
schedule applies a linear increase per optimizer step to our learning
rate from `initial_learning_rate` to `warmup_target` for a duration of
`warmup_steps`. Afterwards, it applies a cosine decay function taking
our learning rate from `warmup_target` to `alpha` for a duration of
`decay_steps`. If `warmup_target` is NULL we skip warmup and our decay
will take our learning rate from `initial_learning_rate` to `alpha`. It
requires a `step` value to compute the learning rate. You can just pass
a backend variable that you increment at each training step.

The schedule is a 1-arg callable that produces a warmup followed by a
decayed learning rate when passed the current optimizer step. This can
be useful for changing the learning rate value across different
invocations of optimizer functions.

Our warmup is computed as:

    warmup_learning_rate <- function(step) {
      completed_fraction <- step / warmup_steps
      total_delta <- target_warmup - initial_learning_rate
      completed_fraction * total_delta
    }

And our decay is computed as:

    if (is.null(warmup_target)) {
      initial_decay_lr <- initial_learning_rate
    } else {
      initial_decay_lr <- warmup_target
    }

    decayed_learning_rate <- function(step) {
      step <- min(step, decay_steps)
      cosine_decay <- 0.5 * (1 + cos(pi * step / decay_steps))
      decayed <- (1 - alpha) * cosine_decay + alpha
      initial_decay_lr * decayed
    }

Example usage without warmup:

    decay_steps <- 1000
    initial_learning_rate <- 0.1
    lr_decayed_fn <- learning_rate_schedule_cosine_decay(
        initial_learning_rate, decay_steps)

Example usage with warmup:

    decay_steps <- 1000
    initial_learning_rate <- 0
    warmup_steps <- 1000
    target_learning_rate <- 0.1
    lr_warmup_decayed_fn <- learning_rate_schedule_cosine_decay(
        initial_learning_rate, decay_steps, warmup_target = target_learning_rate,
        warmup_steps = warmup_steps
    )

You can pass this schedule directly into a `optimizer` as the learning
rate. The learning rate schedule is also serializable and deserializable
using `keras$optimizers$schedules$serialize` and
`keras$optimizers$schedules$deserialize`.

## Usage

``` r
learning_rate_schedule_cosine_decay(
  initial_learning_rate,
  decay_steps,
  alpha = 0,
  name = "CosineDecay",
  warmup_target = NULL,
  warmup_steps = 0L
)
```

## Arguments

- initial_learning_rate:

  A float. The initial learning rate.

- decay_steps:

  A int. Number of steps to decay over.

- alpha:

  A float. Minimum learning rate value for decay as a fraction of
  `initial_learning_rate`.

- name:

  String. Optional name of the operation. Defaults to `"CosineDecay"`.

- warmup_target:

  A float. The target learning rate for our warmup phase. Will cast to
  the `initial_learning_rate` datatype. Setting to `NULL` will skip
  warmup and begins decay phase from `initial_learning_rate`. Otherwise
  scheduler will warmup from `initial_learning_rate` to `warmup_target`.

- warmup_steps:

  A int. Number of steps to warmup over.

## Value

A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the same
type as `initial_learning_rate`.

## See also

- <https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay#cosinedecay-class>

Other optimizer learning rate schedules:  
[`LearningRateSchedule()`](https://keras3.posit.co/dev/reference/LearningRateSchedule.md)  
[`learning_rate_schedule_cosine_decay_restarts()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_cosine_decay_restarts.md)  
[`learning_rate_schedule_exponential_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_exponential_decay.md)  
[`learning_rate_schedule_inverse_time_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_inverse_time_decay.md)  
[`learning_rate_schedule_piecewise_constant_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_piecewise_constant_decay.md)  
[`learning_rate_schedule_polynomial_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_polynomial_decay.md)  
