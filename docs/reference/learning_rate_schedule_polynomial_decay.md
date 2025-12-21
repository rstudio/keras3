# A `LearningRateSchedule` that uses a polynomial decay schedule.

It is commonly observed that a monotonically decreasing learning rate,
whose degree of change is carefully chosen, results in a better
performing model. This schedule applies a polynomial decay function to
an optimizer step, given a provided `initial_learning_rate`, to reach an
`end_learning_rate` in the given `decay_steps`.

It requires a `step` value to compute the decayed learning rate. You can
just pass a backend variable that you increment at each training step.

The schedule is a 1-arg callable that produces a decayed learning rate
when passed the current optimizer step. This can be useful for changing
the learning rate value across different invocations of optimizer
functions. It is computed as:

    decayed_learning_rate <- function(step) {
      step = min(step, decay_steps)
      ((initial_learning_rate - end_learning_rate) *
        (1 - step / decay_steps) ^ (power)) +
        end_learning_rate
    }

If `cycle` is TRUE then a multiple of `decay_steps` is used, the first
one that is bigger than `step`.

    decayed_learning_rate <- function(step) {
      decay_steps = decay_steps * ceil(step / decay_steps)
      ((initial_learning_rate - end_learning_rate) *
          (1 - step / decay_steps) ^ (power)) +
        end_learning_rate
    }

You can pass this schedule directly into a `Optimizer` as the learning
rate.

## Usage

``` r
learning_rate_schedule_polynomial_decay(
  initial_learning_rate,
  decay_steps,
  end_learning_rate = 0.0001,
  power = 1,
  cycle = FALSE,
  name = "PolynomialDecay"
)
```

## Arguments

- initial_learning_rate:

  A float. The initial learning rate.

- decay_steps:

  A integer. Must be positive. See the decay computation above.

- end_learning_rate:

  A float. The minimal end learning rate.

- power:

  A float. The power of the polynomial. Defaults to `1.0`.

- cycle:

  A boolean, whether it should cycle beyond decay_steps.

- name:

  String. Optional name of the operation. Defaults to
  `"PolynomialDecay"`.

## Value

A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the same
type as `initial_learning_rate`.

## Examples

Fit a model while decaying from 0.1 to 0.01 in 10000 steps using sqrt
(i.e. power=0.5):

    ...
    starter_learning_rate <- 0.1
    end_learning_rate <- 0.01
    decay_steps <- 10000
    learning_rate_fn <- learning_rate_schedule_polynomial_decay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)

    model %>% compile(
      optimizer = optimizer_sgd(learning_rate=learning_rate_fn),
      loss = 'sparse_categorical_crossentropy',
      metrics = 'accuracy'
    )

    model %>% fit(data, labels, epochs=5)

The learning rate schedule is also serializable and deserializable using
`keras$optimizers$schedules$serialize` and
`keras$optimizers$schedules$deserialize`.

## See also

- <https://keras.io/api/optimizers/learning_rate_schedules/polynomial_decay#polynomialdecay-class>

Other optimizer learning rate schedules:  
[`LearningRateSchedule()`](https://keras3.posit.co/reference/LearningRateSchedule.md)  
[`learning_rate_schedule_cosine_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_cosine_decay.md)  
[`learning_rate_schedule_cosine_decay_restarts()`](https://keras3.posit.co/reference/learning_rate_schedule_cosine_decay_restarts.md)  
[`learning_rate_schedule_exponential_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_exponential_decay.md)  
[`learning_rate_schedule_inverse_time_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_inverse_time_decay.md)  
[`learning_rate_schedule_piecewise_constant_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_piecewise_constant_decay.md)  
