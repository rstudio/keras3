# A `LearningRateSchedule` that uses an inverse time decay schedule.

When training a model, it is often useful to lower the learning rate as
the training progresses. This schedule applies the inverse decay
function to an optimizer step, given a provided initial learning rate.
It requires a `step` value to compute the decayed learning rate. You can
just pass a backend variable that you increment at each training step.

The schedule is a 1-arg callable that produces a decayed learning rate
when passed the current optimizer step. This can be useful for changing
the learning rate value across different invocations of optimizer
functions. It is computed as:

    decayed_learning_rate <- function(step) {
      initial_learning_rate / (1 + decay_rate * step / decay_step)
    }

or, if `staircase` is `TRUE`, as:

    decayed_learning_rate <- function(step) {
      initial_learning_rate /
               (1 + decay_rate * floor(step / decay_step))
    }

You can pass this schedule directly into a `optimizer_*` as the learning
rate.

## Usage

``` r
learning_rate_schedule_inverse_time_decay(
  initial_learning_rate,
  decay_steps,
  decay_rate,
  staircase = FALSE,
  name = "InverseTimeDecay"
)
```

## Arguments

- initial_learning_rate:

  A float. The initial learning rate.

- decay_steps:

  How often to apply decay.

- decay_rate:

  A number. The decay rate.

- staircase:

  Whether to apply decay in a discrete staircase, as o pposed to
  continuous, fashion.

- name:

  String. Optional name of the operation. Defaults to
  `"InverseTimeDecay"`.

## Value

A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the same
type as `initial_learning_rate`.

## Examples

Fit a Keras model when decaying 1/t with a rate of 0.5:

    ...
    initial_learning_rate <- 0.1
    decay_steps <- 1.0
    decay_rate <- 0.5
    learning_rate_fn <- learning_rate_schedule_inverse_time_decay(
        initial_learning_rate, decay_steps, decay_rate)

    model %>% compile(
      optimizer = optimizer_sgd(learning_rate=learning_rate_fn),
      loss = 'sparse_categorical_crossentropy',
      metrics = 'accuracy')
    )

    model %>% fit(data, labels, epochs=5)

## See also

- <https://keras.io/api/optimizers/learning_rate_schedules/inverse_time_decay#inversetimedecay-class>

Other optimizer learning rate schedules:  
[`LearningRateSchedule()`](https://keras3.posit.co/reference/LearningRateSchedule.md)  
[`learning_rate_schedule_cosine_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_cosine_decay.md)  
[`learning_rate_schedule_cosine_decay_restarts()`](https://keras3.posit.co/reference/learning_rate_schedule_cosine_decay_restarts.md)  
[`learning_rate_schedule_exponential_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_exponential_decay.md)  
[`learning_rate_schedule_piecewise_constant_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_piecewise_constant_decay.md)  
[`learning_rate_schedule_polynomial_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_polynomial_decay.md)  
