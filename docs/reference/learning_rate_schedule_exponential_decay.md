# A `LearningRateSchedule` that uses an exponential decay schedule.

When training a model, it is often useful to lower the learning rate as
the training progresses. This schedule applies an exponential decay
function to an optimizer step, given a provided initial learning rate.

The schedule is a 1-arg callable that produces a decayed learning rate
when passed the current optimizer step. This can be useful for changing
the learning rate value across different invocations of optimizer
functions. It is computed as:

    decayed_learning_rate <- function(step) {
      initial_learning_rate * decay_rate ^ (step / decay_steps)
    }

If the argument `staircase` is `TRUE`, then `step / decay_steps` is an
integer division and the decayed learning rate follows a staircase
function.

You can pass this schedule directly into a `optimizer` as the learning
rate.

## Usage

``` r
learning_rate_schedule_exponential_decay(
  initial_learning_rate,
  decay_steps,
  decay_rate,
  staircase = FALSE,
  name = "ExponentialDecay"
)
```

## Arguments

- initial_learning_rate:

  A float. The initial learning rate.

- decay_steps:

  A integer. Must be positive. See the decay computation above.

- decay_rate:

  A float. The decay rate.

- staircase:

  Boolean. If `TRUE` decay the learning rate at discrete intervals.

- name:

  String. Optional name of the operation. Defaults to
  `"ExponentialDecay`".

## Value

A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the same
type as `initial_learning_rate`.

## Examples

When fitting a Keras model, decay every 100000 steps with a base of
0.96:

    initial_learning_rate <- 0.1
    lr_schedule <- learning_rate_schedule_exponential_decay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=TRUE)

    model %>% compile(
      optimizer = optimizer_sgd(learning_rate = lr_schedule),
      loss = 'sparse_categorical_crossentropy',
      metrics = c('accuracy'))

    model %>% fit(data, labels, epochs=5)

The learning rate schedule is also serializable and deserializable using
`keras$optimizers$schedules$serialize` and
`keras$optimizers$schedules$deserialize`.

## See also

- <https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay#exponentialdecay-class>

Other optimizer learning rate schedules:  
[`LearningRateSchedule()`](https://keras3.posit.co/reference/LearningRateSchedule.md)  
[`learning_rate_schedule_cosine_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_cosine_decay.md)  
[`learning_rate_schedule_cosine_decay_restarts()`](https://keras3.posit.co/reference/learning_rate_schedule_cosine_decay_restarts.md)  
[`learning_rate_schedule_inverse_time_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_inverse_time_decay.md)  
[`learning_rate_schedule_piecewise_constant_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_piecewise_constant_decay.md)  
[`learning_rate_schedule_polynomial_decay()`](https://keras3.posit.co/reference/learning_rate_schedule_polynomial_decay.md)  
