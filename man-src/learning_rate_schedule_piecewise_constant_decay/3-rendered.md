A `LearningRateSchedule` that uses a piecewise constant decay schedule.

@description
The function returns a 1-arg callable to compute the piecewise constant
when passed the current optimizer step. This can be useful for changing the
learning rate value across different invocations of optimizer functions.

# Examples
use a learning rate that's 1.0 for the first 100001 steps, 0.5
for the next 10000 steps, and 0.1 for any additional steps.


```r
step <- 0
boundaries <- c(100000, 110000)
values <- c(1.0, 0.5, 0.1)
learning_rate_fn <- learning_rate_schedule_piecewise_constant_decay(
  boundaries, values)

# Later, whenever we perform an optimization step, we pass in the step.
learning_rate <- learning_rate_fn(step)
```

You can pass this schedule directly into a `optimizer`
as the learning rate. The learning rate schedule is also serializable and
deserializable using `keras$optimizers$schedules$serialize` and
`keras$optimizers$schedules$deserialize`.

# Raises
ValueError: if the number of elements in the `boundaries` and `values`
lists do not match.

@returns
A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the
same type as the boundary tensors.

The output of the 1-arg function that takes the `step`
is `values[0]` when `step <= boundaries[0]`,
`values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`,
..., and `values[-1]` when `step > boundaries[-1]`.

@param boundaries
A list of Python numbers with strictly increasing
entries, and with all elements having the same type as the
optimizer step.

@param values
A list of Python numbers that specifies the values for the
intervals defined by `boundaries`. It should have one more
element than `boundaries`, and all elements should have the same
type.

@param name
A string. Optional name of the operation. Defaults to
`"PiecewiseConstant"`.

@export
@family learning rate schedule optimizers
@family schedule optimizers
@seealso
+ <https:/keras.io/api/optimizers/learning_rate_schedules/piecewise_constant_decay#piecewiseconstantdecay-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay>

