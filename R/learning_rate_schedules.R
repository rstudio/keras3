

#' A LearningRateSchedule that uses an exponential decay schedule
#'
#' @details
#' When training a model, it is often useful to lower the learning rate as
#' the training progresses. This schedule applies an exponential decay function
#' to an optimizer step, given a provided initial learning rate.
#'
#' The schedule is a 1-arg callable that produces a decayed learning
#' rate when passed the current optimizer step. This can be useful for changing
#' the learning rate value across different invocations of optimizer functions.
#' It is computed as:
#'
#' ````r
#' decayed_learning_rate <- function(step)
#'   initial_learning_rate * decay_rate ^ (step / decay_steps)
#' ````
#'
#' If the argument `staircase` is `TRUE`, then `step / decay_steps` is
#' an integer division (`%/%`) and the decayed learning rate follows a
#' staircase function.
#'
#' You can pass this schedule directly into a optimizer
#' as the learning rate (see example)
#' Example: When fitting a Keras model, decay every 100000 steps with a base
#' of 0.96:
#'
#' ```R
#' initial_learning_rate <- 0.1
#' lr_schedule <- learning_rate_schedule_exponential_decay(
#'     initial_learning_rate,
#'     decay_steps = 100000,
#'     decay_rate = 0.96,
#'     staircase = TRUE)
#'
#' model %>% compile(
#'   optimizer= optimizer_sgd(learning_rate = lr_schedule),
#'   loss = 'sparse_categorical_crossentropy',
#'   metrics = 'accuracy')
#'
#' model %>% fit(data, labels, epochs = 5)
#' ```
#'
#' @param ... For backwards and forwards compatibility
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay>
#'
#' @param initial_learning_rate A scalar `float32` or `float64` `Tensor` or a R
#'   number. The initial learning rate.
#' @param decay_steps A scalar `int32` or `int64` `Tensor` or an R number. Must
#'   be positive.  See the decay computation above.
#' @param decay_rate A scalar `float32` or `float64` `Tensor` or an R number.
#'   The decay rate.
#' @param staircase Boolean.  If `TRUE` decay the learning rate at discrete
#'   intervals.
#' @param name String. Optional name of the operation.  Defaults to
#'   'ExponentialDecay'.
#' @export
learning_rate_schedule_exponential_decay <-
function(initial_learning_rate, decay_steps, decay_rate, staircase = FALSE,
         ..., name = NULL)
{
  args <- capture_args(match.call(), list(
    decay_steps = as_integer))
  do.call(keras$optimizers$schedules$ExponentialDecay, args)
}

#' A LearningRateSchedule that uses a cosine decay schedule
#'
#' @details
#' See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
#' SGDR: Stochastic Gradient Descent with Warm Restarts.
#'
#' When training a model, it is often useful to lower the learning rate as
#' the training progresses. This schedule applies a cosine decay function
#' to an optimizer step, given a provided initial learning rate.
#' It requires a `step` value to compute the decayed learning rate. You can
#' just pass a TensorFlow variable that you increment at each training step.
#'
#' The schedule is a 1-arg callable that produces a decayed learning
#' rate when passed the current optimizer step. This can be useful for changing
#' the learning rate value across different invocations of optimizer functions.
#' It is computed as:
#'
#' ```r
#' decayed_learning_rate <- function(step) {
#'   step <- min(step, decay_steps)
#'   cosine_decay = <- 0.5 * (1 + cos(pi * step / decay_steps))
#'   decayed <- (1 - alpha) * cosine_decay + alpha
#'   initial_learning_rate * decayed
#' }
#' ```
#'
#' Example usage:
#' ```R
#' decay_steps <- 1000
#' lr_decayed_fn <-
#'   learning_rate_schedule_cosine_decay(initial_learning_rate, decay_steps)
#' ```
#'
#' You can pass this schedule directly into a keras Optimizer
#' as the `learning_rate`.
#'
#' @param initial_learning_rate A scalar `float32` or `float64` Tensor or a
#'   R number. The initial learning rate.
#' @param decay_steps A scalar `int32` or `int64` `Tensor` or an R number.
#'   Number of steps to decay over.
#' @param alpha A scalar `float32` or `float64` Tensor or an R number.
#'   Minimum learning rate value as a fraction of initial_learning_rate.
#' @param name String. Optional name of the operation.  Defaults to
#'   'CosineDecay'.
#'
#' @param ... For backwards and forwards compatibility
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>
#' @export
learning_rate_schedule_cosine_decay <-
  function(initial_learning_rate, decay_steps, alpha = 0, ..., name = NULL)
  {
    args <- capture_args(match.call(), list(
      decay_steps = as_integer))
    do.call(keras$optimizers$schedules$CosineDecay, args)
  }


#' A LearningRateSchedule that uses a cosine decay schedule with restarts
#'
#' @details
#' See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
#' SGDR: Stochastic Gradient Descent with Warm Restarts.
#'
#' When training a model, it is often useful to lower the learning rate as
#' the training progresses. This schedule applies a cosine decay function with
#' restarts to an optimizer step, given a provided initial learning rate.
#' It requires a `step` value to compute the decayed learning rate. You can
#' just pass a TensorFlow variable that you increment at each training step.
#'
#' The schedule is a 1-arg callable that produces a decayed learning
#' rate when passed the current optimizer step. This can be useful for changing
#' the learning rate value across different invocations of optimizer functions.
#'
#' The learning rate multiplier first decays
#' from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
#' restart is performed. Each new warm restart runs for `t_mul` times more
#' steps and with `m_mul` times initial learning rate as the new learning rate.
#'
#'
#' You can pass this schedule directly into a keras Optimizer
#' as the `learning_rate`.
#'
#' @param initial_learning_rate A scalar `float32` or `float64` Tensor or an R
#'   number. The initial learning rate.
#' @param first_decay_steps A scalar `int32` or `int64` `Tensor` or an R
#'   number. Number of steps to decay over.
#' @param t_mul A scalar `float32` or `float64` `Tensor` or an R number. Used
#'   to derive the number of iterations in the i-th period.
#' @param m_mul A scalar `float32` or `float64` `Tensor` or an R number. Used
#'   to derive the initial learning rate of the i-th period.
#' @param alpha A scalar `float32` or `float64` Tensor or an R number. Minimum
#'   learning rate value as a fraction of the initial_learning_rate.
#' @param name String. Optional name of the operation.  Defaults to
#'   'SGDRDecay'.
#'
#' @param ... For backwards and forwards compatibility
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts>
#' @export
learning_rate_schedule_cosine_decay_restarts <-
function(initial_learning_rate, first_decay_steps, t_mul = 2,
         m_mul = 1, alpha = 0, ..., name = NULL)
{
  args <- capture_args(match.call(), list(first_decay_steps = as_integer))
  do.call(keras$optimizers$schedules$CosineDecayRestarts, args)
}


#' A LearningRateSchedule that uses an inverse time decay schedule
#'
#' @details
#' When training a model, it is often useful to lower the learning rate as
#' the training progresses. This schedule applies the inverse decay function
#' to an optimizer step, given a provided initial learning rate.
#' It requires a `step` value to compute the decayed learning rate. You can
#' just pass a TensorFlow variable that you increment at each training step.
#'
#' The schedule is a 1-arg callable that produces a decayed learning
#' rate when passed the current optimizer step. This can be useful for changing
#' the learning rate value across different invocations of optimizer functions.
#' It is computed as:
#'
#' ```R
#' decayed_learning_rate <- function(step) {
#'   initial_learning_rate / (1 + decay_rate * step / decay_step)
#' }
#' ```
#'
#' or, if `staircase` is `TRUE`, as:
#'
#' ```R
#' decayed_learning_rate function(step) {
#'  initial_learning_rate / (1 + decay_rate * floor(step / decay_step))
#' }
#' ```
#'
#' You can pass this schedule directly into a keras Optimizer
#' as the `learning_rate`.
#'
#' Example: Fit a Keras model when decaying `1/t` with a rate of `0.5`:
#'
#' ```R
#' ...
#' initial_learning_rate <- 0.1
#' decay_steps <- 1.0
#' decay_rate <- 0.5
#' learning_rate_fn <- learning_rate_schedule_inverse_time_decay(
#'   initial_learning_rate, decay_steps, decay_rate)
#'
#' model %>%
#'   compile(optimizer = optimizer_sgd(learning_rate = learning_rate_fn),
#'           loss = 'sparse_categorical_crossentropy',
#'           metrics = 'accuracy')
#'
#' model %>% fit(data, labels, epochs = 5)
#' ```
#'
#' @param initial_learning_rate A scalar `float32` or `float64` `Tensor` or an
#'   R number. The initial learning rate.
#' @param decay_steps A scalar `int32` or `int64` `Tensor` or an R number. How
#'   often to apply decay.
#' @param decay_rate An R number. The decay rate.
#' @param staircase Boolean. Whether to apply decay in a discrete staircase, as
#'   opposed to continuous, fashion.
#' @param name String.  Optional name of the operation.  Defaults to
#'   'InverseTimeDecay'.
#' @param ... For backwards and forwards compatibility
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>
#' @export
learning_rate_schedule_inverse_time_decay <-
function(initial_learning_rate, decay_steps, decay_rate, staircase = FALSE, ...,
         name = NULL) {
  args <- capture_args(match.call(), list(decay_steps = as_integer))
  do.call(keras$optimizers$schedules$InverseTimeDecay, args)
}


#' A LearningRateSchedule that uses a piecewise constant decay schedule
#'
#' @details
#' The function returns a 1-arg callable to compute the piecewise constant
#' when passed the current optimizer step. This can be useful for changing the
#' learning rate value across different invocations of optimizer functions.
#'
#' Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
#'   for the next 10000 steps, and 0.1 for any additional steps.
#'
#' ```R
#' step <- tf$Variable(0, trainable=FALSE)
#' boundaries <- as.integer(c(100000, 110000))
#' values <- c(1.0, 0.5, 0.1)
#' learning_rate_fn <- learning_rate_schedule_piecewise_constant_decay(
#'     boundaries, values)
#'
#' # Later, whenever we perform an optimization step, we pass in the step.
#' learning_rate <- learning_rate_fn(step)
#' ```
#'
#' You can pass this schedule directly into a keras Optimizer
#' as the `learning_rate`.
#'
#' @param boundaries A list of `Tensor`s or R numerics with strictly increasing
#'   entries, and with all elements having the same type as the optimizer step.
#' @param values A list of `Tensor`s or R numerics that specifies the
#'   values for the intervals defined by `boundaries`. It should have one more
#'   element than `boundaries`, and all elements should have the same type.
#' @param name A string. Optional name of the operation. Defaults to
#'   'PiecewiseConstant'.
#' @param ... For backwards and forwards compatibility
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay>
#' @export
learning_rate_schedule_piecewise_constant_decay <-
function(boundaries, values, ..., name = NULL) {
  args <- capture_args(match.call())
  do.call(keras$optimizers$schedules$PiecewiseConstantDecay,
          args)
}


#' A LearningRateSchedule that uses a polynomial decay schedule
#'
#' @details
#' It is commonly observed that a monotonically decreasing learning rate, whose
#' degree of change is carefully chosen, results in a better performing model.
#' This schedule applies a polynomial decay function to an optimizer step,
#' given a provided `initial_learning_rate`, to reach an `end_learning_rate`
#' in the given `decay_steps`.
#'
#' It requires a `step` value to compute the decayed learning rate. You
#' can just pass a TensorFlow variable that you increment at each training
#' step.
#'
#' The schedule is a 1-arg callable that produces a decayed learning rate
#' when passed the current optimizer step. This can be useful for changing the
#' learning rate value across different invocations of optimizer functions.
#' It is computed as:
#'
#' ```R
#' decayed_learning_rate <- function(step) {
#'   step <- min(step, decay_steps)
#'   ((initial_learning_rate - end_learning_rate) *
#'       (1 - step / decay_steps) ^ (power)
#'     ) + end_learning_rate
#' }
#' ```
#'
#' If `cycle` is `TRUE` then a multiple of `decay_steps` is used, the first one
#' that is bigger than `step`.
#'
#' ```python
#' decayed_learning_rate <- function(step) {
#'   decay_steps <- decay_steps * ceiling(step / decay_steps)
#'   ((initial_learning_rate - end_learning_rate) *
#'       (1 - step / decay_steps) ^ (power)
#'     ) + end_learning_rate
#' }
#' ```
#'
#' You can pass this schedule directly into a keras Optimizer
#' as the `learning_rate`.
#'
#' Example: Fit a model while decaying from 0.1 to 0.01 in 10000 steps using
#' sqrt (i.e. power=0.5):
#'
#' ```R
#' ...
#' starter_learning_rate <- 0.1
#' end_learning_rate <- 0.01
#' decay_steps <- 10000
#' learning_rate_fn <- learning_rate_schedule_polynomial_decay(
#'   starter_learning_rate, decay_steps, end_learning_rate, power = 0.5)
#'
#' model %>%
#'   compile(optimizer = optimizer_sgd(learning_rate = learning_rate_fn),
#'           loss = 'sparse_categorical_crossentropy',
#'           metrics = 'accuracy')
#'
#' model %>% fit(data, labels, epochs = 5)
#' ```
#'
#' @param initial_learning_rate A scalar `float32` or `float64` `Tensor` or an
#'   R number.  The initial learning rate.
#' @param decay_steps A scalar `int32` or `int64` `Tensor` or an R number.
#'   Must be positive.  See the decay computation above.
#' @param end_learning_rate A scalar `float32` or `float64` `Tensor` or an
#'   R number.  The minimal end learning rate.
#' @param power A scalar `float32` or `float64` `Tensor` or an R number.
#'   The power of the polynomial. Defaults to linear, 1.0.
#' @param cycle A boolean,
#'   whether or not it should cycle beyond decay_steps.
#' @param name String.  Optional name of the operation. Defaults to
#'   'PolynomialDecay'.
#'
#' @param ... For backwards and forwards compatibility
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay>
#' @export
learning_rate_schedule_polynomial_decay <-
function(initial_learning_rate, decay_steps, end_learning_rate = 1e-04,
         power = 1, cycle = FALSE, ..., name = NULL)
{
  args <- capture_args(match.call(), list(decay_steps = as_integer))
  do.call(keras$optimizers$schedules$PolynomialDecay, args)
}



# TODO: still need to add tests for all these.
# TODO: should all optimizer accept a plain R function to `learning_rate`?


#' Create a new learning rate schedule type
#'
#' @param classname string
#' @param ... methods and properties of the schedule class
#' @param call function which takes a step argument (scalar integer tensor, the
#'   current training step count, and returns the new learning rate). For
#'   tracking additional state, objects `self` and `private` are automatically
#'   injected into the scope of the function.
#' @param initialize,get_config Additional recommended methods to implement.
#'
#'  +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule>
#' @return A `LearningRateSchedule` class generator.
#' @export
new_learning_rate_schedule_class <-
function(classname, ..., initialize = NULL, call, get_config = NULL) {
  members <- capture_args(match.call(), ignore = "classname")
  members <- drop_nulls(members)
  members <- rename_to_dunder(members, "call")
  if (!is.null(members[["call"]])) {
    if ("__call__" %in% names(members))
      warning("`call()` method is ignored, superceded by `__call__`() method.")
    else
      names(members)[match("call", names(members))] <- "__call__"
  }

  new_py_class(
    classname,
    members = members,
    inherit = keras$optimizers$schedules$LearningRateSchedule,
    parent_env = parent.frame(),
    convert = TRUE
  )
}


rename_to_dunder <- function(members, nms) {
  if(anyDuplicated(names(members)))
    stop("All names must be unique")
  for (nm in nms) {
    .__nm__ <- paste0("__", nm, "__")
    if (nm %in% names(members)) {
      if (.__nm__ %in% names(members))
        warning("`", nm, "` method is ignored, superceded by `", .__nm__, "` method.")
      else
        names(members)[match(nm, names(members))] <- .__nm__
    }
  }
  members
}
