

#' A `LearningRateSchedule` that uses a cosine decay with optional warmup.
#'
#' @description
#' See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
#' SGDR: Stochastic Gradient Descent with Warm Restarts.
#'
#' For the idea of a linear warmup of our learning rate,
#' see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).
#'
#' When we begin training a model, we often want an initial increase in our
#' learning rate followed by a decay. If `warmup_target` is an int, this
#' schedule applies a linear increase per optimizer step to our learning rate
#' from `initial_learning_rate` to `warmup_target` for a duration of
#' `warmup_steps`. Afterwards, it applies a cosine decay function taking our
#' learning rate from `warmup_target` to `alpha` for a duration of
#' `decay_steps`. If `warmup_target` is NULL we skip warmup and our decay
#' will take our learning rate from `initial_learning_rate` to `alpha`.
#' It requires a `step` value to  compute the learning rate. You can
#' just pass a backend variable that you increment at each training step.
#'
#' The schedule is a 1-arg callable that produces a warmup followed by a
#' decayed learning rate when passed the current optimizer step. This can be
#' useful for changing the learning rate value across different invocations of
#' optimizer functions.
#'
#' Our warmup is computed as:
#'
#' ```{r}
#' warmup_learning_rate <- function(step) {
#'   completed_fraction <- step / warmup_steps
#'   total_delta <- target_warmup - initial_learning_rate
#'   completed_fraction * total_delta
#' }
#' ```
#'
#' And our decay is computed as:
#'
#' ```{r, eval=FALSE}
#' if (is.null(warmup_target)) {
#'   initial_decay_lr <- initial_learning_rate
#' } else {
#'   initial_decay_lr <- warmup_target
#' }
#'
#' decayed_learning_rate <- function(step) {
#'   step <- min(step, decay_steps)
#'   cosine_decay <- 0.5 * (1 + cos(pi * step / decay_steps))
#'   decayed <- (1 - alpha) * cosine_decay + alpha
#'   initial_decay_lr * decayed
#' }
#' ```
#'
#' Example usage without warmup:
#'
#' ```{r}
#' decay_steps <- 1000
#' initial_learning_rate <- 0.1
#' lr_decayed_fn <- learning_rate_schedule_cosine_decay(
#'     initial_learning_rate, decay_steps)
#' ```
#'
#' Example usage with warmup:
#'
#' ```{r}
#' decay_steps <- 1000
#' initial_learning_rate <- 0
#' warmup_steps <- 1000
#' target_learning_rate <- 0.1
#' lr_warmup_decayed_fn <- learning_rate_schedule_cosine_decay(
#'     initial_learning_rate, decay_steps, warmup_target = target_learning_rate,
#'     warmup_steps = warmup_steps
#' )
#' ```
#'
#' You can pass this schedule directly into a `optimizer`
#' as the learning rate. The learning rate schedule is also serializable and
#' deserializable using `keras$optimizers$schedules$serialize` and
#' `keras$optimizers$schedules$deserialize`.
#'
#' @returns
#' A 1-arg callable learning rate schedule that takes the current optimizer
#' step and outputs the decayed learning rate, a scalar tensor of the
#' same type as `initial_learning_rate`.
#'
#' @param initial_learning_rate
#' A float. The initial learning rate.
#'
#' @param decay_steps
#' A int. Number of steps to decay over.
#'
#' @param alpha
#' A float. Minimum learning rate value for decay as a
#' fraction of `initial_learning_rate`.
#'
#' @param name
#' String. Optional name of the operation.  Defaults to
#' `"CosineDecay"`.
#'
#' @param warmup_target
#' A float. The target learning rate for our
#' warmup phase. Will cast to the `initial_learning_rate` datatype.
#' Setting to `NULL` will skip warmup and begins decay phase from
#' `initial_learning_rate`. Otherwise scheduler will warmup from
#' `initial_learning_rate` to `warmup_target`.
#'
#' @param warmup_steps
#' A int. Number of steps to warmup over.
#'
#' @export
#' @family optimizer learning rate schedules
#' @seealso
#' + <https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay#cosinedecay-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>
#'
#' @tether keras.optimizers.schedules.CosineDecay
learning_rate_schedule_cosine_decay <-
function (initial_learning_rate, decay_steps, alpha = 0, name = "CosineDecay",
    warmup_target = NULL, warmup_steps = 0L)
{
    args <- capture_args(list(decay_steps = as_integer, warmup_steps = as_integer))
    do.call(keras$optimizers$schedules$CosineDecay, args)
}


#' A `LearningRateSchedule` that uses a cosine decay schedule with restarts.
#'
#' @description
#' See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
#' SGDR: Stochastic Gradient Descent with Warm Restarts.
#'
#' When training a model, it is often useful to lower the learning rate as
#' the training progresses. This schedule applies a cosine decay function with
#' restarts to an optimizer step, given a provided initial learning rate.
#' It requires a `step` value to compute the decayed learning rate. You can
#' just pass a backend variable that you increment at each training step.
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
#' Example usage:
#' ```{r, eval = TRUE}
#' first_decay_steps <- 1000
#' lr_decayed_fn <- learning_rate_schedule_cosine_decay_restarts(
#'         0.001,
#'         first_decay_steps)
#' ```
#'
#' You can pass this schedule directly into a `optimizer`
#' as the learning rate. The learning rate schedule is also serializable and
#' deserializable using `keras$optimizers$schedules$serialize` and
#' `keras$optimizers$schedules$deserialize`.
#'
#' @returns
#' A 1-arg callable learning rate schedule that takes the current optimizer
#' step and outputs the decayed learning rate, a scalar tensor of the
#' same type as `initial_learning_rate`.
#'
#' @param initial_learning_rate
#' A float. The initial learning rate.
#'
#' @param first_decay_steps
#' An integer. Number of steps to decay over.
#'
#' @param t_mul
#' A float. Used to derive the number of iterations in
#' the i-th period.
#'
#' @param m_mul
#' A float. Used to derive the initial learning rate of
#' the i-th period.
#'
#' @param alpha
#' A float. Minimum learning rate value as a fraction of
#' the `initial_learning_rate`.
#'
#' @param name
#' String. Optional name of the operation. Defaults to
#' `"SGDRDecay"`.
#'
#' @export
#' @family optimizer learning rate schedules
#' @seealso
#' + <https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay_restarts#cosinedecayrestarts-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts>
#'
#' @tether keras.optimizers.schedules.CosineDecayRestarts
learning_rate_schedule_cosine_decay_restarts <-
function (initial_learning_rate, first_decay_steps, t_mul = 2,
    m_mul = 1, alpha = 0, name = "SGDRDecay")
{
    args <- capture_args(list(first_decay_steps = as_integer))
    do.call(keras$optimizers$schedules$CosineDecayRestarts, args)
}


#' A `LearningRateSchedule` that uses an exponential decay schedule.
#'
#' @description
#' When training a model, it is often useful to lower the learning rate as
#' the training progresses. This schedule applies an exponential decay function
#' to an optimizer step, given a provided initial learning rate.
#'
#' The schedule is a 1-arg callable that produces a decayed learning
#' rate when passed the current optimizer step. This can be useful for changing
#' the learning rate value across different invocations of optimizer functions.
#' It is computed as:
#'
#' ```{r}
#' decayed_learning_rate <- function(step) {
#'   initial_learning_rate * decay_rate ^ (step / decay_steps)
#' }
#' ```
#'
#' If the argument `staircase` is `TRUE`, then `step / decay_steps` is
#' an integer division and the decayed learning rate follows a
#' staircase function.
#'
#' You can pass this schedule directly into a `optimizer`
#' as the learning rate.
#'
#' # Examples
#' When fitting a Keras model, decay every 100000 steps with a base
#' of 0.96:
#'
#' ```{r, eval=FALSE}
#' initial_learning_rate <- 0.1
#' lr_schedule <- learning_rate_schedule_exponential_decay(
#'     initial_learning_rate,
#'     decay_steps=100000,
#'     decay_rate=0.96,
#'     staircase=TRUE)
#'
#' model %>% compile(
#'   optimizer = optimizer_sgd(learning_rate = lr_schedule),
#'   loss = 'sparse_categorical_crossentropy',
#'   metrics = c('accuracy'))
#'
#' model %>% fit(data, labels, epochs=5)
#' ```
#'
#' The learning rate schedule is also serializable and deserializable using
#' `keras$optimizers$schedules$serialize` and
#' `keras$optimizers$schedules$deserialize`.
#'
#' @returns
#' A 1-arg callable learning rate schedule that takes the current optimizer
#' step and outputs the decayed learning rate, a scalar tensor of the
#' same type as `initial_learning_rate`.
#'
#' @param initial_learning_rate
#' A float. The initial learning rate.
#'
#' @param decay_steps
#' A integer. Must be positive. See the decay
#' computation above.
#'
#' @param decay_rate
#' A float. The decay rate.
#'
#' @param staircase
#' Boolean.  If `TRUE` decay the learning rate at discrete
#' intervals.
#'
#' @param name
#' String.  Optional name of the operation.  Defaults to
#' `"ExponentialDecay`".
#'
#' @export
#' @family optimizer learning rate schedules
#' @seealso
#' + <https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay#exponentialdecay-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay>
#'
#' @tether keras.optimizers.schedules.ExponentialDecay
learning_rate_schedule_exponential_decay <-
function (initial_learning_rate, decay_steps, decay_rate, staircase = FALSE,
    name = "ExponentialDecay")
{
    args <- capture_args(list(decay_steps = as_integer))
    do.call(keras$optimizers$schedules$ExponentialDecay, args)
}


#' A `LearningRateSchedule` that uses an inverse time decay schedule.
#'
#' @description
#' When training a model, it is often useful to lower the learning rate as
#' the training progresses. This schedule applies the inverse decay function
#' to an optimizer step, given a provided initial learning rate.
#' It requires a `step` value to compute the decayed learning rate. You can
#' just pass a backend variable that you increment at each training step.
#'
#' The schedule is a 1-arg callable that produces a decayed learning
#' rate when passed the current optimizer step. This can be useful for changing
#' the learning rate value across different invocations of optimizer functions.
#' It is computed as:
#'
#' ```{r}
#' decayed_learning_rate <- function(step) {
#'   initial_learning_rate / (1 + decay_rate * step / decay_step)
#' }
#' ```
#'
#' or, if `staircase` is `TRUE`, as:
#'
#' ```{r}
#' decayed_learning_rate <- function(step) {
#'   initial_learning_rate /
#'            (1 + decay_rate * floor(step / decay_step))
#' }
#' ```
#'
#' You can pass this schedule directly into a `optimizer_*`
#' as the learning rate.
#'
#' # Examples
#' Fit a Keras model when decaying 1/t with a rate of 0.5:
#'
#' ```{r, eval=FALSE}
#' ...
#' initial_learning_rate <- 0.1
#' decay_steps <- 1.0
#' decay_rate <- 0.5
#' learning_rate_fn <- learning_rate_schedule_inverse_time_decay(
#'     initial_learning_rate, decay_steps, decay_rate)
#'
#' model %>% compile(
#'   optimizer = optimizer_sgd(learning_rate=learning_rate_fn),
#'   loss = 'sparse_categorical_crossentropy',
#'   metrics = 'accuracy')
#' )
#'
#' model %>% fit(data, labels, epochs=5)
#' ```
#'
#' @returns
#' A 1-arg callable learning rate schedule that takes the current optimizer
#' step and outputs the decayed learning rate, a scalar tensor of the
#' same type as `initial_learning_rate`.
#'
#' @param initial_learning_rate
#' A float. The initial learning rate.
#'
#' @param decay_steps
#' How often to apply decay.
#'
#' @param decay_rate
#' A number.  The decay rate.
#'
#' @param staircase
#' Whether to apply decay in a discrete staircase, as o
#' pposed to continuous, fashion.
#'
#' @param name
#' String.  Optional name of the operation.  Defaults to
#' `"InverseTimeDecay"`.
#'
#' @export
#' @family optimizer learning rate schedules
#' @seealso
#' + <https://keras.io/api/optimizers/learning_rate_schedules/inverse_time_decay#inversetimedecay-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>
#'
#' @tether keras.optimizers.schedules.InverseTimeDecay
learning_rate_schedule_inverse_time_decay <-
function (initial_learning_rate, decay_steps, decay_rate, staircase = FALSE,
    name = "InverseTimeDecay")
{
    args <- capture_args()
    do.call(keras$optimizers$schedules$InverseTimeDecay, args)
}


#' A `LearningRateSchedule` that uses a piecewise constant decay schedule.
#'
#' @description
#' The function returns a 1-arg callable to compute the piecewise constant
#' when passed the current optimizer step. This can be useful for changing the
#' learning rate value across different invocations of optimizer functions.
#'
#' # Examples
#' use a learning rate that's 1.0 for the first 100001 steps, 0.5
#' for the next 10000 steps, and 0.1 for any additional steps.
#'
#' ```{r}
#' step <- 0
#' boundaries <- c(100000, 110000)
#' values <- c(1.0, 0.5, 0.1)
#' learning_rate_fn <- learning_rate_schedule_piecewise_constant_decay(
#'   boundaries, values)
#'
#' # Later, whenever we perform an optimization step, we pass in the step.
#' learning_rate <- learning_rate_fn(step)
#' ```
#'
#' You can pass this schedule directly into a `optimizer`
#' as the learning rate. The learning rate schedule is also serializable and
#' deserializable using `keras$optimizers$schedules$serialize` and
#' `keras$optimizers$schedules$deserialize`.
#'
#' # Raises
#' ValueError: if the number of elements in the `boundaries` and `values`
#' lists do not match.
#'
#' @returns
#' A 1-arg callable learning rate schedule that takes the current optimizer
#' step and outputs the decayed learning rate, a scalar tensor of the
#' same type as the boundary tensors.
#'
#' The output of the 1-arg function that takes the `step`
#' is `values[0]` when `step <= boundaries[0]`,
#' `values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`,
#' ..., and `values[-1]` when `step > boundaries[-1]`.
#'
#' @param boundaries
#' A list of Python numbers with strictly increasing
#' entries, and with all elements having the same type as the
#' optimizer step.
#'
#' @param values
#' A list of Python numbers that specifies the values for the
#' intervals defined by `boundaries`. It should have one more
#' element than `boundaries`, and all elements should have the same
#' type.
#'
#' @param name
#' A string. Optional name of the operation. Defaults to
#' `"PiecewiseConstant"`.
#'
#' @export
#' @family optimizer learning rate schedules
#' @seealso
#' + <https://keras.io/api/optimizers/learning_rate_schedules/piecewise_constant_decay#piecewiseconstantdecay-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay>
#'
#' @tether keras.optimizers.schedules.PiecewiseConstantDecay
learning_rate_schedule_piecewise_constant_decay <-
function (boundaries, values, name = "PiecewiseConstant")
{
    args <- capture_args()
    do.call(keras$optimizers$schedules$PiecewiseConstantDecay,
        args)
}


#' A `LearningRateSchedule` that uses a polynomial decay schedule.
#'
#' @description
#' It is commonly observed that a monotonically decreasing learning rate, whose
#' degree of change is carefully chosen, results in a better performing model.
#' This schedule applies a polynomial decay function to an optimizer step,
#' given a provided `initial_learning_rate`, to reach an `end_learning_rate`
#' in the given `decay_steps`.
#'
#' It requires a `step` value to compute the decayed learning rate. You
#' can just pass a backend variable that you increment at each training
#' step.
#'
#' The schedule is a 1-arg callable that produces a decayed learning rate
#' when passed the current optimizer step. This can be useful for changing the
#' learning rate value across different invocations of optimizer functions.
#' It is computed as:
#'
#' ```{r}
#' decayed_learning_rate <- function(step) {
#'   step = min(step, decay_steps)
#'   ((initial_learning_rate - end_learning_rate) *
#'     (1 - step / decay_steps) ^ (power)) +
#'     end_learning_rate
#' }
#' ```
#'
#' If `cycle` is TRUE then a multiple of `decay_steps` is used, the first one
#' that is bigger than `step`.
#'
#' ```{r}
#' decayed_learning_rate <- function(step) {
#'   decay_steps = decay_steps * ceil(step / decay_steps)
#'   ((initial_learning_rate - end_learning_rate) *
#'       (1 - step / decay_steps) ^ (power)) +
#'     end_learning_rate
#' }
#' ```
#'
#' You can pass this schedule directly into a `Optimizer`
#' as the learning rate.
#'
#' # Examples
#' Fit a model while decaying from 0.1 to 0.01 in 10000 steps using
#' sqrt (i.e. power=0.5):
#'
#' ```{r, eval=FALSE}
#' ...
#' starter_learning_rate <- 0.1
#' end_learning_rate <- 0.01
#' decay_steps <- 10000
#' learning_rate_fn <- learning_rate_schedule_polynomial_decay(
#'     starter_learning_rate,
#'     decay_steps,
#'     end_learning_rate,
#'     power=0.5)
#'
#' model %>% compile(
#'   optimizer = optimizer_sgd(learning_rate=learning_rate_fn),
#'   loss = 'sparse_categorical_crossentropy',
#'   metrics = 'accuracy'
#' )
#'
#' model %>% fit(data, labels, epochs=5)
#' ```
#'
#' The learning rate schedule is also serializable and deserializable using
#' `keras$optimizers$schedules$serialize` and
#' `keras$optimizers$schedules$deserialize`.
#'
#' @returns
#' A 1-arg callable learning rate schedule that takes the current optimizer
#' step and outputs the decayed learning rate, a scalar tensor of the
#' same type as `initial_learning_rate`.
#'
#' @param initial_learning_rate
#' A float. The initial learning rate.
#'
#' @param decay_steps
#' A integer. Must be positive. See the decay
#' computation above.
#'
#' @param end_learning_rate
#' A float. The minimal end learning rate.
#'
#' @param power
#' A float. The power of the polynomial. Defaults to
#' `1.0`.
#'
#' @param cycle
#' A boolean, whether it should cycle beyond decay_steps.
#'
#' @param name
#' String.  Optional name of the operation. Defaults to
#' `"PolynomialDecay"`.
#'
#' @export
#' @family optimizer learning rate schedules
#' @seealso
#' + <https://keras.io/api/optimizers/learning_rate_schedules/polynomial_decay#polynomialdecay-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay>
#'
#' @tether keras.optimizers.schedules.PolynomialDecay
learning_rate_schedule_polynomial_decay <-
function (initial_learning_rate, decay_steps, end_learning_rate = 1e-04,
    power = 1, cycle = FALSE, name = "PolynomialDecay")
{
    args <- capture_args(list(decay_steps = as_integer))
    do.call(keras$optimizers$schedules$PolynomialDecay, args)
}
