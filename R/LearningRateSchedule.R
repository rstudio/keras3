#' Define a custom `LearningRateSchedule` class
#'
#' @description
#' Subclass the keras learning rate schedule base class.
#'
#' You can use a learning rate schedule to modulate how the learning rate
#' of your optimizer changes over time.
#'
#' Several built-in learning rate schedules are available, such as
#' [`learning_rate_schedule_exponential_decay()`] or
#' [`learning_rate_schedule_piecewise_constant_decay()`]:
#'
#' ```{r}
#' lr_schedule <- learning_rate_schedule_exponential_decay(
#'   initial_learning_rate = 1e-2,
#'   decay_steps = 10000,
#'   decay_rate = 0.9
#' )
#' optimizer <- optimizer_sgd(learning_rate = lr_schedule)
#' ```
#'
#' A `LearningRateSchedule()` instance can be passed in as the `learning_rate`
#' argument of any optimizer.
#'
#' To implement your own schedule object, you should implement the `call`
#' method, which takes a `step` argument (a scalar integer backend tensor, the
#' current training step count).
#' Note that `step` is 0-based (i.e., the first step is `0`).
#' Like for any other Keras object, you can also optionally
#' make your object serializable by implementing the `get_config()`
#' and `from_config()` methods.
#'
#' # Example
#'
#' ```{r}
#' my_custom_learning_rate_schedule <- LearningRateSchedule(
#'   classname = "MyLRSchedule",
#'   initialize = function( initial_learning_rate) {
#'
#'     self$initial_learning_rate <- initial_learning_rate
#'   },
#'
#'
#'   call = function(step) {
#'     # note that `step` is a tensor
#'     # and call() will be traced via tf_function() or similar.
#'
#'     str(step) # <KerasVariable shape=(), dtype=int64, path=SGD/iteration>
#'
#'
#'     # print 'step' every 1000 steps
#'     op_cond((step %% 1000) == 0,
#'             \() {tensorflow::tf$print(step); NULL},
#'             \() {NULL})
#'     self$initial_learning_rate / (step + 1)
#'   }
#' )
#'
#' optimizer <- optimizer_sgd(
#'   learning_rate = my_custom_learning_rate_schedule(0.1)
#' )
#'
#' # You can also call schedule instances directly
#' # (e.g., for interactive testing, or if implementing a custom optimizer)
#' schedule <- my_custom_learning_rate_schedule(0.1)
#' step <- keras$Variable(initializer = op_ones,
#'                        shape = shape(),
#'                        dtype = "int64")
#' schedule(step)
#' ```
#'
#' # Methods available:
#'
#' * ```
#'   get_config()
#'   ```
#'
#' @param call,initialize,get_config
#' Recommended methods to implement. See description and details sections.
#' @returns A function that returns `LearningRateSchedule` instances, similar to the
#'   built-in `learning_rate_schedule_*` family of functions.
#' @tether keras.optimizers.schedules.LearningRateSchedule
#' @inheritSection Layer Symbols in scope
#' @inheritParams Layer
#' @family optimizer learning rate schedules
LearningRateSchedule <- function(classname,
                                 call = NULL,
                                 initialize = NULL,
                                 get_config = NULL,
                                 ...,
                                 public = list(),
                                 private = list(),
                                 inherit = NULL,
                                 parent_env = parent.frame()) {

  members <- drop_nulls(named_list(initialize, call, get_config))
  members <- modifyList(members, list2(...), keep.null = TRUE)
  members <- modifyList(members, public, keep.null = TRUE)

  members <- rename(members, "__call__" = "call",
                    .skip_existing = TRUE)

  members <- modify_intersection(members, list(
    from_config = function(x) decorate_method(x, "classmethod")
  ))

  inherit <- substitute(inherit) %||%
    quote(base::asNamespace("keras3")$keras$optimizers$schedules$LearningRateSchedule)

  new_wrapped_py_class(
    classname = classname,
    members = members,
    inherit = inherit,
    parent_env = parent_env,
    private = private
  )

}

# TODO: should all optimizer accept a plain R function to `learning_rate`?
