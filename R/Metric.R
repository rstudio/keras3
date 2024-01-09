
#' Define a custom Metric
#'
#' A `Metric` object encapsulates metric logic and state that can be used to
#' track model performance during training. It is what is returned by the family
#' of metric functions that start with prefix `metric_*`, as well as what is
#' returned by custom metrics defined with `Metric()`.
#'
#' # Examples
#'
#' ## Usage with `compile()`:
#' ```r
#' model |> compile(
#'   optimizer = 'sgd',
#'   loss = 'mse',
#'   metrics = list(metric_SOME_METRIC(), metric_SOME_OTHER_METRIC())
#' )
#' ```
#'
#' ## Standalone usage:
#' ```r
#' m <- metric_SOME_METRIC()
#' for (e in seq(epochs)) {
#'   for (i in seq(train_steps)) {
#'     c(y_true, y_pred, sample_weight = NULL) %<-% ...
#'     m$update_state(y_true, y_pred, sample_weight)
#'   }
#'   cat('Final epoch result: ', as.numeric(m$result()), "\n")
#'   m$reset_state()
#' }
#' ```
#'
#' # Full Examples
#'
#' ## Usage with `compile()`:
#' ```{r}
#' model <- keras_model_sequential()
#' model |>
#'   layer_dense(64, activation = "relu") |>
#'   layer_dense(64, activation = "relu") |>
#'   layer_dense(10, activation = "softmax")
#' model |>
#'   compile(optimizer = optimizer_rmsprop(0.01),
#'           loss = loss_categorical_crossentropy(),
#'           metrics = metric_categorical_accuracy())
#'
#' data <- random_uniform(c(1000, 32))
#' labels <- random_uniform(c(1000, 10))
#'
#' model |> fit(data, labels, verbose = 0)
#' ```
#'
#' To be implemented by subclasses (custom metrics):
#'
#' * `initialize()`: All state variables should be created in this method by
#'   calling `self$add_variable()` like: `self$var <- self$add_variable(...)`.
#' * `update_state()`: Updates all the state variables like:
#'   `self$var$assign(...)`.
#' * `result()`: Computes and returns a scalar value or a named list of scalar values
#'   for the metric from the state variables.
#'
#' Example subclass implementation:
#'
#' ```{r}
#' metric_binary_true_positives <- Metric(
#'   classname = "BinaryTruePositives",
#'
#'   initialize = function(name = 'binary_true_positives', ...) {
#'     super$initialize(name = name, ...)
#'     self$true_positives <-
#'       self$add_weight(shape = shape(),
#'                       initializer = 'zeros',
#'                       name = 'true_positives')
#'   },
#'
#'   update_state = function(y_true, y_pred, sample_weight = NULL) {
#'     y_true <- op_cast(y_true, "bool")
#'     y_pred <- op_cast(y_pred, "bool")
#'
#'     values <- y_true & y_pred # `&` calls op_logical_and()
#'     values <- op_cast(values, self$dtype)
#'     if (!is.null(sample_weight)) {
#'       sample_weight <- op_cast(sample_weight, self$dtype)
#'       sample_weight <- op_broadcast_to(sample_weight, shape(values))
#'       values <- values * sample_weight # `*` calls op_multiply()
#'     }
#'     self$true_positives$assign(self$true_positives + op_sum(values))
#'   },
#'
#'   result = function() {
#'     self$true_positives
#'   }
#' )
#' model <- keras_model_sequential(input_shape = 32) |> layer_dense(10)
#' model |> compile(loss = loss_binary_crossentropy(),
#'                  metrics = list(metric_binary_true_positives()))
#' model |> fit(data, labels, verbose = 0)
#' ```
#'
#' # Methods defined by the base `Metric` class:
#'
#' * ```
#'   __call__(...)
#'   ````
#'   Calling a metric instance self like `m(...)` is equivalent to calling:
#'   ```r
#'   function(...) {
#'     m$update_state(...)
#'     m$result()
#'   }
#'   ```
#'
#' * ```r
#'   initialize(dtype=NULL, name=NULL)
#'   ```
#'   Initialize self.
#'
#'   Args:
#'   * `name`: (Optional) string name of the metric instance.
#'   * `dtype`: (Optional) data type of the metric result.
#'
#' * ```r
#'   add_variable(shape, initializer, dtype=NULL, name=NULL)
#'   ```
#'
#' * ```r
#'   add_weight(shape=shape(), initializer=NULL, dtype=NULL, name=NULL)
#'   ```
#'
#' * ```r
#'   get_config()
#'   ```
#'   Return the serializable config of the metric.
#'
#' * ```r
#'   reset_state()
#'   ```
#'   Reset all of the metric state variables.
#'
#'   This function is called between epochs/steps,
#'   when a metric is evaluated during training.
#'
#' * ```r
#'   result()
#'   ```
#'   Compute the current metric value.
#'
#'   Returns:
#'   A scalar tensor, or a named list of scalar tensors.
#'
#' * ```r
#'   stateless_result(metric_variables)
#'   ```
#' * ```r
#'   stateless_update_state(metric_variables, ...)
#'   ```
#' * ```r
#'   update_state(...)
#'   ```
#'   Accumulate statistics for the metric.
#'
#' # Readonly properties
#'
#' * `dtype`
#'
#' * `variables`
#'
#' @inheritSection Layer Symbols in scope
#' @inheritParams Layer
#' @export
#' @family metrics
#' @tether keras.metrics.Metric
Metric <-
function(classname,
         initialize = NULL,
         update_state = NULL,
         result = NULL,
         ...,
         public = list(),
         private = list(),
         inherit = NULL,
         parent_env = parent.frame()) {

  members <- drop_nulls(named_list(initialize, update_state, result))
  members <- modifyList(members, list2(...), keep.null = TRUE)
  members <- modifyList(members, public, keep.null = TRUE)

  members <- modify_intersection(members, list(
    from_config = function(x) decorate_method(x, "classmethod")
  ))

  inherit <- substitute(inherit) %||%
    quote(base::asNamespace("keras3")$keras$Metric)

  new_wrapped_py_class(
    classname = classname,
    members = members,
    inherit = inherit,
    parent_env = parent_env,
    private = private
  )
}
