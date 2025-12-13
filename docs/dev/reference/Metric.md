# Subclass the base `Metric` class

A `Metric` object encapsulates metric logic and state that can be used
to track model performance during training. It is what is returned by
the family of metric functions that start with prefix `metric_*`, as
well as what is returned by custom metrics defined with `Metric()`.

## Usage

``` r
Metric(
  classname,
  initialize = NULL,
  update_state = NULL,
  result = NULL,
  ...,
  public = list(),
  private = list(),
  inherit = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- classname:

  String, the name of the custom class. (Conventionally, CamelCase).

- initialize, update_state, result:

  Recommended methods to implement. See description section.

- ..., public:

  Additional methods or public members of the custom class.

- private:

  Named list of R objects (typically, functions) to include in instance
  private environments. `private` methods will have all the same symbols
  in scope as public methods (See section "Symbols in Scope"). Each
  instance will have it's own `private` environment. Any objects in
  `private` will be invisible from the Keras framework and the Python
  runtime.

- inherit:

  What the custom class will subclass. By default, the base keras class.

- parent_env:

  The R environment that all class methods will have as a grandparent.

## Value

A function that returns `Metric` instances, similar to the builtin
metric functions.

## Examples

### Usage with [`compile()`](https://generics.r-lib.org/reference/compile.html):

    model |> compile(
      optimizer = 'sgd',
      loss = 'mse',
      metrics = c(metric_SOME_METRIC(), metric_SOME_OTHER_METRIC())
    )

### Standalone usage:

    m <- metric_SOME_METRIC()
    for (e in seq(epochs)) {
      for (i in seq(train_steps)) {
        c(y_true, y_pred, sample_weight = NULL) %<-% ...
        m$update_state(y_true, y_pred, sample_weight)
      }
      cat('Final epoch result: ', as.numeric(m$result()), "\n")
      m$reset_state()
    }

## Full Examples

### Usage with [`compile()`](https://generics.r-lib.org/reference/compile.html):

    model <- keras_model_sequential()
    model |>
      layer_dense(64, activation = "relu") |>
      layer_dense(64, activation = "relu") |>
      layer_dense(10, activation = "softmax")
    model |>
      compile(optimizer = optimizer_rmsprop(0.01),
              loss = loss_categorical_crossentropy(),
              metrics = metric_categorical_accuracy())

    data <- random_uniform(c(1000, 32))
    labels <- random_uniform(c(1000, 10))

    model |> fit(data, labels, verbose = 0)

To be implemented by subclasses (custom metrics):

- `initialize()`: All state variables should be created in this method
  by calling `self$add_variable()` like:
  `self$var <- self$add_variable(...)`.

- `update_state()`: Updates all the state variables like:
  `self$var$assign(...)`.

- `result()`: Computes and returns a scalar value or a named list of
  scalar values for the metric from the state variables.

Example subclass implementation:

    metric_binary_true_positives <- Metric(
      classname = "BinaryTruePositives",

      initialize = function(name = 'binary_true_positives', ...) {
        super$initialize(name = name, ...)
        self$true_positives <-
          self$add_weight(shape = shape(),
                          initializer = 'zeros',
                          name = 'true_positives')
      },

      update_state = function(y_true, y_pred, sample_weight = NULL) {
        y_true <- op_cast(y_true, "bool")
        y_pred <- op_cast(y_pred, "bool")

        values <- y_true & y_pred # `&` calls op_logical_and()
        values <- op_cast(values, self$dtype)
        if (!is.null(sample_weight)) {
          sample_weight <- op_cast(sample_weight, self$dtype)
          sample_weight <- op_broadcast_to(sample_weight, shape(values))
          values <- values * sample_weight # `*` calls op_multiply()
        }
        self$true_positives$assign(self$true_positives + op_sum(values))
      },

      result = function() {
        self$true_positives
      }
    )
    model <- keras_model_sequential(input_shape = 32) |> layer_dense(10)
    model |> compile(loss = loss_binary_crossentropy(),
                     metrics = list(metric_binary_true_positives()))
    model |> fit(data, labels, verbose = 0)

## Methods defined by the base `Metric` class:

- __call__(...)

  Calling a metric instance self like `m(...)` is equivalent to calling:

      function(...) {
        m$update_state(...)
        m$result()
      }

- initialize(dtype=NULL, name=NULL)

  Initialize self.

  Args:

  - `name`: Optional name for the metric instance.

  - `dtype`: The dtype of the metric's computations. Defaults to `NULL`,
    which means using
    [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md).
    [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)
    is a `"float32"` unless set to different value (via
    [`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)).
    If a `keras$DTypePolicy` is provided, then the `compute_dtype` will
    be utilized.

- add_variable(shape, initializer, dtype=NULL, aggregation = 'sum', name=NULL)

- add_weight(shape=shape(), initializer=NULL, dtype=NULL, name=NULL)

- get_config()

  Return the serializable config of the metric.

- reset_state()

  Reset all of the metric state variables.

  This function is called between epochs/steps, when a metric is
  evaluated during training.

- result()

  Compute the current metric value.

  Returns: A scalar tensor, or a named list of scalar tensors.

- stateless_result(metric_variables)

- stateless_reset_state()

- stateless_update_state(metric_variables, ...)

- update_state(...)

  Accumulate statistics for the metric.

## Readonly properties

- `dtype`

- `variables`

## Symbols in scope

All R function custom methods (public and private) will have the
following symbols in scope:

- `self`: The custom class instance.

- `super`: The custom class superclass.

- `private`: An R environment specific to the class instance. Any
  objects assigned here are invisible to the Keras framework.

- `__class__` and `as.symbol(classname)`: the custom class type object.

## See also

Other metrics:  
[`custom_metric()`](https://keras3.posit.co/dev/reference/custom_metric.md)  
[`metric_auc()`](https://keras3.posit.co/dev/reference/metric_auc.md)  
[`metric_binary_accuracy()`](https://keras3.posit.co/dev/reference/metric_binary_accuracy.md)  
[`metric_binary_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_crossentropy.md)  
[`metric_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_focal_crossentropy.md)  
[`metric_binary_iou()`](https://keras3.posit.co/dev/reference/metric_binary_iou.md)  
[`metric_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_categorical_accuracy.md)  
[`metric_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_crossentropy.md)  
[`metric_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_focal_crossentropy.md)  
[`metric_categorical_hinge()`](https://keras3.posit.co/dev/reference/metric_categorical_hinge.md)  
[`metric_concordance_correlation()`](https://keras3.posit.co/dev/reference/metric_concordance_correlation.md)  
[`metric_cosine_similarity()`](https://keras3.posit.co/dev/reference/metric_cosine_similarity.md)  
[`metric_f1_score()`](https://keras3.posit.co/dev/reference/metric_f1_score.md)  
[`metric_false_negatives()`](https://keras3.posit.co/dev/reference/metric_false_negatives.md)  
[`metric_false_positives()`](https://keras3.posit.co/dev/reference/metric_false_positives.md)  
[`metric_fbeta_score()`](https://keras3.posit.co/dev/reference/metric_fbeta_score.md)  
[`metric_hinge()`](https://keras3.posit.co/dev/reference/metric_hinge.md)  
[`metric_huber()`](https://keras3.posit.co/dev/reference/metric_huber.md)  
[`metric_iou()`](https://keras3.posit.co/dev/reference/metric_iou.md)  
[`metric_kl_divergence()`](https://keras3.posit.co/dev/reference/metric_kl_divergence.md)  
[`metric_log_cosh()`](https://keras3.posit.co/dev/reference/metric_log_cosh.md)  
[`metric_log_cosh_error()`](https://keras3.posit.co/dev/reference/metric_log_cosh_error.md)  
[`metric_mean()`](https://keras3.posit.co/dev/reference/metric_mean.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_iou()`](https://keras3.posit.co/dev/reference/metric_mean_iou.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_mean_wrapper()`](https://keras3.posit.co/dev/reference/metric_mean_wrapper.md)  
[`metric_one_hot_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_iou.md)  
[`metric_one_hot_mean_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_mean_iou.md)  
[`metric_pearson_correlation()`](https://keras3.posit.co/dev/reference/metric_pearson_correlation.md)  
[`metric_poisson()`](https://keras3.posit.co/dev/reference/metric_poisson.md)  
[`metric_precision()`](https://keras3.posit.co/dev/reference/metric_precision.md)  
[`metric_precision_at_recall()`](https://keras3.posit.co/dev/reference/metric_precision_at_recall.md)  
[`metric_r2_score()`](https://keras3.posit.co/dev/reference/metric_r2_score.md)  
[`metric_recall()`](https://keras3.posit.co/dev/reference/metric_recall.md)  
[`metric_recall_at_precision()`](https://keras3.posit.co/dev/reference/metric_recall_at_precision.md)  
[`metric_root_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_root_mean_squared_error.md)  
[`metric_sensitivity_at_specificity()`](https://keras3.posit.co/dev/reference/metric_sensitivity_at_specificity.md)  
[`metric_sparse_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_accuracy.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_sparse_top_k_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_sparse_top_k_categorical_accuracy.md)  
[`metric_specificity_at_sensitivity()`](https://keras3.posit.co/dev/reference/metric_specificity_at_sensitivity.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/dev/reference/metric_squared_hinge.md)  
[`metric_sum()`](https://keras3.posit.co/dev/reference/metric_sum.md)  
[`metric_top_k_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_top_k_categorical_accuracy.md)  
[`metric_true_negatives()`](https://keras3.posit.co/dev/reference/metric_true_negatives.md)  
[`metric_true_positives()`](https://keras3.posit.co/dev/reference/metric_true_positives.md)  
