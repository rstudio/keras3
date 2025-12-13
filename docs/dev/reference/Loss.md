# Subclass the base `Loss` class

Use this to define a custom loss class. Note, in most cases you do not
need to subclass `Loss` to define a custom loss: you can also pass a
bare R function, or a named R function defined with
[`custom_metric()`](https://keras3.posit.co/dev/reference/custom_metric.md),
as a loss function to
[`compile()`](https://generics.r-lib.org/reference/compile.html).

## Usage

``` r
Loss(
  classname,
  call = NULL,
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

- call:

      function(y_true, y_pred)

  Method to be implemented by subclasses: Function that contains the
  logic for loss calculation using `y_true`, `y_pred`.

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

A function that returns `Loss` instances, similar to the builtin loss
functions.

## Details

Example subclass implementation:

    loss_custom_mse <- Loss(
      classname = "CustomMeanSquaredError",
      call = function(y_true, y_pred) {
        op_mean(op_square(y_pred - y_true), axis = -1)
      }
    )

    # Usage in compile()
    model <- keras_model_sequential(input_shape = 10) |> layer_dense(10)
    model |> compile(loss = loss_custom_mse())

    # Standalone usage
    mse <- loss_custom_mse(name = "my_custom_mse_instance")

    y_true <- op_arange(20) |> op_reshape(c(4, 5))
    y_pred <- op_arange(20) |> op_reshape(c(4, 5)) * 2
    (loss <- mse(y_true, y_pred))

    ## tf.Tensor(143.5, shape=(), dtype=float32)

    loss2 <- (y_pred - y_true)^2 |>
      op_mean(axis = -1) |>
      op_mean()

    stopifnot(all.equal(as.array(loss), as.array(loss2)))

    sample_weight <-array(c(.25, .25, 1, 1))
    (weighted_loss <- mse(y_true, y_pred, sample_weight = sample_weight))

    ## tf.Tensor(129.0625, shape=(), dtype=float32)

    weighted_loss2 <- (y_true - y_pred)^2 |>
      op_mean(axis = -1) |>
      op_multiply(sample_weight) |>
      op_mean()

    stopifnot(all.equal(as.array(weighted_loss),
                        as.array(weighted_loss2)))

## Methods defined by base `Loss` class:

- initialize(name=NULL, reduction="sum_over_batch_size", dtype=NULL)

  Args:

  - `name`: Optional name for the loss instance.

  - `reduction`: Type of reduction to apply to the loss. In almost all
    cases this should be `"sum_over_batch_size"`. Supported options are
    `"sum"`, `"sum_over_batch_size"`, `"mean"`,
    `"mean_with_sample_weight"` or `NULL`. `"sum"` sums the loss,
    `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
    sample size, and `"mean_with_sample_weight"` sums the loss and
    divides by the sum of the sample weights. `"none"` and `NULL`
    perform no aggregation. Defaults to `"sum_over_batch_size"`.

  - `dtype`: The dtype of the loss's computations. Defaults to `NULL`,
    which means using
    [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md).
    [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)
    is a `"float32"` unless set to different value (via
    [`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)).
    If a `keras$DTypePolicy` is provided, then the `compute_dtype` will
    be utilized.

- __call__(y_true, y_pred, sample_weight=NULL)

  Call the loss instance as a function, optionally with `sample_weight`.

- get_config()

## Readonly properties:

- dtype

## Symbols in scope

All R function custom methods (public and private) will have the
following symbols in scope:

- `self`: The custom class instance.

- `super`: The custom class superclass.

- `private`: An R environment specific to the class instance. Any
  objects assigned here are invisible to the Keras framework.

- `__class__` and `as.symbol(classname)`: the custom class type object.

## See also

Other losses:  
[`loss_binary_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_crossentropy.md)  
[`loss_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_focal_crossentropy.md)  
[`loss_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_crossentropy.md)  
[`loss_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_focal_crossentropy.md)  
[`loss_categorical_generalized_cross_entropy()`](https://keras3.posit.co/dev/reference/loss_categorical_generalized_cross_entropy.md)  
[`loss_categorical_hinge()`](https://keras3.posit.co/dev/reference/loss_categorical_hinge.md)  
[`loss_circle()`](https://keras3.posit.co/dev/reference/loss_circle.md)  
[`loss_cosine_similarity()`](https://keras3.posit.co/dev/reference/loss_cosine_similarity.md)  
[`loss_ctc()`](https://keras3.posit.co/dev/reference/loss_ctc.md)  
[`loss_dice()`](https://keras3.posit.co/dev/reference/loss_dice.md)  
[`loss_hinge()`](https://keras3.posit.co/dev/reference/loss_hinge.md)  
[`loss_huber()`](https://keras3.posit.co/dev/reference/loss_huber.md)  
[`loss_kl_divergence()`](https://keras3.posit.co/dev/reference/loss_kl_divergence.md)  
[`loss_log_cosh()`](https://keras3.posit.co/dev/reference/loss_log_cosh.md)  
[`loss_mean_absolute_error()`](https://keras3.posit.co/dev/reference/loss_mean_absolute_error.md)  
[`loss_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/loss_mean_absolute_percentage_error.md)  
[`loss_mean_squared_error()`](https://keras3.posit.co/dev/reference/loss_mean_squared_error.md)  
[`loss_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/loss_mean_squared_logarithmic_error.md)  
[`loss_poisson()`](https://keras3.posit.co/dev/reference/loss_poisson.md)  
[`loss_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/loss_sparse_categorical_crossentropy.md)  
[`loss_squared_hinge()`](https://keras3.posit.co/dev/reference/loss_squared_hinge.md)  
[`loss_tversky()`](https://keras3.posit.co/dev/reference/loss_tversky.md)  
[`metric_binary_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_crossentropy.md)  
[`metric_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_focal_crossentropy.md)  
[`metric_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_crossentropy.md)  
[`metric_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_focal_crossentropy.md)  
[`metric_categorical_hinge()`](https://keras3.posit.co/dev/reference/metric_categorical_hinge.md)  
[`metric_hinge()`](https://keras3.posit.co/dev/reference/metric_hinge.md)  
[`metric_huber()`](https://keras3.posit.co/dev/reference/metric_huber.md)  
[`metric_kl_divergence()`](https://keras3.posit.co/dev/reference/metric_kl_divergence.md)  
[`metric_log_cosh()`](https://keras3.posit.co/dev/reference/metric_log_cosh.md)  
[`metric_mean_absolute_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_error.md)  
[`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_percentage_error.md)  
[`metric_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_error.md)  
[`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_logarithmic_error.md)  
[`metric_poisson()`](https://keras3.posit.co/dev/reference/metric_poisson.md)  
[`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_crossentropy.md)  
[`metric_squared_hinge()`](https://keras3.posit.co/dev/reference/metric_squared_hinge.md)  
