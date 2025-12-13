# Configure a model for training.

Configure a model for training.

## Usage

``` r
# S3 method for class 'keras.src.models.model.Model'
compile(
  object,
  optimizer = "rmsprop",
  loss = NULL,
  metrics = NULL,
  ...,
  loss_weights = NULL,
  weighted_metrics = NULL,
  run_eagerly = FALSE,
  steps_per_execution = 1L,
  jit_compile = "auto",
  auto_scale_loss = TRUE
)
```

## Arguments

- object:

  Keras model object

- optimizer:

  String (name of optimizer) or optimizer instance. See `optimizer_*`
  family.

- loss:

  Loss function. May be:

  - a string (name of builtin loss function),

  - a custom function, or

  - a [`Loss`](https://keras3.posit.co/dev/reference/Loss.md) instance
    (returned by the `loss_*` family of functions).

  A loss function is any callable with the signature
  `loss = fn(y_true, y_pred)`, where `y_true` are the ground truth
  values, and `y_pred` are the model's predictions. `y_true` should have
  shape `(batch_size, d1, .. dN)` (except in the case of sparse loss
  functions such as sparse categorical crossentropy which expects
  integer arrays of shape `(batch_size, d1, .. dN-1)`). `y_pred` should
  have shape `(batch_size, d1, .. dN)`. The loss function should return
  a float tensor.

- metrics:

  List of metrics to be evaluated by the model during training and
  testing. Each of these can be:

  - a string (name of a built-in function),

  - a function, optionally with a `"name"` attribute or

  - a [`Metric()`](https://keras3.posit.co/dev/reference/Metric.md)
    instance. See the `metric_*` family of functions.

  Typically you will use `metrics = c('accuracy')`. A function is any
  callable with the signature `result = fn(y_true, y_pred)`. To specify
  different metrics for different outputs of a multi-output model, you
  could also pass a named list, such as
  `metrics = list(a = 'accuracy', b = c('accuracy', 'mse'))`. You can
  also pass a list to specify a metric or a list of metrics for each
  output, such as `metrics = list(c('accuracy'), c('accuracy', 'mse'))`
  or `metrics = list('accuracy', c('accuracy', 'mse'))`. When you pass
  the strings `'accuracy'` or `'acc'`, we convert this to one of
  [`metric_binary_accuracy()`](https://keras3.posit.co/dev/reference/metric_binary_accuracy.md),
  [`metric_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_categorical_accuracy.md),
  [`metric_sparse_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_accuracy.md)
  based on the shapes of the targets and of the model output. A similar
  conversion is done for the strings `"crossentropy"` and `"ce"` as
  well. The metrics passed here are evaluated without sample weighting;
  if you would like sample weighting to apply, you can specify your
  metrics via the `weighted_metrics` argument instead.

  If providing an anonymous R function, you can customize the printed
  name during training by assigning
  `attr(<fn>, "name") <- "my_custom_metric_name"`, or by calling
  [`custom_metric("my_custom_metric_name", <fn>)`](https://keras3.posit.co/dev/reference/custom_metric.md)

- ...:

  Additional arguments passed on to the
  [`compile()`](https://generics.r-lib.org/reference/compile.html) model
  method.

- loss_weights:

  Optional list (named or unnamed) specifying scalar coefficients (R
  numerics) to weight the loss contributions of different model outputs.
  The loss value that will be minimized by the model will then be the
  *weighted sum* of all individual losses, weighted by the
  `loss_weights` coefficients. If an unnamed list, it is expected to
  have a 1:1 mapping to the model's outputs. If a named list, it is
  expected to map output names (strings) to scalar coefficients.

- weighted_metrics:

  List of metrics to be evaluated and weighted by `sample_weight` or
  `class_weight` during training and testing.

- run_eagerly:

  Bool. If `TRUE`, this model's forward pass will never be compiled. It
  is recommended to leave this as `FALSE` when training (for best
  performance), and to set it to `TRUE` when debugging.

- steps_per_execution:

  Int. The number of batches to run during each a single compiled
  function call. Running multiple batches inside a single compiled
  function call can greatly improve performance on TPUs or small models
  with a large R/Python overhead. At most, one full epoch will be run
  each execution. If a number larger than the size of the epoch is
  passed, the execution will be truncated to the size of the epoch. Note
  that if `steps_per_execution` is set to `N`, `Callback$on_batch_begin`
  and `Callback$on_batch_end` methods will only be called every `N`
  batches (i.e. before/after each compiled function execution). Not
  supported with the PyTorch backend.

- jit_compile:

  Bool or `"auto"`. Whether to use XLA compilation when compiling a
  model. For `jax` and `tensorflow` backends, `jit_compile="auto"`
  enables XLA compilation if the model supports it, and disabled
  otherwise. For `torch` backend, `"auto"` will default to eager
  execution and `jit_compile=True` will run with `torch.compile` with
  the `"inductor"` backend.

- auto_scale_loss:

  Bool. If `TRUE` and the model dtype policy is `"mixed_float16"`, the
  passed optimizer will be automatically wrapped in a
  `LossScaleOptimizer`, which will dynamically scale the loss to prevent
  underflow.

## Value

This is called primarily for the side effect of modifying `object`
in-place. The first argument `object` is also returned, invisibly, to
enable usage with the pipe.

## Examples

    model |> compile(
      optimizer = optimizer_adam(learning_rate = 1e-3),
      loss = loss_binary_crossentropy(),
      metrics = c(metric_binary_accuracy(),
                  metric_false_negatives())
    )

## See also

- <https://keras.io/api/models/model_training_apis#compile-method>

Other model training:  
[`evaluate.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/evaluate.keras.src.models.model.Model.md)  
[`predict.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/predict.keras.src.models.model.Model.md)  
[`predict_on_batch()`](https://keras3.posit.co/dev/reference/predict_on_batch.md)  
[`test_on_batch()`](https://keras3.posit.co/dev/reference/test_on_batch.md)  
[`train_on_batch()`](https://keras3.posit.co/dev/reference/train_on_batch.md)  
