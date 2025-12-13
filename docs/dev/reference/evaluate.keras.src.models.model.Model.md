# Evaluate a Keras Model

This functions returns the loss value and metrics values for the model
in test mode. Computation is done in batches (see the `batch_size` arg.)

## Usage

``` r
# S3 method for class 'keras.src.models.model.Model'
evaluate(
  object,
  x = NULL,
  y = NULL,
  ...,
  batch_size = NULL,
  verbose = getOption("keras.verbose", default = "auto"),
  sample_weight = NULL,
  steps = NULL,
  callbacks = NULL
)
```

## Arguments

- object:

  Keras model object

- x:

  Input data. It can be:

  - An R array (or array-like), or a list of arrays (in case the model
    has multiple inputs).

  - A backend-native tensor, or a list of tensors (in case the model has
    multiple inputs).

  - A named list mapping input names to the corresponding array/tensors,
    if the model has named inputs.

  - A `tf.data.Dataset`. Should return a tuple of either
    `(inputs, targets)` or `(inputs, targets, sample_weights)`.

  - A generator returning `(inputs, targets)` or
    `(inputs, targets, sample_weights)`.

- y:

  Target data. Like the input data `x`, it could be either R array(s) or
  backend-native tensor(s). If `x` is a `tf.data.Dataset` or generator
  function, `y` should not be specified (since targets will be obtained
  from the iterator/dataset).

- ...:

  For forward/backward compatability.

- batch_size:

  Integer or `NULL`. Number of samples per batch of computation. If
  unspecified, `batch_size` will default to `32`. Do not specify the
  `batch_size` if your data is in the form of a a tf dataset or
  generator (since they generate batches).

- verbose:

  `"auto"`, `0`, `1`, or `2`. Verbosity mode. `0` = silent, `1` =
  progress bar, `2` = single line. `"auto"` becomes `1` for most cases,
  `2` if in a knitr render or running on a distributed training server.
  Note that the progress bar is not particularly useful when logged to a
  file, so `verbose=2` is recommended when not running interactively
  (e.g. in a production environment). Defaults to `"auto"`.

- sample_weight:

  Optional array or tensor of weights for the training samples, used for
  weighting the loss function (during training only). You can either
  pass a flat (1D) array or tensor with the same length as the input
  samples (1:1 mapping between weights and samples), or in the case of
  temporal data, you can pass a 2D array or tensor with shape
  `(samples, sequence_length)` to apply a different weight to every
  timestep of every sample. This argument is not supported when `x` is a
  `tf.data.Dataset`, or Python generator function. Instead, provide
  `sample_weights` as the third element of `x`. Note that sample
  weighting does not apply to metrics specified via the `metrics`
  argument in
  [`compile()`](https://generics.r-lib.org/reference/compile.html). To
  apply sample weighting to your metrics, you can specify them via the
  `weighted_metrics` in
  [`compile()`](https://generics.r-lib.org/reference/compile.html)
  instead.

- steps:

  Integer or `NULL`. Total number of steps (batches of samples) before
  declaring the evaluation round finished. Ignored with the default
  value of `NULL`. If `x` is a `tf.data.Dataset` and `steps` is `NULL`,
  evaluation will run until the dataset is exhausted. In the case of an
  infinitely repeating dataset, it will run indefinitely.

- callbacks:

  List of `Callback` instances. List of callbacks to apply during
  evaluation.

## Value

Scalar test loss (if the model has a single output and no metrics) or
list of scalars (if the model has multiple outputs and/or metrics). The
attribute `model$metrics_names` will give you the display labels for the
scalar outputs.

## See also

- <https://keras.io/api/models/model_training_apis#evaluate-method>

Other model training:  
[`compile.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/compile.keras.src.models.model.Model.md)  
[`predict.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/predict.keras.src.models.model.Model.md)  
[`predict_on_batch()`](https://keras3.posit.co/dev/reference/predict_on_batch.md)  
[`test_on_batch()`](https://keras3.posit.co/dev/reference/test_on_batch.md)  
[`train_on_batch()`](https://keras3.posit.co/dev/reference/train_on_batch.md)  
